"""
Core of the parity harness: paths, metrics, and artifact writing.

The harness has exactly one job - turn a run of the *reference* implementation
into a file that a machine with no Python can check the Swift port against.

Two roles, deliberately kept apart:

  generation    needs Python, MLX/torch, the reference code under Models/python
                and the weights. Runs here, rarely, on the one machine that has
                all of it.

  verification  needs Swift, the weights and the artifact. Runs anywhere,
                including CI, and never imports any of this.

The artifact carries the *input tensor*, not a path to a wav. Decoding the same
file through soundfile and through AVFoundation can differ in the last bit or
two, and a parity harness that reports those as model divergence is worse than
no harness at all. Swift reads float32 samples straight out of the artifact.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Parity/ -> AudioToolSwift/ -> checkout root
PARITY_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PARITY_DIR.parent
CHECKOUT_ROOT = PACKAGE_DIR.parent
ARTIFACT_DIR = PARITY_DIR / "artifacts"


# MARK: - Metrics


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Signal-to-noise ratio of `candidate` against `reference`, in dB.

    This is the metric the conversion work already used - `Models/python/
    frcrn_se_mlx/README.md` records 115.98 dB for MLX against PyTorch, and the
    CoreML GAN README records 23.8 dB - so numbers produced here are directly
    comparable to the ones you recorded then. float32 round-off alone lands
    somewhere around 110-130 dB; anything below ~60 dB is a real difference.

    Returns `inf` when the two are bit-identical.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    candidate = np.asarray(candidate, dtype=np.float64).ravel()
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    noise = float(np.sum((reference - candidate) ** 2))
    if noise == 0.0:
        return float("inf")
    signal = float(np.sum(reference**2))
    return 10.0 * np.log10(signal / noise)


def correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Pearson correlation, the second number the GAN README records."""
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(candidate, dtype=np.float64).ravel()
    if a.std() == 0.0 or b.std() == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def max_abs_diff(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


# MARK: - Cases


@dataclass
class ParityCase:
    """One model's reference run, ready to be written out.

    `tensors` must contain `input`; everything else is an output the Swift side
    will be held to. Multi-output models (SS, Demucs) name each one, so a
    failure says *which* stem or speaker moved.
    """

    name: str
    sample_rate: int
    tensors: dict[str, np.ndarray]
    weights: dict[str, Path] = field(default_factory=dict)
    source_audio: Path | None = None
    reference_files: list[Path] = field(default_factory=list)
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "input" not in self.tensors:
            raise ValueError(f"{self.name}: tensors must include 'input'")
        if len(self.tensors) < 2:
            raise ValueError(f"{self.name}: tensors must include at least one output")

        # Caught a real mistake the first time it ran: USS's test_speech.wav opens
        # on six seconds of silence, so the first four seconds produced an input at
        # 2e-5 RMS and two "separated" outputs that agreed with each other and with
        # nothing in particular. Refuse to write a case that cannot discriminate.
        level = _rms(self.tensors["input"])
        if level < MIN_INPUT_RMS:
            raise ValueError(
                f"{self.name}: input RMS is {level:.2e}, below {MIN_INPUT_RMS:.0e} - "
                "this window is effectively silent and would pass against anything. "
                "Use `offset` to take a window with signal in it."
            )
        for name, value in self.tensors.items():
            if name != "input" and _rms(value) == 0.0:
                raise ValueError(f"{self.name}: output '{name}' is all zeros")


@dataclass
class Context:
    """Where the generator finds things. All of it is outside the package."""

    checkout_root: Path = CHECKOUT_ROOT

    def reference(self, relative: str) -> Path:
        """A path under `Models/`, e.g. `"python/frcrn_se_mlx"`."""
        path = self.checkout_root / "Models" / relative
        if not path.exists():
            raise FileNotFoundError(
                f"reference not found: {path}\n"
                "Generation needs the research checkout; verification does not."
            )
        return path

    def fixture(self, relative: str) -> Path:
        """A committed test fixture, e.g. `"AudioToolFluidAudioTests/Fixtures/test.wav"`."""
        path = PACKAGE_DIR / "Tests" / relative
        if not path.exists():
            raise FileNotFoundError(f"fixture not found: {path}")
        return path

    @contextlib.contextmanager
    def on_path(self, directory: Path):
        """Import the reference modules without installing or copying them.

        They are written to be run from their own directory - `run_mlx.py` does
        `from frcrn_mlx import ...` - so this reproduces that rather than
        restructuring code that is frozen reference material.

        Modules imported from `directory` are dropped from `sys.modules` on the
        way out. `sys.path` scopes where an import *looks*, but `sys.modules` is
        global and is consulted first, so without this the second adapter to say
        `from generate import ...` silently receives the first one's module. That
        is not hypothetical: `mossformer2_se_mlx` and `mosforrmer2_ss_mlx` both
        ship a `generate`, and `--all` failed on
        `ImportError: cannot import name 'MODEL_CONFIGS' from 'generate'` naming
        the SE file, because SE runs first.
        """
        directory = str(directory)
        resolved = os.path.realpath(directory)
        previous_cwd = os.getcwd()
        before = set(sys.modules)
        sys.path.insert(0, directory)
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous_cwd)
            with contextlib.suppress(ValueError):
                sys.path.remove(directory)
            for name in set(sys.modules) - before:
                origin = getattr(sys.modules.get(name), "__file__", None)
                if not origin:
                    continue
                if os.path.realpath(origin).startswith(resolved + os.sep):
                    del sys.modules[name]


# MARK: - Audio


def load_audio(
    path: Path,
    *,
    mono: bool = True,
    seconds: float | None = None,
    offset: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Read a wav at its native rate. No resampling, ever - that is the point.

    `offset` exists because several of the reference clips open on several
    seconds of near-silence, and a window taken from there produces a case that
    passes against anything.

    Returns `(samples, sample_rate)`; shape is `(n,)` for mono and `(channels, n)`
    otherwise, matching what the reference runners expect.
    """
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.T  # (channels, n)
    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)
    start = int(round(offset * rate))
    stop = start + int(round(seconds * rate)) if seconds is not None else audio.shape[1]
    audio = audio[:, start:stop]
    if audio.shape[1] == 0:
        raise ValueError(f"{path.name}: offset {offset}s is past the end of the file")
    return (audio[0] if mono else audio), rate


# A case whose input is near-silent agrees with almost any implementation, so it
# proves nothing while looking green. Speech fixtures here sit around 0.1 RMS;
# the quietest legitimate window is two orders of magnitude above this floor.
MIN_INPUT_RMS = 1e-3


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


# MARK: - Chunking


def chunked(
    ctx: "Context",
    audio: np.ndarray,
    sample_rate: int,
    *,
    chunk_duration: float,
    overlap_ratio: float,
    strategy: str,
    process_fn,
) -> np.ndarray:
    """Run `process_fn` through `Models/python/benchmark_chunking.py`.

    Every one of these providers chunks above some duration, and the first draft
    of this harness compared a chunked Swift result against an unchunked
    reference. That is not a model comparison - it is a comparison of two
    different algorithms, and it read as 15-41 dB of "port error" that was
    nothing of the kind.

    `benchmark_chunking.process_with_chunking` is the code the Swift chunking was
    ported from - `MLXEnhancerProvider.swift` says so in as many words - so this
    reuses it rather than reimplementing the seams and hoping they line up.

    - Parameter strategy: `"discard_edges"`, `"triangular_blend"`, `"hann_blend"`,
      `"overlap_add"` or `"no_overlap"`, matching the provider's `ChunkingConfig`.
    """
    with ctx.on_path(ctx.reference("python")):
        from benchmark_chunking import ChunkingConfig, ChunkingStrategy, process_with_chunking

        config = ChunkingConfig(
            chunk_duration=chunk_duration,
            overlap_ratio=overlap_ratio,
            strategy=ChunkingStrategy(strategy),
        )
        result, _, _ = process_with_chunking(
            np.ascontiguousarray(audio, dtype=np.float32), sample_rate, config, process_fn
        )
    return np.asarray(result, dtype=np.float32)


# MARK: - Provenance


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(directory: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _portable(path: Path) -> str:
    """Path relative to the checkout, so artifacts carry no one's home directory.

    These get published alongside the weights; an absolute `/Users/...` in a
    public file is both noise and a small leak.
    """
    with contextlib.suppress(ValueError):
        return str(path.resolve().relative_to(CHECKOUT_ROOT))
    return path.name


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("mlx", "torch", "torchaudio", "numpy", "coremltools"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            pass
    with contextlib.suppress(Exception):
        import mlx.core as mx
        versions["mlx"] = mx.__version__
    return versions


# MARK: - Writing


def write(case: ParityCase, *, output_dir: Path = ARTIFACT_DIR) -> tuple[Path, Path]:
    """Write `<name>.safetensors` and its `<name>.json` sidecar.

    safetensors rather than npy because Swift already reads it - the
    verification side needs no new parser and no new dependency.
    """
    import mlx.core as mx

    output_dir.mkdir(parents=True, exist_ok=True)
    tensors_path = output_dir / f"{case.name}.safetensors"
    sidecar_path = output_dir / f"{case.name}.json"

    arrays = {k: mx.array(np.ascontiguousarray(v, dtype=np.float32)) for k, v in case.tensors.items()}
    mx.save_safetensors(str(tensors_path), arrays, metadata={"parity_case": case.name})

    sidecar = {
        "case": case.name,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference": "MLX Python (Models/python) - the implementation the Swift was ported from",
        "sample_rate": case.sample_rate,
        "notes": case.notes,
        "tensors": {
            name: {
                "shape": list(np.shape(value)),
                "dtype": "float32",
                "rms": float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2))),
            }
            for name, value in case.tensors.items()
        },
        "weights": {
            name: {"path": _portable(p), "sha256": sha256(p)} for name, p in case.weights.items()
        },
        "source_audio": (
            {"path": _portable(case.source_audio), "sha256": sha256(case.source_audio)}
            if case.source_audio else None
        ),
        "reference_files": [
            {"path": _portable(p), "sha256": sha256(p)} for p in case.reference_files
        ],
        "checkout_revision": _git_revision(PACKAGE_DIR),
        "versions": _versions(),
        "artifact_sha256": sha256(tensors_path),
        **case.extra,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    return tensors_path, sidecar_path
