"""
MossFormer2 speech enhancement, 48 kHz.

Reference: `Models/python/mossformer2_se_mlx/generate.py`.
Swift:     `Sources/Models/MossFormer2SE`, driven by `MLXEnhancerProvider`.

Calls the reference's own `load_model` and `decode_one_audio` rather than
reimplementing them - including the `nn.LayerNorm.__call__` monkeypatch, which
is part of how the reference computes, not an optimisation detail.

Weights come from the HF cache the reference already populates
(`starkdmi/MossFormer2_SE_48K_MLX`, fp32), so this uses the published bits
rather than a local copy that might have drifted from them.
"""

from __future__ import annotations

import numpy as np

from harness import Context, ParityCase, chunked, load_audio

NAME = "mossformer2_se_48k"
SAMPLE_RATE = 48_000
MAX_DIRECT_SECONDS = 4.0  # MossFormer2SE48KProvider.maxDirectDuration
DIRECT_SECONDS = 3.0      # comfortably under it, so both sides do one pass
CHUNK_DURATION = 4.0      # ChunkingConfig.mossformer2SE48K
OVERLAP_RATIO = 0.25
STRATEGY = "discard_edges"


def build(ctx: Context) -> list[ParityCase]:
    reference_dir = ctx.reference("python/mossformer2_se_mlx")
    source = ctx.fixture("AudioToolFluidAudioTests/Fixtures/test_48k.wav")

    audio, rate = load_audio(source, mono=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{NAME} is a {SAMPLE_RATE} Hz model; {source.name} is {rate} Hz")
    short = audio[: int(DIRECT_SECONDS * SAMPLE_RATE)]

    with ctx.on_path(reference_dir):
        from generate import MODEL_CONFIG, decode_one_audio, load_model

        model = load_model("fp32")

        def enhance(chunk: np.ndarray) -> np.ndarray:
            output = np.asarray(
                decode_one_audio(model, chunk.reshape(1, -1), MODEL_CONFIG), dtype=np.float32
            ).reshape(-1)
            # Whole STFT hops only, so a 192000-sample chunk returns 191, 999-ish.
            if len(output) < len(chunk):
                output = np.pad(output, (0, len(chunk) - len(output)))
            return output[: len(chunk)]

        direct = enhance(short)

    shared = {
        "source_audio": source,
        "reference_files": [
            reference_dir / "generate.py",
            reference_dir / "mossformer2_se_wrapper.py",
        ],
    }

    return [
        # Chunked against chunked: the path the provider takes for anything over
        # four seconds, which is most real audio.
        ParityCase(
            name=NAME,
            sample_rate=SAMPLE_RATE,
            tensors={
                "input": audio,
                "enhanced": chunked(
                    ctx, audio, SAMPLE_RATE,
                    chunk_duration=CHUNK_DURATION, overlap_ratio=OVERLAP_RATIO,
                    strategy=STRATEGY, process_fn=enhance,
                ),
            },
            notes=(
                f"Chunked at {CHUNK_DURATION}s / {OVERLAP_RATIO:.0%} / {STRATEGY}, matching "
                "ChunkingConfig.mossformer2SE48K. Input is already 48 kHz, so no "
                "resampler is inside this comparison."
            ),
            extra={
                "precision": "fp32", "repo": "starkdmi/MossFormer2_SE_48K_MLX",
                "chunk_duration": CHUNK_DURATION, "overlap_ratio": OVERLAP_RATIO,
                "strategy": STRATEGY,
            },
            **shared,
        ),
        # Whole file against whole file, under maxDirectDuration. Isolates the
        # model from the chunking wrapper: if this one is exact and the chunked
        # one is not, the seams are where to look.
        ParityCase(
            name=f"{NAME}_direct",
            sample_rate=SAMPLE_RATE,
            tensors={"input": short, "enhanced": direct},
            notes=(
                f"{DIRECT_SECONDS:g}s, under the provider's {MAX_DIRECT_SECONDS:g}s "
                "maxDirectDuration, so neither side chunks. Model only."
            ),
            extra={"precision": "fp32", "seconds": DIRECT_SECONDS, "chunked": False},
            **shared,
        ),
    ]
