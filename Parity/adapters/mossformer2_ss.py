"""
MossFormer2 speaker separation - three configurations.

Reference: `Models/python/mosforrmer2_ss_mlx/generate.py`.
Swift:     `Sources/Models/MossFormer2SS`, driven by `MLXSeparatorProvider`.

Each configuration is its own case, so a failure names the config rather than
"SS". They differ in more than a weights file - 2spk runs at 16 kHz, both 8 kHz
variants at 8 kHz, and WHAMR skips the mask multiplication entirely.

Speakers are stored as separate named tensors. Which speaker lands in which
slot is itself part of the port's behaviour: a permutation would be silent under
any single combined metric and obvious here.

Inputs are the committed CC0 mixtures, not the WSJ0/WHAMR-derived `mix*.wav` in
the research checkout these cases first used. An artifact stores its input
samples rather than a path, so a case built from restricted audio carries that
audio wherever the artifact goes - and these are published alongside the
weights. Same three overlapping mixtures, each at its model's own rate.
"""

from __future__ import annotations

import numpy as np

from harness import Context, ParityCase, chunked, load_audio

NAME = "mossformer2_ss"

MAX_DIRECT_SECONDS = 4.0  # MossFormer2SSProvider.maxDirectDuration
DIRECT_SECONDS = 3.0
CHUNK_DURATION = 4.0      # ChunkingConfig.mossformer2SS
OVERLAP_RATIO = 0.25
STRATEGY = "triangular_blend"

CONFIGS = {
    "mossformer2_ss_2spk_16k": {
        "model": "2spk",
        "fixture": "AudioToolFluidAudioTests/Fixtures/mix_16k.wav",
        "sample_rate": 16_000,
    },
    "mossformer2_ss_2spk_whamr_8k": {
        "model": "2spk-whamr",
        "fixture": "AudioToolFluidAudioTests/Fixtures/mix_8k.wav",
        "sample_rate": 8_000,
    },
    "mossformer2_ss_3spk_8k": {
        "model": "3spk",
        "fixture": "AudioToolFluidAudioTests/Fixtures/mix3_8k.wav",
        "sample_rate": 8_000,
    },
}


def _normalize(source: np.ndarray, input_rms: float) -> np.ndarray:
    """`save_sources`' normalisation, which the Swift provider reproduces inline."""
    output_rms = float(np.sqrt(np.mean(source.astype(np.float64) ** 2)))
    scaled = source * (input_rms / output_rms) if output_rms > 1e-8 else source.copy()
    peak = float(np.abs(scaled).max())
    if peak > 1.0:
        scaled = scaled / peak
    return np.asarray(scaled, dtype=np.float32)


def build(ctx: Context) -> list[ParityCase]:
    reference_dir = ctx.reference("python/mosforrmer2_ss_mlx")
    cases: list[ParityCase] = []

    for case_name, spec in CONFIGS.items():
        source = ctx.fixture(spec["fixture"])
        rate_expected = spec["sample_rate"]
        audio, rate = load_audio(source, mono=True)
        if rate != rate_expected:
            raise ValueError(
                f"{case_name} is a {rate_expected} Hz model; {source.name} is {rate} Hz"
            )
        short = audio[: int(DIRECT_SECONDS * rate_expected)]

        with ctx.on_path(reference_dir):
            from generate import MODEL_CONFIGS, create_model, download_model, separate_audio

            weights_path, config = download_model(spec["model"])
            model = create_model(
                num_spks=config["num_spks"],
                weights_path=weights_path,
                is_whamr=config["is_whamr"],
            )
            speakers = config["num_spks"]

            def separate(chunk: np.ndarray, *, index: int) -> np.ndarray:
                out = np.asarray(separate_audio(model, chunk)[index], dtype=np.float32)
                if len(out) < len(chunk):
                    out = np.pad(out, (0, len(chunk) - len(out)))
                return out[: len(chunk)]

            direct_sources = [
                np.asarray(source_array, dtype=np.float32)
                for source_array in separate_audio(model, short)
            ]

        # Each speaker is blended across chunks independently, which is what the
        # provider does too (MLXSeparatorProvider.swift:255-262). Running the
        # chunker once per speaker index is wasteful but reproduces that exactly,
        # including any per-chunk speaker permutation - a real behaviour worth
        # pinning rather than papering over.
        chunked_sources = [
            chunked(
                ctx, audio, rate_expected,
                chunk_duration=CHUNK_DURATION, overlap_ratio=OVERLAP_RATIO,
                strategy=STRATEGY,
                process_fn=lambda chunk, index=index: separate(chunk, index=index),
            )
            for index in range(speakers)
        ]

        shared = {
            "source_audio": source,
            "reference_files": [
                reference_dir / "generate.py",
                reference_dir / "mossformer2_ss_16k.py",
            ],
        }
        base_extra = {
            "repo": MODEL_CONFIGS[spec["model"]]["repo_id"],
            "num_speakers": speakers,
            "whamr": config["is_whamr"],
        }

        # Raw model output *and* what the public API returns. The reference
        # normalises in `save_sources`, at write time; the Swift provider does the
        # same thing inline (MLXSeparatorProvider.swift:178-195, RMS match then
        # peak clamp). Storing only raw would fail a correct port; storing only
        # normalised would hide a uniform gain error, since the scaling is
        # self-cancelling. So both.
        def tensors_for(mixture: np.ndarray, sources: list[np.ndarray]) -> dict:
            level = float(np.sqrt(np.mean(mixture.astype(np.float64) ** 2)))
            built = {"input": mixture}
            for index, separated in enumerate(sources, start=1):
                built[f"speaker_{index}"] = separated
                built[f"speaker_{index}_normalized"] = _normalize(separated, level)
            return built

        cases.append(ParityCase(
            name=case_name,
            sample_rate=rate_expected,
            tensors=tensors_for(audio, chunked_sources),
            notes=(
                f"Chunked at {CHUNK_DURATION}s / {OVERLAP_RATIO:.0%} / {STRATEGY}, matching "
                "ChunkingConfig.mossformer2SS. `speaker_N` is raw model output; "
                "`speaker_N_normalized` is what a caller sees, after the RMS match "
                "and peak clamp both sides apply. Speaker order is itself pinned."
            ),
            extra={**base_extra, "chunk_duration": CHUNK_DURATION,
                   "overlap_ratio": OVERLAP_RATIO, "strategy": STRATEGY},
            **shared,
        ))

        cases.append(ParityCase(
            name=f"{case_name}_direct",
            sample_rate=rate_expected,
            tensors=tensors_for(short, direct_sources),
            notes=(
                f"{DIRECT_SECONDS:g}s, under the provider's {MAX_DIRECT_SECONDS:g}s "
                "maxDirectDuration, so neither side chunks. Model only."
            ),
            extra={**base_extra, "seconds": DIRECT_SECONDS, "chunked": False},
            **shared,
        ))

    return cases
