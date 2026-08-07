"""
FRCRN speech enhancement, 16 kHz.

Reference: `Models/python/frcrn_se_mlx/frcrn_mlx.py` (`FRCRN_SE_16K_MLX`),
           chunked through `Models/python/benchmark_chunking.py`.
Swift:     `Sources/Models/FRCRN`, driven by `FRCRNSE16KProvider`.

`FRCRNSE16KProvider.process` has no direct path - it says "Always use chunking
for consistent quality" and routes everything through 4-second chunks at 25%
overlap with discard-edges. So the reference has to chunk identically or the two
are not comparable. There is no `_direct` case here for that reason; the public
API cannot be asked for one.

Comparing whole-file Python against this gave 41.1 dB with correlation 1.000 and
zero lag, with 65% of the error energy sitting in the final 2400 samples - the
last partial chunk, which Swift zero-pads to 64000 and the unchunked reference
never saw. That is a chunking difference wearing a parity number's clothes.
"""

from __future__ import annotations

import numpy as np

from harness import Context, ParityCase, chunked, load_audio

NAME = "frcrn_se_16k"
SAMPLE_RATE = 16_000
CHUNK_DURATION = 4.0      # ChunkingConfig.frcrnSE16K
OVERLAP_RATIO = 0.25
STRATEGY = "discard_edges"


def build(ctx: Context) -> ParityCase:
    reference_dir = ctx.reference("python/frcrn_se_mlx")
    weights = reference_dir / "frcrn_se_16k.safetensors"
    source = ctx.fixture("AudioToolFluidAudioTests/Fixtures/test.wav")

    audio, rate = load_audio(source, mono=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{NAME} is a {SAMPLE_RATE} Hz model; {source.name} is {rate} Hz")

    with ctx.on_path(reference_dir):
        import mlx.core as mx
        import mlx.nn as nn
        from frcrn_mlx import FRCRN_SE_16K_MLX

        model = FRCRN_SE_16K_MLX()
        nn.Module.load_weights(model, str(weights))
        model.eval()

        def enhance(chunk: np.ndarray) -> np.ndarray:
            output = model(mx.array(chunk)[None, :, None])
            mx.eval(output)
            result = np.array(output.squeeze())
            # The model emits whole STFT hops, so a chunk of 64000 comes back as
            # 63x320. The chunking caller indexes up to chunk length, so pad the
            # remainder rather than letting it silently shorten the chunk.
            if len(result) < len(chunk):
                result = np.pad(result, (0, len(chunk) - len(result)))
            return result[: len(chunk)]

        whole = enhance(audio)

    return ParityCase(
        name=NAME,
        sample_rate=SAMPLE_RATE,
        tensors={
            "input": audio,
            "enhanced": chunked(
                ctx, audio, SAMPLE_RATE,
                chunk_duration=CHUNK_DURATION, overlap_ratio=OVERLAP_RATIO,
                strategy=STRATEGY, process_fn=enhance,
            ),
            # Kept for reference, not for the API comparison: what the model does
            # in one pass over the whole file. The difference between this and
            # `enhanced` is the cost of chunking, measured rather than assumed.
            "enhanced_whole_file": whole,
        },
        weights={"model": weights},
        source_audio=source,
        reference_files=[reference_dir / "frcrn_mlx.py"],
        notes=(
            f"Chunked at {CHUNK_DURATION}s / {OVERLAP_RATIO:.0%} / {STRATEGY}, matching "
            "ChunkingConfig.frcrnSE16K. Compare the provider against `enhanced`; "
            "`enhanced_whole_file` is the unchunked model for reference only."
        ),
        extra={
            "chunk_duration": CHUNK_DURATION,
            "overlap_ratio": OVERLAP_RATIO,
            "strategy": STRATEGY,
            "direct_path": False,
        },
    )
