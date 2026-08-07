"""
Universal source separation (ResUNet30), 32 kHz, query-conditioned.

Reference: `Models/python/uss_mlx/inference.py`.
Swift:     `Sources/Models/USS`, driven by `USSMLXProvider`.

Conditioned separation, so the case pins two things at once: the model, and that
the 527-d AudioSet query embedding reaches it intact. Two different conditions
are run against the same mixture - if the port ever ignored or mis-shaped the
condition, both outputs would agree with each other while disagreeing with the
reference, which no single-condition case would reveal.

Mirrors `separate()`: 2-second segments, zero-padded up to a whole number of
them, concatenated and trimmed back. The reference reads the embedding with
`mx.load`, which returns a dict for safetensors, so the array is unwrapped here.
"""

from __future__ import annotations

import numpy as np

from harness import Context, ParityCase, load_audio

NAME = "uss_resunet30_32k"
SAMPLE_RATE = 32_000
SEGMENT_SECONDS = 2.0
SECONDS = 4.0  # exactly two segments, so padding is not silently part of the case
# test_speech.wav opens on six seconds of silence and test_music.wav has a gap at
# 4-6 s. This window has signal in both; see MIN_INPUT_RMS in harness.py.
OFFSET_SECONDS = 6.0
CONDITIONS = ["speech", "music"]


def build(ctx: Context) -> ParityCase:
    package_dir = ctx.reference("python")
    reference_dir = ctx.reference("python/uss_mlx")
    weights = ctx.reference("uss_mlx_swift/USSSwift/Models/resunet30_fp32.safetensors")
    embeddings_dir = ctx.reference("uss_mlx_swift/USSSwift/Embeddings")
    source = ctx.reference("uss_mlx_swift/test_music.wav")

    audio, rate = load_audio(source, mono=True, seconds=SECONDS, offset=OFFSET_SECONDS)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{NAME} is a {SAMPLE_RATE} Hz model; {source.name} is {rate} Hz")

    tensors: dict[str, np.ndarray] = {"input": audio}
    used_embeddings = {}

    with ctx.on_path(package_dir):
        import mlx.core as mx
        from uss_mlx.inference import create_compiled_forward, load_model

        model = load_model(str(weights))
        forward = create_compiled_forward(model, compile=True)

        segment_samples = int(SAMPLE_RATE * SEGMENT_SECONDS)
        segments_count = -(-len(audio) // segment_samples)
        padded = np.pad(audio, (0, segments_count * segment_samples - len(audio)))
        segments = mx.array(padded).reshape(segments_count, segment_samples)

        for condition_name in CONDITIONS:
            embedding_path = embeddings_dir / f"{condition_name}_embedding_527d.safetensors"
            loaded = mx.load(str(embedding_path))
            condition = loaded["embedding"] if isinstance(loaded, dict) else loaded
            used_embeddings[condition_name] = embedding_path

            outputs = [
                forward(mx.expand_dims(mx.expand_dims(segments[i], axis=0), axis=1), condition)
                for i in range(segments_count)
            ]
            separated = mx.concatenate(outputs, axis=0).flatten()[: len(audio)]
            mx.eval(separated)
            tensors[f"separated_{condition_name}"] = np.asarray(separated, dtype=np.float32)

    return ParityCase(
        name=NAME,
        sample_rate=SAMPLE_RATE,
        tensors=tensors,
        weights={"model": weights, **used_embeddings},
        source_audio=source,
        reference_files=[reference_dir / "inference.py", reference_dir / "resunet.py"],
        notes=(
            f"{SECONDS:g} s at {SAMPLE_RATE} Hz - exactly {int(SECONDS / SEGMENT_SECONDS)} "
            "segments, so no zero padding is folded into the comparison. Two query "
            "conditions against one mixture; both must match, and they must differ "
            "from each other."
        ),
        extra={
            "segment_seconds": SEGMENT_SECONDS,
            "offset_seconds": OFFSET_SECONDS,
            "conditions": CONDITIONS,
            "loader_note": (
                "load_weights.py reports '26 parameters not found' and an error on "
                "base.stft.window_func, then takes its fallback path. Verified "
                "cosmetic: all 329 checkpoint tensors are bit-equal in the loaded "
                "model. The warning comes from flatten_params not walking lists - "
                "the same quirk demucs_mlx/verify_weights.py documents."
            ),
        },
    )
