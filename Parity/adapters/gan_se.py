"""
MossFormerGAN speech enhancement, 16 kHz, CoreML.

Reference: `Models/python/mossformer_gan_se_coreml/run.py`.
Swift:     `Sources/AudioToolCoreML/MossFormerGANCoreMLProvider.swift`.

The tightest expected bound of the eight, and therefore the most sensitive
detector of wrapper drift. Both sides run the *same compiled `.mlpackage`* and
both do STFT/ISTFT in MLX, so the only things that can differ are framing,
normalisation and segment stitching - all Swift wrapper code, all of it touched
this session.

Do not confuse this with the 23.8 dB in `Models/python/mossformer_gan_se_coreml/
README.md`. That is CoreML against PyTorch - the conversion loss, already
measured and accepted when the 256-frame variant was chosen. It is not the
budget for this comparison, which should land at float32 round-off.

Two tensors, so a failure localises without needing internal taps:

  enhanced_segment  one exact 25500-sample segment. Model + STFT only.
  enhanced_full     the whole clip through the segmenting path. Adds stitching.
"""

from __future__ import annotations

import numpy as np

from harness import Context, ParityCase, load_audio

NAME = "mossformer_gan_se_16k_coreml"
SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 25_500  # 256 frames; matches the Swift provider's constant


def build(ctx: Context) -> ParityCase:
    reference_dir = ctx.reference("python/mossformer_gan_se_coreml")
    model_package = reference_dir / "MossFormerGAN_256frames.mlpackage"
    source = ctx.fixture("AudioToolFluidAudioTests/Fixtures/test.wav")

    audio, rate = load_audio(source, mono=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{NAME} is a {SAMPLE_RATE} Hz model; {source.name} is {rate} Hz")

    with ctx.on_path(reference_dir):
        import coremltools as ct
        import mlx.core as mx
        from run import process_audio, process_segment
        from stft import create_periodic_hann_window_mlx

        model = ct.models.MLModel(
            str(model_package), compute_units=ct.ComputeUnit.CPU_AND_GPU
        )
        window = create_periodic_hann_window_mlx(400)

        segment_in = audio[:SEGMENT_SAMPLES]
        segment_out = process_segment(segment_in, model, window)

        full_out, _ = process_audio(
            audio, model, window,
            overlap=0.0,
            segment_seconds=SEGMENT_SAMPLES / SAMPLE_RATE,
            verbose=False,
        )

    return ParityCase(
        name=NAME,
        sample_rate=SAMPLE_RATE,
        tensors={
            "input": audio,
            "input_segment": segment_in,
            "enhanced_segment": np.asarray(segment_out, dtype=np.float32),
            "enhanced_full": np.asarray(full_out, dtype=np.float32),
        },
        weights={"coreml": model_package / "Manifest.json"},
        source_audio=source,
        reference_files=[reference_dir / "run.py"],
        notes=(
            "Segmented without overlap, 25500 samples per segment, matching the "
            "Swift provider. Expected agreement is round-off, not the 23.8 dB "
            "CoreML-vs-PyTorch conversion figure. Note one known asymmetry: the "
            "Python short-final-segment path pads with mode='reflect' while Swift "
            "pads with zeros - only reachable for inputs under 1.594 s."
        ),
        extra={"segment_samples": SEGMENT_SAMPLES, "compute_units": "CPU_AND_GPU"},
    )
