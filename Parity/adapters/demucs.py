"""
Demucs music separation, 44.1 kHz stereo.

Reference: `Models/python/demucs_mlx/api.py` (`DemucsSeparator`).
Swift:     `Sources/Models/Demucs`, driven by `MLXSeparatorProvider`.

Stereo is the whole point of this case. The Swift port was reading interleaved
frames as planar - the fix landed this session in both the batch and streaming
paths - and a mono comparison would not have caught it. So the input stays
stereo, and each stem is stored per channel: a left/right swap or a re-collapse
to mono shows up as a specific tensor moving, not as a vague loss of quality.

HTDemucs ships one checkpoint per stem and each emits all four sources, and
`DemucsProvider.separate(_:stem:)` uses *stem X's own checkpoint* and then takes
slot X out of it. So the reference has to do the same: four model loads, each
contributing one stem.

Getting this wrong is what the first version of this adapter did - it ran the
`vocals` checkpoint once and took all four of its outputs. Vocals then agreed
with Swift at 74.7 dB while drums, bass and other came back at -0.1, 0.0 and
-7.8 dB, because those three were being compared against a different model's
opinion of them. The near-perfect stem is what gave it away.

Weights come from `Models/demucs_mlx_swift/Weights/`, the only copy with a
`.json` config beside every stem.
"""

from __future__ import annotations

import numpy as np

from harness import Context, ParityCase, load_audio

NAME = "demucs_vocals_44k"
SAMPLE_RATE = 44_100
SECONDS = 3.0  # four stereo stems at float32; 3 s keeps the artifact near 10 MB


def build(ctx: Context) -> ParityCase:
    reference_dir = ctx.reference("python/demucs_mlx")
    weights_dir = ctx.reference("demucs_mlx_swift/Weights")
    source = ctx.fixture("AudioToolFluidAudioTests/Fixtures/music.wav")

    audio, rate = load_audio(source, mono=False, seconds=SECONDS)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{NAME} is a {SAMPLE_RATE} Hz model; {source.name} is {rate} Hz")
    if audio.shape[0] != 2:
        raise ValueError(f"{NAME} needs stereo; {source.name} has {audio.shape[0]} channel(s)")

    stem_names = ["drums", "bass", "other", "vocals"]
    per_stem: dict[str, np.ndarray] = {}
    used_weights: dict[str, object] = {}

    with ctx.on_path(reference_dir):
        import mlx.core as mx
        from api import DemucsSeparator

        for stem in stem_names:
            checkpoint = weights_dir / f"{stem}.safetensors"
            separator = DemucsSeparator(str(checkpoint))
            index = list(separator.sources).index(stem)
            output = separator.separate(mx.array(audio), overlap=0.25, split=True)
            mx.eval(output)
            per_stem[stem] = np.asarray(output, dtype=np.float32)[0, index]  # (C, T)
            used_weights[stem] = checkpoint

    tensors = {"input": audio}
    for stem in stem_names:
        left = per_stem[stem][0]
        right = per_stem[stem][1]
        tensors[f"{stem}_left"] = left
        tensors[f"{stem}_right"] = right
        # DemucsProvider.separate(_:stem:) means over channels before returning,
        # so the reference for the public API is the mono mix. Recorded here
        # rather than derived in the test: the reference should be a hashed
        # artifact, not something the test computes and then compares to itself.
        tensors[f"{stem}_mono"] = ((left + right) / 2).astype(np.float32)

    return ParityCase(
        name=NAME,
        sample_rate=SAMPLE_RATE,
        tensors=tensors,
        weights=used_weights,
        source_audio=source,
        reference_files=[reference_dir / "api.py", reference_dir / "model.py"],
        notes=(
            f"First {SECONDS:g} s, stereo, 25% overlap, split=True - the default "
            "chunked path. Stems are stored per channel so an interleaved/planar "
            "mix-up is unmissable. Known gap: music.wav is instrumental, so the "
            "vocals stem sits near 1e-4 RMS and discriminates weakly - the stem the "
            "API exposes is the one this clip tests least. drums/bass/other carry "
            "real signal and differ between channels, which is what the channel-layout "
            "regression needs. A 44.1 kHz CC0 clip with vocals would close it."
        ),
        extra={"stems": stem_names, "overlap": 0.25, "seconds": SECONDS},
    )
