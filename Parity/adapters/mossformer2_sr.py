"""
MossFormer2 super-resolution, 16 kHz in -> 48 kHz out.

Reference: `Models/python/mossformer2_sr_mlx/generate.py`, chunked through
           `Models/python/benchmark_chunking.py`.
Swift:     `Sources/Models/MossFormer2SR`, driven by `MLXSuperResolutionProvider`.

The awkward one, because two things happen before the model does:

1. The input is upsampled 16 -> 48 kHz. Swift uses AVAudioConverter at Mastering
   quality; librosa uses soxr_hq. They agree at about 45 dB, and SR amplifies
   that - reconstructing the band the upsample has to preserve is the entire job -
   so an end-to-end comparison measured the two resamplers rather than the port.
   It read 22 dB with everything else at round-off.

   So the 48 kHz input is not re-derived here: `Parity/inputs/sr_upsampled_48k*.wav`
   is Swift's own resampler output, exported by ParityInputExportTests and loaded
   below. Both sides now start from identical samples and what is left in the
   comparison is the model.

   That moves the resampler out of the comparison, so it needs its own yardstick
   or it is simply untested. `upsampled_48k` is the shared input - Swift's - and
   comparing Swift against it measures nothing at all. `upsampled_48k_librosa` is
   librosa's output on the same 16 kHz samples, kept precisely so the seam can be
   checked against an implementation that is not the one under test.

2. Above four seconds the 48 kHz signal is chunked at 25% with give_up_length
   discard-edges assembly - `stride = int(window * 0.75)` and
   `give_up_length = (window - stride) // 2`, exactly as the published
   `generate.py` does it.

   This said 50% and "hann_blend" until 2026-08-10, mirroring what the Swift
   provider did rather than what the reference does. That is the wrong direction
   for an adapter to face: it made the suite compare Swift against a Python
   reimplementation of Swift, which passes by construction and cannot report a
   divergence in the algorithm itself. Both sides now follow generate.py.

   Artifacts generated before that date encode the old assembly and must be
   regenerated; a stale one will fail rather than mislead, which is the intent.
"""

from __future__ import annotations

import json

import numpy as np

from harness import PARITY_DIR, Context, ParityCase, chunked, load_audio

NAME = "mossformer2_sr_48k"
INPUT_SAMPLE_RATE = 16_000
SAMPLE_RATE = 48_000
MAX_DIRECT_SECONDS = 4.0  # MLXSuperResolutionProvider.maxDirectDuration
DIRECT_SECONDS = 3.0
CHUNK_DURATION = 4.0      # ChunkingConfig.mossformer2SR48K
OVERLAP_RATIO = 0.25
STRATEGY = "discard_edges"


def _swift_upsampled(ctx, audio, short, rate):
    """The 48 kHz signal Swift produced, not one librosa re-derives.

    Falls back to librosa only if the export is missing, and says so loudly -
    a silent fallback would put the resampler back in the comparison and the
    number would drop ~70 dB with no indication why.
    """
    import librosa
    import soundfile as sf

    directory = PARITY_DIR / "inputs"
    full_path = directory / "sr_upsampled_48k.wav"
    short_path = directory / "sr_upsampled_48k_direct.wav"

    if not full_path.exists() or not short_path.exists():
        print(
            f"  WARNING: {directory} is missing Swift's upsampled input; falling back to\n"
            "  librosa. Expect ~22 dB instead of round-off - that gap is the two\n"
            "  resamplers, not the port. Regenerate with ParityInputExportTests."
        )
        return (
            np.asarray(librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE), dtype=np.float32),
            np.asarray(librosa.resample(short, orig_sr=rate, target_sr=SAMPLE_RATE), dtype=np.float32),
        )

    def read(path, expected):
        data, sr = sf.read(str(path), dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{path.name} is {sr} Hz, expected {SAMPLE_RATE}")
        if len(data) != expected:
            raise ValueError(f"{path.name} has {len(data)} samples, expected {expected}")
        return np.ascontiguousarray(data, dtype=np.float32)

    return read(full_path, len(audio) * 3), read(short_path, len(short) * 3)


def build(ctx: Context) -> list[ParityCase]:
    reference_dir = ctx.reference("python/mossformer2_sr_mlx")
    source = ctx.fixture("AudioToolFluidAudioTests/Fixtures/sr_input_16k.wav")

    audio, rate = load_audio(source, mono=True)
    if rate != INPUT_SAMPLE_RATE:
        raise ValueError(f"{NAME} takes {INPUT_SAMPLE_RATE} Hz; {source.name} is {rate} Hz")
    short = audio[: int(DIRECT_SECONDS * INPUT_SAMPLE_RATE)]

    with ctx.on_path(reference_dir):
        import librosa
        import mlx.core as mx
        from huggingface_hub import hf_hub_download
        from mlx.utils import tree_unflatten

        from bandwidth_sub import bandwidth_sub
        from mel_spec import mel_spectrogram
        from mossformer2_sr_wrapper import AttrDict, MossFormer2_SR_48K

        repo = "starkdmi/MossFormer2_SR_48K_MLX"
        weights_path = hf_hub_download(repo_id=repo, filename="model_fp32.safetensors")
        config_path = hf_hub_download(repo_id=repo, filename="config.json")

        with open(config_path) as handle:
            args = AttrDict(json.load(handle))
        args.one_time_decode_length = 20.0
        args.decode_window = 4.0

        model = MossFormer2_SR_48K(args)
        model.update(tree_unflatten(list(mx.load(weights_path).items())))

        def upscale(chunk: np.ndarray, *, pad: bool = True) -> np.ndarray:
            """One pass of the provider's processChunk, on 48 kHz input.

            `pad` only for the chunked path, whose caller indexes up to the chunk
            length. The direct path must not pad: the model emits whole hops, so
            144000 samples in gives 143872 out, and the provider returns exactly
            that. Padding here would manufacture a length mismatch that is not real.
            """
            inputs = mx.array(np.ascontiguousarray(chunk, dtype=np.float32))
            mel = mel_spectrogram(
                mx.expand_dims(inputs, axis=0),
                n_fft=args.n_fft, num_mels=args.num_mels,
                sampling_rate=args.sampling_rate, hop_size=args.hop_size,
                win_size=args.win_size, fmin=args.fmin, fmax=args.fmax,
            )
            raw = mx.squeeze(model(mel))
            out = bandwidth_sub(inputs, raw, fs=SAMPLE_RATE)
            mx.eval(out)
            result = np.asarray(out, dtype=np.float32)
            if not pad:
                return result
            if len(result) < len(chunk):
                result = np.pad(result, (0, len(chunk) - len(result)))
            return result[: len(chunk)]

        upsampled, upsampled_short = _swift_upsampled(ctx, audio, short, rate)
        direct = upscale(upsampled_short, pad=False)

        # The independent reading of the same seam. Stored at exactly 3x the input
        # length so the Swift side can compare it sample for sample; librosa
        # returns that already for an integer ratio, and an off-by-one here would
        # be a length mismatch reported as a content difference.
        def _librosa_upsampled(signal: np.ndarray) -> np.ndarray:
            resampled = librosa.resample(
                signal, orig_sr=rate, target_sr=SAMPLE_RATE, res_type="soxr_hq"
            )
            expected = len(signal) * (SAMPLE_RATE // rate)
            if len(resampled) != expected:
                raise ValueError(
                    f"librosa returned {len(resampled)} samples for {len(signal)} at "
                    f"{rate} -> {SAMPLE_RATE} Hz; expected {expected}"
                )
            return np.asarray(resampled, dtype=np.float32)

        librosa_upsampled = _librosa_upsampled(audio)
        librosa_upsampled_short = _librosa_upsampled(short)

        # Inside the path context, not in the return below. `bandwidth_sub` reaches
        # `mlx_filters.detect_bandwidth`, which does `from stft import ...` lazily -
        # on the first chunk, not at import time. `on_path` drops what the block
        # imported when it exits, so a `chunked` call placed in the return statement
        # re-triggers that import with the reference directory no longer on the
        # path, and generation dies on ModuleNotFoundError after the model has run.
        enhanced = chunked(
            ctx, upsampled, SAMPLE_RATE,
            chunk_duration=CHUNK_DURATION, overlap_ratio=OVERLAP_RATIO,
            strategy=STRATEGY, process_fn=upscale,
        )

    shared = {
        "source_audio": source,
        "reference_files": [
            reference_dir / "generate.py",
            reference_dir / "mossformer2_sr_wrapper.py",
            reference_dir / "bandwidth_sub.py",
        ],
        "extra_common": None,
    }
    del shared["extra_common"]

    return [
        ParityCase(
            name=NAME,
            sample_rate=SAMPLE_RATE,
            tensors={
                "input": audio,
                "upsampled_48k": upsampled,
                "upsampled_48k_librosa": librosa_upsampled,
                "enhanced": enhanced,
            },
            notes=(
                f"Chunked at {CHUNK_DURATION}s / {OVERLAP_RATIO:.0%} / {STRATEGY}, matching "
                "ChunkingConfig.mossformer2SR48K. The resampler is not in this "
                "comparison: both sides start from `upsampled_48k`, which is Swift's "
                "own 48 kHz output. `upsampled_48k_librosa` is librosa's, for checking "
                "that seam against something other than the code under test."
            ),
            extra={
                "input_sample_rate": INPUT_SAMPLE_RATE, "repo": repo,
                "resampler": "Swift AVAudioConverter (Mastering/.max), via Parity/inputs",
                "resampler_reference": "librosa.resample soxr_hq, in upsampled_48k_librosa",
                "chunk_duration": CHUNK_DURATION, "overlap_ratio": OVERLAP_RATIO,
                "strategy": STRATEGY,
            },
            **shared,
        ),
        ParityCase(
            name=f"{NAME}_direct",
            sample_rate=SAMPLE_RATE,
            tensors={
                "input": short,
                "upsampled_48k": upsampled_short,
                "upsampled_48k_librosa": librosa_upsampled_short,
                "enhanced": direct,
            },
            notes=(
                f"{DIRECT_SECONDS:g}s in, {DIRECT_SECONDS:g}s out - under the provider's "
                f"{MAX_DIRECT_SECONDS:g}s maxDirectDuration, so neither side chunks. "
                "The resampler is not in this comparison either; "
                "`upsampled_48k_librosa` is where it gets measured."
            ),
            extra={
                "input_sample_rate": INPUT_SAMPLE_RATE, "repo": repo,
                "resampler": "Swift AVAudioConverter (Mastering/.max), via Parity/inputs",
                "resampler_reference": "librosa.resample soxr_hq, in upsampled_48k_librosa",
                "seconds": DIRECT_SECONDS, "chunked": False,
            },
            **shared,
        ),
    ]
