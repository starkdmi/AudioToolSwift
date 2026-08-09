"""
Chatterbox multilingual TTS - the conditioning front half, 24 kHz output model.

Reference: `Models/python/chatterbox_mlx/chatterbox/model.py`, `prepare_conditionals`.
Swift:     `Sources/Models/Chatterbox`, driven by `ChatterboxTTSProvider`.

Everything downstream of this is autoregressive and sampled. `t3.inference` draws
from a distribution with `temperature`, `min_p` and `repetition_penalty`, and the
same seed does not buy the same RNG stream across Python MLX and Swift MLX, so a
whole-pipeline comparison would measure the samplers rather than the port.

`prepare_conditionals` is the part with no sampling in it, and it is not a small
part: the voice encoder, the S3 tokenizer, the log-mel frontend and CAMPPlus all
run here, and every one of them feeds both branches downstream. If conditioning
agrees, a difference in the generated audio is the sampler; if it does not, the
generated audio was never going to agree.

Five tensors, cut where the two implementations can be told apart:

| tensor | covers |
|---|---|
| `ve_speaker_emb` | VoiceEncoder over the full 16 kHz reference |
| `t3_cond_prompt_tokens` | S3 tokenizer at the 6 s encoder-conditioning length |
| `s3gen_prompt_token` | S3 tokenizer at the 10 s decoder-conditioning length |
| `s3gen_prompt_feat` | the 24 kHz mel `embed_ref` derives |
| `s3gen_embedding` | CAMPPlus speaker embedding |

The two token tensors come through different chains - 16 kHz direct for T3, 24 kHz
then down to 16 kHz for S3Gen - at different conditioning lengths, and that is only
true when the clip is long enough for the lengths to differ. On a 3 s prompt the
6 s and 10 s windows are both no-ops and the two tensors come out bit-identical
(checked, not assumed), which is why there are two cases below.

A note on which side is authoritative, because it is not the usual answer. Every
other adapter here treats `Models/python` as the reference and the Swift as the
port. This case found a defect in the reference instead: `model.py` resampled with
librosa, which on librosa >= 0.10 is soxr - a fine filter and an LGPL one, so the
Swift could not reproduce it inside an Apache package. `model.py` now runs
resampy's `kaiser_best` design, which the Swift runs too, and the speech tokens are
bit-identical across the two. Both sides moved to a third thing; neither was the
reference of record.

That trade is deliberate and has a cost: this no longer reproduces upstream
chatterbox, which still resamples with librosa. Revisit if upstream fidelity ever
matters more than having one specified algorithm on both sides.
"""

from __future__ import annotations

import numpy as np

from harness import Context, ParityCase, load_audio

EXAGGERATION = 0.5  # `generate`'s default, so the case pins the shipped path.

# Two cases, because one clip cannot exercise this model's conditioning.
#
#   24k    watson_short.wav, 3 s of 24 kHz mono. The source rate is already S3Gen's,
#          so the 24 kHz leg resamples nothing and both conditioning windows are
#          no-ops. What it does cover is the shipped 24 kHz path end to end.
#
#   22k    speech_long.wav, tiled to 12 s at 22050 Hz. Deliberately awkward on every
#          axis the first case leaves flat: past both the 6 s encoder and 10 s
#          decoder windows, so the truncations run and the two token tensors stop
#          being the same tensor; and 22050 shares only gcd 150 with 24000, giving
#          up/down of 160/147 where watson_short gets 1/1. A polyphase filter with
#          147 phases is a different piece of arithmetic from one with none.
CASES = (
    ("chatterbox_conditionals_24k", "chatterbox_swift/watson_short.wav", None),
    ("chatterbox_conditionals_22k_long", None, 12.0),
)
LONG_FIXTURE = "AudioToolFluidAudioTests/Fixtures/speech_long.wav"


def build(ctx: Context) -> list[ParityCase]:
    reference_dir = ctx.reference("python/chatterbox_mlx")
    # Absent until the weights are fetched; generate.py reports FileNotFoundError
    # as a skip, so this adapter costs nothing on a machine without them.
    weights = ctx.reference("python/chatterbox_mlx/weights")

    cases: list[ParityCase] = []

    with ctx.on_path(reference_dir):
        import mlx.core as mx
        from chatterbox import model as model_module
        from chatterbox.model import Model

        # Loaded once for both cases. It is 2.7 GB of fp32 and nothing in
        # `prepare_conditionals` mutates it.
        model = Model.from_pretrained(str(weights))

        # Read off the reference rather than restated here, so the sidecar cannot
        # claim constants the reference has since changed.
        resampler_provenance = {
            "used": "resampy kaiser_best design, via scipy resample_poly",
            "instead_of": "librosa.resample, which dispatches to soxr on librosa>=0.10",
            "reason": (
                "soxr is LGPL-2.1-or-later and cannot ship inside an Apache Swift "
                "package except dynamically, so both sides run one published design. "
                "Alias rejection on a sweep from above Nyquist: soxr HQ -57.7 dB, "
                "this -55.8 dB, scipy resample_poly defaults -20.9 dB"
            ),
            "zero_crossings": model_module.KAISER_BEST_ZERO_CROSSINGS,
            "beta": model_module.KAISER_BEST_BETA,
            "rolloff": model_module.KAISER_BEST_ROLLOFF,
            "scope": (
                "model.py only. s3gen.py has its own resample_audio on scipy's "
                "polyphase defaults, because embed_ref's CAMPPlus input must match it"
            ),
        }

        def take(value) -> np.ndarray:
            mx.eval(value)
            return np.asarray(value, dtype=np.float32)

        for name, reference_clip, tile_seconds in CASES:
            if reference_clip is not None:
                source = ctx.reference(reference_clip)
            else:
                source = ctx.fixture(LONG_FIXTURE)
            audio, rate = load_audio(source, mono=True)

            if tile_seconds is not None:
                # Tiled rather than shipped as a new fixture: the point is a prompt
                # past both conditioning windows, and repeating an existing clip
                # gets there without adding a binary to the repository. The seam it
                # introduces lands inside the 10 s window, where the truncation can
                # see it.
                wanted = int(tile_seconds * rate)
                repeats = -(-wanted // len(audio))
                audio = np.tile(audio, repeats)[:wanted]

            conds = model.prepare_conditionals(
                mx.array(audio), ref_sr=rate, exaggeration=EXAGGERATION
            )

            tensors: dict[str, np.ndarray] = {"input": audio}
            tensors["ve_speaker_emb"] = take(conds.t3.speaker_emb)
            if conds.t3.cond_prompt_speech_tokens is not None:
                tensors["t3_cond_prompt_tokens"] = take(conds.t3.cond_prompt_speech_tokens)
            for key in ("prompt_token", "prompt_feat", "embedding"):
                if key in conds.gen:
                    tensors[f"s3gen_{key}"] = take(conds.gen[key])

            tokens_coincide = (
                "t3_cond_prompt_tokens" in tensors
                and tensors["t3_cond_prompt_tokens"].shape
                == tensors["s3gen_prompt_token"].shape
                and np.array_equal(
                    tensors["t3_cond_prompt_tokens"], tensors["s3gen_prompt_token"]
                )
            )

            cases.append(
                ParityCase(
                    name=name,
                    # The rate of the `input` tensor, which is all the Swift side
                    # reads. Not the model's 24 kHz output rate - for the long case
                    # those differ, and handing Swift the wrong one would resample
                    # the prompt before conditioning ever started.
                    sample_rate=rate,
                    tensors=tensors,
                    # The checkpoint file, not the directory `from_pretrained` takes:
                    # the sidecar hashes whatever it is handed, and hashing a
                    # directory raises.
                    weights={"model": weights / "model.safetensors"},
                    source_audio=source,
                    reference_files=[
                        reference_dir / "chatterbox" / "model.py",
                        reference_dir / "chatterbox" / "voice_encoder" / "voice_encoder.py",
                        reference_dir / "chatterbox" / "s3tokenizer" / "model_v2.py",
                    ],
                    notes=(
                        "Conditioning only - the deterministic half. "
                        f"`prepare_conditionals` at exaggeration {EXAGGERATION:g}, "
                        "which is `generate`'s default. Nothing here samples, so a "
                        "mismatch is the port and not the RNG. Token tensors are "
                        "integer-valued and stored as float32; they are small enough "
                        "that the round trip is exact, and any difference at all is a "
                        "real one. `model.py` resamples with resampy's kaiser_best "
                        "design rather than librosa; see resampler_16k, whose "
                        "constants are read off the reference."
                    ),
                    extra={
                        "exaggeration": EXAGGERATION,
                        "resampler_16k": resampler_provenance,
                        "source_seconds": round(len(audio) / rate, 3),
                        "source_tiled": tile_seconds is not None,
                        "enc_cond_seconds": 6,
                        "dec_cond_seconds": 10,
                        # Recorded rather than left implicit: on a short prompt the
                        # two token tensors are the same tensor, and a reader
                        # comparing them without knowing that would call it
                        # corroboration when it is a tautology.
                        "token_tensors_coincide": bool(tokens_coincide),
                        "sampling": (
                            "excluded on purpose - t3.inference draws with "
                            "temperature/min_p/repetition_penalty and MLX's RNG "
                            "stream is not shared across the two runtimes, so it is "
                            "not comparable at the token level"
                        ),
                        "authority": (
                            "neither side is the reference of record: this case found "
                            "the reference resampling with soxr, which the Swift "
                            "cannot reproduce under its licence, and both sides were "
                            "moved to one published design. Not "
                            "upstream-chatterbox-faithful, on purpose"
                        ),
                    },
                )
            )

    return cases
