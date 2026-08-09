#!/usr/bin/env python3
"""
Listen to what the resampler choice actually costs.

The parity case measures conditioning; this renders speech. Three runs, identical
model, identical text, identical seed - the only variable is which resampler
`chatterbox.model.prepare_conditionals` uses for its 24 -> 16 kHz conversions:

  soxr      what librosa dispatches to, and what the reference artifact recorded
  scipy     `resample_poly` defaults, which is what Swift does today (31.7 dB)
  kaiser    a from-scratch 385-tap design, the proposed replacement (51.8 dB)

`chatterbox.s3gen.s3gen.resample_audio` is deliberately left alone in all three:
the reference uses scipy's polyphase there and Swift already matches it.

Read the generated speech with a caveat. `t3.inference` samples, so once the
prompt tokens differ the sequences diverge and keep diverging - the three files
will not be small perturbations of each other even where the resampler barely
matters. What is directly comparable is `prompt_16k_*.wav`: the same three
resamplers, no model, no sampling. That pair of comparisons is the point - one
isolates the resampler, the other shows what survives to the ear.

    python Parity/listen_resamplers.py --out /tmp/resampler_listen
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]
REFERENCE = RESEARCH / "Models" / "python" / "chatterbox_mlx"
WEIGHTS = REFERENCE / "weights"
PROMPT = RESEARCH / "Models" / "chatterbox_swift" / "watson_short.wav"

TEXT = (
    "The quick brown fox jumps over the lazy dog, "
    "and the sixth sick sheikh's sixth sheep is sick."
)
SEED = 1234


def resamplers():
    """The three 24 -> 16 kHz conversions, as drop-in replacements."""
    import librosa
    import mlx.core as mx
    import numpy as np
    from scipy import signal

    def soxr(audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        return mx.array(
            librosa.resample(np.array(audio), orig_sr=orig_sr, target_sr=target_sr)
        )

    def polyphase(audio, orig_sr, target_sr, window=None):
        if orig_sr == target_sr:
            return audio
        g = math.gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        kwargs = {"window": window} if window is not None else {}
        out = signal.resample_poly(
            np.array(audio), up, down, padtype="edge", **kwargs
        )
        return mx.array(out.astype(np.float32))

    def scipy_default(audio, orig_sr, target_sr):
        return polyphase(audio, orig_sr, target_sr)

    def kaiser(audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        g = math.gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        max_rate = max(up, down)
        # resampy's kaiser_best, quoted from that project: beta 12.9846, roll
        # 0.917347, 50 zeros, designed for -120 dB. This is what both the adapter
        # and Swift now run, so `kaiser` here is the shipped path and not a proposal.
        #
        # Deliberately not a filter fitted to track soxr. One tuned that way scores
        # better on token agreement and is not otherwise a better filter - see the
        # note in Tests/AudioToolParityTests/ParityThresholds.swift.
        taps = 2 * 50 * max_rate + 1
        h = signal.firwin(taps, 0.917347 / max_rate, window=("kaiser", 12.9846))
        return polyphase(audio, orig_sr, target_sr, window=h)

    return {"soxr": soxr, "scipy": scipy_default, "kaiser": kaiser}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/resampler_listen")
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REFERENCE))
    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    from chatterbox import model as model_module
    from chatterbox.model import Model

    prompt, prompt_sr = sf.read(PROMPT, dtype="float32")
    print(f"prompt: {PROMPT.name}, {len(prompt)} samples at {prompt_sr} Hz")

    model = Model.from_pretrained(str(WEIGHTS))
    original = model_module.resample_audio
    variants = resamplers()

    # The resampler on its own: no model, no sampling, directly comparable.
    for name, fn in variants.items():
        sf.write(
            out_dir / f"prompt_16k_{name}.wav",
            np.array(fn(mx.array(prompt), prompt_sr, 16_000)),
            16_000,
        )
    print(f"wrote prompt_16k_*.wav ({len(variants)} files)")

    for name, fn in variants.items():
        model_module.resample_audio = fn
        try:
            conds = model.prepare_conditionals(
                mx.array(prompt), ref_sr=prompt_sr, exaggeration=0.5
            )
            tokens = conds.gen["prompt_token"]
            mx.eval(tokens)
            mx.random.seed(args.seed)
            audio = None
            for result in model.generate(
                args.text, conds=conds, temperature=0.8, cfg_weight=0.5,
                min_p=0.05, repetition_penalty=2.0, verbose=False,
            ):
                chunk = np.array(result.audio).reshape(-1)
                audio = chunk if audio is None else np.concatenate([audio, chunk])
                rate = result.sample_rate
        finally:
            model_module.resample_audio = original

        path = out_dir / f"speech_{name}.wav"
        sf.write(path, audio, rate)
        print(f"  {name:7} {len(audio)/rate:5.2f}s -> {path.name}")

    print(f"\noutput in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
