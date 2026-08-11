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

# Quantized checkpoints published alongside the fp32 one, and the group size they
# were produced with - a literal in the generator, recorded nowhere in the flat
# repository. `QuantizationParameters.generatorGroupSize` is the Swift side of the
# same constant; if one moves the other has to.
QUANTIZED_BITS = (8, 6, 4)
GROUP_SIZE = 64


def build(ctx: Context) -> list[ParityCase]:
    reference_dir = ctx.reference("python/mossformer2_se_mlx")
    source = ctx.fixture("AudioToolFluidAudioTests/Fixtures/test_48k.wav")

    audio, rate = load_audio(source, mono=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{NAME} is a {SAMPLE_RATE} Hz model; {source.name} is {rate} Hz")
    short = audio[: int(DIRECT_SECONDS * SAMPLE_RATE)]

    with ctx.on_path(reference_dir):
        import mlx.core as mx
        import mlx.nn as nn
        from huggingface_hub import hf_hub_download
        from mlx.utils import tree_unflatten

        from generate import MODEL_CONFIG, MODEL_REPO, decode_one_audio, load_model
        from mossformer2_se_wrapper import MossFormer2_SE_48K

        def enhance_with(model):
            def enhance(chunk: np.ndarray) -> np.ndarray:
                output = np.asarray(
                    decode_one_audio(model, chunk.reshape(1, -1), MODEL_CONFIG), dtype=np.float32
                ).reshape(-1)
                # Whole STFT hops only, so a 192000-sample chunk returns 191, 999-ish.
                if len(output) < len(chunk):
                    output = np.pad(output, (0, len(chunk) - len(output)))
                return output[: len(chunk)]

            return enhance

        model = load_model("fp32")
        enhance = enhance_with(model)
        direct = enhance(short)

        def load_quantized(bits: int):
            """The fp32 loader with `nn.quantize` in the middle.

            `load_model` cannot be reused: a quantized checkpoint stores each
            `Linear` as packed integers plus `scales` and `biases`, so the modules
            have to be replaced *before* the weights arrive, and `load_model`
            builds and loads in one step. This is `public/python/generate.py`'s
            quantized branch, which the frozen copy under `Models/python/`
            predates - and `Models/` is reference, not something to edit.

            The monkeypatch is repeated rather than assumed: it is applied by
            `load_model`, which this path does not call, and it is part of how the
            reference computes rather than an optimisation.
            """
            nn.LayerNorm.__call__ = lambda self, x: mx.fast.layer_norm(
                x, self.weight, self.bias, self.eps
            )
            wrapper = MossFormer2_SE_48K(MODEL_CONFIG)
            nn.quantize(wrapper, group_size=GROUP_SIZE, bits=bits)
            weights = mx.load(
                hf_hub_download(repo_id=MODEL_REPO, filename=f"model_int{bits}.safetensors")
            )
            wrapper.update(tree_unflatten(list(weights.items())))
            return wrapper.model

        quantized_direct = {
            bits: enhance_with(load_quantized(bits))(short) for bits in QUANTIZED_BITS
        }
        # fp16 goes through the ordinary loader - it is a dtype, not a quantization,
        # and needs no module surgery. It is here because it is the row the
        # trade-off table is decided by: on this architecture it is both faster and
        # closer to fp32 than any of the quantized widths, and a table missing it
        # would recommend int8 by default.
        fp16_direct = enhance_with(load_model("fp16"))(short)

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
        ParityCase(
            name=f"{NAME}_direct_fp16",
            sample_rate=SAMPLE_RATE,
            tensors={"input": short, "enhanced": fp16_direct},
            notes=(
                f"{DIRECT_SECONDS:g}s at fp16, unchunked. Half precision throughout "
                "rather than quantized layers, so every weight narrows - unlike the "
                "int cases, where the convolutions stay float."
            ),
            extra={"precision": "fp16", "seconds": DIRECT_SECONDS, "chunked": False},
            **shared,
        ),
    ] + [
        # One per quantized checkpoint, direct only. Chunking is orthogonal to
        # precision and the fp32 case above already covers it; running it again at
        # four precisions would quadruple the artifact set to re-test the seams.
        #
        # What this catches that a Swift-fp32 comparison cannot: a wrong group
        # size, a different set of quantized layers, or a dequantization that
        # disagrees between the runtimes. All three produce a Swift output that
        # still degrades smoothly from Swift fp32 - plausible-looking, and wrong.
        ParityCase(
            name=f"{NAME}_direct_int{bits}",
            sample_rate=SAMPLE_RATE,
            tensors={"input": short, "enhanced": quantized_direct[bits]},
            notes=(
                f"{DIRECT_SECONDS:g}s at {bits}-bit, group size {GROUP_SIZE}, unchunked. "
                f"Only FFConvM.linear and UniDeepFSMN.linear/project quantize - they are "
                "the model's only Linear layers - so the convolutions stay float on both "
                "sides and the size saving is smaller than the bit width suggests."
            ),
            extra={
                "precision": f"int{bits}", "bits": bits, "group_size": GROUP_SIZE,
                "repo": "starkdmi/MossFormer2_SE_48K_MLX",
                "seconds": DIRECT_SECONDS, "chunked": False,
            },
            **shared,
        )
        for bits in QUANTIZED_BITS
    ]
