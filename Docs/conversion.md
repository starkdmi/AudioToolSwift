# Conversion and parity

None of these models were trained here. Each one was converted from a PyTorch
original to MLX or Core ML, re-implemented in Swift, and then held to a number
that says the Swift agrees with the reference. This page describes that chain and
how to check it.

## The chain

```
PyTorch original  →  MLX Python port  →  Swift port
   (upstream)         (reference)         (this package)
                                    ↘
                                      Core ML package   (MossFormerGAN only)
```

Two conversions happen, and they are different kinds of work:

**PyTorch → MLX** is a weight remap plus a re-implementation of the forward pass.
Its cost was measured once, at conversion time, and accepted: FRCRN's MLX port
records 115.98 dB against PyTorch, which is float32 round-off. That number is not
re-litigated here.

**MLX Python → Swift** is what this repository owns, and it is the one that has
to keep being true. The MLX Python implementation is the reference; the Swift is
held against it.

**Core ML** is its own path. `MossFormerGAN_256frames.mlpackage` was compiled from
PyTorch directly, with a measured 23.8 dB conversion loss against the original —
accepted when the 256-frame variant was chosen over the alternatives. That number
is a property of the package, not a tolerance for anything in Swift.

## Checkpoint layout

Converted weights are `safetensors`, one file per precision, named
`model_<precision>.safetensors` in a single repository — `model_fp32`,
`model_fp16`, `model_int8` and so on. Demucs is the exception: one file per stem,
since asking for vocals alone should not download the other three.

Quantization is `mlx.nn.quantize` at group size 64. `Scripts/convert-sr-precisions.py`
is the worked example, and its docstring is worth reading before quantizing
anything here, because the result is usually disappointing:

> Only 168 `Linear` modules quantize against a Generator that is almost entirely
> convolutions, so the file barely shrinks and nothing gets faster.

That is the general shape on these models. Integer quantization reaches the
linear layers and leaves the convolutions float, so int4 lands at 0.30x the size
rather than the 0.125x a naive multiplier predicts — and it is often *slower*
than fp16. Which precisions are worth publishing is decided by measurement:
[MODEL-PRECISIONS.md](MODEL-PRECISIONS.md) has the tables, and
`Scripts/quantization-report.py` regenerates them.

Some conversions do not survive contact with their model at all. MossFormer2 SR's
fp16 checkpoint holds no non-finite weight and still returns NaN for every
sample, because the forward pass overflows; it is not published, and
`supportedPrecisions` does not offer it.

## Parity

`Parity/` holds the evidence that the ports compute what the references compute:
a fixed input, a recorded reference output, and a number.

The work splits in two. **Generation** needs Python, MLX, the reference code and
the weights; it runs rarely, on a machine that has all of them, and produces
artifacts. **Verification** needs only Swift and those artifacts, so it runs
anywhere — `Tests/AudioToolParityTests` reads them and skips cleanly when they
are absent.

An artifact holds the **input tensor itself**, not a path to a WAV. Decoding the
same file through `soundfile` and through AVFoundation can differ in the last
bit, and a harness that reports that as model divergence is worse than none. It
also means an artifact contains its input audio, so an artifact can only be
published if that audio can be — which is why every case reads a cleared fixture.

Outputs are cut at the seams that matter rather than only at the end: the
super-resolution case exposes the resampled 16 → 48 kHz signal separately from
the enhanced result, the Core ML case separates the model-plus-STFT output from
the stitched one, and Chatterbox compares conditioning tensors because everything
past `prepare_conditionals` samples. A single end-to-end number gives pass/fail;
these give *where*.

### The metric

SNR in dB of the Swift output against the reference. For scale: float32 round-off
alone lands at 110–130 dB, and anything under roughly 60 dB is a real difference
in the port. Thresholds are derived from a first run and recorded in
`ParityThresholds.swift` with the reasoning attached — a threshold chosen before
measuring only encodes what someone hoped for.

Representative measurements:

| Case | Measured |
| --- | ---: |
| USS ResUNet30 | 139.3 / 119.3 dB |
| FRCRN, chunked | 119.8 dB |
| MossFormer2 SR | 114.2 dB direct, 110.1 dB chunked |
| MossFormer2 SE | 80.7 dB direct, 67.1 dB chunked |
| MossFormer2 SE, int8 / int6 / int4 | 81.3 / 81.8 / 77.4 dB |
| MossFormerGAN, Core ML FP32 | 129.0 / 129.6 dB |
| MossFormerGAN, Core ML FP16 | 61.4 / 61.8 dB |

A near-silent input agrees with almost any implementation while looking green, so
the harness refuses to write a case whose input RMS is too low. That guard caught
a real mistake: a USS case built on a clip that opens with six seconds of silence
produced two "separated" outputs that agreed with each other and with nothing
else.

### What parity cannot tell you

Every precision of MossFormer2 SE is held to the *same* threshold as fp32, and
all of them pass in one band. That is correct — both sides load identical packed
bytes and dequantize through the same kernels, so the quantization error is
common to both and cancels — and it is exactly why a green quantized case is
**not** a recommendation to ship that precision.

What a precision costs is a different measurement, against the model's own fp32
output rather than across languages:

```bash
Scripts/quantization-report.py --bench 'BenchmarkResults/bench-*.json'
```

Size comes from the checkpoints, speed and memory from an `audio-tool-bench`
report, quality from the parity artifacts.

### Running generation

```bash
python Parity/generate.py --list
python Parity/generate.py --all
```

Artifacts land in `Parity/artifacts/` as `<case>.safetensors` plus a JSON
sidecar pinning the weights hash, the source-audio hash, the reference module's
hash, the package revision and every library version. Paths in the sidecar are
checkout-relative, because these get published.

Nothing in `Parity/` is a dependency of the Swift package, and no third-party
reference code is vendored into this repository — the adapters import the
reference modules in place.

## Adding a model

1. Convert the weights and publish them, one file per precision.
2. Port the forward pass into `Sources/Models/<Model>/`, source only, carrying
   the upstream `LICENSE` into that directory.
3. Write a provider in the matching backend target: an actor conforming to the
   task protocol, exact about its sample rate, with both a repository-based and a
   path-based initializer.
4. Name the repository in `ModelRepository`, add its variants to `ModelCatalog`,
   and pin it in `ModelPins` — `ModelPinTests` fails the build if you skip this.
5. Add a parity case, measure it, and record the threshold you got.
6. Add a benchmark case if the model's cost is worth publishing, and measure each
   precision you intend to offer before offering it.
