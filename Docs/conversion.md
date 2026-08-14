# Conversion validation

None of these models were trained here. Each one was converted from a PyTorch
original to MLX or Core ML, re-implemented in Swift, and then held to a number
that says the Swift agrees with the reference. This page records that chain and
its completed validation.

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

Quantization uses `mlx.nn.quantize` at group size 64, but it is model-specific:
each conversion has to construct the exact reference model before rewriting its
parameters. Integer quantization reaches the linear layers and leaves the
convolutions float, so int4 can land around 0.30x the size rather than the 0.125x
a naive multiplier predicts — and it is often *slower* than fp16. Which
precisions are worth publishing is decided by measurement; the recorded tables
are in [model-precisions.md](model-precisions.md).

Some conversions do not survive contact with their model at all. MossFormer2 SR's
fp16 checkpoint holds no non-finite weight and still returns NaN for every
sample, because the forward pass overflows; it is not published, and
`supportedPrecisions` does not offer it.

## Conversion validation

Before release, fixed inputs were run through both the Python MLX references and
the Swift ports. Outputs were compared at meaningful internal seams as well as at
the public API boundary. This caught differences in resampling, chunk assembly,
Core ML tensor strides, conditioning, and quantized checkpoint loading.

The metric was SNR in dB of the Swift output against the Python MLX reference.
Float32 round-off alone lands around 110–130 dB; results below roughly 60 dB
indicate a material implementation difference.

Representative completed measurements:

| Case | Measured |
| --- | ---: |
| USS ResUNet30 | 139.3 / 119.3 dB |
| FRCRN, chunked | 119.8 dB |
| MossFormer2 SR | 114.2 dB direct, 110.1 dB chunked |
| MossFormer2 SE | 80.7 dB direct, 67.1 dB chunked |
| MossFormer2 SE, int8 / int6 / int4 | 81.3 / 81.8 / 77.4 dB |
| MossFormerGAN, Core ML FP32 | 129.0 / 129.6 dB |
| MossFormerGAN, Core ML FP16 | 61.4 / 61.8 dB |

These measurements establish the completed conversion chain; they do not measure
model quality against clean audio. Precision quality, speed, and memory are
separate measurements reported in [model-precisions.md](model-precisions.md).
The generation harness and unpublished Python reference ports remain maintainer
material and are not presented as a reproducible public test suite.

## Adding a model

1. Convert the weights and publish them, one file per precision.
2. Port the forward pass into `Sources/Models/<Model>/`, source only, carrying
   the upstream `LICENSE` into that directory.
3. Write a provider in the matching backend target: an actor conforming to the
   task protocol, exact about its sample rate, with both a repository-based and a
   path-based initializer.
4. Name the repository in `ModelRepository`, add its variants to `ModelCatalog`,
   and pin it in `ModelPins` — `ModelPinTests` fails the build if you skip this.
5. Validate the Swift port against its reference implementation and record the
   measured result.
6. Add a benchmark case if the model's cost is worth publishing, and measure each
   precision you intend to offer before offering it.
