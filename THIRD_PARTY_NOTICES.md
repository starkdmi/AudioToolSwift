# Third-party notices

AudioToolSwift itself is licensed under Apache-2.0 (see `LICENSE`). It also
redistributes source derived from the projects below, and resolves the SwiftPM
dependencies listed at the end.

If you ship a compiled application built on AudioToolSwift, the permissive
licences here still require you to reproduce these notices in your product
documentation or an "acknowledgements" screen. Source-level distribution is
covered by the per-directory `LICENSE` files.

`Docs/licenses.md` is the fuller reference: it also covers model **weights**,
which carry licences independent of the code that runs them.

## Vendored model ports

Each directory carries its own `LICENSE` with the full text.

| Directory | Upstream | Licence |
|---|---|---|
| `Sources/Models/MossFormer2SE/` | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | Apache-2.0 |
| `Sources/Models/MossFormer2SS/` | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | Apache-2.0 |
| `Sources/Models/MossFormer2SR/` | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | Apache-2.0 |
| `Sources/Models/FRCRN/` | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | Apache-2.0 |
| `Sources/Models/USS/` | [ByteDance USS](https://github.com/bytedance/uss) | Apache-2.0, © 2023 ByteDance |
| `Sources/Models/Demucs/` | [adefossez/demucs](https://github.com/adefossez/demucs) | MIT, © Meta Platforms, Inc. and affiliates |
| `Sources/Models/Chatterbox/` | [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) | MIT, © 2025 Resemble AI |
| `Sources/Models/Kokoro/` | [mlalma/kokoro-ios](https://github.com/mlalma/kokoro-ios) | MIT, © 2025 Lassi Maksimainen |

### Mixed-provenance file

`Sources/Models/Chatterbox/S3Gen/HiFiGAN/HiFiGAN.swift` sits in the Chatterbox
tree but its signal-processing and building-block helpers (`interpolate`,
`interpolate1d`, `unwrap`, `mlxStft`, `mlxIstft`, `MLXSTFT`, `InstanceNorm1d`,
`ConvWeighted`, `weightNorm`, `ReflectionPad1d`, `Upsample`, `SineGen`,
`SourceModuleHnNSF`) are adapted from the Kokoro Swift implementation. Both the
Resemble AI and Lassi Maksimainen notices therefore apply to the Chatterbox
directory; `Sources/Models/Chatterbox/LICENSE` reproduces both.

The remaining Chatterbox source, including `S3Gen/Conformer/`, is an original
port written for this project and needs no third-party attribution.

## SwiftPM dependencies

Resolved revisions are pinned in `Package.resolved`.

| Package | Licence |
|---|---|
| `mlx-swift`, `mlx-swift-lm` | MIT |
| `swift-transformers` | Apache-2.0 |
| `SwiftAudio` | Apache-2.0 |
| `MisakiSwift` (fork) | Apache-2.0 |
| `MLXUtilsLibrary` | Apache-2.0 |
| `FluidAudio` | Apache-2.0 |
| `swift-asn1`, `swift-collections`, `swift-crypto`, `swift-numerics` | Apache-2.0 |
| `swift-jinja` | Apache-2.0 |
| `ZIPFoundation`, `yyjson` | MIT |
