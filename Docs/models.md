# Models

Every model this package can run, what it is for, and where its weights come
from. Sizes are the default precision's download;
[MODEL-PRECISIONS.md](MODEL-PRECISIONS.md) covers the alternatives and says which
one to pick. Speeds are in [benchmarks.md](benchmarks.md).

Weights are never in this repository. They download from Hugging Face on first
use into the shared cache (`~/.cache/huggingface/hub`), and the providers for
models this package owns also accept an explicit path, so an app that ships its
own copies never touches the network — see
[Running from local weights](#running-from-local-weights) for which ones do not.

## Speech enhancement

Removes noise and reverberation from a single voice.

| Model | Provider | Rate | Default | Size | Repository |
| --- | --- | ---: | --- | ---: | --- |
| MossFormer2 SE 48K | `MossFormer2SE48KProvider` | 48 kHz | fp32 | 211 MiB | `starkdmi/MossFormer2_SE_48K_MLX` |
| FRCRN SE 16K | `FRCRNSE16KProvider` | 16 kHz | fp32 | 53 MiB | `starkdmi/FRCRN_SE_16K_MLX` |
| MossFormerGAN SE 16K | `MossFormerGANCoreMLProvider` | 16 kHz | FP16 | 7.6 MiB | `starkdmi/MossFormer_GAN_SE_16K_CoreML` |

MossFormer2 SE publishes fp32, fp16, int8, int6 and int4; **fp16 is the one to
use** — fastest, half the size, and 70 dB against fp32. FRCRN is fp32 only.
MossFormerGAN is compiled for Core ML, so its precision is fixed at conversion
time and the two variants are separate packages; **FP16 is faster and uses 5.5x
less memory** than FP32, the clearest precision win in the package.

Both MossFormer2 SE and FRCRN can return the removed background as a second
signal instead of discarding it (`-bg` in the CLI), which is what makes them
usable for background extraction rather than only for cleanup.

## Speaker separation

Splits overlapping speakers into one track each.

| Model | Speakers | Rate | Size | Repository |
| --- | ---: | ---: | ---: | --- |
| MossFormer2 SS 2-speaker | 2 | 16 kHz | 213 MiB | `starkdmi/MossFormer2_SS_2SPK_16K_MLX` |
| MossFormer2 SS 3-speaker | 3 | 8 kHz | 214 MiB | `starkdmi/MossFormer2_SS_3SPK_8K_MLX` |
| MossFormer2 SS WHAMR | 2 | 8 kHz | 213 MiB | `starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX` |

All three run through `MossFormer2SSProvider`, selected by its `Model` argument,
and all are fp32 only. Pick WHAMR for noisy or reverberant recordings — it is
trained on them — and the 16 kHz 2-speaker model for clean ones. The WHAMR
checkpoint is MIT-licensed where the other two are Apache-2.0.

## Super-resolution

Rebuilds the missing high band of narrowband audio.

| Model | Provider | In → out | Default | Size | Repository |
| --- | --- | --- | --- | ---: | --- |
| MossFormer2 SR 48K | `MossFormer2SR48KProvider` | 16 → 48 kHz | fp32 | 418 MiB | `starkdmi/MossFormer2_SR_48K_MLX` |

int8 is published as a download-size option only: quantization here reaches just
the linear layers of a convolution-heavy generator, so it changes neither speed
nor memory. fp16 is **not** published — its forward pass overflows and returns
NaN for every sample.

This is the heaviest model in the package: ~4 GiB peak footprint, and at 1.9x
realtime on an M1 Pro it has little headroom — 30 s of audio takes about 15 s.

## Source separation

| Model | Provider | Rate | Size | Repository |
| --- | --- | ---: | ---: | --- |
| Demucs v4 (HTDemucs-ft) | `DemucsProvider` | 44.1 kHz | 4 × 80 MiB | `starkdmi/Demucs_MLX` |
| USS ResUNet30 | `USSMLXProvider` | 32 kHz | 102 MiB | `starkdmi/USS_MLX` |

**Demucs** splits music into drums, bass, other and vocals. It is the fine-tuned
variant, one checkpoint per stem, so asking for vocals alone downloads and runs a
quarter of the work.

**USS** extracts whatever class you ask for rather than a fixed stem set. It is
conditioned on a 527-dimensional vector over the AudioSet classes;
`SoundEmbedding` ships seven presets — speech, music, animal, nature, noise,
things, human — and you can build your own from any set of class indices. Use
fp32: fp16 is slower, uses *more* memory, and costs 57 dB, because the weights
are a rounding error next to the segmented forward pass.

## Text to speech

| Model | Provider | Rate | Default | Size | Repository |
| --- | --- | ---: | --- | ---: | --- |
| Kokoro 82M | `KokoroTTSProvider` | 24 kHz | bf16 | 312 MiB | `mlx-community/Kokoro-82M-bf16` |
| Chatterbox | `ChatterboxTTSProvider` | 24 kHz | fp32 | 2.5 GiB | `starkdmi/chatterbox` |
| Apple | `AppleTTSProvider` | system | — | — | on-device, no download |

**Kokoro** is small, fast and multi-voice, with bf16/8bit/6bit/4bit checkpoints
and seven working language profiles (Japanese and Chinese are declared but not
yet wired up). **Chatterbox** is the multilingual one — 23 languages — and does
voice cloning: its speaker encoder stays at full precision in every quantized
checkpoint, so cloning quality is unaffected by the bit width — 8bit matches fp32
speed at 1.7 GiB less memory. **Apple TTS** wraps `AVSpeechSynthesizer` for the
cases where a 2.5 GiB download is not justified.

## Transcription, VAD and diarization

These come from [FluidAudio](https://github.com/FluidInference/FluidAudio) or
Apple's own frameworks. FluidAudio downloads its own weights from a mutable
default branch, outside this package's pins and hash checks — see
[Downloads, pinning and pre-fetching](#downloads-pinning-and-pre-fetching).

| Task | Provider | Backing |
| --- | --- | --- |
| Transcription | `FluidAudioTranscriber` | Parakeet, Core ML |
| Transcription | `AppleSpeechTranscriber` (iOS 26+) | `SpeechAnalyzer` |
| Voice activity | `FluidAudioVADProvider` | Silero VAD, Core ML |
| Diarization | `FluidAudioDiarizationProvider` | Pyannote |
| Diarization | `FluidAudioSortformerProvider` | NVIDIA Sortformer |
| Speaker embedding | `SpeakerEmbeddingProvider` | Pyannote embeddings |

`ModelCatalog` has no transcription entries: both providers fetch their own
weights, so there is nothing for a host download screen to offer. Whisper used to
be listed there — two MLX variants, no `Transcriber` that could load them — and
was removed along with the packages that bundled it. `Transcriber` is the
extension point if you want another engine; register it with
`AudioEngine.register(transcriber:for:)` and add catalog rows only if you also
download the weights yourself.

Translation is available through `AudioToolTranslation` (Apple's Translation
framework) and `AudioToolMLXTranslation` (TranslateGemma, 55+ languages). The
Gemma weights are governed by the Gemma Terms rather than an open-source licence
— see [licenses.md](licenses.md).

## Downloads, pinning and pre-fetching

`ModelCatalog` describes every variant — id, display name, precision, download
size, repository and the exact files needed to load it — which is what a host app
shows in a download screen. `ModelPackage` groups them into bundles ("Speech
Studio Essentials" and friends) for a one-tap pre-download.

Every repository *this package* fetches is pinned in `ModelPins` by commit and by
the SHA-256 of each file. `ModelDownloader` resolves each download against that
revision and verifies the bytes it wrote, throwing `modelIntegrityFailed` on a
mismatch — so an upstream change to a default branch cannot alter either the
weights or their terms underneath you. `ModelPinTests` fails the build if a
catalog repository is neither pinned nor explicitly declared unpublished.

Two exceptions, both in the FluidAudio-backed providers:

- **Transcription, diarization and Sortformer** weights are downloaded by
  FluidAudio itself, and are **neither pinned nor verified**. Pinning the
  FluidAudio package in `Package.resolved` freezes its downloader code and the
  repository names it uses, not the bytes: `ModelRegistry` builds
  `…/resolve/main/…` URLs, so an upstream push to a default branch reaches every
  install. If that matters for your product, fetch those models yourself and hand
  them to FluidAudio.
- **Silero VAD** is fetched through `ModelDownloader` against the pin, but
  `PinnedVADModel` degrades to FluidAudio's own unpinned download when the pinned
  snapshot cannot be fetched or opened — offline, or a repository layout change.
  A 1.1 MB model several pipelines depend on is better unpinned than absent, and
  the fallback is logged so "am I pinned?" has an answer. A **hash mismatch is
  not** one of the fallback cases: that rethrows, since answering "these bytes are
  not what was pinned" by fetching whatever the branch serves today would invert
  the point of pinning.

Memory is accounted per provider by `ModelResidency`, which evicts least-recently
used models when the configured limit is reached. The figures are measured
footprints, not checkpoint sizes: the two differ by 3–55x, because what dominates
is the forward pass rather than the weights.

Residency covers providers that conform to `ManagedModel` — every model this
package owns, across MLX, Core ML, USS, TTS and TranslateGemma. The
FluidAudio-backed and Apple-framework providers do not conform, so `AudioEngine`
calls them directly: they are neither charged against the memory limit nor
evicted, and their footprint is invisible to the budget you configure.

## Running from local weights

The providers for models this package owns take a path and skip the network
entirely:

```swift
MossFormer2SE48KProvider(weightsPath: "…/model_fp16.safetensors")
FRCRNSE16KProvider(weightsPath: "…/model_fp32.safetensors")
MossFormer2SSProvider(model: .twoSpeaker, weightsPath: "…")
MossFormer2SR48KProvider(weightsPath: "…", configPath: "…")
MossFormerGANCoreMLProvider(modelPath: "…")  // compiled .mlmodelc
DemucsProvider(weightsDirectory: "…")        // one .safetensors per stem
USSMLXProvider(weightsPath: "…")
KokoroTTSProvider(modelPath: url)
ChatterboxTTSProvider(modelPath: url)        // weights directory
```

`FluidAudioTranscriber`, the FluidAudio VAD and diarization providers, the Apple
providers and `TranslateGemmaProvider` have no such initializer — they take a
version, a locale or a repository, and their weights arrive through their own
downloader. An offline install has to prime the relevant cache instead.
