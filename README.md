# AudioToolSwift

On-device speech and audio ML for Apple platforms: speech enhancement, speaker
separation, super-resolution, source separation, text-to-speech, transcription
and diarization — one Swift API over MLX and Core ML.

[![Swift 6.2](https://img.shields.io/badge/Swift-6.2-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/platforms-iOS%2018%20%7C%20macOS%2015-blue.svg)](#requirements)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

Everything runs locally; nothing is sent anywhere. Weights download from Hugging
Face on first use — every repository this package manages is pinned to a commit
and verified by SHA-256. The FluidAudio-backed transcription and diarization
models are the exception: FluidAudio fetches them itself from a mutable default
branch, so those bytes are neither pinned nor verified, and the Silero VAD falls
back to that path if its pinned snapshot cannot be fetched. See
[Docs/models.md](Docs/models.md#downloads-pinning-and-pre-fetching).

## Speed

Realtime factor — 21.9x means a minute of audio takes under three seconds.
Apple M1 Pro (16 GB), 30 s of audio, median of three timed runs.

| Task | Model | Rate | Speed |
| --- | --- | ---: | ---: |
| Speech enhancement | MossFormer2 SE (fp16) | 48 kHz | 25.5x |
| Speech enhancement | FRCRN | 16 kHz | 8.7x |
| Speech enhancement | MossFormerGAN, Core ML (FP16) | 16 kHz | 4.2x |
| Speaker separation | MossFormer2 SS, 2 speakers | 16 kHz | 1.7x |
| Speaker separation | MossFormer2 SS, 3 speakers | 8 kHz | 3.1x |
| Super-resolution | MossFormer2 SR, 16 → 48 kHz | 48 kHz | 1.9x |
| Music separation | Demucs, vocals only | 44.1 kHz | 13.4x |
| Music separation | Demucs, four stems | 44.1 kHz | 3.4x |
| Sound separation | USS ResUNet30 | 32 kHz | 28.8x |
| Text to speech | Chatterbox (8bit) | 24 kHz | 1.0x¹ |

¹ TTS generates rather than consumes audio, so its rate has a different
denominator and is not comparable to the rows above.

Absolute numbers are hardware-specific; the ordering has been stable across the
models measured so far. [Docs/benchmarks.md](Docs/benchmarks.md) has the full
table with run dates, memory and methodology, and
[Docs/model-precisions.md](Docs/model-precisions.md) says which checkpoint to
pick — the smallest is rarely the right one.

## Install

```swift
.package(url: "https://github.com/starkdmi/AudioToolSwift.git", branch: "main")
```

There is no tagged release yet; track `main` until `0.1.0` ships, then pin with
`from: "0.1.0"`.

Then depend on the libraries you need — each backend is its own product, so an
app that only transcribes never compiles a MossFormer2:

```swift
.product(name: "AudioTool", package: "AudioToolSwift"),      // facade + pipelines
.product(name: "AudioToolMLX", package: "AudioToolSwift"),   // MLX providers
.product(name: "AudioToolCoreML", package: "AudioToolSwift") // Core ML providers
```

`AudioToolUSS`, `AudioToolTTS`, `AudioToolFluidAudio`, `AudioToolSpeech`,
`AudioToolTranslation` and `AudioToolMLXTranslation` are the rest.

`AudioToolCoreML` links MLX too, despite the name: the Core ML enhancer runs its
STFT and ISTFT through MLX and only the model itself through Core ML. The
MLX-free products are `AudioToolFluidAudio`, `AudioToolSpeech` and
`AudioToolTranslation`.

## Quick start

```swift
import AudioTool
import AudioToolMLX

let engine = AudioEngine()
await engine.register(enhancer: MossFormer2SE48KProvider(precision: .fp16),
                      for: .mossformerSE48k)

let noisy = try await engine.loadAudio(from: inputURL)
let clean = try await engine.enhance(noisy, model: .mossformerSE48k)
try await engine.saveAudio(clean, to: outputURL)
```

`AudioEngine` is an actor. Registering a provider does not load it: the model
loads on first use, stays resident while in use, and is evicted under memory
pressure by an LRU policy you can bound through `AudioToolConfiguration`.

That applies to the providers this package owns, which conform to `ManagedModel`.
The FluidAudio and Apple-framework providers do not, so the engine calls them
straight through — nothing loads them for you, and you `load()` them yourself
before registering.

Input is adapted at the boundary — a 48 kHz stereo file handed to a 16 kHz mono
model is resampled and downmixed once, and you get your own rate back.

### Pipelines

Stages compose, and the intermediate buffers stay inside the engine. Every stage
needs its provider registered first — a pipeline stage whose provider is missing
throws `modelNotLoaded` rather than loading something for you:

```swift
import AudioToolFluidAudio

// Not `ManagedModel`, so load before registering - see above.
let vad = FluidAudioVADProvider()
let transcriber = FluidAudioTranscriber()
try await vad.load()
try await transcriber.load()

await engine.register(vad: vad)
await engine.register(transcriber: transcriber, for: .parakeet)

let result = try await engine.pipeline()
    .detect()                       // voice activity
    .enhance(.mossformerSE48k)      // registered above
    .transcribe(.parakeet)
    .process(source: .file(url))
```

`stream(source:)` runs the same pipeline as an `AsyncThrowingStream` of events,
so a UI can show segments as they land instead of waiting for the whole file.

### Other tasks

```swift
// Separate overlapping speakers
await engine.register(separator: MossFormer2SSProvider(model: .twoSpeaker),
                      for: .mossformer2spk)
let speakers = try await engine.separate(mixture, model: .mossformer2spk)

// Upscale 16 kHz to 48 kHz
await engine.register(upscaler: MossFormer2SR48KProvider())
let wide = try await engine.upscale(narrow)

// Speak
let voice = SynthesisModel.kokoro(language: .americanEnglish, voice: "af_heart")
await engine.register(synthesizer: KokoroTTSProvider(), for: voice)
let speech = try await engine.synthesize("Hello.", voice: "af_heart", model: voice)
```

## Command line

```bash
swift run audio-tool -m se48k -i noisy.wav -o clean.wav
```

Models: `frcrn`, `frcrn-bg`, `se48k`, `se48k-bg`, `demucs`, `ss_2spk`,
`ss_3spk`, `ss_whamr`, `sr48k`, `transcribe`. The `-bg` variants keep the
background as a second output instead of discarding it.

`audio-tool-bench` measures the MLX and Core ML models on your own machine — 22
cases, including Chatterbox TTS. The FluidAudio-backed and Apple-framework
providers are deliberately out of scope, since their numbers describe those
projects rather than this one. See [Docs/running-benchmarks.md](Docs/running-benchmarks.md).

## Requirements

- iOS 18+ / macOS 15+, Apple silicon, Swift 6.2 (Xcode 26.0 or newer)
- Swift 6.2 is a hard floor, not a preference: MLXUtilsLibrary, which Kokoro's
  G2P pulls in, declares `swift-tools-version: 6.2` in every published tag, so an
  older toolchain cannot load the dependency graph. Xcode 26.0 through 26.3 run on
  macOS Sequoia 15.6, so this does not require Tahoe.
- MLX models need a compiled Metal library, which Xcode and `xcodebuild` produce
  automatically. Plain `swift build` does not: run
  `./Scripts/build_mlx_metallib.sh` once if you work through SwiftPM directly.
  [AGENTS.md](AGENTS.md) explains why.
- The Apple-framework providers (Speech, Translation, AVSpeechSynthesizer) and
  the FluidAudio-backed ones have no such constraint. The Core ML enhancer does:
  its STFT runs in MLX.

## Documentation

| | |
| --- | --- |
| [models.md](Docs/models.md) | Every model: what it is for, its rate, size and repository |
| [architecture.md](Docs/architecture.md) | How the facade, providers and pipeline fit together |
| [model-precisions.md](Docs/model-precisions.md) | Which checkpoint to download, measured |
| [benchmarks.md](Docs/benchmarks.md) | Speed and memory, with methodology |
| [running-benchmarks.md](Docs/running-benchmarks.md) | Running the benchmark harness yourself |
| [conversion.md](Docs/conversion.md) | How the PyTorch models became MLX ports, and how parity is proven |
| [testing.md](Docs/testing.md) | Test layers and what each one needs |
| [licenses.md](Docs/licenses.md) | Provenance for every vendored port and every weight |

## Licence

Apache-2.0 — see [LICENSE](LICENSE).

The vendored model ports carry their upstream licences (Apache-2.0 and MIT); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which you must reproduce if you
ship a binary built on this package. Model **weights** are licensed separately
from the code that runs them: no model here carries a non-commercial clause, but
each has attribution terms of its own, and TranslateGemma is governed by the
Gemma Terms rather than an open-source licence.
[Docs/licenses.md](Docs/licenses.md) records what a host must carry, per model.
