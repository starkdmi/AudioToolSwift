# Architecture

Three layers, in dependency order: core types that know nothing about ML,
provider targets that each wrap one backend, and a facade that composes them.

```
AudioTool                 facade: AudioEngine actor, pipeline builder, residency
   ↑
AudioToolMLX  AudioToolCoreML  AudioToolUSS  AudioToolTTS  AudioToolFluidAudio
AudioToolSpeech  AudioToolTranslation  AudioToolMLXTranslation
   ↑                                            ↑
AudioToolCore             AudioBuffer, protocols, catalog, downloader, errors
                          Sources/Models/*      vendored model implementations
```

Each provider target is a separate SwiftPM product, so an app that only needs
transcription never compiles a MossFormer2. The split is by *provider family*
rather than strictly by backend: `AudioToolCoreML` also depends on MLX, because
the Core ML enhancer runs its STFT and ISTFT there and only the model itself
through Core ML. `AudioToolFluidAudio`, `AudioToolSpeech` and
`AudioToolTranslation` are the products that pull in no MLX — and therefore the
only ones free of the Metal library constraint below.

`AudioToolCore` is the only target everything shares, and it has no ML dependency
at all — which is what keeps `swift test` on the core types fast and hermetic.

## AudioToolCore

**`AudioBuffer`** is the currency: an immutable value type holding interleaved
float samples, a sample rate and a channel count. Conversions (`resampled(to:)`,
`converted(toChannels:)`) return new buffers.

**Protocols** describe capability, not implementation. `AudioProcessor` is the
root — it declares the rate, channel counts and resampling quality a model
requires — and the task protocols refine it:

| Protocol | Shape |
| --- | --- |
| `AudioTransform` | one buffer in, one buffer out (enhancement, super-resolution) |
| `SpeechEnhancer` | `AudioTransform` + streaming |
| `SpeechSeparator`, `MusicSeparator`, `UniversalSoundSeparator` | one buffer in, *many* out |
| `Transcriber`, `SoundClassifier`, `DiarizationProvider`, `VADProvider` | audio in, structured data out |
| `SpeechSynthesizer`, `TextTranslator`, `TextPreprocessor` | text in |
| `ManagedModel` | `load()` / `unload()` / `checkIfLoaded()` + a memory estimate |

The split between `AudioProcessor` and `AudioTransform` matters: anything whose
natural output is a list or a transcript is not a 1-to-1 transform, and typing it
as one forces callers to unwrap a first element that has no meaning.

**A provider validates its input rate rather than resampling it.** Handing a
48 kHz buffer to a 16 kHz model throws `sampleRateMismatch`. Adaptation is the
facade's job, and doing it there means a chain of five stages converts once
instead of five times. `preferredResamplingQuality` is part of that contract: it
records the resampler each model was validated against, so it is a correctness
setting rather than a speed dial.

## Providers

Every provider is an actor, which makes concurrent use safe without a lock. The
providers for models this package owns can be constructed either from a
repository (downloading on first load) or from an explicit path (no network at
all); the FluidAudio-backed and Apple-framework providers offer no path
initializer, because they do not own their own downloads.

Loading is separate from registration. `register(...)` is bookkeeping and returns
immediately; the weights arrive on first use. An earlier design kicked off a
download from `register` in a detached task, which meant a call that looked like
configuration could quietly pull hundreds of megabytes and swallow the error.

`ModelLoadGate` serialises loads within a provider, so two concurrent first-use
calls do not both materialise the weights.

### Chunking

Long input is processed in overlapping chunks, and the seam handling is part of
what each port was validated against — the amount of overlap, whether edges are
discarded, and how the pieces are cross-faded all change the output. Providers
declare `minChunkSize` and `recommendedChunkSize`; `MLXOverlap` implements the
shared assembly; the super-resolution path uses 25% overlap with discarded edges
because that is what its reference implementation does.

### Memory

`MLXCachePolicy` installs two process-global MLX limits — a 512 MiB allocator
cache and a memory ceiling of 60% of RAM, capped by the GPU's recommended working
set — inside every provider's `load()`. MLX's own defaults scale with the machine
rather than with the work, and the cache limit defaults to the *memory* limit,
which on a 16 GB Mac lets a long chunked job climb until the OS pushes back.

Above that, `ModelResidency` tracks which models are loaded, charges each one its
measured footprint, and evicts least-recently-used models when the configured
limit is reached. `AudioEngine` engages residency at the point of use rather than
at registration, which is both where the LRU timestamp means something and the
only place an evicted model can be brought back before inference is attempted.

The footprints are measured, not inferred from checkpoint size: the two differ by
3–55x, because the forward pass dominates.

## AudioEngine

The facade is an actor holding a registry of providers by task and model id. It
does four things a caller would otherwise repeat:

1. **Adapts input** to whatever the target provider declares, once.
2. **Brackets residency** around each call, so a model is protected from eviction
   while it is running.
3. **Restores the caller's sample rate** on the way out (`preservingSampleRate`).
4. **Composes stages** into pipelines.

## Pipelines

`PipelineBuilder` is a value type describing stages; nothing runs until
`process(audio:)`, `process(source:)` or `stream(source:)`.

```swift
try await engine.pipeline()
    .analyze()                                     // VAD + diarization
    .conditionally({ ($0.analysis?.speakers.maxOverlappingSpeakers ?? 0) > 1 },
                   then: { $0.separateOverlap() })
    .enhance(.mossformerSE48k)
    .transcribe(.parakeet)
    .mergeTranscriptionWithDiarization()
    .process(source: .file(url))
```

Each of those stages resolves to a registered provider and throws
`modelNotLoaded` if there is none — so this chain assumes a VAD, a diarizer, a
separator, an enhancer and a transcriber were all registered first. Registration
is per capability, and every capability the pipeline can express has a public
`register(...)` overload.

Beyond linear chains it supports `conditionally`, `parallel` and `forEach`, and
`stream(source:)` yields `PipelineEvent`s as segments complete so a UI can render
progressively.

The builder holds its engine **weakly**: a builder is a value callers keep, and a
strong reference would pin the engine and every loaded model for as long as the
value lived. The cost is that dropping the engine while holding a builder
surfaces as an error at `process` time.

## Vendored model implementations

`Sources/Models/` holds the Swift ports themselves — MossFormer2 SE/SS/SR, FRCRN,
Demucs, USS, Kokoro, Chatterbox — as source only, no weights. They keep their
original module names, and they build in **Swift 5 language mode** while
everything above them is Swift 6.

That is deliberate rather than pending work. Each was written as a standalone
package declaring `swift-tools-version: 5.9`, and promoting them turns MLX's
non-Sendable `MLXArray` and the models' kernel caches into roughly twenty hard
errors. They are inference internals behind a fully Swift 6 public API; migrating
them risks changing numerical behaviour and belongs in its own pass with
benchmarks either side.

Each directory carries its upstream `LICENSE` — see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Weights

`ModelRepository` names every Hugging Face coordinate once, below both consumers,
because the catalog and the providers used to keep separate copies that drifted:
a user could download from one repository and have inference fail looking in
another.

`ModelCatalog` describes variants for a download UI; `ModelPins` records the
commit and per-file SHA-256 of every repository; `ModelDownloader` resolves
against the pin and verifies the bytes it wrote. `ModelFiles` distinguishes what
to *fetch* from what must be present to *load* — a distinction that matters,
because gating the presence check on the download manifest made every load pay a
repository round-trip for a file it never opened.

## Build constraint

MLX needs a compiled Metal library that Xcode's build system produces and plain
SwiftPM does not. Build through Xcode or `xcodebuild` and run binaries from the
products directory, or run `Scripts/build_mlx_metallib.sh` once to stage the
library for `swift build` and `swift test`. [AGENTS.md](../AGENTS.md) has the
detail and the upstream references.
