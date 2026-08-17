# AGENTS.md - AudioTool Swift Framework

Guidelines for AI coding agents working on this Swift audio processing framework.

## Project Overview

AudioTool is a Swift package for audio processing: speech enhancement, speaker separation, 
super-resolution, TTS, transcription, and diarization. Uses MLX for GPU acceleration on Apple Silicon.

## Build Commands

### Warnings are errors, on request

`-warnings-as-errors` is opt-in through an environment variable and CI sets it:

```bash
AUDIOTOOL_STRICT_WARNINGS=1 swift build --build-tests
```

It cannot be unconditional. The flag reaches the compiler through
`.unsafeFlags`, and SwiftPM refuses to let a product whose targets carry unsafe
flags be used as a versioned dependency - a consumer writing `from: "0.1.0"`
fails resolution with "contains unsafe build flags". Same reason no dependency
here is pinned by branch or revision. `Scripts/check-publishable.sh` fails if
either creeps back, and CI runs it.

Set it for a whole session rather than one step: the manifest is evaluated per
invocation, so a `swift build` that sees it and a `swift test` that does not will
rebuild everything between them.

### Checking the declared Swift floor

```bash
./Scripts/check-swift-floor.sh
```

The manifest declares Swift 6.2 and every local build uses whatever Xcode is
installed, which on macOS Tahoe is 6.3 or newer - Xcode 26.4.1 dropped Sequoia
and took the 6.2 toolchains with it. So code that only compiles on 6.3 looks
fine here and fails for everyone on the floor.

That is not hypothetical. It has happened twice, both times in code that had just
been written: an expression 6.2's type checker refuses ("unable to type-check
this expression in reasonable time") where 6.3 handles it in under 100ms, and
`Test.cancel`, which does not exist in the Swift Testing 6.2 ships. Each cost a
CI round trip. The script needs a swift.org 6.2 toolchain installed alongside
Xcode - one `installer` command, no sudo, path in the script header - and then
reproduces the CI job locally in about two minutes.

### Tagging a release

```bash
python3 Scripts/check-release.py 0.1.0
```

Three things in the tree describe the *current* release and are wrong the moment
a tag exists, so they belong in the release commit itself rather than after it,
and that script is what checks them:

1. `README.md` - the install snippet in the `## Install` section must read
   `from: "<version>"`, and nothing may still say a tagged release does not exist.
2. `CHANGELOG.md` - the top heading needs an ISO date, not `unreleased`.
3. `Docs/licenses.md` - the resolved dependency table must match
   `Package.resolved`.

Then cut the release with the **Release** workflow, giving it the version. It
re-runs those checks, `Scripts/check-publishable.sh`, and a check that CI is
green for that exact commit, and creates the tag only if all of them pass.

Order matters and is the whole point. A workflow triggered by a pushed tag runs
after GitHub has accepted it, and a tag is immutable in every resolver cache that
has already seen it - so validation that happens afterwards reports the mistake
instead of preventing it.

### Metal/MLX: the build paths that work

MLX needs a compiled Metal shader library. Bare command-line SwiftPM cannot build
one: mlx-swift ships its kernels under `Source/Cmlx/mlx-generated/metal`, and
Xcode's build system compiles and bundles them automatically while `swift build`,
`swift test` and `swift run` do not. So `swift build` *compiles* fine and then any
GPU operation fails at runtime with:

```
MLX error: Failed to load the default metallib. library not found
```

This is documented by mlx-swift itself
([README SwiftPM note](https://github.com/ml-explore/mlx-swift#swiftpm),
[issue #36](https://github.com/ml-explore/mlx-swift/issues/36)) and is inherited
by mlx-swift-lm. It is not an extra step for an Xcode or `xcodebuild` workflow.
mlx-swift's own README answers it with `xcodebuild`, and mlx-swift-examples ships
an `mlx-run` wrapper that resolves `BUILT_PRODUCTS_DIR` and runs the binary from
there. (mlx-swift#349 is a separate Tuist resource-bundle issue, not this.)

#### Primary: xcodebuild (upstream's answer)

Xcode compiles the shaders as part of the build, so nothing extra is needed.

```bash
xcodebuild build -scheme audio-tool -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
.build/DerivedData/Build/Products/Release/audio-tool -m se48k -i noisy.wav -o clean.wav
```

Note you must run the binary **from the products directory** — that is where the
metallib lives (`Build/Products/<config>/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`).

#### Secondary: build the metallib once, then plain SwiftPM works

Useful because `swift test` is far faster than `xcodebuild test` and its output is
legible. Requires the Metal Toolchain, which ships separately from Xcode.

```bash
xcodebuild -downloadComponent MetalToolchain   # one-off
swift build --build-tests
./Scripts/build_mlx_metallib.sh debug          # compiles + installs mlx.metallib
swift test
swift run audio-tool -m se48k -i noisy.wav -o clean.wav
```

`Scripts/build_mlx_metallib.sh` compiles the kernels with `xcrun metal` and places
`mlx.metallib` next to the SwiftPM binaries and inside each `.xctest` bundle. MLX
looks for a colocated `mlx.metallib` before it tries any bundle, so that is enough.
It content-hashes the sources and is a no-op when nothing changed.

This path is ours, not upstream's. It is verified working here, but it compiles the
shaders independently of Xcode, so if you ever see behaviour that differs between
the two paths, trust xcodebuild and say so.

#### Naming constraint

The CLI product is `audio-tool`, not `audiotool`. An `audiotool` product collides
case-insensitively with the `AudioTool` library target; Xcode then folds the CLI's
`main.swift` into module `AudioTool` and the link fails with a missing `_main`.
`swift build` tolerates the clash silently, which is how it went unnoticed.

### Running Tests

```bash
# Run ALL tests via Xcode (alternative to Scripts/build_mlx_metallib.sh + swift test)
xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' -derivedDataPath .build/DerivedData

# Run specific test suite
xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData \
  -only-testing:AudioToolMLXIntegrationTests

# Run single test class
xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData \
  -only-testing:AudioToolMLXIntegrationTests/FRCRNChunkingIntegrationTests

# Run single test method
xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData \
  -only-testing:AudioToolMLXIntegrationTests/FRCRNChunkingIntegrationTests/testFRCRNWithChunking
```

### Test layers

`swift test` is meant to be the fast signal - mocks, no models, no network:

```bash
SKIP_MLX_TESTS=1 swift test         # ~10s, no GPU and no metallib needed
swift test                          # same plus the hermetic MLX suites (needs a metallib)
RUN_INTEGRATION_TESTS=1 swift test  # adds Apple Speech/TTS/Translation, RUAccent
```

The first line is the one that works on a fresh clone. Bare `swift test` also
runs suites that evaluate MLX arrays, and without a staged metallib MLX aborts
the process rather than failing a test.

Integration suites in the AudioToolTests target are opt-in. They drive Apple
SpeechAnalyzer, Apple TTS, Apple Translation and RUAccent, take 38-60s each and
download models on first use; two of them used to exceed Swift Testing's 60-second
limit and fail the whole run.

Model-backed suites in the other targets need weights, and skip without them - see
below.

### Benchmarks

Two layers, answering different questions.

#### Whole models: `audio-tool-bench`

What each model costs on this machine - load time, RTF, peak memory, CPU time -
written as JSON plus a markdown summary that carries the machine and the settings,
so runs from different machines are comparable. See `Docs/running-benchmarks.md`.

```bash
xcodebuild build -scheme audio-tool-bench -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
.build/DerivedData/Build/Products/Release/audio-tool-bench --prefetch   # once per machine
.build/DerivedData/Build/Products/Release/audio-tool-bench --list
.build/DerivedData/Build/Products/Release/audio-tool-bench -t "$(git rev-parse --short HEAD)"
```

Run it from the products directory, same metallib reason as `audio-tool`.

Every case runs in **its own child process, one at a time**, with a cooldown
between them. That is not ceremony: MLX's allocator is process-global and its cache
survives `unload()`, so two models measured in one process share a high-water mark
and the second inherits the first's. It also means a model that exhausts memory
kills one case rather than the run. `--in-process` turns it off and the report
notes that the memory columns no longer mean anything.

Two MLX caps are applied before any model loads, because MLX's defaults scale with
the machine rather than the work - the cache limit defaults to the *memory* limit,
which is ~16 GB on a 16 GB Mac. The benchmark defaults to the same limits the
package applies in production - a 512 MB cache and a memory limit of 60% of RAM
(`MLXCachePolicy`, mirrored by `USSInference` and `MemoryBudget`, pinned together
by `MLXProcessLimitsTests`). `--gpu-cache-limit` and `--gpu-memory-limit` override,
but only until a provider's `load()` reinstalls the production values - the report
records both the requested and the effective figure.

Scope is MLX and CoreML - 22 cases. FluidAudio-backed providers are excluded on
purpose: they measure a third-party pipeline. TTS *is* measured, in its own
column, because its RTF counts audio generated rather than consumed
(`RateBasis.output`) and the two denominators are not comparable.

Case ids are an interface. Renaming one breaks comparison with every report already
recorded; bump `BenchmarkReport.currentSchemaVersion` when a field changes meaning.

#### Below a model: gated test suites

`ResamplerBenchmarkTests` and `USSPrewarmBenchmarkTests` measure a resampling
kernel and an embedding-swap strategy - things that need `@testable` access and a
fixture. Gated on `RUN_BENCHMARKS=1` *and* on being compiled with optimisation,
because a debug build of numeric code measures the compiler rather than the code:
`ResamplerBenchmarkTests` runs ~70x slower in debug and moves its two kernels apart
by 17x instead of 3x. The suite skips rather than print that.

```bash
./Scripts/build_mlx_metallib.sh release
RUN_BENCHMARKS=1 swift test -c release -Xswiftc -enable-testing \
  --filter ResamplerBenchmarkTests
```

`-Xswiftc -enable-testing` is required: release builds do not enable testability, and
`@testable import` fails without it.

### Running models from local weights

Weights are fetched from HuggingFace at runtime, but every provider also takes an
explicit path, so development does not depend on any repo being published:

```swift
FRCRNSE16KProvider(weightsPath: "...frcrn_se_16k.safetensors")
MossFormer2SE48KProvider(weightsPath: "...")
MossFormer2SSProvider(model: .twoSpeaker, weightsPath: "...")
MossFormer2SR48KProvider(weightsPath: "...", configPath: "...")
DemucsProvider(weightsDirectory: "...")          // one .safetensors per stem
USSProviders.separation(weightsPath: "...resunet30_fp16.safetensors")
```

USS model tests read that path from the environment and skip without it:

```bash
./Scripts/build_mlx_metallib.sh debug     # once, or MLX aborts under SwiftPM
AUDIOTOOL_USS_WEIGHTS=/path/to/resunet30_fp32.safetensors \
  RUN_INTEGRATION_TESTS=1 swift test --filter AudioToolUSSTests
```

All three parts matter: the suite is gated on `RUN_INTEGRATION_TESTS` before it
looks at the weights path, so passing only the path gives a green run in which
every test skipped.

### Test Suites

| Suite | Target | Description |
|-------|--------|-------------|
| `AudioToolMLXIntegrationTests` | MLX models | FRCRN, MossFormer2 SE/SS/SR |
| `AudioToolUSSTests` | USS/Demucs/GAN | Background extraction tests |
| `AudioToolFluidAudioTests` | VAD/Diarization | Silero VAD, Pyannote, Sortformer |
| `AudioToolTests` | Core types | AudioBuffer, chunking, protocols |

### Running the audio-tool CLI

```bash
swift run audio-tool -m <model> -i input.wav [-o output.wav]
```

Available models: `frcrn`, `frcrn-bg`, `se48k`, `se48k-bg`, `demucs`, `ss_2spk`, `ss_3spk`,
`ss_whamr`, `sr48k`, `streaming_verify`, `transcribe`

The TTS and diarization subcommands were removed: they took no arguments and drove
hardcoded fixtures rather than input files. Use `AudioToolTTS` directly until a real
subcommand exists.

## Code Style Guidelines

### File Header
```swift
//
//  FileName.swift
//  ModuleName
//
//  Brief description of purpose
//
```

### Imports Order
1. Foundation/System frameworks
2. AudioTool modules (AudioTool, AudioToolCore, AudioToolMLX)
3. External dependencies with `@preconcurrency` for non-Sendable types

```swift
import Foundation
import AudioTool
import AudioToolCore
@preconcurrency import MLX
@preconcurrency import MLXNN
@preconcurrency import AudioUtils
```

### Actor-Based Providers

All ML providers are actors for thread safety:

```swift
public actor MossFormer2SE48KProvider: SpeechEnhancer {
    public nonisolated let sampleRate: Int = 48000
    public nonisolated let inputChannels: Int = 1
    
    private var model: SomeModel?
    
    public func load() async throws { ... }
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer { ... }
}
```

### Naming Conventions

- **Types**: PascalCase (`AudioBuffer`, `AudioToolError`)
- **Functions/Properties**: camelCase (`sampleRate`, `processWithChunking`)
- **Provider classes**: `<Model><Task>Provider` (e.g., `FRCRNSE16KProvider`)
- **Constants**: camelCase or SCREAMING_SNAKE for ML config
- **Test functions**: `test<Feature>` or `test<Model><Behavior>`

### Error Handling

Use `AudioToolError` enum for all framework errors:

```swift
public enum AudioToolError: Error, Sendable {
    case modelNotFound(String)
    case modelNotLoaded(String)
    case sampleRateMismatch(expected: Int, found: Int)
    case pipelineConfigurationInvalid(String)
    // ...
}

// Usage
throw AudioToolError.modelNotLoaded("MossFormer2_SE_48K")
```

### AudioBuffer

Core immutable audio type - always use this for audio data:

```swift
let input = AudioBuffer(samples: floatArray, sampleRate: 16000, channels: 1)
print("Duration: \(input.duration)s, Frames: \(input.frameCount)")
```

### Project Root Resolution

For tests and CLI tools, compute project root from `#filePath`:

```swift
private let projectRoot: String = {
    var url = URL(fileURLWithPath: #filePath)
    for _ in 0..<4 { url.deleteLastPathComponent() }  // Adjust levels as needed
    return url.path
}()
```

### Test Structure (Swift Testing)

```swift
import Testing
@testable import AudioToolMLX

@Suite("Model Name Tests", .tags(.integration))
struct ModelNameTests {
    
    @Test("Descriptive test name")
    func testFeature() async throws {
        guard TestConfig.shouldRunIntegrationTests else { return }
        
        // Arrange
        let provider = SomeProvider()
        try await provider.load()
        
        // Act
        let output = try await provider.process(input)
        
        // Assert
        #expect(output.sampleRate == 48000)
        #expect(output.frameCount > 0)
    }
}
```

### Output Paths

Write test and CLI output to a temporary directory or an explicit `-o` path. Do not
write into `Sources/Models/` - those are vendored model sources, not scratch space,
and everything there is source-only by design.

Model *weights* are never in the tree at all; they download from HuggingFace at
runtime via `ModelDownloader`. See `Tests/*/Fixtures/README.md` for input audio.

### MLX Best Practices

```swift
// Always eval() after MLX operations
let output = model(input)
eval(output)

// Clear GPU cache for memory management
GPU.clearCache()

// Set memory limits for large models
GPU.set(memoryLimit: 6 * 1024 * 1024 * 1024)  // 6GB
```

### HuggingFace Model Downloads

```swift
// Auto-download with ModelDownloader
if let cached = ModelDownloader.shared.localPath(for: "starkdmi/ModelName") {
    // Use cached path
} else {
    let path = try await ModelDownloader.shared.downloadAndGetPath(
        repo: "starkdmi/ModelName",
        matching: ["*.safetensors", "config.json"]
    )
}
```

## Directory Structure

```
AudioToolSwift/
├── Sources/
│   ├── AudioTool/           # Facade + pipeline API (AudioEngine actor)
│   ├── AudioToolCore/       # Core types (AudioBuffer, Errors, Protocols)
│   ├── AudioToolMLX/        # MLX providers (SE, SS, SR, Demucs)
│   ├── AudioToolCoreML/     # CoreML providers (MossFormerGAN)
│   ├── AudioToolUSS/        # Universal sound separation
│   ├── AudioToolTTS/        # Kokoro, Chatterbox TTS
│   ├── AudioToolFluidAudio/ # VAD, transcription, diarization
│   ├── AudioToolSpeech/     # Apple SpeechAnalyzer (iOS 26+)
│   ├── AudioToolTranslation/     # Apple Translation framework
│   ├── AudioToolMLXTranslation/  # TranslateGemma
│   ├── AudioToolCLI/        # `audio-tool` executable
│   ├── AudioToolBenchmark/     # Benchmark harness: profile, probes, catalog, report
│   ├── AudioToolBenchmarkCLI/  # `audio-tool-bench` executable
│   └── Models/              # Vendored model implementations, source only
│       ├── MossFormer2SE/ MossFormer2SS/ MossFormer2SR/
│       ├── FRCRN/ Demucs/ USS/ Kokoro/ Chatterbox/
├── Tests/
│   ├── AudioToolTests/                  # Unit tests with mocks, no MLX
│   ├── AudioToolMLXIntegrationTests/
│   ├── AudioToolUSSTests/
│   ├── AudioToolFluidAudioTests/
│   └── AudioToolMLXTranslationTests/
├── Scripts/build_mlx_metallib.sh
└── Package.swift
```

There is no sibling `../Models/` directory and no local path dependencies. Model
implementations are vendored under `Sources/Models/` as source only; weights are
fetched from HuggingFace at runtime.

## Common Issues

1. **metallib not found**: run `Scripts/build_mlx_metallib.sh`, or build via xcodebuild — see the Metal/MLX section above for why
2. **Model not found**: Check HuggingFace cache at `~/.cache/huggingface/hub/`
3. **Path errors after machine migration**: Use `#filePath` based project root, not hardcoded paths
4. **Memory issues**: Use `GPU.clearCache()` between chunks, set memory limits
