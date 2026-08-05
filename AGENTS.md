# AGENTS.md - AudioTool Swift Framework

Guidelines for AI coding agents working on this Swift audio processing framework.

## Project Overview

AudioTool is a Swift package for audio processing: speech enhancement, speaker separation, 
super-resolution, TTS, transcription, and diarization. Uses MLX for GPU acceleration on Apple Silicon.

## Build Commands

### Metal/MLX: why `swift test` fails, and how to make it work

`swift build` works fine, including every MLX target. What fails is **running**:

```
MLX error: Failed to load the default metallib. library not found
```

**Cause.** SwiftPM cannot compile `.metal` sources. mlx-swift ships 32 Metal
kernels under `Source/Cmlx/mlx/mlx/backend/metal/kernels`, but its `Package.swift`
declares no resource, plugin or binary target that turns them into a `.metallib` —
only Xcode's build system does that. So the shader library is never produced and
anything touching the GPU throws at runtime. This is upstream and unfixed:
[ml-explore/mlx-swift#349](https://github.com/ml-explore/mlx-swift/issues/349) is
open, unassigned, with no maintainer response.

**Two working paths.** Pick either:

```bash
# 1. Build the shader library once, then plain SwiftPM works
xcodebuild -downloadComponent MetalToolchain    # one-off, ships separately from Xcode
swift build --build-tests
./Scripts/build_mlx_metallib.sh debug           # compiles + installs mlx.metallib
swift test
```

`Scripts/build_mlx_metallib.sh` compiles the kernels with `xcrun metal` and drops
`mlx.metallib` next to each test binary. MLX searches for a colocated
`mlx.metallib` before it tries any bundle, so that is all it takes. The script
content-hashes the kernel sources and skips recompiling when nothing changed.

```bash
# 2. Or let Xcode do it, which compiles the shaders as part of the build
xcodebuild build -scheme AudioToolSwift-Package -destination 'platform=macOS' -derivedDataPath .build/DerivedData
xcodebuild build -scheme audiotool -destination 'platform=macOS' -derivedDataPath .build/DerivedData
```

Path 1 is preferred for CI and for anything iterative — `swift test` is far faster
than `xcodebuild test` and gives usable output. Path 2 needs no extra component.

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

### Test Suites

| Suite | Target | Description |
|-------|--------|-------------|
| `AudioToolMLXIntegrationTests` | MLX models | FRCRN, MossFormer2 SE/SS/SR |
| `AudioToolUSSTests` | USS/Demucs/GAN | Background extraction tests |
| `AudioToolFluidAudioTests` | VAD/Diarization | Silero VAD, Pyannote, Sortformer |
| `AudioToolTests` | Core types | AudioBuffer, chunking, protocols |

### Running the audiotool CLI

```bash
swift run audiotool -m <model> -i input.wav [-o output.wav]
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
│   ├── AudioToolCLI/        # `audiotool` executable
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
