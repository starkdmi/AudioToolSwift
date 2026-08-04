// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// Common Swift settings for all targets
let commonSwiftSettings: [SwiftSetting] = [
    .unsafeFlags(["-warnings-as-errors"])
]

// FluidAudio wrapper has Sendable issues in upstream dependency - exclude from strict mode
let fluidAudioSwiftSettings: [SwiftSetting] = []

let package = Package(
    name: "AudioToolSwift",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .library(
            name: "AudioTool",
            targets: ["AudioTool"]
        ),
        .library(
            name: "AudioToolMLX",
            targets: ["AudioToolMLX"]
        ),
        .library(
            name: "AudioToolCoreML",
            targets: ["AudioToolCoreML"]
        ),
        .library(
            name: "AudioToolFluidAudio",
            targets: ["AudioToolFluidAudio"]
        ),
        .library(
            name: "AudioToolUSS",
            targets: ["AudioToolUSS"]
        ),
        .library(
            name: "AudioToolTTS",
            targets: ["AudioToolTTS"]
        ),
        .library(
            name: "AudioToolSpeech",
            targets: ["AudioToolSpeech"]
        ),
        .library(
            name: "AudioToolTranslation",
            targets: ["AudioToolTranslation"]
        ),
        .library(
            name: "AudioToolMLXTranslation",
            targets: ["AudioToolMLXTranslation"]
        ),
        .executable(
            name: "audiotool",
            targets: ["AudioToolCLI"]
        ),
    ],
    dependencies: [
        // SwiftAudio for resampling, STFT, and audio I/O (local with buffer capacity fix)
        .package(name: "SwiftAudio", path: "../SwiftAudio"),
        
        // MLX Model packages (local development)
        .package(name: "Mossformer2MLXSwift", path: "../Models/mossformer2_se_mlx_swift"),
        .package(name: "FRCRNMLXSwift", path: "../Models/frcrn_se_mlx_swift"),
        .package(name: "MossFormer2SS", path: "../Models/mosforrmer2_ss_mlx_swift"),
        .package(name: "MossFormer2SR", path: "../Models/mossformer2_sr_mlx_swift"),
        .package(name: "DemucsMLXSwift", path: "../Models/demucs_mlx_swift"),
        .package(name: "USSMLXSwift", path: "../Models/uss_mlx_swift"),
        
        // MLX for neural engine operations
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.29.1")),
        
        // FluidAudio for VAD, transcription, diarization (local fork with mel context fix)
        // Fixes: Chunk boundary truncation bug where audio at exact chunk boundaries caused all-blank predictions
        // TODO: Switch back to hosted version after PR is merged: https://github.com/FluidInference/FluidAudio/pull/XXX
        .package(name: "FluidAudio", path: "../Docs/temp/FluidAudio"),
        
        // Kokoro TTS with MisakiSwift G2P (MIT license, no ESpeakNG)
        // Using local fork with extended misaki[en] language support
        .package(name: "KokoroSwift", path: "../Models/kokoro-ios"),
        
        // ChatterBox Multilingual TTS (MLX, 25 languages)
        .package(name: "ChatterboxMLXSwift", path: "../Models/chatterbox_swift"),
        
        // HuggingFace Hub for model downloading 
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMinor(from: "1.1.6")),
        
        // MLX LLM for TranslateGemma translation
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "2.29.3")),
    ],
    targets: [
        // Core shared infrastructure
        .target(
            name: "AudioToolCore",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/AudioToolCore",
            swiftSettings: commonSwiftSettings
        ),
        
        // Main public API (no MLX/CoreML dependency)
        .target(
            name: "AudioTool",
            dependencies: [
                "AudioToolCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/AudioTool",
            swiftSettings: commonSwiftSettings
        ),
        
        // MLX Backend providers
        .target(
            name: "AudioToolMLX",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                .product(name: "Mossformer2MLXSwift", package: "Mossformer2MLXSwift"),
                .product(name: "FRCRNMLXSwift", package: "FRCRNMLXSwift"),
                .product(name: "MossFormer2SS", package: "MossFormer2SS"),
                .product(name: "MossFormer2SR", package: "MossFormer2SR"),
                .product(name: "DemucsMLXSwift", package: "DemucsMLXSwift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/AudioToolMLX",
            swiftSettings: commonSwiftSettings
        ),
        
        // CoreML Backend providers
        .target(
            name: "AudioToolCoreML",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/AudioToolCoreML",
            swiftSettings: commonSwiftSettings
        ),
        
        // FluidAudio Backend providers (VAD, transcription, diarization)
        // Note: FluidAudio has Sendable issues in upstream - excluded from strict warnings
        .target(
            name: "AudioToolFluidAudio",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/AudioToolFluidAudio",
            swiftSettings: fluidAudioSwiftSettings
        ),
        
        // USS MLX Backend providers (speech separation)
        .target(
            name: "AudioToolUSS",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                .product(name: "USSMLXSwift", package: "USSMLXSwift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/AudioToolUSS",
            swiftSettings: commonSwiftSettings
        ),
        
        // TTS Backend providers (Kokoro with MisakiSwift G2P)
        .target(
            name: "AudioToolTTS",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                "AudioToolFluidAudio",  // For VAD-based audio trimming
                .product(name: "KokoroSwift", package: "KokoroSwift"),
                .product(name: "ChatterboxMLXSwift", package: "ChatterboxMLXSwift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/AudioToolTTS",
            swiftSettings: commonSwiftSettings
        ),
        
        // Speech-to-Text Backend (Apple SpeechAnalyzer, iOS 26+)
        .target(
            name: "AudioToolSpeech",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
            ],
            path: "Sources/AudioToolSpeech",
            swiftSettings: commonSwiftSettings
        ),
        
        // Translation Backend (Apple Translation framework)
        .target(
            name: "AudioToolTranslation",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
            ],
            path: "Sources/AudioToolTranslation",
            swiftSettings: commonSwiftSettings
        ),
        
        // MLX Translation Backend (TranslateGemma, 55+ languages)
        .target(
            name: "AudioToolMLXTranslation",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/AudioToolMLXTranslation",
            swiftSettings: commonSwiftSettings
        ),
        
        // Unit tests with mocks (swift test compatible)
        .testTarget(
            name: "AudioToolTests",
            dependencies: ["AudioTool", "AudioToolTTS", "AudioToolSpeech"],
            path: "Tests/AudioToolTests",
            resources: [
                .copy("Fixtures/")
            ],
            swiftSettings: commonSwiftSettings
        ),
        
        // MLX Integration tests (requires xcodebuild for Metal)
        // Run with: xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS'
        .testTarget(
            name: "AudioToolMLXIntegrationTests",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                "AudioToolMLX",
                "AudioToolTTS",
                "AudioToolFluidAudio",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Tests/AudioToolMLXIntegrationTests",
            resources: [
                .copy("Fixtures/")
            ],
            swiftSettings: commonSwiftSettings
        ),
        
        // FluidAudio Integration tests (VAD, transcription, diarization)
        // Run with: xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' -only-testing:AudioToolFluidAudioTests
        .testTarget(
            name: "AudioToolFluidAudioTests",
            dependencies: [
                "AudioTool",
                "AudioToolFluidAudio",
                "AudioToolMLX",
                "AudioToolCore",
                "AudioToolUSS",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Tests/AudioToolFluidAudioTests",
            resources: [
                .copy("Fixtures/")
            ],
            swiftSettings: commonSwiftSettings
        ),
        
        // USS MLX Integration tests (speech separation)
        // Run with: xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' -only-testing:AudioToolUSSTests
        .testTarget(
            name: "AudioToolUSSTests",
            dependencies: [
                "AudioToolUSS",
                "AudioToolCoreML",
                "AudioToolMLX",
                "AudioToolCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Tests/AudioToolUSSTests",
            resources: [
                .copy("Fixtures/")
            ],
            swiftSettings: commonSwiftSettings
        ),
        
        // MLX Translation Integration tests (TranslateGemma)
        // Run with: xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' -only-testing:AudioToolMLXTranslationTests
        .testTarget(
            name: "AudioToolMLXTranslationTests",
            dependencies: [
                "AudioToolMLXTranslation",
                "AudioTool",
                "AudioToolCore",
            ],
            path: "Tests/AudioToolMLXTranslationTests",
            swiftSettings: commonSwiftSettings
        ),
        // CLI executable for testing MLX providers (use xcodebuild + run directly)
        // Build: xcodebuild build -scheme audiotool -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
        // Run: .build/DerivedData/Build/Products/Release/audiotool --model frcrn --input test.wav --output enhanced.wav
        // Note: AudioToolSpeech (Apple SpeechAnalyzer) requires macOS 26+ and is only conditionally imported
        .executableTarget(
            name: "AudioToolCLI",
            dependencies: [
                "AudioToolMLX",
                "AudioToolTTS",
                "AudioToolFluidAudio",  // For voice matching
                "AudioToolSpeech", // Required for linking even with conditional import
                "AudioToolCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/AudioToolCLI",
            swiftSettings: commonSwiftSettings
        ),
    ],
    swiftLanguageModes: [.v6]
)
