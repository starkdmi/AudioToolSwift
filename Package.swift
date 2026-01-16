// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ClearVoice",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .library(
            name: "ClearVoice",
            targets: ["ClearVoice"]
        ),
        .library(
            name: "ClearVoiceMLX",
            targets: ["ClearVoiceMLX"]
        ),
        .library(
            name: "ClearVoiceCoreML",
            targets: ["ClearVoiceCoreML"]
        ),
        .library(
            name: "ClearVoiceFluidAudio",
            targets: ["ClearVoiceFluidAudio"]
        ),
        .library(
            name: "ClearVoiceUSS",
            targets: ["ClearVoiceUSS"]
        ),
        .library(
            name: "ClearVoiceTTS",
            targets: ["ClearVoiceTTS"]
        ),
        .library(
            name: "ClearVoiceSpeech",
            targets: ["ClearVoiceSpeech"]
        ),
        .library(
            name: "ClearVoiceTranslation",
            targets: ["ClearVoiceTranslation"]
        ),
        .library(
            name: "ClearVoiceMLXTranslation",
            targets: ["ClearVoiceMLXTranslation"]
        ),
        .executable(
            name: "Generate",
            targets: ["Generate"]
        ),
    ],
    dependencies: [
        // SwiftAudio for resampling, STFT, and audio I/O
        .package(url: "https://github.com/starkdmi/SwiftAudio.git", exact: "1.0.0"),
        
        // MLX Model packages (local development)
        .package(name: "Mossformer2MLXSwift", path: "../Models/mossformer2_se_mlx_swift"),
        .package(name: "FRCRNMLXSwift", path: "../Models/frcrn_se_mlx_swift"),
        .package(name: "MossFormer2SS", path: "../Models/mosforrmer2_ss_mlx_swift"),
        .package(name: "MossFormer2SR", path: "../Models/mossformer2_sr_mlx_swift"),
        .package(name: "DemucsMLXSwift", path: "../Models/demucs_mlx_swift"),
        .package(name: "USSMLXSwift", path: "../Models/uss_mlx_swift"),
        
        // MLX for neural engine operations
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.2"),
        
        // FluidAudio for VAD, transcription, diarization (v0.10.0+ for Sortformer)
        .package(url: "https://github.com/FluidInference/FluidAudio", from: "0.10.0"),
        
        // Kokoro TTS with MisakiSwift G2P (MIT license, no ESpeakNG)
        // Using local fork with extended misaki[en] language support
        .package(name: "KokoroSwift", path: "../Models/kokoro-ios"),
        
        // ChatterBox Multilingual TTS (MLX, 25 languages)
        .package(name: "ChatterboxMLXSwift", path: "../Models/chatterbox_swift"),
        
        // HuggingFace Hub for model downloading (1.1.6+ required by mlx-swift-lm)
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.6"),
        
        // MLX LLM for TranslateGemma translation
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.29.3"),
    ],
    targets: [
        // Core shared infrastructure
        .target(
            name: "ClearVoiceCore",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/ClearVoiceCore"
        ),
        
        // Main public API (no MLX/CoreML dependency)
        .target(
            name: "ClearVoice",
            dependencies: ["ClearVoiceCore"],
            path: "Sources/ClearVoice"
        ),
        
        // MLX Backend providers
        .target(
            name: "ClearVoiceMLX",
            dependencies: [
                "ClearVoice",
                "ClearVoiceCore",
                .product(name: "Mossformer2MLXSwift", package: "Mossformer2MLXSwift"),
                .product(name: "FRCRNMLXSwift", package: "FRCRNMLXSwift"),
                .product(name: "MossFormer2SS", package: "MossFormer2SS"),
                .product(name: "MossFormer2SR", package: "MossFormer2SR"),
                .product(name: "DemucsMLXSwift", package: "DemucsMLXSwift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/ClearVoiceMLX"
        ),
        
        // CoreML Backend providers
        .target(
            name: "ClearVoiceCoreML",
            dependencies: [
                "ClearVoice",
                "ClearVoiceCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/ClearVoiceCoreML"
        ),
        
        // FluidAudio Backend providers (VAD, transcription, diarization)
        .target(
            name: "ClearVoiceFluidAudio",
            dependencies: [
                "ClearVoice",
                "ClearVoiceCore",
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/ClearVoiceFluidAudio"
        ),
        
        // USS MLX Backend providers (speech separation)
        .target(
            name: "ClearVoiceUSS",
            dependencies: [
                "ClearVoice",
                "ClearVoiceCore",
                .product(name: "USSMLXSwift", package: "USSMLXSwift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/ClearVoiceUSS"
        ),
        
        // TTS Backend providers (Kokoro with MisakiSwift G2P)
        .target(
            name: "ClearVoiceTTS",
            dependencies: [
                "ClearVoice",
                "ClearVoiceCore",
                "ClearVoiceFluidAudio",  // For VAD-based audio trimming
                .product(name: "KokoroSwift", package: "KokoroSwift"),
                .product(name: "ChatterboxMLXSwift", package: "ChatterboxMLXSwift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/ClearVoiceTTS"
        ),
        
        // Speech-to-Text Backend (Apple SpeechAnalyzer, iOS 26+)
        .target(
            name: "ClearVoiceSpeech",
            dependencies: [
                "ClearVoice",
                "ClearVoiceCore",
            ],
            path: "Sources/ClearVoiceSpeech"
        ),
        
        // Translation Backend (Apple Translation framework)
        .target(
            name: "ClearVoiceTranslation",
            dependencies: [
                "ClearVoice",
                "ClearVoiceCore",
            ],
            path: "Sources/ClearVoiceTranslation"
        ),
        
        // MLX Translation Backend (TranslateGemma, 55+ languages)
        .target(
            name: "ClearVoiceMLXTranslation",
            dependencies: [
                "ClearVoice",
                "ClearVoiceCore",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/ClearVoiceMLXTranslation"
        ),
        
        // Unit tests with mocks (swift test compatible)
        .testTarget(
            name: "ClearVoiceTests",
            dependencies: ["ClearVoice", "ClearVoiceTTS", "ClearVoiceSpeech"],
            path: "Tests/ClearVoiceTests",
            resources: [
                .copy("Fixtures/")
            ]
        ),
        
        // MLX Integration tests (requires xcodebuild for Metal)
        // Run with: xcodebuild test -scheme ClearVoice-Package -destination 'platform=macOS'
        .testTarget(
            name: "ClearVoiceMLXIntegrationTests",
            dependencies: [
                "ClearVoice",
                "ClearVoiceCore",
                "ClearVoiceMLX",
                "ClearVoiceTTS",
                "ClearVoiceFluidAudio",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Tests/ClearVoiceMLXIntegrationTests",
            resources: [
                .copy("Fixtures/")
            ]
        ),
        
        // FluidAudio Integration tests (VAD, transcription, diarization)
        // Run with: xcodebuild test -scheme ClearVoice-Package -destination 'platform=macOS' -only-testing:ClearVoiceFluidAudioTests
        .testTarget(
            name: "ClearVoiceFluidAudioTests",
            dependencies: [
                "ClearVoiceFluidAudio",
                "ClearVoiceMLX",
                "ClearVoiceCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Tests/ClearVoiceFluidAudioTests",
            resources: [
                .copy("Fixtures/")
            ]
        ),
        
        // USS MLX Integration tests (speech separation)
        // Run with: xcodebuild test -scheme ClearVoice-Package -destination 'platform=macOS' -only-testing:ClearVoiceUSSTests
        .testTarget(
            name: "ClearVoiceUSSTests",
            dependencies: [
                "ClearVoiceUSS",
                "ClearVoiceCoreML",
                "ClearVoiceMLX",
                "ClearVoiceCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Tests/ClearVoiceUSSTests",
            resources: [
                .copy("Fixtures/")
            ]
        ),
        
        // MLX Translation Integration tests (TranslateGemma)
        // Run with: xcodebuild test -scheme ClearVoice-Package -destination 'platform=macOS' -only-testing:ClearVoiceMLXTranslationTests
        .testTarget(
            name: "ClearVoiceMLXTranslationTests",
            dependencies: [
                "ClearVoiceMLXTranslation",
                "ClearVoice",
                "ClearVoiceCore",
            ],
            path: "Tests/ClearVoiceMLXTranslationTests"
        ),
        // CLI executable for testing MLX providers (use xcodebuild + run directly)
        // Build: xcodebuild build -scheme Generate -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
        // Run: .build/DerivedData/Build/Products/Release/Generate --model frcrn --input test.wav --output enhanced.wav
        // Note: ClearVoiceSpeech (Apple SpeechAnalyzer) requires macOS 26+ and is only conditionally imported
        .executableTarget(
            name: "Generate",
            dependencies: [
                "ClearVoiceMLX",
                "ClearVoiceTTS",
                "ClearVoiceFluidAudio",  // For voice matching
                "ClearVoiceSpeech", // Required for linking even with conditional import
                "ClearVoiceCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/Generate"
        ),
    ],
    swiftLanguageModes: [.v6]
)
