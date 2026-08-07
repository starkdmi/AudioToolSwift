// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// Common Swift settings for all targets
let commonSwiftSettings: [SwiftSetting] = [
    .unsafeFlags(["-warnings-as-errors"])
]

// FluidAudio wrapper has Sendable issues in upstream dependency - exclude from strict mode
let fluidAudioSwiftSettings: [SwiftSetting] = []

// Model implementation targets are vendored under Sources/Models. They keep their
// original module names so imports elsewhere are unchanged, but live in directories
// named after the model rather than after the package they came from.
//
// They build in Swift 5 language mode. That is not a concession: as separate packages
// they all declared swift-tools-version 5.9, so Swift 5 is the mode they were written
// and validated against. Folding them into a package whose root is swiftLanguageModes
// [.v6] would silently promote them and turn MLX's non-Sendable MLXArray and the
// models' kernel caches into ~20 hard errors. The public API in Sources/AudioTool* is
// fully Swift 6; these are inference internals behind it. Migrating them is real work
// with real risk of changing numerical behaviour, so it belongs in its own pass with
// benchmarks either side, not smuggled into a repo move.
let modelSwiftSettings: [SwiftSetting] = [
    .swiftLanguageMode(.v5)
]

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
        // Named with a hyphen deliberately: an "audiotool" product collides
        // case-insensitively with the AudioTool library target, and Xcode then
        // folds the CLI's main.swift into module AudioTool and fails to link.
        // swift build tolerates the clash; xcodebuild does not.
        .executable(
            name: "audio-tool",
            targets: ["AudioToolCLI"]
        ),
    ],
    dependencies: [
        // MLX for neural engine operations
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.29.1")),

        // MLX LLM for TranslateGemma translation
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "2.29.3")),

        // HuggingFace Hub for model downloading
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMinor(from: "1.1.6")),

        // Resampling, STFT, and audio I/O.
        .package(url: "https://github.com/starkdmi/SwiftAudio", exact: "1.1.0"),

        // G2P for Kokoro TTS. A fork of mlalma/MisakiSwift whose only change is the
        // manifest - the Swift source at this revision is byte-identical to upstream
        // 1.0.5, and its 18 MB of pronunciation dictionaries and BART weights stay
        // out of this repository.
        //
        // Two problems with upstream's manifest, both fixed on the fork:
        //
        //   type: .dynamic. MisakiSwift links MLX, MLXNN, MLXUtilsLibrary and
        //   ZIPFoundation, so a dylib carries its own copies of their Objective-C
        //   classes. Anything that also links MLX - which is every reason to use a
        //   Kokoro G2P library - then loads two sets of the same classes. The runtime
        //   warns that this "may cause spurious casting failures and mysterious
        //   crashes", and here it did: a segfault partway through the XCTest process,
        //   and a CLI that linked but could not launch because MLX.framework was not
        //   on its rpath. That is why AudioToolCLI still excludes AudioToolTTS below.
        //   Verified: with the fork, 8+ duplicate-class warnings and the signal 11
        //   both go to zero and AudioToolMLXIntegrationTests completes.
        //
        //   swift-tools-version 6.2, which would raise this package's real minimum
        //   toolchain to Xcode 26 while the manifest here claims 6.0. Nothing in
        //   MisakiSwift needs 6.2; the fork declares 6.0 and compiles unchanged.
        //
        // Pinned by revision rather than version because the fork carries no tag of
        // its own, and pinned off 1.0.5 rather than 1.0.6 because 1.0.6 requires
        // mlx-swift exactly 0.30.2, which conflicts with the 0.29.x every model
        // target here was built against.
        //
        // Send the same one-line change upstream and this becomes a version range
        // again: github.com/mlalma/MisakiSwift, remove `type: .dynamic`.
        .package(
            url: "https://github.com/starkdmi/MisakiSwift",
            revision: "1ecaf9a6057ed8bdd69e5a37cdcc0b5cb30eb901"
        ),
        .package(url: "https://github.com/mlalma/MLXUtilsLibrary.git", from: "0.0.6"),

        // VAD, transcription, diarization. Upstream, not a fork.
        //
        // Pinned to a revision rather than a version range because the newest tag,
        // v0.9.1, predates FluidInference/FluidAudio#264 - the mel-context fix for
        // chunk-boundary transcription loss, authored here and merged upstream on
        // 2026-01-23. Without it, audio landing on a chunk boundary produces
        // all-blank predictions. No tagged release contains that commit yet, so a
        // revision pin is the only way to depend on correct behaviour. Switch to a
        // version range once upstream cuts a release past it.
        .package(
            url: "https://github.com/FluidInference/FluidAudio",
            revision: "5390df9752c8fc583596018360c5fd70d6fa6c75"
        ),
    ],
    targets: [
        // MARK: - Vendored model implementations
        //
        // Previously six separate local SwiftPM packages plus a fork checkout. Source
        // only: every weight is fetched at runtime from HuggingFace, never committed.

        .target(
            name: "Mossformer2MLXSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/Models/MossFormer2SE",
            swiftSettings: modelSwiftSettings
        ),
        .target(
            name: "FRCRNMLXSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/Models/FRCRN",
            swiftSettings: modelSwiftSettings
        ),
        .target(
            name: "MossFormer2SS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/Models/MossFormer2SS",
            swiftSettings: modelSwiftSettings
        ),
        .target(
            name: "MossFormer2SR",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/Models/MossFormer2SR",
            swiftSettings: modelSwiftSettings
        ),
        .target(
            name: "DemucsMLXSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/Models/Demucs",
            swiftSettings: modelSwiftSettings
        ),
        .target(
            name: "USSMLXSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/Models/USS",
            // The ResUNet30 weights this target used to bundle (~106 MB) now come from
            // HuggingFace. What stays are the 527-d AudioSet class vectors, 2 KB each,
            // which select what to separate - they are inputs, not weights.
            resources: [
                .copy("Embeddings")
            ],
            swiftSettings: modelSwiftSettings
        ),
        .target(
            name: "KokoroSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MisakiSwift", package: "MisakiSwift"),
                .product(name: "MLXUtilsLibrary", package: "MLXUtilsLibrary"),
            ],
            path: "Sources/Models/Kokoro",
            exclude: ["LICENSE"],
            resources: [
                .copy("Resources")
            ],
            swiftSettings: modelSwiftSettings
        ),
        .target(
            name: "ChatterboxMLXSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/Models/Chatterbox",
            swiftSettings: modelSwiftSettings
        ),

        // MARK: - Core shared infrastructure

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

        // MARK: - Backends

        .target(
            name: "AudioToolMLX",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                "Mossformer2MLXSwift",
                "FRCRNMLXSwift",
                "MossFormer2SS",
                "MossFormer2SR",
                "DemucsMLXSwift",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/AudioToolMLX",
            swiftSettings: commonSwiftSettings
        ),
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
        // Note: FluidAudio has Sendable issues upstream - excluded from strict warnings
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
        .target(
            name: "AudioToolUSS",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                "USSMLXSwift",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/AudioToolUSS",
            swiftSettings: commonSwiftSettings
        ),
        .target(
            name: "AudioToolTTS",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                "AudioToolFluidAudio",  // For VAD-based audio trimming
                "KokoroSwift",
                "ChatterboxMLXSwift",
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

        // MARK: - Tests

        // Gating shared by every test target: which suites may touch models, the
        // network, or the sibling research checkout, and where artifacts go.
        // Plain Foundation, no XCTest, so it can be an ordinary library target.
        .target(
            name: "AudioToolTestSupport",
            path: "Tests/TestSupport",
            swiftSettings: commonSwiftSettings
        ),

        // Unit tests with mocks (swift test compatible)
        .testTarget(
            name: "AudioToolTests",
            dependencies: ["AudioTool", "AudioToolTTS", "AudioToolSpeech", "AudioToolTestSupport"],
            path: "Tests/AudioToolTests",
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
                "AudioToolUSS",
                "AudioToolFluidAudio",
                .product(name: "AudioUtils", package: "SwiftAudio"),
                "AudioToolTestSupport",
            ],
            path: "Tests/AudioToolMLXIntegrationTests",
            swiftSettings: commonSwiftSettings
        ),
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
                            "AudioToolTestSupport",
            ],
            path: "Tests/AudioToolFluidAudioTests",
            // Declared so SwiftPM emits a resource bundle and Bundle.module exists.
            // The audio itself is gitignored - see Fixtures/README.md.
            resources: [
                .copy("Fixtures")
            ],
            swiftSettings: commonSwiftSettings
        ),
        // Run with: xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' -only-testing:AudioToolUSSTests
        .testTarget(
            name: "AudioToolUSSTests",
            dependencies: [
                "AudioToolUSS",
                "USSMLXSwift",
                "AudioToolCoreML",
                "AudioToolMLX",
                "AudioToolCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
                            "AudioToolTestSupport",
            ],
            path: "Tests/AudioToolUSSTests",
            // Declared so SwiftPM emits a resource bundle and Bundle.module exists.
            // The audio itself is gitignored - see Fixtures/README.md.
            resources: [
                .copy("Fixtures")
            ],
            swiftSettings: commonSwiftSettings
        ),
        // Parity against the MLX Python references. Opt-in twice over:
        // RUN_PARITY_TESTS=1 and the artifacts being present on disk. See
        // Parity/README.md for how the artifacts are generated and why they are
        // published with the weights rather than committed.
        //
        // Measure first, then assert. Note the TEST_RUNNER_ prefix: xcodebuild
        // consumes bare variables itself and does not forward them to the test
        // process, so the unprefixed form runs and skips everything while still
        // reporting success.
        //   TEST_RUNNER_RUN_PARITY_TESTS=1 TEST_RUNNER_PARITY_RECORD=1 \
        //     xcodebuild test -only-testing:AudioToolParityTests
        // Under `swift test`, use the names without the prefix.
        .testTarget(
            name: "AudioToolParityTests",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                "AudioToolMLX",
                "AudioToolCoreML",
                "AudioToolUSS",
                "USSMLXSwift",
                "AudioToolTestSupport",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Tests/AudioToolParityTests",
            swiftSettings: commonSwiftSettings
        ),
        // Run with: xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' -only-testing:AudioToolMLXTranslationTests
        .testTarget(
            name: "AudioToolMLXTranslationTests",
            dependencies: [
                "AudioToolMLXTranslation",
                "AudioTool",
                "AudioToolCore",
                            "AudioToolTestSupport",
            ],
            path: "Tests/AudioToolMLXTranslationTests",
            swiftSettings: commonSwiftSettings
        ),

        // MARK: - CLI

        // Build: xcodebuild build -scheme audio-tool -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
        // Run: .build/DerivedData/Build/Products/Release/audio-tool --model frcrn --input test.wav --output enhanced.wav
        // Note: AudioToolSpeech (Apple SpeechAnalyzer) requires macOS 26+ and is only conditionally imported
        .executableTarget(
            name: "AudioToolCLI",
            dependencies: [
                "AudioToolMLX",
                // Deliberately NOT AudioToolTTS or AudioToolFluidAudio: neither is
                // used since the scratch subcommands were removed, so linking them
                // costs build time and binary size for nothing.
                //
                // It used to be that AudioToolTTS *could* not be linked - it drags in
                // MisakiSwift, which upstream builds as a dynamic library, and under
                // xcodebuild that produced an executable which linked but could not
                // launch because MLX.framework was not on its rpath. That is fixed at
                // the source: the forked manifest omits `type:`, so MLX and friends
                // are no longer embedded as dynamic frameworks at all. Verified by
                // adding AudioToolTTS back and building with xcodebuild - it links,
                // launches, and the Frameworks directory is empty. Add it here freely
                // if the CLI ever needs TTS.
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
