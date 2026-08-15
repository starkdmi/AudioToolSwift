// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
// 6.2, not the 6.0 this said for a long time, because 6.0 was never true. Every
// published tag of MLXUtilsLibrary - 0.0.1 through 0.0.7 - declares
// swift-tools-version 6.2, and MisakiSwift depends on it too, so a Swift 6.0 or
// 6.1 toolchain cannot even load this graph's manifests, let alone compile it.
// Declaring 6.0 only moved the error onto a transitive dependency.
//
// The practical floor is therefore Xcode 26.0, the first release carrying Swift
// 6.2. That still runs on macOS Sequoia 15.6; Xcode 26.4.1 and later are the ones
// that require macOS Tahoe.

import PackageDescription
import Foundation

// Common Swift settings for all targets.
//
// `-warnings-as-errors` goes through `.unsafeFlags`, and SwiftPM refuses to let a
// product whose targets carry unsafe flags be used as a *versioned* dependency:
// a consumer declaring `from: "0.1.0"` fails resolution with "the target 'AudioTool'
// in product 'AudioTool' contains unsafe build flags". Only the root package may
// use them. So the flag is opt-in through the environment - CI sets it, developers
// can, and a package that depends on this one never sees it.
//
//   AUDIOTOOL_STRICT_WARNINGS=1 swift build
let commonSwiftSettings: [SwiftSetting] =
    ProcessInfo.processInfo.environment["AUDIOTOOL_STRICT_WARNINGS"] == "1"
    ? [.unsafeFlags(["-warnings-as-errors"])]
    : []

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
        // Same hyphenation rule as `audio-tool`, for the same reason: a product
        // whose name folds case-insensitively onto a library target confuses
        // Xcode's module layout. `audio-tool-bench` collides with nothing.
        .executable(
            name: "audio-tool-bench",
            targets: ["AudioToolBenchmarkCLI"]
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
        .package(url: "https://github.com/starkdmi/SwiftAudio", exact: "1.5.0"),

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
        //   on its rpath. Both are fixed on the fork - see the note on AudioToolCLI
        //   below, which no longer excludes AudioToolTTS for that reason. Verified:
        //   with the fork, 8+ duplicate-class warnings and the signal 11 both go to
        //   zero and AudioToolMLXIntegrationTests completes.
        //
        //   swift-tools-version 6.2. This was the second reason for the fork, back
        //   when the floor here was thought to be 6.0. It is no longer a reason for
        //   anything: MLXUtilsLibrary declares 6.2 in every tag and MisakiSwift
        //   depends on it, so the graph requires 6.2 whichever manifest is used,
        //   and this package now says so at the top. The dylib problem above is
        //   what keeps the fork alive.
        //
        // Pinned off 1.0.5 rather than 1.0.6 because 1.0.6 requires mlx-swift exactly
        // 0.30.2, which conflicts with the 0.29.x every model target here was built
        // against.
        //
        // `exact:` on a tag rather than `revision:` because a package that carries a
        // revision requirement cannot be depended on by version at all - SwiftPM
        // rejects the graph with "required using a stable-version but depends on an
        // unstable-version package". The tag names the same commit the revision pin
        // used, 1ecaf9a6057ed8bdd69e5a37cdcc0b5cb30eb901; the prerelease suffix keeps
        // it distinct from upstream's own 1.0.5.
        //
        // Send the same one-line change upstream and this becomes a version range
        // again: github.com/mlalma/MisakiSwift, remove `type: .dynamic`.
        .package(
            url: "https://github.com/starkdmi/MisakiSwift",
            exact: "1.0.5-static.1"
        ),
        .package(url: "https://github.com/mlalma/MLXUtilsLibrary.git", from: "0.0.6"),

        // VAD, transcription, diarization. Upstream, not a fork.
        //
        // This was pinned to a revision because the newest tag at the time, v0.9.1,
        // predated FluidInference/FluidAudio#264 - the mel-context fix for
        // chunk-boundary transcription loss, authored here and merged upstream on
        // 2026-01-23, without which audio landing on a chunk boundary produces
        // all-blank predictions. That reason expired: every release from v0.10 on
        // carries the fix, and `ChunkProcessor.melContextSamples` is present in
        // v0.15.5.
        //
        // The pin had to go regardless of the fix. A package holding a revision
        // requirement cannot itself be depended on by version - SwiftPM rejects the
        // graph with "required using a stable-version but depends on an
        // unstable-version package" - so a SHA here made this package unpublishable.
        // The revision it named was 32 commits past v0.15.5 on main; those commits
        // are upstream's own download and Kokoro-ANE work, none of which this package
        // calls into.
        .package(url: "https://github.com/FluidInference/FluidAudio", .upToNextMinor(from: "0.15.5")),
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
            exclude: ["LICENSE"],
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
            exclude: ["LICENSE"],
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
            exclude: ["LICENSE"],
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
            exclude: ["LICENSE"],
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
            exclude: ["LICENSE"],
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
            exclude: ["LICENSE"],
            // No resources. The ResUNet30 weights this target used to bundle (~106 MB)
            // come from HuggingFace, and the seven 527-d AudioSet conditioning vectors
            // that replaced them are gone too: each was a normalised multi-hot vector
            // over a fixed class list, so `SoundEmbedding`'s presets reproduce them
            // exactly from the indices alone. Shipping both a generated artifact and
            // the thing that generates it meant carrying binary files whose provenance
            // had to be explained separately from the code that made them.
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
            exclude: ["LICENSE"],
            swiftSettings: modelSwiftSettings
        ),

        // MARK: - Core shared infrastructure

        .target(
            name: "AudioToolCore",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/AudioToolCore",
            // Apple requires an SDK to declare its own required-reason API use
            // rather than lean on the consuming app's manifest. This target reads
            // volume capacity before a download and file modification dates on the
            // cache; both are declared there.
            resources: [
                .copy("PrivacyInfo.xcprivacy")
            ],
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
            // Reads a compiled model's modification date; see AudioToolCore above.
            resources: [
                .copy("PrivacyInfo.xcprivacy")
            ],
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

        // MARK: - Benchmarking

        // The harness: system profiling, resource sampling, the case catalog and
        // the report format. A library rather than part of the executable so the
        // report types can be read by anything that wants to diff two runs.
        //
        // Depends on every backend it measures, which is what keeps the catalog
        // honest - a provider that stops compiling cannot quietly drop out of the
        // benchmark. Scope is MLX and CoreML; see the note in BenchmarkCatalog for
        // why the FluidAudio-backed providers are not here.
        .target(
            name: "AudioToolBenchmark",
            dependencies: [
                "AudioTool",
                "AudioToolCore",
                "AudioToolMLX",
                "AudioToolCoreML",
                "AudioToolUSS",
                // For the `tts` category. Adding this is what the note in
                // BenchmarkCatalog meant by "a small change when someone wants it";
                // the dependency was never the obstacle, the shared RTF column was.
                "AudioToolTTS",
                .product(name: "AudioUtils", package: "SwiftAudio"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/AudioToolBenchmark",
            swiftSettings: commonSwiftSettings
        ),

        // Build: xcodebuild build -scheme audio-tool-bench -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
        // Run:   .build/DerivedData/Build/Products/Release/audio-tool-bench --list
        //
        // Re-executes itself once per case so each model gets a clean process.
        // That only works when the binary can find itself, which it can - but it
        // still has to be run from the products directory, because that is where
        // MLX's metallib lives.
        .executableTarget(
            name: "AudioToolBenchmarkCLI",
            dependencies: ["AudioToolBenchmark"],
            path: "Sources/AudioToolBenchmarkCLI",
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
                // For the MLX STFT/ISTFT pair, which lives in AudioToolCoreML but is
                // a plain MLX transform: hermetic, no CoreML model, no weights.
                "AudioToolCoreML",
                "AudioToolTTS",
                "AudioToolUSS",
                "AudioToolFluidAudio",
                "KokoroSwift",
                "USSMLXSwift",
                "DemucsMLXSwift",
                "ChatterboxMLXSwift",
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
            // Fixture licensing and provenance are documented in Fixtures/README.md.
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
            // Fixture licensing and provenance are documented in Fixtures/README.md.
            resources: [
                .copy("Fixtures")
            ],
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
