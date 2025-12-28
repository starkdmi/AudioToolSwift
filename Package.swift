// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ClearVoice",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
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
    ],
    targets: [
        // Core shared infrastructure
        .target(
            name: "ClearVoiceCore",
            dependencies: [],
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
            ],
            path: "Sources/ClearVoiceCoreML"
        ),
        
        // Unit tests with mocks (swift test compatible)
        .testTarget(
            name: "ClearVoiceTests",
            dependencies: ["ClearVoice"],
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
                "ClearVoiceMLX",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Tests/ClearVoiceMLXIntegrationTests",
            resources: [
                .copy("Fixtures/")
            ]
        ),
        
        // CLI executable for testing MLX providers (use xcodebuild + run directly)
        // Build: xcodebuild build -scheme Generate -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
        // Run: .build/DerivedData/Build/Products/Release/Generate --model frcrn --input test.wav --output enhanced.wav
        .executableTarget(
            name: "Generate",
            dependencies: [
                "ClearVoiceMLX",
                "ClearVoiceCore",
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/Generate"
        ),
    ],
    swiftLanguageModes: [.v6]
)
