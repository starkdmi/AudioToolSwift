# Running MLX Code from CLI

**IMPORTANT**: MLX Swift code built with the bare command-line SwiftPM workflow
requires special handling for Metal shader resource bundling. Xcode and
`xcodebuild` handle this automatically.

## The Problem

When running MLX binaries built with bare `swift build`, `swift test`, or
`swift run`:
```
MLX error: Failed to load the default metallib. library not found
```

This happens because the command-line SwiftPM build does not compile and bundle
MLX's Metal shaders. It is an `mlx-swift` limitation inherited by
`mlx-swift-lm`, not a separate setup requirement for that higher-level package.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SKIP_MLX_TESTS=1` | Skip MLX integration tests (default: run them) |
| `SKIP_INTEGRATION_TESTS=1` | Skip all integration tests |
| `CI=1` | Indicates CI environment (adjusts performance thresholds) |

## Solution: Run via Xcode Product

Instead of running the binary directly, use **xcodebuild test** or **run from Xcode IDE**:

### Option 1: Run Tests via xcodebuild (Recommended)

```bash
cd ~/Downloads/clear_voice_research/AudioToolSwift

# Run all MLX integration tests
xcodebuild test \
  -scheme AudioToolSwift-Package \
  -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData \
  -only-testing:AudioToolMLXIntegrationTests

# Skip MLX tests
SKIP_MLX_TESTS=1 xcodebuild test \
  -scheme AudioToolSwift-Package \
  -destination 'platform=macOS'
```

### Option 2: Run CVGenerate from Xcode IDE

1. Open `AudioTool` in Xcode
2. Select the `CVGenerate` scheme
3. Edit Scheme → Run → Arguments: `-m chatterbox`
4. Run (⌘R)

### Option 3: Build and Run Product Directly (if you have Xcode configured)

```bash
# Build
xcodebuild build \
  -scheme CVGenerate \
  -configuration Debug \
  -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData

# Run from derived data with proper bundle (must run from Build directory)
cd .build/DerivedData/Build/Products/Debug
./CVGenerate -m chatterbox
```

## Available Test Commands (CVGenerate)

| Command | Description |
|---------|-------------|
| `-m frcrn` | FRCRN speech enhancement |
| `-m se48k` | MossFormer2 SE 48K |
| `-m demucs` | Demucs vocals separation |
| `-m kokoro` | Kokoro TTS multilingual |
| `-m chatterbox` | ChatterBox TTS (23 languages) |
| `-m voice_match` | Voice matching test |

## Troubleshooting

If you still get the metallib error:
1. Clean build: `rm -rf .build/DerivedData`
2. Ensure you're running from the correct directory
3. Try Debug configuration instead of Release
4. Run directly from Xcode IDE as a last resort
