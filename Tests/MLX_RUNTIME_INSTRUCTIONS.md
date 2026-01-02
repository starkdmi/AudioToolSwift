# Running MLX Code from CLI

**IMPORTANT**: MLX Swift code requires special handling due to Metal shader resource bundling.

## The Problem

When running MLX binaries built with Swift Package Manager:
```
MLX error: Failed to load the default metallib. library not found
```

This happens because Metal `.metallib` shaders aren't bundled correctly with SPM.

## Solution: Run via Xcode Product

Instead of running the binary directly, use **xcodebuild test** or **run from Xcode IDE**:

### Option 1: Run Tests via xcodebuild (Recommended)

```bash
cd /path/to/clear_voice_research/ClearVoice

# Run all MLX integration tests
RUN_MLX_TESTS=1 xcodebuild test \
  -scheme ClearVoice-Package \
  -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData \
  -only-testing:ClearVoiceMLXIntegrationTests
```

### Option 2: Run CVGenerate from Xcode IDE

1. Open `ClearVoice` in Xcode
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
