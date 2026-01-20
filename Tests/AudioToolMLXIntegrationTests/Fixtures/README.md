# Fixtures

This directory contains test audio fixtures for MLX integration tests.

## Required Files

For integration tests to run, you need:
- `noisy.wav` - A noisy speech sample for testing enhancement

## Environment Variables

Set these before running tests:
- `SKIP_MLX_TESTS=1` - Skip MLX integration tests (default: run them)
- `SKIP_INTEGRATION_TESTS=1` - Skip all integration tests
- `CI=1` - Indicates CI environment (adjusts performance thresholds)
- `FRCRN_WEIGHTS=/path/to/model.safetensors` - FRCRN weights path (optional, auto-detected)
- `MOSSFORMER_GAN_WEIGHTS=/path/to/weights.npz` - MossFormer GAN weights path (optional)
- `TEST_AUDIO=/path/to/noisy.wav` - Test audio file (optional, uses fixtures)

## Running Tests

```bash
# Build with xcodebuild (required for Metal/MLX)
xcodebuild test \
  -scheme ClearVoice-Package \
  -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData

# Or for specific test suite
RUN_MLX_TESTS=1 xcodebuild test \
  -scheme ClearVoice-Package \
  -destination 'platform=macOS' \
  -only-testing:ClearVoiceMLXIntegrationTests
```

## Running MLX CLI Commands (Generate)

**Important:** MLX code requires xcodebuild, NOT `swift run`, due to Metal shader resource bundling.

```bash
# Build the CLI with xcodebuild
xcodebuild build \
  -scheme Generate \
  -configuration Release \
  -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData \
  -quiet

# Run from the build products (with DYLD paths for frameworks)
.build/DerivedData/Build/Products/Release/Generate -m chatterbox

# Available models:
# - frcrn, frcrn-bg     - Speech enhancement
# - se48k, se48k-bg     - MossFormer2 SE
# - demucs              - Vocals separation
# - ss_2spk, ss_3spk    - Speaker separation
# - sr48k               - Super resolution
# - kokoro              - Kokoro TTS
# - chatterbox, cb      - ChatterBox TTS
```

**Note:** `swift run` will fail with "Failed to load the default metallib" error.
This is a known MLX Swift issue with SPM Metal resource bundling.

