# Fixtures

This directory contains test audio fixtures for MLX integration tests.

## Required Files

For integration tests to run, you need:
- `noisy.wav` - A noisy speech sample for testing enhancement

## Environment Variables

Set these before running tests:
- `RUN_MLX_TESTS=1` - Enable MLX integration tests
- `FRCRN_WEIGHTS=/path/to/model.safetensors` - FRCRN weights path
- `MOSSFORMER_GAN_WEIGHTS=/path/to/weights.npz` - MossFormer GAN weights path  
- `TEST_AUDIO=/path/to/noisy.wav` - Test audio file

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
