# AudioTool Testing Guide

## Running Tests

### Unit Tests (SPM)

Basic unit tests that don't require Metal/GPU:

```bash
swift test
```

### MLX Integration Tests (Xcodebuild Required)

MLX-based tests require Metal hardware and must be run via `xcodebuild`:

```bash
# Full test suite
xcodebuild test \
  -scheme AudioToolSwift-Package \
  -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData

# MLX tests only
TEST_RUNNER_RUN_MLX_TESTS=1 xcodebuild test \
  -scheme AudioToolSwift-Package \
  -destination 'platform=macOS' \
  -only-testing:AudioToolMLXIntegrationTests
```

> **The `TEST_RUNNER_` prefix is required under `xcodebuild`.** `xcodebuild`
> consumes bare environment variables itself and does not forward them to the
> test process, so `RUN_MLX_TESTS=1 xcodebuild test` runs, skips every gated
> test, and still reports success. Prefix each variable below with
> `TEST_RUNNER_` when using `xcodebuild`; use the bare names under `swift test`.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `RUN_MLX_TESTS=1` | Enable MLX integration tests |
| `RUN_PARITY_TESTS=1` | Enable parity tests against the MLX Python references |
| `PARITY_RECORD=1` | Measure and report parity SNR instead of asserting |
| `PARITY_DUMP_DIR=/path` | Write both sides of each parity comparison as wav |
| `SKIP_MLX_TESTS=1` | Skip MLX tests in command-line builds |
| `FRCRN_WEIGHTS=/path/to/model.safetensors` | FRCRN weights path |
| `MOSSFORMER_GAN_WEIGHTS=/path/to/weights.npz` | MossFormer GAN weights path |
| `TEST_AUDIO=/path/to/noisy.wav` | Test audio file |

### Specific Test Suites

```bash
# Voice Matching Tests
xcodebuild test \
  -scheme AudioToolSwift-Package \
  -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData \
  -only-testing:AudioToolMLXIntegrationTests/VoiceMatchingIntegrationTests

# Enhancer Tests
xcodebuild test \
  -scheme AudioToolSwift-Package \
  -destination 'platform=macOS' \
  -only-testing:AudioToolMLXIntegrationTests/MLXEnhancerIntegrationTests
```

## Why xcodebuild?

SPM's `swift test` doesn't properly initialize Metal resources needed by MLX. Use `xcodebuild` for any tests that:

- Use MLX/Metal operations
- Load CoreML models
- Perform GPU-accelerated audio processing

## Test Fixtures

Place test audio files in `Tests/AudioToolMLXIntegrationTests/Fixtures/`:
- `noisy.wav` - Noisy speech sample for enhancement tests
