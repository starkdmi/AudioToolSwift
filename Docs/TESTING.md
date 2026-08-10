# AudioTool Testing Guide

## Running Tests

### Unit Tests (SPM)

Basic unit tests that don't require Metal/GPU:

```bash
swift test
```

### MLX Integration Tests

MLX-based tests require Metal hardware. The normal path is `xcodebuild`, which
compiles and bundles MLX's Metal library automatically:

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

This is an underlying `mlx-swift` build-system constraint, not a special
`mlx-swift-lm` setup step. Bare command-line SwiftPM can compile the Swift and
C++ targets but cannot compile MLX's Metal shaders; the first MLX operation then
fails because `default.metallib` is absent. Xcode and `xcodebuild` perform that
build and resource-bundling step, which is why ordinary Xcode projects work
without a custom script.

Use `xcodebuild` for tests that execute MLX/Metal operations. If CI deliberately
uses bare `swift test`, run `Scripts/build_mlx_metallib.sh` after building the
test bundle; that script is the repository's SwiftPM-only workaround. Pure
CoreML execution is a separate path and does not, by itself, require MLX's
metallib.

Upstream reference: [mlx-swift installation notes](https://github.com/ml-explore/mlx-swift#swiftpm).

## Test Fixtures

Redistributable fixtures live in
`Tests/AudioToolFluidAudioTests/Fixtures/` and
`Tests/AudioToolUSSTests/Fixtures/`. They are CC0 or CC BY 4.0, and their exact
sources, transformations, hashes and model coverage are recorded in the fixture
README and `Docs/licenses.md`. Regenerate the derived files with
`Scripts/fetch-fixtures.sh`; it fetches individual small files, not datasets.

Exact-content regression media that cannot be redistributed stays in the
sibling research checkout. Tests reach it through `TestGate.reference(...)` and
skip cleanly in a standalone public clone. Do not copy private-pool media into a
package `Fixtures/` directory.

## Performance

Correctness is here; cost is in [BENCHMARKING.md](BENCHMARKING.md).
`audio-tool-bench` measures whole models through their public API - load time,
RTF, peak memory - one model per process, and writes a report carrying the
machine it ran on. The two gated timing suites in the test targets
(`ResamplerBenchmarkTests`, `USSPrewarmBenchmarkTests`) stay here because they
measure things below a model and need `@testable` access.
