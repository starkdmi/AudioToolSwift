# AudioTool Testing Guide

## Running Tests

### Unit Tests (SPM)

```bash
SKIP_MLX_TESTS=1 swift test    # no GPU, no metallib, no weights, no network
```

`swift test` on its own is *not* the no-Metal command. Several suites are
hermetic in every other sense - synthetic input, no weights, no network - but
still evaluate MLX arrays, and under plain SwiftPM that needs the prebuilt
metallib described below. Without one they do not fail politely; MLX aborts the
process. `SKIP_MLX_TESTS=1` is what skips them, and it is what CI sets when its
Metal probe comes back negative.

With a metallib staged once (`./Scripts/build_mlx_metallib.sh debug`), bare
`swift test` runs those suites too and is the fuller local signal.

### MLX Integration Tests

MLX-based tests require Metal hardware. The normal path is `xcodebuild`, which
compiles and bundles MLX's Metal library automatically:

```bash
# Full test suite
xcodebuild test \
  -scheme AudioToolSwift-Package \
  -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData

# MLX tests only, including the ones that load real weights
TEST_RUNNER_RUN_INTEGRATION_TESTS=1 xcodebuild test \
  -scheme AudioToolSwift-Package \
  -destination 'platform=macOS' \
  -only-testing:AudioToolMLXIntegrationTests
```

> **The `TEST_RUNNER_` prefix is required under `xcodebuild`.** `xcodebuild`
> consumes bare environment variables itself and does not forward them to the
> test process, so `RUN_INTEGRATION_TESTS=1 xcodebuild test` runs, skips every
> gated test, and still reports success. Prefix each variable below with
> `TEST_RUNNER_` when using `xcodebuild`; use the bare names under `swift test`.

### Environment Variables

The gates live in `TestGate` (`Tests/TestSupport/TestGate.swift`):

| Variable | Description |
|----------|-------------|
| `RUN_INTEGRATION_TESTS=1` | Opt in to tests that need model weights, network, or the sibling research checkout. Off by default, so bare `swift test` stays fast and offline. |
| `SKIP_INTEGRATION_TESTS=1` | Wins over the above. For a CI job that must stay hermetic. |
| `SKIP_MLX_TESTS=1` | Skip the MLX-backed suites. Needed without a Metal device, and under plain SwiftPM without a staged metallib. CI sets it when either its metallib build or its Metal execution probe fails. |
| `RUN_BENCHMARKS=1` | Run the two gated timing suites. They report rather than assert. |
| `CI=1` | Loosen timing assertions. |
| `AUDIOTOOL_TEST_OUTPUT_DIR=/path` | Where test artifacts go. Default is a per-run temporary directory the OS reaps. |

Two suites additionally want weights you already have on disk, and **skip
silently without them** - a green run is not evidence they executed:

| Variable | Description |
|----------|-------------|
| `AUDIOTOOL_USS_WEIGHTS=/path/resunet30_fp32.safetensors` | A resunet30 checkpoint. Without it the USS model tests skip (`Tests/AudioToolUSSTests/USSTestWeights.swift`). |
| `AUDIOTOOL_DEMUCS_WEIGHTS=/path/to/dir` | Directory holding `<stem>.safetensors`. Without it `DemucsShapeTests` skips. |

```bash
AUDIOTOOL_USS_WEIGHTS=~/weights/resunet30_fp32.safetensors \
  RUN_INTEGRATION_TESTS=1 swift test --filter AudioToolUSSTests
```

Under `xcodebuild` these take the `TEST_RUNNER_` prefix like everything else:
`TEST_RUNNER_AUDIOTOOL_USS_WEIGHTS=...`.

Nothing else is configurable by path: every other provider fetches its own
weights from HuggingFace, and non-redistributable reference media is found
through `TestGate.reference(...)` in the sibling checkout or skipped.

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

## What CI runs

Four jobs, in `.github/workflows/ci.yml`:

- **Build and test (macOS)** - `swift test` with no opt-ins: mocks, hermetic MLX
  transforms, no network. Also runs `Scripts/check-publishable.sh`, which reads
  `swift package dump-package` to reject unsafe flags and unstable dependency
  pins, and `Scripts/test-check-release.py`, which is pure Python and checks that
  the release checker still rejects what it is supposed to. The fast gate.
- **Build for iOS** - every library product for `generic/platform=iOS`. Catches
  the macOS-only API that kept this package from compiling for its declared
  deployment target.
- **Model-backed tests** - `RUN_INTEGRATION_TESTS=1` over the FluidAudio suites
  and `ModelDownloadIntegrationTests` (real HuggingFace fetches of a 10 KB
  fixture, including the download-cancellation contract) with a cached weights
  directory, then `Scripts/smoke-cli.sh`, which runs FRCRN end to end and checks
  the output is the right length, audible, and different from its input. This job
  needs a Metal toolchain and fails rather than skipping if it has none, because
  a silent skip here proves nothing.
- **Build on the declared Swift floor** - compiles with an Xcode carrying Swift
  6.2 exactly, since every other job takes the newest toolchain on the image and
  would not notice a 6.3-only construct.

Run the model-backed job locally with:

```bash
./Scripts/build_mlx_metallib.sh debug          # once; MLX aborts without it
RUN_INTEGRATION_TESTS=1 swift test --filter AudioToolFluidAudioTests
RUN_INTEGRATION_TESTS=1 swift test --filter ModelDownloadIntegrationTests
./Scripts/smoke-cli.sh
```

Releases are cut by `.github/workflows/release.yml`, which validates and only
then creates the tag; `python3 Scripts/check-release.py <version>` runs the same
metadata checks locally.

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

Correctness is here; cost is in [running-benchmarks.md](running-benchmarks.md).
`audio-tool-bench` measures whole models through their public API - load time,
RTF, peak memory - one model per process, and writes a report carrying the
machine it ran on. The two gated timing suites in the test targets
(`ResamplerBenchmarkTests`, `USSPrewarmBenchmarkTests`) stay here because they
measure things below a model and need `@testable` access.
