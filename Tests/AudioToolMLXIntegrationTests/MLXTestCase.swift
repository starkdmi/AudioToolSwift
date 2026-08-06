//
//  MLXTestCase.swift
//  AudioToolMLXIntegrationTests
//
//  Base for suites that evaluate MLX arrays but need no weights and no network.
//

import XCTest
import AudioToolTestSupport

/// Skips when MLX is unavailable, without requiring `RUN_INTEGRATION_TESTS`.
///
/// Distinct from ``IntegrationTestCase``, and the distinction matters for CI. These
/// suites are hermetic - synthetic input, no weights, no network - so they belong in
/// the default `swift test` signal. But they call `eval`, which needs a Metal device
/// and, under SwiftPM, the prebuilt metallib that `Scripts/build_mlx_metallib.sh`
/// produces. A runner without either would fail them rather than skip, which is the
/// same "fresh environment reports a failure it cannot act on" problem the gating
/// work set out to remove.
///
/// So: `SKIP_MLX_TESTS=1 swift test` is the offline, no-GPU configuration, and it has
/// to reach these too. Before this it did not - they were plain `XCTestCase` and ran
/// regardless.
class MLXTestCase: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipUnless(!TestGate.skipMLXTests, TestGate.mlxDisabled)
    }
}
