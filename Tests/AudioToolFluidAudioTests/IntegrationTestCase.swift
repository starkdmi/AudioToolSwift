//
//  IntegrationTestCase.swift
//  AudioToolFluidAudioTests
//
//  Base class for suites that need model weights, the network, or the sibling
//  research checkout.
//

import XCTest
import AudioToolTestSupport

/// Skips the whole suite unless integration tests are opted into.
///
/// XCTest has no per-target gate, so this is the per-suite one: subclass it
/// instead of `XCTestCase` and every test in the class skips cleanly when the
/// machine has no weights and no network. Suites that additionally need a
/// specific file from the sibling checkout should still guard it with
/// `TestGate.reference(_:)`, since having opted in does not mean having the file.
class IntegrationTestCase: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipUnless(TestGate.runIntegrationTests, TestGate.integrationDisabled)
    }

    /// A file or directory in the sibling research checkout, or a skip.
    ///
    /// - Parameter relativePath: path under `Models/`, e.g. `"uss_mlx_swift/test.wav"`.
    func reference(_ relativePath: String) throws -> URL {
        guard let url = TestGate.reference(relativePath) else {
            throw XCTSkip(TestGate.missingReference(relativePath))
        }
        return url
    }

    /// A writable directory for this suite's artifacts, outside the source tree.
    func outputDirectory(_ name: String = #function) throws -> URL {
        try TestGate.outputDirectory("\(Self.self)/\(name)")
    }
}
