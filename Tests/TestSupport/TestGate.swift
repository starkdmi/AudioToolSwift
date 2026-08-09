//
//  TestGate.swift
//  AudioToolTestSupport
//
//  Shared gating for tests that need models, network or the sibling research
//  checkout. Available to every test target, unlike TestConfiguration which only
//  ever existed inside AudioToolTests.
//

import Foundation

/// Preconditions for tests that reach outside the package.
///
/// Several suites here are integration tests in the honest sense: they load real
/// weights, hit the network, and read fixtures from the sibling research checkout
/// that this package was extracted from. None of that is present in a fresh clone,
/// and until this existed those suites *failed* there rather than skipping - and
/// wrote their outputs into the sibling directories on the machine that did have
/// them, overwriting files no one had asked them to touch.
///
/// Two independent gates, both of which must pass:
///
/// 1. `RUN_INTEGRATION_TESTS=1` - opt in. Keeps bare `swift test` fast and offline.
/// 2. The specific fixture the test needs is actually on disk.
///
/// ```bash
/// swift test                          # unit tests only, no models, no network
/// RUN_INTEGRATION_TESTS=1 swift test  # everything the machine has fixtures for
/// ```
///
/// Deliberately does not import XCTest, so it can be a plain library that every
/// test target depends on. Callers turn a `nil` or a `false` into `XCTSkip`
/// themselves - see ``missingReference(_:)`` for the message to use.
public enum TestGate {

    // MARK: - Environment

    /// Running under CI. Loosens timing assertions.
    public static var isCI: Bool {
        ProcessInfo.processInfo.environment["CI"] == "1"
    }

    /// Integration tests are opted into, not out of.
    ///
    /// `SKIP_INTEGRATION_TESTS=1` still wins, so an existing CI job that sets it
    /// keeps working.
    public static var runIntegrationTests: Bool {
        let env = ProcessInfo.processInfo.environment
        guard env["SKIP_INTEGRATION_TESTS"] != "1" else { return false }
        return env["RUN_INTEGRATION_TESTS"] == "1"
    }

    /// MLX tests need a Metal device and, under SwiftPM, a prebuilt metallib.
    public static var skipMLXTests: Bool {
        ProcessInfo.processInfo.environment["SKIP_MLX_TESTS"] == "1"
    }

    /// Benchmarks are opted into, like everything else that takes real time.
    ///
    /// They report rather than assert, so running them by default would cost seconds
    /// and buy nothing; and a machine under load produces numbers worth ignoring.
    public static var runBenchmarks: Bool {
        ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"
    }

    /// Message for `XCTSkip` when ``runIntegrationTests`` is false.
    public static let integrationDisabled =
        "integration test - set RUN_INTEGRATION_TESTS=1 to run (needs model weights and network)"

    /// Message for `XCTSkip` when MLX tests are off.
    public static let mlxDisabled = "MLX tests disabled via SKIP_MLX_TESTS=1"

    /// Message for `XCTSkip` when ``runBenchmarks`` is false.
    public static let benchmarksDisabled =
        "benchmark - set RUN_BENCHMARKS=1 to run (reports timings, asserts almost nothing)"

    // MARK: - The sibling research checkout

    /// Root of the research checkout this package was extracted from, if present.
    ///
    /// The reference Python implementations, converted weights and the original
    /// sample audio live in the parent directory's `Models/` and `Docs/`, which are
    /// deliberately *not* part of this package and are absent in any normal clone.
    /// The presence of `Models/` is what distinguishes the two.
    public static let checkoutRoot: URL? = {
        // TestGate.swift -> TestSupport -> Tests -> AudioToolSwift -> checkout root
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        var isDirectory: ObjCBool = false
        let models = url.appendingPathComponent("Models")
        guard FileManager.default.fileExists(atPath: models.path, isDirectory: &isDirectory),
              isDirectory.boolValue else { return nil }
        return url
    }()

    /// A path under ``checkoutRoot``, or nil if either the root or the entry is missing.
    ///
    /// - Parameter relativePath: path relative to the checkout root, e.g.
    ///   `"Models/demucs_mlx_swift/test.wav"` or `"Docs/harry_potter.wav"`.
    public static func reference(_ relativePath: String) -> URL? {
        guard let root = checkoutRoot else { return nil }
        let candidate = root.appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: candidate.path) else { return nil }
        return candidate
    }

    /// Message for `XCTSkip` when ``reference(_:)`` returns nil.
    public static func missingReference(_ relativePath: String) -> String {
        guard checkoutRoot != nil else {
            return "sibling research checkout not present - \(relativePath) is unavailable in a standalone clone"
        }
        return "\(relativePath) not found in the sibling research checkout"
    }

    // MARK: - Output

    /// A writable directory for test artifacts.
    ///
    /// Never the source tree and never the sibling checkout. Set
    /// `AUDIOTOOL_TEST_OUTPUT_DIR` to keep artifacts somewhere you can find them;
    /// otherwise they go to a per-run temporary directory and the OS reaps them.
    ///
    /// - Parameter name: subdirectory name, usually the suite's.
    public static func outputDirectory(_ name: String) throws -> URL {
        let base: URL
        if let override = ProcessInfo.processInfo.environment["AUDIOTOOL_TEST_OUTPUT_DIR"] {
            base = URL(fileURLWithPath: override)
        } else {
            base = scratchRoot
        }
        let directory = base.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// One temporary root per process, so a run's artifacts stay together.
    private static let scratchRoot: URL = {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("AudioToolTests-\(ProcessInfo.processInfo.processIdentifier)")
    }()
}
