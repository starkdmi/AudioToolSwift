//
//  TestConfiguration.swift
//  AudioToolTests
//
//  Shared test configuration and skip utilities for consistent test behavior.
//  Use these helpers to properly skip integration tests instead of silent returns.
//

import Foundation
import Testing
import AudioToolTestSupport

// MARK: - Test Configuration

/// Centralized test configuration for AudioTool tests.
///
/// Environment variables for controlling test execution:
/// - `RUN_INTEGRATION_TESTS=1`: Run the integration suites. They are **off** by
///   default - this used to say the opposite - so a bare run skips them.
/// - `SKIP_INTEGRATION_TESTS=1`: Force them off even when the above is set.
/// - `SKIP_MLX_TESTS=1`: Skip the MLX-backed suites. Needed on any machine
///   without a Metal device or, under plain SwiftPM, without a staged metallib.
/// - `CI=1`: Indicates running in CI environment (adjusts performance thresholds)
public enum TestConfiguration {
    
    // MARK: - Environment
    //
    // One source of truth: these forward to ``TestGate``, which every test target
    // shares. Duplicating the env checks per target is how the integration suites
    // ended up ungated in the first place.

    /// Whether integration tests are explicitly suppressed (SKIP_INTEGRATION_TESTS=1).
    ///
    /// Rarely what a test wants: it is false in a default environment, where
    /// integration tests are off anyway for want of the opt-in. Gate on
    /// ``runIntegrationTests`` instead.
    public static var skipIntegrationTests: Bool {
        ProcessInfo.processInfo.environment["SKIP_INTEGRATION_TESTS"] == "1"
    }

    /// Whether to run integration tests. Opt-in: set `RUN_INTEGRATION_TESTS=1`.
    ///
    /// The AudioToolTests target is meant to be the fast signal - mocks, no models,
    /// no network. Several suites in it drive Apple SpeechAnalyzer, Apple TTS, Apple
    /// Translation and RUAccent, which take 38-60s each and download models on first
    /// use; two of them exceeded Swift Testing's 60-second limit and failed the whole
    /// run. Opting in keeps `swift test` usable while leaving them one variable away.
    ///
    /// ```bash
    /// SKIP_MLX_TESTS=1 swift test         # fast: mocks, no GPU
    /// swift test                          # same plus hermetic MLX (needs a metallib)
    /// RUN_INTEGRATION_TESTS=1 swift test  # everything
    /// ```
    public static var runIntegrationTests: Bool { TestGate.runIntegrationTests }

    /// Whether to skip MLX tests (set SKIP_MLX_TESTS=1)
    public static var skipMLXTests: Bool { TestGate.skipMLXTests }

    /// Whether running in CI environment (set CI=1)
    public static var isCI: Bool { TestGate.isCI }

    // MARK: - Performance Thresholds
    
    /// RTF threshold for VAD (adjusted for CI)
    public static var vadRTFThreshold: Double {
        isCI ? 5.0 : 10.0
    }
    
    /// RTF threshold for transcription (adjusted for CI)
    public static var transcriptionRTFThreshold: Double {
        isCI ? 2.0 : 5.0
    }
    
    /// RTF threshold for speech enhancement (adjusted for CI)
    public static var seRTFThreshold: Double {
        isCI ? 0.5 : 1.0
    }
    
    /// Maximum time for performance tests in ms (adjusted for CI)
    public static var maxMatchTimeMs: Double {
        isCI ? 20.0 : 5.0
    }
}

// MARK: - Skip Helpers for Swift Testing

/// Cancels a `@Test` unless integration tests were opted into.
///
/// Swift Testing has no `XCTSkip`; `Test.cancel` is the runtime equivalent, and
/// a cancelled test is reported as cancelled while the run still passes. What
/// this did before was record an `Issue` and throw - both of which are ordinary
/// failures - so a test following the example below failed rather than skipped,
/// and it consulted `SKIP_INTEGRATION_TESTS` rather than the opt-in every
/// `@Suite` here gates on, meaning it also ran by default.
///
/// Prefer the suite-level trait where you can; this is for a single test inside a
/// suite that is otherwise hermetic.
///
/// Example:
/// ```swift
/// @Test func testIntegration() async throws {
///     try requireIntegrationTests()
///     // ... test code
/// }
/// ```
public func requireIntegrationTests() throws {
    if !TestConfiguration.runIntegrationTests {
        try Test.cancel(Comment(rawValue: TestGate.integrationDisabled))
    }
}

/// Cancels a `@Test` when MLX is unavailable. See ``requireIntegrationTests()``.
public func requireMLXTests() throws {
    if TestConfiguration.skipMLXTests {
        try Test.cancel(Comment(rawValue: TestGate.mlxDisabled))
    }
}

/// Cancels a `@Test` when a file it needs is absent.
public func requireFile(at path: String, description: String = "Required file") throws {
    guard FileManager.default.fileExists(atPath: path) else {
        try Test.cancel(Comment(rawValue: "\(description) not found at: \(path)"))
    }
}

/// Cancels a `@Test` when a directory it needs is absent.
public func requireDirectory(at path: String, description: String = "Required directory") throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
        try Test.cancel(Comment(rawValue: "\(description) not found at: \(path)"))
    }
}


// MARK: - Deterministic Test Data

/// Provides deterministic pseudo-random data for testing.
/// Uses a fixed seed for reproducibility.
public struct DeterministicRandom {
    private var state: UInt64
    
    public init(seed: UInt64 = 42) {
        self.state = seed
    }
    
    /// Generate next random UInt64 using xorshift64
    public mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
    
    /// Generate random Float in range
    public mutating func float(in range: ClosedRange<Float> = -1.0...1.0) -> Float {
        let normalized = Float(next() % 1_000_000) / 1_000_000.0
        return range.lowerBound + normalized * (range.upperBound - range.lowerBound)
    }
    
    /// Generate array of random floats
    public mutating func floats(count: Int, in range: ClosedRange<Float> = -1.0...1.0) -> [Float] {
        (0..<count).map { _ in float(in: range) }
    }
    
    /// Generate deterministic "speech-like" audio samples
    public mutating func speechLikeSamples(count: Int, amplitude: Float = 0.3) -> [Float] {
        var samples = [Float]()
        samples.reserveCapacity(count)
        
        for i in 0..<count {
            // Combine low and high frequency components for speech-like pattern
            let t = Float(i) / 16000.0
            let lowFreq = sin(2.0 * .pi * 200.0 * t)
            let highFreq = sin(2.0 * .pi * 1500.0 * t) * 0.3
            let noise = float(in: -0.1...0.1)
            
            samples.append((lowFreq + highFreq + noise) * amplitude)
        }
        
        return samples
    }
    
    /// Generate deterministic silence with optional noise floor
    public mutating func silenceSamples(count: Int, noiseFloor: Float = 0.001) -> [Float] {
        floats(count: count, in: -noiseFloor...noiseFloor)
    }
}

// MARK: - Audio Quality Metrics

/// Calculates audio quality metrics for test assertions
public enum AudioMetrics {
    
    /// Calculate RMS (Root Mean Square) of audio samples
    public static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sumSquares = samples.reduce(Float(0)) { $0 + $1 * $1 }
        return sqrt(sumSquares / Float(samples.count))
    }
    
    /// Calculate peak amplitude
    public static func peak(_ samples: [Float]) -> Float {
        samples.map { abs($0) }.max() ?? 0
    }
    
    /// Calculate SNR in dB (signal vs noise floor)
    public static func snrDB(signal: [Float], noise: [Float]) -> Float {
        let signalRMS = rms(signal)
        let noiseRMS = rms(noise)
        guard noiseRMS > 0 else { return Float.infinity }
        return 20 * log10(signalRMS / noiseRMS)
    }
    
    /// Calculate correlation between two signals (for comparing outputs)
    public static func correlation(_ a: [Float], _ b: [Float]) -> Float {
        let minLen = min(a.count, b.count)
        guard minLen > 0 else { return 0 }
        
        var sum: Float = 0
        for i in 0..<minLen {
            sum += a[i] * b[i]
        }
        
        let normA = sqrt(a.prefix(minLen).reduce(Float(0)) { $0 + $1 * $1 })
        let normB = sqrt(b.prefix(minLen).reduce(Float(0)) { $0 + $1 * $1 })
        
        guard normA > 0 && normB > 0 else { return 0 }
        return sum / (normA * normB)
    }
}
