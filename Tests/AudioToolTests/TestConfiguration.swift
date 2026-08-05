//
//  TestConfiguration.swift
//  AudioToolTests
//
//  Shared test configuration and skip utilities for consistent test behavior.
//  Use these helpers to properly skip integration tests instead of silent returns.
//

import Foundation
import Testing

// MARK: - Test Configuration

/// Centralized test configuration for AudioTool tests.
///
/// Environment variables for controlling test execution:
/// - `SKIP_INTEGRATION_TESTS=1`: Skip all integration tests (default: run them)
/// - `SKIP_MLX_TESTS=1`: Skip MLX-specific tests (default: run them)
/// - `CI=1`: Indicates running in CI environment (adjusts performance thresholds)
public enum TestConfiguration {
    
    // MARK: - Project Paths
    
    /// Project root computed from source file location
    public static let projectRoot: String = {
        let filePath = #filePath
        var url = URL(fileURLWithPath: filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()
    
    // MARK: - Environment Checks
    
    /// Whether to skip integration tests (set SKIP_INTEGRATION_TESTS=1)
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
    /// swift test                          # fast: mocks only
    /// RUN_INTEGRATION_TESTS=1 swift test  # everything
    /// ```
    public static var runIntegrationTests: Bool {
        guard !skipIntegrationTests else { return false }
        return ProcessInfo.processInfo.environment["RUN_INTEGRATION_TESTS"] == "1"
    }
    
    /// Whether to skip MLX tests (set SKIP_MLX_TESTS=1)
    public static var skipMLXTests: Bool {
        ProcessInfo.processInfo.environment["SKIP_MLX_TESTS"] == "1"
    }
    
    /// Whether running in CI environment (set CI=1)
    public static var isCI: Bool {
        ProcessInfo.processInfo.environment["CI"] == "1"
    }
    
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

/// Checks if integration tests should run, throws Issue.record if skipped.
/// Use in Swift Testing (@Test) functions.
///
/// Example:
/// ```swift
/// @Test func testIntegration() async throws {
///     try requireIntegrationTests()
///     // ... test code
/// }
/// ```
public func requireIntegrationTests(file: StaticString = #filePath, line: Int = #line) throws {
    if TestConfiguration.skipIntegrationTests {
        Issue.record("Integration tests skipped (SKIP_INTEGRATION_TESTS=1)", sourceLocation: SourceLocation(fileID: "\(file)", filePath: "\(file)", line: line, column: 0))
        throw TestSkipped(reason: "Integration tests disabled via SKIP_INTEGRATION_TESTS=1")
    }
}

/// Checks if MLX tests should run, throws Issue.record if skipped.
/// Use in Swift Testing (@Test) functions.
public func requireMLXTests(file: StaticString = #filePath, line: Int = #line) throws {
    if TestConfiguration.skipMLXTests {
        Issue.record("MLX tests skipped (SKIP_MLX_TESTS=1)", sourceLocation: SourceLocation(fileID: "\(file)", filePath: "\(file)", line: line, column: 0))
        throw TestSkipped(reason: "MLX tests disabled via SKIP_MLX_TESTS=1")
    }
}

/// Checks if a required file exists, throws with proper skip message if not.
public func requireFile(at path: String, description: String = "Required file", file: StaticString = #filePath, line: Int = #line) throws {
    guard FileManager.default.fileExists(atPath: path) else {
        Issue.record("\(description) not found at: \(path)", sourceLocation: SourceLocation(fileID: "\(file)", filePath: "\(file)", line: line, column: 0))
        throw TestSkipped(reason: "\(description) not found at: \(path)")
    }
}

/// Checks if a required directory exists.
public func requireDirectory(at path: String, description: String = "Required directory", file: StaticString = #filePath, line: Int = #line) throws {
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
        Issue.record("\(description) not found at: \(path)", sourceLocation: SourceLocation(fileID: "\(file)", filePath: "\(file)", line: line, column: 0))
        throw TestSkipped(reason: "\(description) not found at: \(path)")
    }
}

/// Error thrown when a test should be skipped.
public struct TestSkipped: Error {
    public let reason: String
    
    public init(reason: String) {
        self.reason = reason
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
