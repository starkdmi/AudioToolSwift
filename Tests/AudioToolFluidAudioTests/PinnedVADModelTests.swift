//
//  PinnedVADModelTests.swift
//  AudioToolFluidAudioTests
//
//  The pinned VAD snapshot loads and detects speech
//

import XCTest
import AudioToolTestSupport
import AudioToolCore
@preconcurrency import FluidAudio
@testable import AudioToolFluidAudio

/// Downloads ~1.1 MB from the pinned Silero repository, so it is gated like every
/// other model-backed suite.
final class PinnedVADModelTests: IntegrationTestCase {

    func testPinnedSnapshotProducesAWorkingManager() async throws {
        let manager = try await PinnedVADModel.makeManager(
            config: VadConfig(defaultThreshold: 0.5)
        )

        // One second of 440 Hz tone, which the model must at least process without
        // throwing. The assertion is on the pinned model loading and running, not on
        // what it decides about a sine wave.
        let samples = (0..<16000).map { index in
            0.3 * sin(2 * Float.pi * 440 * Float(index) / 16000)
        }
        let chunks = try await manager.process(samples)
        XCTAssertFalse(chunks.isEmpty, "pinned VAD returned no chunks")
    }

    func testSnapshotIsTheOnePinned() throws {
        let path = try XCTUnwrap(
            ModelDownloader.shared.localPath(
                for: ModelRepository.sileroVADCoreML,
                matching: ["silero-vad-unified-256ms-v6.2.1.mlmodelc/**"]
            ),
            "pinned VAD snapshot is not installed; run the load test first"
        )
        // Accepted by the pinned lookup at all means its revision or its hashes
        // matched - the point of routing this away from FluidAudio's downloader.
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: path.appendingPathComponent("silero-vad-unified-256ms-v6.2.1.mlmodelc").path
        ))
    }
}
