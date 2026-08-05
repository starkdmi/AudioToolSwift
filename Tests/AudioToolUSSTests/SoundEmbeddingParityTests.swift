//
//  SoundEmbeddingParityTests.swift
//  AudioToolUSSTests
//
//  Pins the SoundEmbedding presets to the original bundled .safetensors weights
//

import XCTest
import AudioToolCore
import USSMLXSwift
import MLX

/// The seven presets used to be seven `.safetensors` resources reachable only via a
/// closed `USSSoundType` enum. They are now plain AudioSet class lists in Core,
/// reconstructed as normalised multi-hot vectors.
///
/// That is only a safe swap if the reconstruction is exact, so this compares each
/// preset against the bytes still shipped in the USS bundle. If these pass, the
/// resource files are redundant and could be dropped; until then they are the
/// ground truth and this is what keeps the two honest.
final class SoundEmbeddingParityTests: XCTestCase {

    private func bundledEmbedding(_ type: EmbeddingLoader.EmbeddingType) throws -> [Float] {
        let directory = try XCTUnwrap(
            USSBundle.embeddingsDirectory,
            "USS embeddings directory missing from the bundle"
        )
        let array = try EmbeddingLoader.loadEmbedding(type: type, from: directory.path)
        // loadEmbedding returns shape [1, 527]; flatten for comparison.
        return array.reshaped([SoundEmbedding.dimension]).asArray(Float.self)
    }

    private func assertParity(
        _ preset: SoundEmbedding,
        _ type: EmbeddingLoader.EmbeddingType,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let bundled = try bundledEmbedding(type)
        XCTAssertEqual(bundled.count, SoundEmbedding.dimension, file: file, line: line)

        // Exact, not approximate: both sides should be the identical float32 values.
        XCTAssertEqual(
            preset.weights, bundled,
            "preset '\(preset.label ?? "?")' differs from the bundled \(type.rawValue) embedding",
            file: file, line: line
        )

        // And identical bit patterns, which `==` on Float would not catch for -0.0.
        let presetBits = preset.weights.map { $0.bitPattern }
        let bundledBits = bundled.map { $0.bitPattern }
        XCTAssertEqual(presetBits, bundledBits, "bit patterns differ", file: file, line: line)
    }

    func testSpeechPresetMatchesBundledWeights() throws { try assertParity(.speech, .speech) }
    func testMusicPresetMatchesBundledWeights() throws { try assertParity(.music, .music) }
    func testAnimalPresetMatchesBundledWeights() throws { try assertParity(.animal, .animal) }
    func testNaturePresetMatchesBundledWeights() throws { try assertParity(.nature, .nature) }
    func testNoisePresetMatchesBundledWeights() throws { try assertParity(.noise, .noise) }
    func testThingsPresetMatchesBundledWeights() throws { try assertParity(.things, .things) }
    func testHumanPresetMatchesBundledWeights() throws { try assertParity(.human, .human) }
}
