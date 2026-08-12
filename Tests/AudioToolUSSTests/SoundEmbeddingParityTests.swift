//
//  SoundEmbeddingParityTests.swift
//  AudioToolUSSTests
//
//  Pins the SoundEmbedding presets to the bytes of the original bundled .safetensors
//

import XCTest
import AudioToolCore
import CryptoKit
import USSMLXSwift

/// The seven presets were once seven `.safetensors` resources in the USS bundle,
/// reachable only through a closed enum. They are plain AudioSet class lists in Core
/// now, reconstructed as normalised multi-hot vectors.
///
/// This used to load those files and compare. They are deleted - reconstruction was
/// proven exact, and a generated artifact shipped next to the code that generates it
/// is a provenance question with no upside - so the ground truth is kept as a
/// SHA-256 of each file's 2108-byte float32 payload instead. Same guarantee, 32 bytes
/// per preset, and it stays valid with nothing binary in the tree.
///
/// A failure here means a preset's class list changed. That is a change in what the
/// separator extracts, not a test detail: update the expected digest only alongside a
/// deliberate, documented change to the classes.
final class SoundEmbeddingParityTests: XCTestCase {

    /// SHA-256 of the `embedding` tensor payload from each original file: 527
    /// float32 values, little-endian, exactly as `mx.save_safetensors` wrote them.
    private static let originalDigests: [EmbeddingLoader.EmbeddingType: String] = [
        .speech: "dca1a0fc8dc644ec9f3e5e32aee7935bbf4775986686b3c04781c52e81aafeb2",
        .music: "3628d021245f02a9a0261f653a7b51e949760be72ac57b21210c567e60f1199c",
        .noise: "f152fbc19a9d22a5c4bd79d0b686b422d38b5d32c1a88f5c46f71c05fcb98417",
        .nature: "e4eb8285919dcde06c62061deedd2461e62f684ad1cc4fa5b2e521985e1ebe3d",
        .things: "e90047502f22ac8cab26ac0b84c63fb287ec42e2054e3912552fe7eff3efcc4d",
        .animal: "8bfe666b4b546f3e60bb400abd924aabf1f51c2f763d64d20b7d06961db27596",
        .human: "42aa6dbc6209db38b598a6e290959af82cc60593226d4f9758b801130226a0d4",
    ]

    /// The preset's weights in the same byte layout the file used.
    private func payload(_ preset: SoundEmbedding) -> Data {
        var data = Data(capacity: SoundEmbedding.dimension * 4)
        for weight in preset.weights {
            withUnsafeBytes(of: weight.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    private func assertParity(
        _ preset: SoundEmbedding,
        _ type: EmbeddingLoader.EmbeddingType,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        XCTAssertEqual(
            preset.weights.count, SoundEmbedding.dimension,
            file: file, line: line
        )

        let bytes = payload(preset)
        XCTAssertEqual(bytes.count, 2108, "float32 payload size", file: file, line: line)

        let digest = SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
        let expected = try XCTUnwrap(
            Self.originalDigests[type],
            "no recorded digest for \(type.rawValue)",
            file: file, line: line
        )
        XCTAssertEqual(
            digest, expected,
            "preset '\(preset.label ?? "?")' no longer reproduces the original "
                + "\(type.rawValue) embedding byte for byte",
            file: file, line: line
        )
    }

    /// Every case is covered, so adding a target cannot silently skip this check.
    func testEveryEmbeddingTypeHasARecordedDigest() {
        for type in EmbeddingLoader.EmbeddingType.allCases {
            XCTAssertNotNil(
                Self.originalDigests[type],
                "missing recorded digest for \(type.rawValue)"
            )
        }
    }

    /// The property the reconstruction relies on: n classes, each exactly 1/n.
    func testPresetsAreNormalisedMultiHot() {
        for preset in SoundEmbedding.presets {
            let nonzero = preset.weights.filter { $0 != 0 }
            XCTAssertFalse(nonzero.isEmpty, "\(preset.label ?? "?") is all zero")
            XCTAssertEqual(
                Set(nonzero).count, 1,
                "\(preset.label ?? "?") is not uniformly weighted"
            )
            XCTAssertEqual(
                Double(preset.weights.reduce(0, +)), 1.0, accuracy: 1e-5,
                "\(preset.label ?? "?") does not sum to 1"
            )
        }
    }

    func testSpeechPresetMatchesOriginalWeights() throws { try assertParity(.speech, .speech) }
    func testMusicPresetMatchesOriginalWeights() throws { try assertParity(.music, .music) }
    func testAnimalPresetMatchesOriginalWeights() throws { try assertParity(.animal, .animal) }
    func testNaturePresetMatchesOriginalWeights() throws { try assertParity(.nature, .nature) }
    func testNoisePresetMatchesOriginalWeights() throws { try assertParity(.noise, .noise) }
    func testThingsPresetMatchesOriginalWeights() throws { try assertParity(.things, .things) }
    func testHumanPresetMatchesOriginalWeights() throws { try assertParity(.human, .human) }
}
