//
//  SoundEmbeddingTests.swift
//  AudioToolTests
//
//  Validation and construction rules for SoundEmbedding
//

import Testing
import Foundation
@testable import AudioToolCore

@Suite("SoundEmbedding")
struct SoundEmbeddingTests {

    // MARK: - Validation

    @Test("Rejects vectors that are not 527-d")
    func rejectsWrongDimension() throws {
        #expect(throws: AudioToolError.self) {
            _ = try SoundEmbedding(weights: [Float](repeating: 0, count: 526))
        }
        #expect(throws: AudioToolError.self) {
            _ = try SoundEmbedding(weights: [Float](repeating: 0, count: 528))
        }
        #expect(throws: Never.self) {
            _ = try SoundEmbedding(weights: [Float](repeating: 0, count: 527))
        }
    }

    @Test("Rejects non-finite weights")
    func rejectsNonFinite() throws {
        var weights = [Float](repeating: 0, count: SoundEmbedding.dimension)
        weights[10] = .nan
        #expect(throws: AudioToolError.self) {
            _ = try SoundEmbedding(weights: weights)
        }
        weights[10] = .infinity
        #expect(throws: AudioToolError.self) {
            _ = try SoundEmbedding(weights: weights)
        }
    }

    @Test("Rejects empty or out-of-range class lists")
    func rejectsBadClasses() throws {
        #expect(throws: AudioToolError.self) { _ = try SoundEmbedding(audioSetClasses: []) }
        #expect(throws: AudioToolError.self) { _ = try SoundEmbedding(audioSetClasses: [-1]) }
        #expect(throws: AudioToolError.self) { _ = try SoundEmbedding(audioSetClasses: [527]) }
        #expect(throws: Never.self) { _ = try SoundEmbedding(audioSetClasses: [0, 526]) }
    }

    // MARK: - Construction

    @Test("Class list produces a normalised multi-hot vector")
    func multiHotIsNormalised() throws {
        let embedding = try SoundEmbedding(audioSetClasses: [3, 7, 11], label: "three")
        #expect(embedding.weights.count == SoundEmbedding.dimension)
        #expect(embedding.activeClasses == [3, 7, 11])
        for index in [3, 7, 11] {
            #expect(embedding.weights[index] == Float(1.0 / 3.0))
        }
        #expect(embedding.weights.reduce(0, +) == 1.0)
        #expect(embedding.weights[0] == 0)
    }

    @Test("Duplicate classes are ignored, not double-counted")
    func duplicatesIgnored() throws {
        let once = try SoundEmbedding(audioSetClasses: [5, 9])
        let twice = try SoundEmbedding(audioSetClasses: [5, 9, 5, 9, 9])
        #expect(once.weights == twice.weights)
    }

    // MARK: - Presets

    @Test("Presets have the documented class counts")
    func presetCounts() {
        #expect(SoundEmbedding.speech.activeClasses.count == 40)
        #expect(SoundEmbedding.music.activeClasses.count == 146)
        #expect(SoundEmbedding.animal.activeClasses.count == 65)
        #expect(SoundEmbedding.nature.activeClasses.count == 15)
        #expect(SoundEmbedding.noise.activeClasses.count == 63)
        #expect(SoundEmbedding.things.activeClasses.count == 166)
        #expect(SoundEmbedding.human.activeClasses.count == 32)
    }

    /// The seven presets were derived from a partition of the AudioSet ontology:
    /// every class belongs to exactly one preset. If a preset is ever edited this
    /// catches overlap or omission immediately.
    @Test("Presets partition all 527 AudioSet classes exactly once")
    func presetsPartitionOntology() {
        let all = SoundEmbedding.presets.flatMap(\.activeClasses)
        #expect(all.count == SoundEmbedding.dimension)
        #expect(Set(all).count == SoundEmbedding.dimension)
        #expect(Set(all) == Set(0..<SoundEmbedding.dimension))
    }

    @Test("A preset equals the same classes passed by hand")
    func presetMatchesManualConstruction() throws {
        for preset in SoundEmbedding.presets {
            let rebuilt = try SoundEmbedding(audioSetClasses: preset.activeClasses,
                                             label: preset.label)
            #expect(rebuilt.weights == preset.weights, "\(preset.label ?? "?") diverged")
            #expect(rebuilt == preset)
        }
    }

    @Test("Presets carry their label and are usable as dictionary keys")
    func presetsHashable() {
        #expect(SoundEmbedding.speech.label == "speech")
        var byTarget: [SoundEmbedding: Int] = [:]
        byTarget[.speech] = 1
        byTarget[.music] = 2
        #expect(byTarget[.speech] == 1)
        #expect(byTarget[.music] == 2)
        #expect(SoundEmbedding.speech != SoundEmbedding.music)
    }
}
