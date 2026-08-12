//
//  SoundEmbedding.swift
//  AudioToolCore
//
//  Conditioning vector selecting what a universal sound separator extracts
//

import Foundation

/// What a universal sound separator should extract, as an AudioSet class vector.
///
/// USS is conditioned on a 527-dimensional vector over the AudioSet ontology - one
/// weight per sound class - which is fed to the model's FiLM layers. Any such vector
/// is valid, so the space of things the model can be asked for is continuous, not a
/// fixed menu.
///
/// The seven ``speech``, ``music``, ``animal``, ``nature``, ``noise``, ``things`` and
/// ``human`` presets are one application's grouping of that space, offered as a
/// convenient starting point rather than as the set of available options. Build your
/// own with ``init(audioSetClasses:label:)`` to target something the presets do not
/// cover - a single AudioSet class, or any combination of them.
///
/// ```swift
/// // A preset
/// let vocals = try await uss.separateSound(audio, target: .speech)
///
/// // Or exactly the classes you want
/// let target = try SoundEmbedding(audioSetClasses: [0, 137, 500], label: "custom")
/// let custom = try await uss.separateSound(audio, target: target)
/// ```
public struct SoundEmbedding: Sendable, Hashable {

    /// Size of the AudioSet class vector. Fixed by the model's conditioning input.
    public static let dimension = 527

    /// Per-class weights over the AudioSet ontology. Always ``dimension`` long.
    public let weights: [Float]

    /// Optional human-readable name, for logging and dictionary keys.
    public let label: String?

    /// Create an embedding from explicit per-class weights.
    ///
    /// - Parameters:
    ///   - weights: Exactly ``dimension`` values, one per AudioSet class.
    ///   - label: Optional name for the target.
    /// - Throws: ``AudioToolError/invalidEmbedding(_:)`` if the count is wrong or any
    ///   value is not finite.
    public init(weights: [Float], label: String? = nil) throws {
        guard weights.count == Self.dimension else {
            throw AudioToolError.invalidEmbedding(
                "expected \(Self.dimension) weights, got \(weights.count)")
        }
        guard weights.allSatisfy({ $0.isFinite }) else {
            throw AudioToolError.invalidEmbedding("weights must all be finite")
        }
        self.weights = weights
        self.label = label
    }

    /// Create an embedding by naming AudioSet classes, weighted equally.
    ///
    /// Produces a normalised multi-hot vector: each listed class gets `1/count` and
    /// everything else zero. Duplicates are ignored. This is how the bundled presets
    /// were built, so a preset and the equivalent class list are the same vector.
    ///
    /// - Parameters:
    ///   - audioSetClasses: Class indices in `0..<`` dimension``. Must be non-empty.
    ///   - label: Optional name for the target.
    /// - Throws: ``AudioToolError/invalidEmbedding(_:)`` if empty or out of range.
    public init(audioSetClasses: [Int], label: String? = nil) throws {
        let unique = Set(audioSetClasses)
        guard !unique.isEmpty else {
            throw AudioToolError.invalidEmbedding("at least one AudioSet class is required")
        }
        guard let lo = unique.min(), let hi = unique.max(), lo >= 0, hi < Self.dimension else {
            throw AudioToolError.invalidEmbedding(
                "AudioSet class indices must be in 0..<\(Self.dimension)")
        }
        self.weights = Self.multiHot(unique)
        self.label = label
    }

    /// Preset construction. Indices are compile-time constants checked by
    /// `SoundEmbeddingTests`, so this skips validation and cannot fail.
    init(presetClasses: [Int], label: String) {
        self.weights = Self.multiHot(Set(presetClasses))
        self.label = label
    }

    /// Normalised multi-hot vector over `classes`.
    ///
    /// This is USS's `at_soft` conditioning: a zero vector with 1.0 at each class in
    /// the group, divided by the group's size. The conversion script did it in
    /// float32 throughout; the reciprocal is taken in `Double` here and narrowed,
    /// which is a different rounding path in principle and identical in fact - the
    /// two agree for every one of the seven group sizes, and `SoundEmbeddingTests`
    /// pins each preset to the SHA-256 of the original conversion's float32 payload.
    private static func multiHot(_ classes: Set<Int>) -> [Float] {
        let value = Float(1.0 / Double(classes.count))
        var weights = [Float](repeating: 0, count: dimension)
        for index in classes { weights[index] = value }
        return weights
    }

    /// AudioSet class indices carrying non-zero weight, ascending.
    public var activeClasses: [Int] {
        weights.enumerated().filter { $0.element != 0 }.map(\.offset)
    }
}

extension SoundEmbedding: CustomStringConvertible {
    public var description: String {
        "SoundEmbedding(\(label ?? "unlabelled"), \(activeClasses.count) classes)"
    }
}
