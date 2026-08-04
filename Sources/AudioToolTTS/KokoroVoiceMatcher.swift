//
//  KokoroVoiceMatcher.swift
//  AudioToolTTS
//
//  Voice matching for Kokoro TTS using speaker embeddings
//

import Foundation
import Accelerate
import MLX
import AudioToolCore

// MARK: - Voice Embedding Table

/// Precomputed voice embedding table for fast matching
///
/// Contains speaker embeddings for all Kokoro voices, computed at build time
/// or on first use. Can be serialized for caching.
public struct VoiceEmbeddingTable: Codable, Sendable {
    
    /// Voice identifiers (e.g., "af_bella", "am_echo")
    public let voiceIds: [String]
    
    /// Speaker embeddings [N_voices][D_embedding]
    public let embeddings: [[Float]]
    
    /// Text used for calibration synthesis
    public let calibrationText: String
    
    /// Embedding dimension (typically 256)
    public var embeddingDimension: Int {
        embeddings.first?.count ?? 256
    }
    
    /// Number of voices in the table
    public var count: Int {
        voiceIds.count
    }
    
    /// Check if table is valid
    public var isValid: Bool {
        !voiceIds.isEmpty &&
        voiceIds.count == embeddings.count &&
        embeddings.allSatisfy { $0.count == embeddingDimension }
    }
    
    /// Initialize voice embedding table
    public init(voiceIds: [String], embeddings: [[Float]], calibrationText: String) {
        self.voiceIds = voiceIds
        self.embeddings = embeddings
        self.calibrationText = calibrationText
    }
    
    /// Get embedding for a specific voice
    public func embedding(for voiceId: String) -> [Float]? {
        guard let index = voiceIds.firstIndex(of: voiceId) else { return nil }
        return embeddings[index]
    }
    
    /// Filter table to only include voices of a specific gender
    ///
    /// Uses Kokoro voice naming convention:
    /// - Female: af_, bf_, ef_, jf_, zf_ (American/British/Spanish/Japanese/Chinese Female)
    /// - Male: am_, bm_, im_, jm_, zm_ (American/British/Italian/Japanese/Chinese Male)
    ///
    /// - Parameter gender: Gender to filter by
    /// - Returns: New table with only voices of the specified gender
    public func filtered(by gender: VoiceGender) -> VoiceEmbeddingTable {
        let filteredPairs = zip(voiceIds, embeddings).filter { voiceId, _ in
            gender.matches(voiceId: voiceId)
        }
        
        return VoiceEmbeddingTable(
            voiceIds: filteredPairs.map(\.0),
            embeddings: filteredPairs.map(\.1),
            calibrationText: calibrationText
        )
    }
}

/// Gender for voice filtering
public enum VoiceGender: String, Sendable, Codable {
    case female
    case male
    case any
    
    /// Check if a voice ID matches this gender
    ///
    /// Kokoro voice naming convention:
    /// - Female: `*f_*` (second char is 'f')
    /// - Male: `*m_*` (second char is 'm')
    public func matches(voiceId: String) -> Bool {
        switch self {
        case .any:
            return true
        case .female:
            // Voice IDs like "af_bella", "bf_emma", "ef_dora", "jf_alpha", "zf_xiaobei"
            guard voiceId.count >= 2 else { return false }
            let secondChar = voiceId[voiceId.index(voiceId.startIndex, offsetBy: 1)]
            return secondChar == "f"
        case .male:
            // Voice IDs like "am_adam", "bm_george", "im_nicola", "jm_kumo", "zm_yunyang"
            guard voiceId.count >= 2 else { return false }
            let secondChar = voiceId[voiceId.index(voiceId.startIndex, offsetBy: 1)]
            return secondChar == "m"
        }
    }
}

/// Language/accent for voice filtering
///
/// Based on Kokoro voice naming convention where the first character indicates language:
/// - a: American English
/// - b: British English
/// - e: Spanish (Español)
/// - f: French
/// - h: Hindi
/// - i: Italian
/// - j: Japanese
/// - p: Portuguese
/// - z: Chinese (Mandarin)
public enum VoiceLanguage: String, Sendable, Codable {
    case american = "a"    // American English (af_, am_)
    case british = "b"     // British English (bf_, bm_)
    case english = "ab"    // All English (American + British)
    case spanish = "e"     // Spanish (ef_, em_)
    case french = "f"      // French (ff_, fm_)
    case hindi = "h"       // Hindi (hf_, hm_)
    case italian = "i"     // Italian (if_, im_)
    case japanese = "j"    // Japanese (jf_, jm_)
    case portuguese = "p"  // Portuguese (pf_, pm_)
    case chinese = "z"     // Chinese/Mandarin (zf_, zm_)
    case any               // All languages
    
    /// Check if a voice ID matches this language
    public func matches(voiceId: String) -> Bool {
        switch self {
        case .any:
            return true
        case .english:
            // Matches both American (a) and British (b)
            guard !voiceId.isEmpty else { return false }
            let firstChar = String(voiceId[voiceId.startIndex])
            return firstChar == "a" || firstChar == "b"
        default:
            guard !voiceId.isEmpty else { return false }
            let firstChar = String(voiceId[voiceId.startIndex])
            return firstChar == self.rawValue
        }
    }
}

// MARK: - VoiceEmbeddingTable Language Filtering

extension VoiceEmbeddingTable {
    /// Filter table to only include voices of a specific language
    ///
    /// - Parameter language: Language to filter by
    /// - Returns: New table with only voices of the specified language
    public func filtered(by language: VoiceLanguage) -> VoiceEmbeddingTable {
        let filteredPairs = zip(voiceIds, embeddings).filter { voiceId, _ in
            language.matches(voiceId: voiceId)
        }
        
        return VoiceEmbeddingTable(
            voiceIds: filteredPairs.map(\.0),
            embeddings: filteredPairs.map(\.1),
            calibrationText: calibrationText
        )
    }
    
    /// Filter table by both gender and language
    ///
    /// - Parameters:
    ///   - gender: Gender to filter by
    ///   - language: Language to filter by
    /// - Returns: New table with only voices matching both criteria
    public func filtered(by gender: VoiceGender, language: VoiceLanguage) -> VoiceEmbeddingTable {
        let filteredPairs = zip(voiceIds, embeddings).filter { voiceId, _ in
            gender.matches(voiceId: voiceId) && language.matches(voiceId: voiceId)
        }
        
        return VoiceEmbeddingTable(
            voiceIds: filteredPairs.map(\.0),
            embeddings: filteredPairs.map(\.1),
            calibrationText: calibrationText
        )
    }
}

// MARK: - Voice Match Result

/// Result of voice matching operation
public struct VoiceMatchResult: Sendable {
    
    /// Voice weights [(voiceId, weight)] sorted by weight descending
    public let weights: [(name: String, weight: Float)]
    
    /// Similarity score between reference and matched blend (0-1)
    public let similarity: Float
    
    /// Time taken for matching in seconds
    public let matchTime: TimeInterval
    
    /// Top contributing voice
    public var primaryVoice: String? {
        weights.first?.name
    }
    
    /// Weights as dictionary for easy lookup
    public var weightsDictionary: [String: Float] {
        Dictionary(uniqueKeysWithValues: weights)
    }
}

// MARK: - Kokoro Voice Matcher

/// Voice matcher for Kokoro TTS
///
/// Finds optimal voice weight mixtures that approximate a reference speaker
/// by solving a constrained optimization problem in speaker embedding space.
///
/// ## Algorithm
/// 1. Extract speaker embedding from reference audio
/// 2. Compare to precomputed Kokoro voice embeddings
/// 3. Select top-K most similar voices
/// 4. Solve simplex-constrained least squares for optimal weights
/// 5. Return weights that minimize embedding distance
///
/// ## Usage
/// ```swift
/// let matcher = KokoroVoiceMatcher()
///
/// // Precompute voice embeddings (once)
/// let table = try await matcher.precomputeEmbeddings(
///     tts: kokoroTTS,
///     embeddingProvider: speakerEmbedding
/// )
///
/// // Match reference audio
/// let result = try await matcher.matchVoice(
///     referenceAudio: referenceAudio,
///     embeddingTable: table,
///     embeddingProvider: speakerEmbedding
/// )
///
/// // Use matched weights with mixVoices
/// let blended = try tts.mixVoices(result.weights)
/// ```
public actor KokoroVoiceMatcher {
    
    /// Protocol for embedding extraction (allows dependency injection)
    public protocol EmbeddingProvider: Sendable {
        func extractEmbedding(_ audio: [Float]) async throws -> [Float]
        var embeddingDimension: Int { get }
    }
    
    // MARK: - Properties
    
    /// Simplex solver for weight optimization
    private let solver: SimplexSolver
    
    /// Number of top voices to consider for blending
    private let defaultTopK: Int
    
    /// Calibration text for embedding computation
    private let calibrationText: String
    
    // MARK: - Initialization
    
    /// Initialize voice matcher
    /// - Parameters:
    ///   - topK: Default number of voices to blend (default: 5)
    ///   - calibrationText: Text for voice embedding calibration
    ///   - solverIterations: Max iterations for optimization (default: 100)
    public init(
        topK: Int = 5,
        calibrationText: String = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore.",
        solverIterations: Int = 100
    ) {
        self.defaultTopK = topK
        self.calibrationText = calibrationText
        self.solver = SimplexSolver(
            learningRate: 0.1,
            maxIterations: solverIterations,
            tolerance: 1e-6
        )
    }
    
    // MARK: - Precomputation
    
    /// Precompute voice embeddings for all loaded voices
    ///
    /// This should be called once to build the lookup table. The table can be
    /// serialized and cached for faster startup.
    ///
    /// - Parameters:
    ///   - tts: Kokoro TTS provider with loaded voices
    ///   - extractEmbedding: Closure to extract embeddings from synthesized audio
    /// - Returns: Voice embedding table
    public func precomputeEmbeddings(
        tts: KokoroTTSProvider,
        extractEmbedding: @Sendable @escaping ([Float]) async throws -> [Float]
    ) async throws -> VoiceEmbeddingTable {
        let voices = tts.availableVoices
        guard !voices.isEmpty else {
            throw AudioToolError.resourceUnavailable("No voices loaded in TTS provider")
        }
        
        var embeddings: [[Float]] = []
        embeddings.reserveCapacity(voices.count)
        
        for voiceId in voices {
            // Synthesize calibration text with this voice
            let audio = try await tts.synthesize(calibrationText, voice: voiceId)
            
            // Resample to 16kHz if needed (TTS outputs 24kHz)
            let samples16k = resampleTo16kHz(audio.samples, from: tts.sampleRate)
            
            // Extract speaker embedding
            let embedding = try await extractEmbedding(samples16k)
            embeddings.append(embedding)
        }
        
        return VoiceEmbeddingTable(
            voiceIds: voices,
            embeddings: embeddings,
            calibrationText: calibrationText
        )
    }
    
    // MARK: - Voice Matching
    
    /// Match reference audio to find optimal voice blend
    ///
    /// - Parameters:
    ///   - referenceAudio: Reference speaker audio samples (16kHz mono)
    ///   - embeddingTable: Precomputed voice embeddings
    ///   - extractEmbedding: Closure to extract embedding from reference
    ///   - topK: Number of top voices to blend (default: uses init value)
    /// - Returns: Voice match result with weights and similarity
    public func matchVoice(
        referenceAudio: [Float],
        embeddingTable: VoiceEmbeddingTable,
        extractEmbedding: @Sendable @escaping ([Float]) async throws -> [Float],
        topK: Int? = nil
    ) async throws -> VoiceMatchResult {
        let startTime = Date()
        let k = min(topK ?? defaultTopK, embeddingTable.count)
        
        guard k > 0 else {
            throw AudioToolError.resourceUnavailable("Empty embedding table")
        }
        
        // Step 1: Extract reference embedding
        let refEmbedding = try await extractEmbedding(referenceAudio)
        
        // Step 2: Compute similarities to all voices
        var similarities: [Float] = []
        similarities.reserveCapacity(embeddingTable.count)
        
        for embedding in embeddingTable.embeddings {
            let sim = SimplexSolver.cosineSimilarity(refEmbedding, embedding)
            similarities.append(sim)
        }
        
        // Step 3: Find top-K most similar voices
        let topIndices = SimplexSolver.topKIndices(similarities, k: k)
        
        // Step 4: Extract subset for optimization
        let topEmbeddings = topIndices.map { embeddingTable.embeddings[$0] }
        let topVoiceIds = topIndices.map { embeddingTable.voiceIds[$0] }
        
        // Step 5: Solve for optimal weights
        let weights = solver.solve(basisVectors: topEmbeddings, target: refEmbedding)
        
        // Step 6: Compute final similarity
        var blendedEmbedding = [Float](repeating: 0, count: refEmbedding.count)
        for (i, embedding) in topEmbeddings.enumerated() {
            for (j, val) in embedding.enumerated() {
                blendedEmbedding[j] += weights[i] * val
            }
        }
        let finalSimilarity = SimplexSolver.cosineSimilarity(refEmbedding, blendedEmbedding)
        
        // Create result with weights sorted by value descending
        var weightPairs: [(name: String, weight: Float)] = zip(topVoiceIds, weights)
            .map { ($0.0, $0.1) }
            .filter { $0.1 > 0.001 }  // Filter out negligible weights
            .sorted { $0.1 > $1.1 }
        
        // Renormalize after filtering
        let totalWeight = weightPairs.map(\.weight).reduce(0, +)
        if totalWeight > 0 {
            weightPairs = weightPairs.map { ($0.name, $0.weight / totalWeight) }
        }
        
        let matchTime = Date().timeIntervalSince(startTime)
        
        return VoiceMatchResult(
            weights: weightPairs,
            similarity: finalSimilarity,
            matchTime: matchTime
        )
    }
    
    /// Match reference audio from file URL
    ///
    /// Convenience method that handles audio loading and resampling.
    ///
    /// - Parameters:
    ///   - url: Path to reference audio file
    ///   - embeddingTable: Precomputed voice embeddings
    ///   - extractEmbedding: Closure to extract embedding from audio URL
    ///   - topK: Number of top voices to blend
    /// - Returns: Voice match result
    public func matchVoice(
        url: URL,
        embeddingTable: VoiceEmbeddingTable,
        extractEmbedding: @Sendable @escaping (URL) async throws -> [Float],
        topK: Int? = nil
    ) async throws -> VoiceMatchResult {
        let startTime = Date()
        let k = min(topK ?? defaultTopK, embeddingTable.count)
        
        // Extract reference embedding from URL
        let refEmbedding = try await extractEmbedding(url)
        
        // Compute similarities
        var similarities: [Float] = []
        for embedding in embeddingTable.embeddings {
            similarities.append(SimplexSolver.cosineSimilarity(refEmbedding, embedding))
        }
        
        // Find top-K and solve
        let topIndices = SimplexSolver.topKIndices(similarities, k: k)
        let topEmbeddings = topIndices.map { embeddingTable.embeddings[$0] }
        let topVoiceIds = topIndices.map { embeddingTable.voiceIds[$0] }
        
        let weights = solver.solve(basisVectors: topEmbeddings, target: refEmbedding)
        
        // Compute blended embedding for similarity
        var blendedEmbedding = [Float](repeating: 0, count: refEmbedding.count)
        for (i, embedding) in topEmbeddings.enumerated() {
            for (j, val) in embedding.enumerated() {
                blendedEmbedding[j] += weights[i] * val
            }
        }
        let finalSimilarity = SimplexSolver.cosineSimilarity(refEmbedding, blendedEmbedding)
        
        var weightPairs: [(name: String, weight: Float)] = zip(topVoiceIds, weights)
            .map { ($0.0, $0.1) }
            .filter { $0.1 > 0.001 }
            .sorted { $0.1 > $1.1 }
        
        let totalWeight = weightPairs.map(\.weight).reduce(0, +)
        if totalWeight > 0 {
            weightPairs = weightPairs.map { ($0.name, $0.weight / totalWeight) }
        }
        
        return VoiceMatchResult(
            weights: weightPairs,
            similarity: finalSimilarity,
            matchTime: Date().timeIntervalSince(startTime)
        )
    }
    
    // MARK: - Audio Utilities
    
    /// Resample audio from source rate to 16kHz using vDSP
    private func resampleTo16kHz(_ samples: [Float], from sourceRate: Int) -> [Float] {
        guard sourceRate != 16000 else { return samples }
        
        let ratio = Float(16000) / Float(sourceRate)
        let outputLength = Int(Float(samples.count) * ratio)
        guard outputLength > 0 else { return [] }
        
        var output = [Float](repeating: 0, count: outputLength)
        
        // Simple linear interpolation resampling
        // For production, consider using vDSP_desamp or AudioConverter
        for i in 0..<outputLength {
            let srcPos = Float(i) / ratio
            let srcIdx = Int(srcPos)
            let frac = srcPos - Float(srcIdx)
            
            let idx0 = min(srcIdx, samples.count - 1)
            let idx1 = min(srcIdx + 1, samples.count - 1)
            
            output[i] = samples[idx0] * (1 - frac) + samples[idx1] * frac
        }
        
        return output
    }
}

// MARK: - VoiceEmbeddingTable Persistence

extension VoiceEmbeddingTable {
    
    /// Save table to file
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
    
    /// Load table from file
    public static func load(from url: URL) throws -> VoiceEmbeddingTable {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(VoiceEmbeddingTable.self, from: data)
    }
}
