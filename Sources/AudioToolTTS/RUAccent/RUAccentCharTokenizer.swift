//
//  RUAccentCharTokenizer.swift
//  ClearVoiceTTS
//
//  Character-level tokenizer for RUAccent accent model
//

import Foundation

/// Character-level tokenizer for accent prediction model
public final class RUAccentCharTokenizer: Sendable {
    private let tokenToId: [String: Int]
    public let padId: Int
    public let unkId: Int
    public let bosId: Int
    public let eosId: Int
    
    public init(vocabURL: URL) throws {
        let vocabText = try String(contentsOf: vocabURL, encoding: .utf8)
        var map: [String: Int] = [:]
        
        for (idx, tok) in vocabText.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            map[String(tok)] = idx
        }
        
        self.tokenToId = map
        self.padId = map["[pad]"] ?? 0
        self.unkId = map["[unk]"] ?? 1
        self.bosId = map["[bos]"] ?? 2
        self.eosId = map["[eos]"] ?? 3
    }
    
    /// Encode text to token IDs
    public func encode(_ text: String) -> [Int] {
        var ids = [bosId]
        for ch in text {
            ids.append(tokenToId[String(ch)] ?? unkId)
        }
        ids.append(eosId)
        return ids
    }
}
