//
//  WordPieceTokenizer.swift
//  ClearVoiceTTS
//
//  WordPiece tokenization for RUAccent sentence-level models
//

import Foundation

/// Token offset for word reconstruction
public struct TokenOffset {
    public let startPos: Int
    public let endPos: Int
    public let isSubword: Bool
    
    public init(startPos: Int, endPos: Int, isSubword: Bool) {
        self.startPos = startPos
        self.endPos = endPos
        self.isSubword = isSubword
    }
}

/// WordPiece tokenization result
public struct WordPieceTokens {
    public let inputIds: [Int]
    public let attentionMask: [Int]
    public let tokenTypeIds: [Int]
    public let offsets: [TokenOffset]
}

/// WordPiece tokenizer for BERT-style models
public final class WordPieceTokenizer: Sendable {
    private let vocab: [String: Int]
    private let idToToken: [Int: String]
    private let unkId: Int
    private let clsId: Int
    private let sepId: Int
    private let padId: Int
    private let maxWordChars: Int
    
    public init(vocabURL: URL, maxWordChars: Int = 100) throws {
        let vocabText = try String(contentsOf: vocabURL, encoding: .utf8)
        var vocab: [String: Int] = [:]
        var idToToken: [Int: String] = [:]
        
        for (idx, line) in vocabText.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let token = String(line)
            vocab[token] = idx
            idToToken[idx] = token
        }
        
        self.vocab = vocab
        self.idToToken = idToToken
        self.unkId = vocab["[UNK]"] ?? 1
        self.clsId = vocab["[CLS]"] ?? 101
        self.sepId = vocab["[SEP]"] ?? 102
        self.padId = vocab["[PAD]"] ?? 0
        self.maxWordChars = maxWordChars
    }
    
    /// Encode text with token offsets for entity alignment
    public func encode(_ text: String, maxLength: Int, pair: String? = nil) -> WordPieceTokens {
        var tokens: [Int] = [clsId]
        var offsets: [TokenOffset] = []
        
        // Tokenize first segment
        let (firstTokens, firstOffsets) = tokenizeWithOffsets(text)
        tokens.append(contentsOf: firstTokens)
        offsets.append(contentsOf: firstOffsets)
        tokens.append(sepId)
        
        // Token type IDs: 0 for first segment
        var typeIds = Array(repeating: 0, count: tokens.count)
        
        // Tokenize pair if present
        if let pair = pair {
            let (pairTokens, _) = tokenizeWithOffsets(pair)
            tokens.append(contentsOf: pairTokens)
            tokens.append(sepId)
            // Type 1 for second segment
            typeIds.append(contentsOf: Array(repeating: 1, count: pairTokens.count + 1))
        }
        
        // Truncate if needed
        if tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength))
            typeIds = Array(typeIds.prefix(maxLength))
            // Ensure ends with SEP
            tokens[maxLength - 1] = sepId
        }
        
        // Create attention mask before padding
        var attention = Array(repeating: 1, count: tokens.count)
        
        // Pad to maxLength
        let padCount = maxLength - tokens.count
        if padCount > 0 {
            tokens.append(contentsOf: Array(repeating: padId, count: padCount))
            attention.append(contentsOf: Array(repeating: 0, count: padCount))
            typeIds.append(contentsOf: Array(repeating: 0, count: padCount))
        }
        
        return WordPieceTokens(
            inputIds: tokens,
            attentionMask: attention,
            tokenTypeIds: typeIds,
            offsets: offsets
        )
    }
    
    private func tokenizeWithOffsets(_ text: String) -> ([Int], [TokenOffset]) {
        var tokens: [Int] = []
        var offsets: [TokenOffset] = []
        
        // Simple whitespace tokenization
        let words = text.split(separator: " ", omittingEmptySubsequences: true)
        var currentPos = 0
        
        for word in words {
            let wordStr = String(word)
            let wordLower = wordStr.lowercased()
            
            // Find word position in original text
            if let range = text.range(of: wordStr, range: text.index(text.startIndex, offsetBy: currentPos)..<text.endIndex) {
                let startPos = text.distance(from: text.startIndex, to: range.lowerBound)
                let endPos = text.distance(from: text.startIndex, to: range.upperBound)
                currentPos = endPos
                
                // WordPiece tokenization
                let wordPieces = wordPieceTokenize(wordLower)
                
                for (i, pieceId) in wordPieces.enumerated() {
                    tokens.append(pieceId)
                    offsets.append(TokenOffset(
                        startPos: startPos,
                        endPos: endPos,
                        isSubword: i > 0
                    ))
                }
            }
        }
        
        return (tokens, offsets)
    }
    
    private func wordPieceTokenize(_ word: String) -> [Int] {
        if word.count > maxWordChars {
            return [unkId]
        }
        
        var tokens: [Int] = []
        var start = 0
        let chars = Array(word)
        
        while start < chars.count {
            var end = chars.count
            var foundToken: Int? = nil
            
            while start < end {
                var substr = String(chars[start..<end])
                if start > 0 {
                    substr = "##" + substr
                }
                
                if let id = vocab[substr] {
                    foundToken = id
                    break
                }
                end -= 1
            }
            
            if let token = foundToken {
                tokens.append(token)
                start = end
            } else {
                tokens.append(unkId)
                start += 1
            }
        }
        
        return tokens
    }
}
