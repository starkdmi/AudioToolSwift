import Foundation

public struct MTLTokenizerConfig {
    public var lowercase: Bool = true
    public var nfkdNormalize: Bool = true

    public init() {}
}

private struct BpePair: Hashable {
    let first: String
    let second: String
}

private struct BpeModel {
    let vocab: [String: Int]
    let merges: [BpePair]
    let mergeRanks: [BpePair: Int]
    let specialTokens: Set<String>
    let unkToken: String
}

public final class MTLTokenizer {
    private let model: BpeModel
    private let config: MTLTokenizerConfig
    private var cache: [String: [String]] = [:]

    public init(tokenizerJSONPath: String, config: MTLTokenizerConfig = MTLTokenizerConfig()) throws {
        self.model = try MTLTokenizer.loadModel(path: tokenizerJSONPath)
        self.config = config
    }

    public func textToTokens(_ text: String, languageId: String? = nil) -> [Int] {
        let tokens = encode(text, languageId: languageId)
        return tokens.map { model.vocab[$0] ?? model.vocab[model.unkToken] ?? 1 }
    }

    public func encode(_ text: String, languageId: String? = nil) -> [String] {
        var txt = preprocess(text)
        if let languageId = languageId, !languageId.isEmpty {
            txt = "[\(languageId.lowercased())]" + txt
        }
        txt = txt.replacingOccurrences(of: " ", with: "[SPACE]")

        let pieces = Self.splitSpecialTokens(txt, specialTokens: model.specialTokens)
        var out: [String] = []
        for piece in pieces {
            if model.specialTokens.contains(piece) {
                out.append(piece)
            } else if piece.isEmpty {
                continue
            } else {
                out.append(contentsOf: bpe(piece))
            }
        }
        return out
    }

    private func preprocess(_ text: String) -> String {
        var out = text
        if config.lowercase {
            out = out.lowercased()
        }
        if config.nfkdNormalize {
            out = out.decomposedStringWithCompatibilityMapping
        }
        return out
    }

    private func bpe(_ token: String) -> [String] {
        if let cached = cache[token] {
            return cached
        }

        var word = token.unicodeScalars.map { String($0) }
        if word.count <= 1 {
            cache[token] = word
            return word
        }

        var pairs = getPairs(word)
        while true {
            var bestPair: BpePair? = nil
            var bestRank = Int.max
            for pair in pairs {
                if let rank = model.mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }
            guard let mergePair = bestPair else { break }

            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1,
                   word[i] == mergePair.first,
                   word[i + 1] == mergePair.second {
                    newWord.append(word[i] + word[i + 1])
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count <= 1 {
                break
            }
            pairs = getPairs(word)
        }

        cache[token] = word
        return word
    }

    private func getPairs(_ word: [String]) -> Set<BpePair> {
        if word.count < 2 {
            return []
        }
        var pairs: Set<BpePair> = []
        for i in 0..<(word.count - 1) {
            pairs.insert(BpePair(first: word[i], second: word[i + 1]))
        }
        return pairs
    }

    private static func splitSpecialTokens(_ text: String, specialTokens: Set<String>) -> [String] {
        var out: [String] = []
        var buffer = ""
        var idx = text.startIndex
        while idx < text.endIndex {
            let ch = text[idx]
            if ch == "[" {
                if let close = text[idx...].firstIndex(of: "]") {
                    let candidate = String(text[idx...close])
                    if specialTokens.contains(candidate) {
                        if !buffer.isEmpty {
                            out.append(buffer)
                            buffer = ""
                        }
                        out.append(candidate)
                        idx = text.index(after: close)
                        continue
                    }
                }
            }
            buffer.append(ch)
            idx = text.index(after: idx)
        }
        if !buffer.isEmpty {
            out.append(buffer)
        }
        return out
    }

    private static func loadModel(path: String) throws -> BpeModel {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
        guard let root = json else {
            throw NSError(domain: "MTLTokenizer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid tokenizer JSON"])
        }
        guard let modelDict = root["model"] as? [String: Any] else {
            throw NSError(domain: "MTLTokenizer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing model section"])
        }
        guard let vocabDict = modelDict["vocab"] as? [String: Any] else {
            throw NSError(domain: "MTLTokenizer", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing vocab section"])
        }
        guard let mergesList = modelDict["merges"] as? [String] else {
            throw NSError(domain: "MTLTokenizer", code: 4, userInfo: [NSLocalizedDescriptionKey: "Missing merges section"])
        }

        var vocab: [String: Int] = [:]
        for (key, value) in vocabDict {
            if let id = value as? Int {
                vocab[key] = id
            } else if let id = value as? NSNumber {
                vocab[key] = id.intValue
            }
        }

        var merges: [BpePair] = []
        merges.reserveCapacity(mergesList.count)
        for line in mergesList {
            let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }
            merges.append(BpePair(first: parts[0], second: parts[1]))
        }

        var mergeRanks: [BpePair: Int] = [:]
        mergeRanks.reserveCapacity(merges.count)
        for (idx, pair) in merges.enumerated() {
            mergeRanks[pair] = idx
        }

        var specialTokens: Set<String> = []
        if let added = root["added_tokens"] as? [[String: Any]] {
            for entry in added {
                if let content = entry["content"] as? String {
                    specialTokens.insert(content)
                }
            }
        }

        let unkToken = (modelDict["unk_token"] as? String) ?? "[UNK]"
        specialTokens.insert(unkToken)

        // Ensure language tags are treated as specials when present in vocab.
        let langTags = ["en", "ru", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "nl", "pl", "tr", "ar", "hi", "id", "th", "vi", "sv", "da", "no", "fi", "cs", "el", "he"]
        for tag in langTags {
            let tok = "[\(tag)]"
            if vocab[tok] != nil {
                specialTokens.insert(tok)
            }
        }

        return BpeModel(vocab: vocab, merges: merges, mergeRanks: mergeRanks, specialTokens: specialTokens, unkToken: unkToken)
    }
}
