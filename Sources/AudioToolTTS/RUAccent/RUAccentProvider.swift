//
//  RUAccentProvider.swift
//  AudioToolTTS
//
//  Russian stress/accent marking using CoreML
//
//  Adds stress marks (+) to Russian text for proper TTS pronunciation.
//  Example: "привет мир" → "прив+ет м+ир"
//

import Foundation
import CoreML
import AudioToolCore
import Compression

/// Russian stress/accent text preprocessor using CoreML
///
/// Uses a multi-model pipeline to add stress marks to Russian text:
/// - **accent**: Character-level stress position prediction
/// - **yo**: ё/е homograph resolution
/// - **stress**: Stress usage prediction (per-word)
/// - **omograph**: Contextual disambiguation
///
/// Usage:
/// ```swift
/// let provider = try RUAccentProvider(
///     profile: .balanced,
///     modelsDir: URL(fileURLWithPath: "path/to/models/balanced"),
///     assetsDir: URL(fileURLWithPath: "path/to/assets/balanced")
/// )
/// let stressed = try provider.process("привет мир")
/// // stressed = "прив+ет м+ир"
/// ```
public final class RUAccentProvider: TextPreprocessor, @unchecked Sendable {
    
    // MARK: - Configuration
    
    private let profile: RUAccentProfile
    private let sentenceLength: Int
    private let wordLength: Int
    private let threshold: Float
    
    // MARK: - Models
    
    private let accentModel: MLModel
    private let yoModel: MLModel
    private let stressModel: MLModel?
    private let omographModel: MLModel
    
    // MARK: - Tokenizers
    
    private let charTokenizer: RUAccentCharTokenizer
    private let wordpieceTokenizer: WordPieceTokenizer
    private let stressTokenizer: WordPieceTokenizer?
    
    // MARK: - Dictionaries
    
    private let id2labelAccent: [String: String]
    private let id2labelYo: [String: String]
    private let id2labelStress: [String: String]?
    
    private let omographs: [String: [String]]
    private let yoWords: [String: String]
    private let yoHomographs: [String: String]
    private let accents: [String: String]
    
    private let lettersAccent: [String: String] = ["о": "+о", "О": "+О"]
    
    // MARK: - Special Words
    
    private let specialWords: Set<String> = [
        "балчуга", "вертела", "волоки", "волоку", "воронью", "выбродите",
        "вывозите", "выносите", "выноситесь", "выходите", "железы", "начала",
        "округа", "перепела", "развитая", "развитого", "развитое", "развитой",
        "развитом", "развитому", "развитою", "развитую", "развитые", "развитым",
        "развитыми", "развитых", "сторожа", "сторожи", "сторожу", "удало",
        "начался", "началась", "началось", "бутиках", "ожила", "создало",
        "коротки", "проклята", "роженица", "роженицы", "рожениц", "роженице",
        "роженицам", "роженицу", "роженицей", "роженицею", "роженицами",
        "роженицах", "пристава", "приставов", "приставам", "приставами",
        "приставах", "пережитое", "пережитого", "пережитые", "пережитых",
        "пережитому", "пережитым", "пережитыми", "пережитом", "нипоняла"
    ]
    
    // MARK: - Initialization
    
    /// Initialize RUAccent provider
    /// - Parameters:
    ///   - profile: Pipeline profile (lightweight/balanced/max)
    ///   - modelsDir: Directory containing *.mlpackage or *.mlmodelc models
    ///   - assetsDir: Directory containing dictionaries and tokenizer configs
    ///   - sentenceLength: Max sentence length in tokens (default: 128)
    ///   - wordLength: Max word length in chars (default: 56)
    ///   - threshold: Stress confidence threshold (default: 0.55)
    public init(
        profile: RUAccentProfile = .balanced,
        modelsDir: URL,
        assetsDir: URL,
        sentenceLength: Int = 128,
        wordLength: Int = 56,
        threshold: Float = 0.55
    ) throws {
        self.profile = profile
        self.sentenceLength = sentenceLength
        self.wordLength = wordLength
        self.threshold = threshold
        
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .all
        
        // Load models
        accentModel = try Self.loadModel(named: "accent", from: modelsDir, configuration: mlConfig)
        yoModel = try Self.loadModel(named: "yo", from: modelsDir, configuration: mlConfig)
        omographModel = try Self.loadModel(named: "omograph", from: modelsDir, configuration: mlConfig)
        
        if profile != .lightweight {
            stressModel = try Self.loadModel(named: "stress", from: modelsDir, configuration: mlConfig)
        } else {
            stressModel = nil
        }
        
        // Load tokenizers
        let accentVocabURL = assetsDir.appendingPathComponent("nn/nn_accent/vocab.txt")
        charTokenizer = try RUAccentCharTokenizer(vocabURL: accentVocabURL)
        
        let wpVocabURL = assetsDir.appendingPathComponent("nn/nn_yo_homograph_resolver/vocab.txt")
        wordpieceTokenizer = try WordPieceTokenizer(vocabURL: wpVocabURL)
        
        if profile != .lightweight {
            let stressVocabURL = assetsDir.appendingPathComponent("nn/nn_stress_usage_predictor/vocab.txt")
            stressTokenizer = try WordPieceTokenizer(vocabURL: stressVocabURL)
        } else {
            stressTokenizer = nil
        }
        
        // Load id2label configs
        let accentConfigURL = assetsDir.appendingPathComponent("nn/nn_accent/config.json")
        id2labelAccent = try Self.loadId2Label(url: accentConfigURL)
        
        let yoConfigURL = assetsDir.appendingPathComponent("nn/nn_yo_homograph_resolver/config.json")
        id2labelYo = try Self.loadId2Label(url: yoConfigURL)
        
        if stressModel != nil {
            let stressConfigURL = assetsDir.appendingPathComponent("nn/nn_stress_usage_predictor/config.json")
            id2labelStress = try Self.loadId2Label(url: stressConfigURL)
        } else {
            id2labelStress = nil
        }
        
        // Load dictionaries
        omographs = try Self.loadGzipJsonDictArray(
            url: assetsDir.appendingPathComponent("dictionary/omographs.json.gz")
        )
        yoWords = try Self.loadGzipJsonDict(
            url: assetsDir.appendingPathComponent("dictionary/yo_words.json.gz")
        )
        yoHomographs = try Self.loadGzipJsonDict(
            url: assetsDir.appendingPathComponent("dictionary/yo_homographs.json.gz")
        )
        
        if profile == .lightweight {
            accents = try Self.loadGzipJsonDict(
                url: assetsDir.appendingPathComponent("dictionary/accents_nn.json.gz")
            )
        } else {
            var full = try Self.loadGzipJsonDict(
                url: assetsDir.appendingPathComponent("dictionary/accents.json.gz")
            )
            for (k, v) in lettersAccent { full[k] = v }
            accents = full
        }
    }
    
    // MARK: - TextPreprocessor Protocol
    
    /// Process Russian text to add stress marks
    /// - Parameter text: Input Russian text
    /// - Returns: Text with stress marks (+) added
    public func process(_ text: String) throws -> String {
        let normalized = text.replacingOccurrences(
            of: "[^a-zA-Z0-9\\sа-яА-ЯёЁ—.,!?:;\"''(){}\\[\\]«»„\"\"-]",
            with: "",
            options: .regularExpression
        )
        
        let (sentences, separators) = Self.splitBySentences(normalized)
        var outputs: [String] = []
        
        for sentence in sentences {
            let (words, remaining) = Self.splitByWords(sentence)
            
            if words.isEmpty {
                outputs.append(remaining.joined())
                continue
            }
            
            // Get stress usage predictions
            let stressUsages: [String]
            if let stressModel = stressModel,
               let id2labelStress = id2labelStress,
               let stressTokenizer = stressTokenizer {
                stressUsages = try predictTokenEntities(
                    text: sentence,
                    model: stressModel,
                    id2label: id2labelStress,
                    tokenizer: stressTokenizer
                )
            } else {
                stressUsages = Array(repeating: "STRESS", count: words.count)
            }
            
            // Process pipeline stages
            var processedWords = try processYo(words: words, sentence: sentence)
            processedWords = try processOmographs(words: processedWords)
            processedWords = try processAccent(words: processedWords, stressUsages: stressUsages)
            
            // Reconstruct sentence
            var sentenceOut = ""
            for (prefix, word) in zip(remaining, processedWords) {
                sentenceOut += prefix + word
            }
            if let last = remaining.last {
                sentenceOut += last
            }
            
            outputs.append(deleteSpacesBeforePunc(sentenceOut))
        }
        
        // Join sentences with their original separators
        if outputs.isEmpty {
            return ""
        }
        var result = outputs[0]
        for (sep, out) in zip(separators.dropFirst(), outputs.dropFirst()) {
            result += sep + out
        }
        return result
    }
    
    // MARK: - Stress Mark Conversion
    
    /// Converts "+" stress notation to Unicode combining acute accent (U+0301)
    ///
    /// The `process()` method outputs stress marks as "+" before the stressed vowel.
    /// This method converts that notation to proper Unicode stress marks for display.
    ///
    /// Example:
    /// ```swift
    /// let stressed = try provider.process("мы закрыли замок")
    /// // stressed = "м+ы закр+ыли зам+ок"
    /// let display = provider.convertToStressMarks(stressed)
    /// // display = "мы́ закры́ли замо́к"
    /// ```
    ///
    /// - Parameter text: Text with "+" stress notation
    /// - Returns: Text with Unicode combining acute accents on stressed vowels
    public func convertToStressMarks(_ text: String) -> String {
        Self.convertToStressMarks(text)
    }
    
    /// Static version of stress mark conversion
    ///
    /// Converts "+" stress notation to Unicode combining acute accent (U+0301)
    ///
    /// - Parameter text: Text with "+" stress notation
    /// - Returns: Text with Unicode combining acute accents on stressed vowels
    public static func convertToStressMarks(_ text: String) -> String {
        let vowels: Set<Character> = [
            "а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я",
            "А", "Е", "Ё", "И", "О", "У", "Ы", "Э", "Ю", "Я"
        ]
        
        var result: [Character] = []
        var accentNext = false
        
        for ch in text {
            if ch == "+" {
                accentNext = true
                continue
            }
            result.append(ch)
            if accentNext && vowels.contains(ch) {
                result.append("\u{0301}")  // Combining acute accent
            }
            accentNext = false
        }
        
        return String(result)
    }
    
    // MARK: - Pipeline Stages
    
    private func processYo(words: [String], sentence: String) throws -> [String] {
        let lowerSentence = sentence.lowercased()
        var yoPredictions: [String]? = nil
        
        if lowerSentence.contains("е") {
            yoPredictions = try predictTokenEntities(
                text: lowerSentence,
                model: yoModel,
                id2label: id2labelYo,
                tokenizer: wordpieceTokenizer
            )
        }
        
        var out = words
        for i in 0..<words.count {
            let lowerWord = words[i].lowercased()
            let yoWord = yoWords[lowerWord] ?? words[i]
            out[i] = fixCapital(source: words[i], target: yoWord)
            
            if let preds = yoPredictions, i < preds.count, preds[i] == "YO" {
                let yoResolved = yoHomographs[lowerWord] ?? words[i]
                out[i] = fixCapital(source: words[i], target: yoResolved)
            }
        }
        
        return out
    }
    
    private func processOmographs(words: [String]) throws -> [String] {
        var split = words
        var found: [[String: Any]] = []
        var texts: [[String]] = []
        var hypotheses: [[String]] = []
        
        for (i, word) in split.enumerated() {
            if let variants = omographs[word] {
                found.append(["word": word, "variants": variants, "position": i])
                texts.append(split)
                hypotheses.append(variants)
            }
        }
        
        if found.isEmpty { return split }
        
        var textsBatch: [String] = []
        let hypothesesBatch = hypotheses.flatMap { $0 }
        let numHypotheses = hypotheses.map { $0.count }
        
        for (idx, entry) in found.enumerated() {
            var text = texts[idx]
            let position = entry["position"] as! Int
            let original = text[position]
            text[position] = " <w>" + text[position] + "</w> "
            
            for _ in 0..<(entry["variants"] as! [String]).count {
                textsBatch.append(deleteSpacesBeforePunc(text.joined(separator: " ")))
            }
            text[position] = original
        }
        
        let chosen = try classifyOmographs(texts: textsBatch, hypotheses: hypothesesBatch, counts: numHypotheses)
        
        for (idx, entry) in found.enumerated() {
            let position = entry["position"] as! Int
            split[position] = chosen[idx]
        }
        
        return split
    }
    
    private func processAccent(words: [String], stressUsages: [String]) throws -> [String] {
        var out = words
        
        for i in 0..<words.count {
            if words[i].contains("+") { continue }
            if i >= stressUsages.count { continue }
            if stressUsages[i] != "STRESS" { continue }
            
            let lowerWord = words[i].lowercased()
            let stressedWord = accents[lowerWord] ?? lowerWord
            
            if stressedWord == lowerWord && !hasPunctuation(lowerWord) && countVowels(lowerWord) > 1 {
                out[i] = try accentWord(words[i])
            } else {
                out[i] = applyPlusPositions(source: words[i], stressed: stressedWord)
            }
        }
        
        return out
    }
    
    private func accentWord(_ word: String) throws -> String {
        let ids = charTokenizer.encode(word.lowercased())
        if ids.count > wordLength { return word }
        
        let inputIds = try makeInt32Array(ids, padTo: wordLength, padValue: charTokenizer.padId)
        let attention = try makeInt32Mask(count: wordLength, ones: ids.count)
        let tokenTypes = try makeInt32Mask(count: wordLength, ones: 0)
        
        let inputFeatures: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attention),
            "token_type_ids": MLFeatureValue(multiArray: tokenTypes)
        ]
        
        let provider = try MLDictionaryFeatureProvider(dictionary: inputFeatures)
        let output = try accentModel.prediction(from: provider)
        
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            return word
        }
        
        return renderStress(word: word, logits: logits, id2label: id2labelAccent)
    }
    
    private func predictTokenEntities(
        text: String,
        model: MLModel,
        id2label: [String: String],
        tokenizer: WordPieceTokenizer
    ) throws -> [String] {
        let tokens = tokenizer.encode(text, maxLength: sentenceLength)
        
        let inputIds = try makeInt32Array(tokens.inputIds, padTo: sentenceLength, padValue: 0)
        let attention = try makeInt32Array(tokens.attentionMask, padTo: sentenceLength, padValue: 0)
        let tokenTypes = try makeInt32Array(tokens.tokenTypeIds, padTo: sentenceLength, padValue: 0)
        
        let inputFeatures: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attention),
            "token_type_ids": MLFeatureValue(multiArray: tokenTypes)
        ]
        
        let provider = try MLDictionaryFeatureProvider(dictionary: inputFeatures)
        let output = try model.prediction(from: provider)
        
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            return []
        }
        
        let scores = softmaxPerToken(logits, classCount: id2label.count)
        return aggregateOffsets(scores: scores, offsets: tokens.offsets, id2label: id2label)
    }
    
    private func classifyOmographs(texts: [String], hypotheses: [String], counts: [Int]) throws -> [String] {
        if counts.allSatisfy({ $0 % 2 == 0 }) {
            return try classifyPairs(texts: texts, hypotheses: hypotheses)
        }
        
        // Group words and classify
        let grouped = groupWords(words: hypotheses)
        let groupedTexts = transferGrouping(groups: grouped, targets: texts)
        
        var outputs: [String] = []
        for (hyps, textGroup) in zip(grouped, groupedTexts) {
            var scores: [Float] = []
            for hyp in hyps {
                scores.append(try scorePair(text: textGroup[0], hypothesis: hyp))
            }
            if let maxIdx = scores.indices.max(by: { scores[$0] < scores[$1] }) {
                outputs.append(hyps[maxIdx])
            }
        }
        
        return outputs
    }
    
    private func classifyPairs(texts: [String], hypotheses: [String]) throws -> [String] {
        var outputs: [String] = []
        var pairs: [(String, String)] = []
        var scores: [Float] = []
        
        for i in stride(from: 0, to: hypotheses.count, by: 2) {
            pairs.append((hypotheses[i], hypotheses[i + 1]))
        }
        
        for (text, hyp) in zip(texts, hypotheses) {
            scores.append(try scorePair(text: text, hypothesis: hyp))
        }
        
        var scorePairs: [(Float, Float)] = []
        for i in stride(from: 0, to: scores.count, by: 2) {
            scorePairs.append((scores[i], scores[i + 1]))
        }
        
        for (pair, score) in zip(pairs, scorePairs) {
            outputs.append(score.0 >= score.1 ? pair.0 : pair.1)
        }
        
        return outputs
    }
    
    private func scorePair(text: String, hypothesis: String) throws -> Float {
        let tokens = wordpieceTokenizer.encode(text, maxLength: sentenceLength, pair: hypothesis)
        let inputIds = tokens.inputIds
        let attention = inputIds.map { $0 == 0 ? 0 : 1 }
        
        let inputArray = try makeInt32Array(inputIds, padTo: sentenceLength, padValue: 0)
        let attentionArray = try makeInt32Array(attention, padTo: sentenceLength, padValue: 0)
        
        let inputFeatures: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputArray),
            "attention_mask": MLFeatureValue(multiArray: attentionArray)
        ]
        
        let provider = try MLDictionaryFeatureProvider(dictionary: inputFeatures)
        let output = try omographModel.prediction(from: provider)
        
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            return 0.0
        }
        
        let probs = softmaxVector(logits)
        return probs[1]
    }
    
    // MARK: - Helper Methods
    
    private func renderStress(word: String, logits: MLMultiArray, id2label: [String: String]) -> String {
        let scores = softmaxPerToken(logits, classCount: 3)
        let chars = Array(word)
        var out: [String] = []
        
        for idx in 0..<chars.count {
            let tokenIdx = idx + 1
            if tokenIdx >= scores.count / 3 { break }
            
            let labelIdx = argmax(scores, tokenIdx: tokenIdx, classCount: 3)
            let label = id2label[String(labelIdx)] ?? "NO"
            let prob = scores[(tokenIdx * 3) + labelIdx]
            
            if label == "STRESS_PRIMARY" && prob >= threshold {
                out.append("+")
            }
            out.append(String(chars[idx]))
        }
        
        return out.joined()
    }
    
    private func applyPlusPositions(source: String, stressed: String) -> String {
        var out = source
        var offset = 0
        
        for (idx, ch) in stressed.enumerated() where ch == "+" {
            let insertIndex = out.index(out.startIndex, offsetBy: idx - offset)
            out.insert("+", at: insertIndex)
            offset += 1
        }
        
        return out
    }
    
    private func fixCapital(source: String, target: String) -> String {
        if source.count != target.count { return target }
        
        let sourceChars = Array(source)
        let targetChars = Array(target)
        var out: [String] = []
        
        for (s, t) in zip(sourceChars, targetChars) {
            out.append(s.isUppercase ? String(t).uppercased() : String(t).lowercased())
        }
        
        return out.joined()
    }
    
    private func hasPunctuation(_ text: String) -> Bool {
        text.rangeOfCharacter(from: CharacterSet.punctuationCharacters) != nil
    }
    
    private func countVowels(_ text: String) -> Int {
        let vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
        return text.filter { vowels.contains($0) }.count
    }
    
    private func deleteSpacesBeforePunc(_ text: String) -> String {
        var result = text
        let punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-"
        
        for ch in punc {
            let s = String(ch)
            if ch == "-" {
                result = result.replacingOccurrences(of: " " + s, with: s)
                result = result.replacingOccurrences(of: s + " ", with: s)
            }
            result = result.replacingOccurrences(of: " " + s, with: s)
        }
        
        return result.replacingOccurrences(of: "~", with: "-")
    }
    
    private func groupWords(words: [String]) -> [[String]] {
        guard let first = words.first else { return [] }
        
        var result: [[String]] = []
        var currentGroup = [first]
        var currentBase = first.replacingOccurrences(of: "+", with: "")
        
        for word in words.dropFirst() {
            let base = word.replacingOccurrences(of: "+", with: "")
            if base == currentBase {
                currentGroup.append(word)
            } else {
                if specialWords.contains(currentBase) && currentGroup.count > 3 {
                    result.append(contentsOf: strideGroups(currentGroup, size: 3))
                } else if currentGroup.count > 3 && currentGroup.count % 2 == 0 {
                    result.append(contentsOf: strideGroups(currentGroup, size: 2))
                } else {
                    result.append(currentGroup)
                }
                currentGroup = [word]
                currentBase = base
            }
        }
        
        if specialWords.contains(currentBase) && currentGroup.count > 3 {
            result.append(contentsOf: strideGroups(currentGroup, size: 3))
        } else if currentGroup.count > 3 && currentGroup.count % 2 == 0 {
            result.append(contentsOf: strideGroups(currentGroup, size: 2))
        } else {
            result.append(currentGroup)
        }
        
        return result
    }
    
    private func strideGroups(_ group: [String], size: Int) -> [[String]] {
        var output: [[String]] = []
        var index = 0
        
        while index < group.count {
            output.append(Array(group[index..<min(index + size, group.count)]))
            index += size
        }
        
        return output
    }
    
    private func transferGrouping(groups: [[String]], targets: [String]) -> [[String]] {
        var newGroups: [[String]] = []
        var start = 0
        
        for group in groups {
            let length = group.count
            let slice = Array(targets[start..<min(start + length, targets.count)])
            newGroups.append(slice)
            start += length
        }
        
        return newGroups
    }
    
    // MARK: - Softmax Utilities
    
    private func softmaxVector(_ logits: MLMultiArray) -> [Float] {
        let count = logits.count
        let ptr = logits.dataPointer.bindMemory(to: Float32.self, capacity: count)
        var values = [Float](repeating: 0, count: count)
        
        for i in 0..<count { values[i] = ptr[i] }
        
        let maxVal = values.max() ?? 0
        var expVals = values.map { Foundation.exp($0 - maxVal) }
        let sum = expVals.reduce(0, +)
        
        if sum == 0 { return values }
        expVals = expVals.map { $0 / sum }
        
        return expVals
    }
    
    private func softmaxPerToken(_ logits: MLMultiArray, classCount: Int) -> [Float] {
        let count = logits.count
        if classCount <= 0 || count == 0 { return [] }
        
        let ptr = logits.dataPointer.bindMemory(to: Float32.self, capacity: count)
        var values = [Float](repeating: 0, count: count)
        
        for i in 0..<count { values[i] = ptr[i] }
        
        let tokenCount = count / classCount
        var scores = [Float](repeating: 0, count: count)
        
        for t in 0..<tokenCount {
            let base = t * classCount
            let slice = Array(values[base..<(base + classCount)])
            let maxVal = slice.max() ?? 0
            var expVals = slice.map { Foundation.exp($0 - maxVal) }
            let sum = expVals.reduce(0, +)
            
            if sum == 0 {
                for i in 0..<classCount { scores[base + i] = slice[i] }
            } else {
                expVals = expVals.map { $0 / sum }
                for i in 0..<classCount { scores[base + i] = expVals[i] }
            }
        }
        
        return scores
    }
    
    private func argmax(_ scores: [Float], tokenIdx: Int, classCount: Int) -> Int {
        let offset = tokenIdx * classCount
        var bestIdx = 0
        var bestVal = scores[offset]
        
        for i in 1..<classCount {
            let val = scores[offset + i]
            if val > bestVal {
                bestVal = val
                bestIdx = i
            }
        }
        
        return bestIdx
    }
    
    private func aggregateOffsets(
        scores: [Float],
        offsets: [TokenOffset],
        id2label: [String: String]
    ) -> [String] {
        var entities: [String] = []
        var currentCount = 0
        var currentLabelScores: [Float] = []
        let classCount = id2label.count
        
        func flushCurrent() {
            guard currentCount > 0 else { return }
            
            var averaged: [Float] = []
            for i in 0..<classCount {
                averaged.append(currentLabelScores[i] / Float(currentCount))
            }
            
            if let maxIdx = averaged.indices.max(by: { averaged[$0] < averaged[$1] }) {
                entities.append(id2label[String(maxIdx)] ?? "NO")
            }
            
            currentLabelScores = []
            currentCount = 0
        }
        
        for (i, offset) in offsets.enumerated() {
            let tokenIdx = i + 1
            let base = tokenIdx * classCount
            
            if !offset.isSubword {
                flushCurrent()
            }
            
            if currentLabelScores.isEmpty {
                currentLabelScores = Array(repeating: 0, count: classCount)
            }
            
            for j in 0..<classCount {
                currentLabelScores[j] += scores[base + j]
            }
            currentCount += 1
        }
        
        flushCurrent()
        return entities
    }
    
    // MARK: - MLMultiArray Helpers
    
    private func makeInt32Array(_ values: [Int], padTo: Int, padValue: Int) throws -> MLMultiArray {
        var padded = values
        if values.count < padTo {
            padded.append(contentsOf: Array(repeating: padValue, count: padTo - values.count))
        }
        
        let array = try MLMultiArray(shape: [1, NSNumber(value: padTo)], dataType: .int32)
        let stride = array.strides[1].intValue
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        
        for i in 0..<padTo {
            ptr[i * stride] = Int32(padded[i])
        }
        
        return array
    }
    
    private func makeInt32Mask(count: Int, ones: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: count)], dataType: .int32)
        let stride = array.strides[1].intValue
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        
        for i in 0..<count {
            ptr[i * stride] = i < ones ? 1 : 0
        }
        
        return array
    }
    
    // MARK: - Model Loading
    
    private static func loadModel(
        named name: String,
        from directory: URL,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        // Try .mlmodelc first (pre-compiled)
        let compiledURL = directory.appendingPathComponent("\(name).mlmodelc")
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }
        
        // Try .mlpackage and compile it
        let packageURL = directory.appendingPathComponent("\(name).mlpackage")
        if FileManager.default.fileExists(atPath: packageURL.path) {
            let compiledModelURL = try MLModel.compileModel(at: packageURL)
            return try MLModel(contentsOf: compiledModelURL, configuration: configuration)
        }
        
        throw AudioToolError.modelNotFound("RUAccent model '\(name)' not found in \(directory.path)")
    }
    
    // MARK: - JSON/Dict Loading
    
    private static func loadId2Label(url: URL) throws -> [String: String] {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data, options: [])
        
        guard let dict = json as? [String: Any],
              let map = dict["id2label"] as? [String: String] else {
            return [:]
        }
        
        return map
    }
    
    private static func loadGzipJsonDict(url: URL) throws -> [String: String] {
        let data = try Data(contentsOf: url)
        let jsonData = try data.gunzipped()
        let obj = try JSONSerialization.jsonObject(with: jsonData, options: [])
        return obj as? [String: String] ?? [:]
    }
    
    private static func loadGzipJsonDictArray(url: URL) throws -> [String: [String]] {
        let data = try Data(contentsOf: url)
        let jsonData = try data.gunzipped()
        let obj = try JSONSerialization.jsonObject(with: jsonData, options: [])
        return obj as? [String: [String]] ?? [:]
    }
    
    // MARK: - Text Splitting
    
    /// Splits text into sentences, preserving inter-sentence separators.
    /// - Returns: Tuple of (sentences, separators) where separators[i] is the whitespace
    ///            between sentences[i-1] and sentences[i]. separators[0] is empty string.
    private static func splitBySentences(_ text: String) -> ([String], [String]) {
        let pattern = "(?<=[.!?])(\\s+)"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return ([text], [""])
        }
        
        let nsRange = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, options: [], range: nsRange)
        
        if matches.isEmpty {
            return text.isEmpty ? ([], []) : ([text], [""])
        }
        
        var sentences: [String] = []
        var separators: [String] = [""]
        var lastEnd = text.startIndex
        
        for match in matches {
            // match.range(at: 1) is the captured separator (whitespace)
            guard let separatorRange = Range(match.range(at: 1), in: text) else { continue }
            
            // Sentence ends at the start of the separator
            let sentenceEnd = separatorRange.lowerBound
            if lastEnd < sentenceEnd {
                sentences.append(String(text[lastEnd..<sentenceEnd]))
            }
            
            separators.append(String(text[separatorRange]))
            lastEnd = separatorRange.upperBound
        }
        
        // Add final sentence
        if lastEnd < text.endIndex {
            sentences.append(String(text[lastEnd...]))
        }
        
        return (sentences, separators)
    }
    
    private static func splitByWords(_ text: String) -> ([String], [String]) {
        let pattern = "\\w*(?:\\+\\w+)*|[^\\w\\s]+"
        let matches = text.matches(regex: pattern)
        
        if matches.isEmpty {
            return ([], ["", ""])
        }
        
        var allWords: [String] = []
        var allRemaining: [String] = []
        var lastIndex = text.startIndex
        
        for match in matches {
            if let range = Range(match.range, in: text) {
                allRemaining.append(String(text[lastIndex..<range.lowerBound]))
                allWords.append(String(text[range]))
                lastIndex = range.upperBound
            }
        }
        allRemaining.append(String(text[lastIndex..<text.endIndex]))
        
        let wordsMask = allWords.enumerated().compactMap { $1.isEmpty ? nil : $0 }
        if wordsMask.isEmpty {
            return ([], ["", ""])
        }
        
        var validWords: [String] = []
        var validRemaining: [String] = []
        
        var startRemaining = ""
        for i in 0...wordsMask[0] {
            startRemaining += allRemaining[i]
        }
        validRemaining.append(startRemaining)
        validWords.append(allWords[wordsMask[0]])
        
        for i in 1..<wordsMask.count {
            let prevIdx = wordsMask[i - 1]
            let currIdx = wordsMask[i]
            var middleRemaining = ""
            for j in (prevIdx + 1)...currIdx {
                middleRemaining += allRemaining[j]
            }
            validRemaining.append(middleRemaining)
            validWords.append(allWords[currIdx])
        }
        
        let lastIdx = wordsMask.last!
        var endRemaining = ""
        for i in (lastIdx + 1)..<allRemaining.count {
            endRemaining += allRemaining[i]
        }
        validRemaining.append(endRemaining)
        
        return (validWords, validRemaining)
    }
}

// MARK: - String Extensions

private extension String {
    func matches(regex: String) -> [NSTextCheckingResult] {
        guard let re = try? NSRegularExpression(pattern: regex, options: []) else { return [] }
        let range = NSRange(startIndex..<endIndex, in: self)
        return re.matches(in: self, options: [], range: range)
    }
    
    func split(regex: String) -> [String] {
        guard let re = try? NSRegularExpression(pattern: regex, options: []) else { return [self] }
        let range = NSRange(startIndex..<endIndex, in: self)
        let matches = re.matches(in: self, options: [], range: range)
        
        var result: [String] = []
        var lastIndex = startIndex
        
        for match in matches {
            if let matchRange = Range(match.range, in: self) {
                result.append(String(self[lastIndex..<matchRange.lowerBound]))
                lastIndex = matchRange.upperBound
            }
        }
        result.append(String(self[lastIndex..<endIndex]))
        
        return result
    }
}

// MARK: - Data Extension for Gzip

private extension Data {
    func gunzipped() throws -> Data {
        let bytes = [UInt8](self)
        
        guard bytes.count > 18, bytes[0] == 0x1f, bytes[1] == 0x8b else {
            return self
        }
        
        let flags = bytes[3]
        var index = 10
        
        if flags & 0x04 != 0 {
            let xlen = Int(bytes[index]) | (Int(bytes[index + 1]) << 8)
            index += 2 + xlen
        }
        if flags & 0x08 != 0 {
            while index < bytes.count && bytes[index] != 0 { index += 1 }
            index += 1
        }
        if flags & 0x10 != 0 {
            while index < bytes.count && bytes[index] != 0 { index += 1 }
            index += 1
        }
        if flags & 0x02 != 0 { index += 2 }
        
        if index >= bytes.count { return self }
        
        let deflated = Data(bytes[index..<(bytes.count - 8)])
        return try deflated.decompressZlib()
    }
    
    private func decompressZlib() throws -> Data {
        var stream = compression_stream(
            dst_ptr: UnsafeMutablePointer<UInt8>(bitPattern: 1)!,
            dst_size: 0,
            src_ptr: UnsafePointer<UInt8>(bitPattern: 1)!,
            src_size: 0,
            state: nil
        )
        
        var status = compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB)
        guard status != COMPRESSION_STATUS_ERROR else {
            throw AudioToolError.resourceUnavailable("zlib init failed")
        }
        defer { compression_stream_destroy(&stream) }
        
        let bufferSize = 64 * 1024
        var output = Data()
        
        return try withUnsafeBytes { (srcBuffer: UnsafeRawBufferPointer) in
            guard let srcPtr = srcBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return Data()
            }
            
            var dst = [UInt8](repeating: 0, count: bufferSize)
            stream.src_ptr = srcPtr
            stream.src_size = count
            
            repeat {
                dst.withUnsafeMutableBufferPointer { dstBuffer in
                    stream.dst_ptr = dstBuffer.baseAddress!
                    stream.dst_size = dstBuffer.count
                    status = compression_stream_process(&stream, 0)
                }
                
                let produced = bufferSize - stream.dst_size
                if produced > 0 {
                    output.append(contentsOf: dst[0..<produced])
                }
            } while status == COMPRESSION_STATUS_OK
            
            if status != COMPRESSION_STATUS_END {
                throw AudioToolError.resourceUnavailable("zlib decode failed")
            }
            
            return output
        }
    }
}
