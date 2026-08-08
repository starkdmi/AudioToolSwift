//
//  TranslateGemmaProvider.swift
//  AudioToolMLXTranslation
//
//  TranslateGemma translation provider using MLX (55+ languages)
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import AudioToolCore

/// TranslateGemma translation provider
///
/// Uses Google's TranslateGemma model (Gemma 3 finetuned for translation) via MLX.
/// Supports 55+ languages with high quality translation.
///
/// Usage:
/// ```swift
/// let translator = TranslationProviders.translateGemma()
/// let result = try await translator.translate("Hello", from: "en", to: "de-DE")
/// print(result.translatedText) // "Hallo"
/// ```
///
/// > Note: Requires macOS 14+ for MLX Metal support.
/// > Model (~4GB) is downloaded automatically on first use.
public actor TranslateGemmaProvider: TextTranslator, ManagedModel {
    
    // MARK: - Properties
    
    /// Model ID on HuggingFace
    private nonisolated let repositoryId: String
    
    /// Maximum tokens to generate
    private let maxTokens: Int
    
    /// Sampling temperature (0 = greedy/deterministic)
    private let temperature: Float
    
    /// Cached model container
    private var modelContainer: ModelContainer?
    
    /// Progress handler for model loading
    private let progressHandler: (@Sendable (Progress) -> Void)?
    
    // MARK: - Language Mapping
    
    /// Language code to language name mapping (from TranslateGemma chat template)
    private static let languageNames: [String: String] = [
        "ar": "Arabic",
        "bg": "Bulgarian",
        "bn": "Bengali",
        "bs": "Bosnian",
        "ca": "Catalan",
        "cs": "Czech",
        "cy": "Welsh",
        "da": "Danish",
        "de-DE": "German",
        "el": "Greek",
        "en": "English",
        "es-419": "Spanish",
        "et": "Estonian",
        "eu": "Basque",
        "fa": "Persian",
        "fi": "Finnish",
        "fr-FR": "French",
        "gl": "Galician",
        "gu": "Gujarati",
        "he": "Hebrew",
        "hi": "Hindi",
        "hr": "Croatian",
        "hu": "Hungarian",
        "hy": "Armenian",
        "id": "Indonesian",
        "is": "Icelandic",
        "it-IT": "Italian",
        "ja": "Japanese",
        "ka": "Georgian",
        "kk": "Kazakh",
        "km": "Khmer",
        "kn": "Kannada",
        "ko": "Korean",
        "lo": "Lao",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mn": "Mongolian",
        "mr": "Marathi",
        "ms": "Malay",
        "mt": "Maltese",
        "my": "Burmese",
        "ne": "Nepali",
        "nl-NL": "Dutch",
        "no": "Norwegian",
        "pl": "Polish",
        "pt-BR": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "si": "Sinhala",
        "sk": "Slovak",
        "sl": "Slovenian",
        "sq": "Albanian",
        "sr": "Serbian",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "th": "Thai",
        "tl": "Tagalog",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "zh-CN": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)",
    ]
    
    /// All supported language codes
    public static var supportedLanguages: [String] {
        Array(languageNames.keys).sorted()
    }
    
    // MARK: - Initialization
    
    /// Create TranslateGemma provider
    /// - Parameters:
    ///   - modelId: HuggingFace model ID (default: mlx-community/translategemma-4b-it-4bit)
    ///   - maxTokens: Maximum tokens to generate (default: 256)
    ///   - temperature: Sampling temperature, 0 = greedy (default: 0.0)
    ///   - progressHandler: Optional callback for model download progress
    public init(
        modelId: String = "mlx-community/translategemma-4b-it-4bit",
        maxTokens: Int = 256,
        temperature: Float = 0.0,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) {
        self.repositoryId = modelId
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.progressHandler = progressHandler
    }
    
    // MARK: - TextTranslator Conformance
    
    public func translate(
        _ text: String,
        from source: String?,
        to target: String
    ) async throws -> TranslationResult {
        let container = try await getOrLoadModel()
        
        // Build translation prompt
        let sourceLang = source ?? "en"
        let prompt = Self.buildTranslationPrompt(text: text, sourceLang: sourceLang, targetLang: target)
        
        // Tokenize
        let tokens = await container.encode(prompt)
        let input = LMInput(tokens: MLXArray(tokens))
        
        // Generate
        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature
        )
        
        var output = ""
        let stream = try await container.generate(input: input, parameters: params)
        
        for await generation in stream {
            try Task.checkCancellation()
            switch generation {
            case .chunk(let text):
                output += text
            case .info, .toolCall:
                break
            }
        }
        
        // Clean up output
        let translatedText = output.trimmingCharacters(in: .whitespacesAndNewlines)
        
        return TranslationResult(
            sourceText: text,
            translatedText: translatedText,
            sourceLanguage: sourceLang,
            targetLanguage: target
        )
    }
    
    public func translateBatch(
        _ texts: [String],
        from source: String?,
        to target: String
    ) async throws -> BatchTranslationResult {
        // Translate each text sequentially (model handles one at a time)
        var translations: [TranslationResult] = []
        translations.reserveCapacity(texts.count)
        
        for text in texts {
            let result = try await translate(text, from: source, to: target)
            translations.append(result)
        }
        
        return BatchTranslationResult(
            translations: translations,
            sourceLanguage: source,
            targetLanguage: target
        )
    }
    
    public func isAvailable(from source: String, to target: String) async -> Bool {
        // Check if both languages are in our supported list
        let sourceSupported = Self.languageNames[source] != nil
        let targetSupported = Self.languageNames[target] != nil
        return sourceSupported && targetSupported
    }
    
    public func prepareLanguagePair(from source: String, to target: String) async throws {
        // Pre-load the model
        _ = try await getOrLoadModel()
    }
    
    // MARK: - Private Methods
    
    private func getOrLoadModel() async throws -> ModelContainer {
        if let container = modelContainer {
            return container
        }
        
        // Configure with extra EOS tokens for proper stopping
        let configuration = ModelConfiguration(
            id: repositoryId,
            extraEOSTokens: ["<end_of_turn>"]
        )
        
        // Load model
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: configuration,
            progressHandler: progressHandler ?? { _ in }
        )
        
        self.modelContainer = container
        return container
    }

    // MARK: - ManagedModel

    public nonisolated var modelId: String {
        "translate_gemma_\(repositoryId.replacingOccurrences(of: "/", with: "_"))"
    }

    /// Includes model weights plus generation working memory.
    public nonisolated var estimatedMemoryBytes: Int { 4_000_000_000 }

    public func load() async throws {
        _ = try await getOrLoadModel()
    }

    public func unload() async {
        modelContainer = nil
        GPU.clearCache()
    }

    public func checkIfLoaded() async -> Bool {
        modelContainer != nil
    }
    
    /// Build a translation prompt matching TranslateGemma's expected format
    private static func buildTranslationPrompt(text: String, sourceLang: String, targetLang: String) -> String {
        let sourceLanguage = languageNames[sourceLang] ?? sourceLang
        let targetLanguage = languageNames[targetLang] ?? targetLang
        
        return "<bos><start_of_turn>user\nYou are a professional \(sourceLanguage) (\(sourceLang)) to \(targetLanguage) (\(targetLang)) translator. Your goal is to accurately convey the meaning and nuances of the original \(sourceLanguage) text while adhering to \(targetLanguage) grammar, vocabulary, and cultural sensitivities.\nProduce only the \(targetLanguage) translation, without any additional explanations or commentary. Please translate the following \(sourceLanguage) text into \(targetLanguage):\n\n\n\(text)<end_of_turn>\n<start_of_turn>model\n"
    }
}
