//
//  KokoroTest.swift
//  AudioTool
//
//  Test for Kokoro TTS integration with multilingual support
//

import Foundation
import AudioTool
import AudioToolCore
@preconcurrency import AudioToolTTS
import AudioToolFluidAudio
@preconcurrency import AudioUtils
@preconcurrency import MLX

// MARK: - Kokoro TTS Test

// Compute project root from source file path
private let kokoroProjectRoot: String = {
    var url = URL(fileURLWithPath: #filePath)
    for _ in 0..<4 { url.deleteLastPathComponent() }
    return url.path
}()

func runKokoroTest() async throws {
    print("\n=== Kokoro TTS Multilingual Test ===")
    print("Using precision: .bf16 → mlx-community/Kokoro-82M-bf16")
    
    // Output directory - save to Models/kokoro-ios for consistency
    let outputDir = "\(kokoroProjectRoot)/Models/kokoro-ios"
    try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    
    // Get model path first (download once)
    print("\nDownloading/locating model...")
    let downloader = ModelDownloader.shared
    let modelPath: URL
    if let cached = downloader.localPath(for: "mlx-community/Kokoro-82M-bf16") {
        modelPath = cached
        print("Using cached model at: \(modelPath.path)")
    } else {
        modelPath = try await downloader.downloadAndGetPath(
            repo: "mlx-community/Kokoro-82M-bf16",
            matching: ["*.safetensors", "voices/*.npy", "config.json"]
        ) { progress in
            print("Downloading: \(Int(progress.fractionCompleted * 100))%")
        }
        print("Downloaded to: \(modelPath.path)")
    }
    
    let voicesDir = modelPath.appendingPathComponent("voices")
    
    // Define all tests - (language, voice, text)
    let allTests: [(KokoroLanguage, KokoroVoice, String)] = [
        // English (US/GB)
        (.americanEnglish, .af_heart, "Hello, this is the best quality American English voice."),
        (.americanEnglish, .af_bella, "This is another high quality female voice from the USA."),
        (.britishEnglish, .bf_emma, "Good day, I'm Emma speaking British English."),
        (.britishEnglish, .bm_george, "Hello, I'm George with a British accent."),
        
        // Italian
        (.italian, .if_sara, "Ciao mondo! Questa è una prova del sistema di sintesi vocale Kokoro in italiano."),
        (.italian, .im_nicola, "Buongiorno, sono Nicola e parlo italiano."),
        
        // Spanish
        (.spanish, .ef_dora, "¡Hola mundo! Esta es una prueba del sistema de síntesis de voz Kokoro en español."),
        (.spanish, .em_alex, "Buenos días, soy Alex y hablo español."),
        
        // French
        (.french, .ff_siwis, "Bonjour le monde! Ceci est un test du système de synthèse vocale Kokoro en français."),
        
        // Portuguese
        (.portuguese, .pf_dora, "Olá mundo! Este é um teste do sistema de síntese de voz Kokoro em português brasileiro."),
    ]
    
    // Group tests by language
    var testsByLanguage: [KokoroLanguage: [(KokoroVoice, String)]] = [:]
    for (lang, voice, text) in allTests {
        testsByLanguage[lang, default: []].append((voice, text))
    }
    
    // Process each language
    for (language, tests) in testsByLanguage {
        print("\n--- Testing \(language.displayName) ---")
        
        // Create provider for this language (using local path to avoid re-download)
        let provider = KokoroTTSProvider(modelPath: modelPath, language: language)
        try await provider.load()
        try await provider.loadVoices(from: voicesDir)
        
        // Run tests for this language
        for (voice, text) in tests {
            guard provider.availableVoices.contains(voice.rawValue) else {
                print("⚠️ Voice '\(voice.rawValue)' not available, skipping")
                continue
            }
            
            print("\n[\(language.displayName)] \(voice.rawValue) Synthesizing...")
            let synthStart = Date()
            let audio = try await provider.synthesize(text, voice: voice.rawValue)
            let synthTime = Date().timeIntervalSince(synthStart)
            
            let rtf = audio.duration / synthTime
            print("[\(voice.rawValue)] Duration: \(String(format: "%.2f", audio.duration))s, Synth: \(String(format: "%.2f", synthTime))s, RTF: \(String(format: "%.2fx", rtf))")
            
            // Save with language code prefix
            let outputPath = "\(outputDir)/kokoro_\(language.rawValue)_\(voice.rawValue).wav"
            let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
            try saver.save(MLXArray(audio.samples), to: outputPath)
            print("[\(voice.rawValue)] ✓ Saved: \(outputPath)")
        }
        
        // Clear GPU cache between languages
        GPU.clearCache()
    }
    
    // Summary
    print("\n=== Kokoro Language Support Summary ===")
    for lang in KokoroLanguage.allCases {
        let voices = lang.availableVoices
        let status = lang.isFullySupported ? "✓" : "⚠️ (needs misaki[\(lang.rawValue)])"
        print("\(lang.displayName) (\(lang.rawValue)): \(voices.count) voices \(status)")
    }
    
    print("\n=== Output files saved to: \(outputDir) ===")
    
    // List output files
    if let files = try? FileManager.default.contentsOfDirectory(atPath: outputDir) {
        print("\nGenerated \(files.count) audio files:")
        for file in files.sorted() where file.hasSuffix(".wav") {
            print("  - \(file)")
        }
    }
}

// MARK: - Voice Mixing Test

func runVoiceMixingTest() async throws {
    print("\n=== Kokoro Voice Mixing Test ===")
    print("Testing: af_bella (50%) + af_sarah (50%)")
    
    // Output directory - save to Models/kokoro-ios for consistency
    let outputDir = "\(kokoroProjectRoot)/Models/kokoro-ios"
    try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    
    // Get model path
    print("\nDownloading/locating model...")
    let downloader = ModelDownloader.shared
    let modelPath: URL
    if let cached = downloader.localPath(for: "mlx-community/Kokoro-82M-bf16") {
        modelPath = cached
        print("Using cached model at: \(modelPath.path)")
    } else {
        modelPath = try await downloader.downloadAndGetPath(
            repo: "mlx-community/Kokoro-82M-bf16",
            matching: ["*.safetensors", "voices/*.npy", "config.json"]
        ) { progress in
            print("Downloading: \(Int(progress.fractionCompleted * 100))%")
        }
        print("Downloaded to: \(modelPath.path)")
    }
    
    let voicesDir = modelPath.appendingPathComponent("voices")
    
    // Create provider
    let provider = KokoroTTSProvider(modelPath: modelPath, language: .americanEnglish)
    try await provider.load()
    try await provider.loadVoices(from: voicesDir)
    
    print("\nLoaded voices: \(provider.availableVoices.sorted().joined(separator: ", "))")
    
    // Test text - something expressive to hear the blend
    let testText = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and showcases the unique characteristics of my blended voice."
    
    // First, generate individual voices for comparison
    print("\n--- Individual Voices ---")
    
    // af_bella solo
    print("\n[af_bella] Synthesizing...")
    var start = Date()
    var audio = try await provider.synthesize(testText, voice: "af_bella")
    var synthTime = Date().timeIntervalSince(start)
    print("[af_bella] Duration: \(String(format: "%.2f", audio.duration))s, RTF: \(String(format: "%.2fx", audio.duration / synthTime))")
    
    var saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
    try saver.save(MLXArray(audio.samples), to: "\(outputDir)/kokoro_bella_solo.wav")
    print("[af_bella] ✓ Saved: \(outputDir)/kokoro_bella_solo.wav")
    
    // af_sarah solo
    print("\n[af_sarah] Synthesizing...")
    start = Date()
    audio = try await provider.synthesize(testText, voice: "af_sarah")
    synthTime = Date().timeIntervalSince(start)
    print("[af_sarah] Duration: \(String(format: "%.2f", audio.duration))s, RTF: \(String(format: "%.2fx", audio.duration / synthTime))")
    
    saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
    try saver.save(MLXArray(audio.samples), to: "\(outputDir)/kokoro_sarah_solo.wav")
    print("[af_sarah] ✓ Saved: \(outputDir)/kokoro_sarah_solo.wav")
    
    // Now test mixed voice
    print("\n--- Mixed Voice (50% Bella + 50% Sarah) ---")
    
    let mixedVoice = try await provider.mixVoices([
        (name: "af_bella", weight: 0.5),
        (name: "af_sarah", weight: 0.5)
    ])
    
    print("\n[bella+sarah] Synthesizing with blended voice...")
    start = Date()
    audio = try await provider.synthesize(testText, voiceEmbedding: mixedVoice)
    synthTime = Date().timeIntervalSince(start)
    print("[bella+sarah] Duration: \(String(format: "%.2f", audio.duration))s, RTF: \(String(format: "%.2fx", audio.duration / synthTime))")
    
    saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
    try saver.save(MLXArray(audio.samples), to: "\(outputDir)/kokoro_bella_sarah_mix.wav")
    print("[bella+sarah] ✓ Saved: \(outputDir)/kokoro_bella_sarah_mix.wav")
    
    // Bonus: Try other mixing ratios
    print("\n--- Bonus Mixes ---")
    
    // 70/30 Bella/Sarah
    let mix70_30 = try await provider.mixVoices([
        (name: "af_bella", weight: 0.7),
        (name: "af_sarah", weight: 0.3)
    ])
    audio = try await provider.synthesize(testText, voiceEmbedding: mix70_30)
    try saver.save(MLXArray(audio.samples), to: "\(outputDir)/kokoro_bella70_sarah30.wav")
    print("[70/30] ✓ Saved: kokoro_bella70_sarah30.wav")
    
    // 30/70 Bella/Sarah
    let mix30_70 = try await provider.mixVoices([
        (name: "af_bella", weight: 0.3),
        (name: "af_sarah", weight: 0.7)
    ])
    audio = try await provider.synthesize(testText, voiceEmbedding: mix30_70)
    try saver.save(MLXArray(audio.samples), to: "\(outputDir)/kokoro_bella30_sarah70.wav")
    print("[30/70] ✓ Saved: kokoro_bella30_sarah70.wav")
    
    print("\n=== Voice Mixing Test Complete ===")
    print("Output files saved to: \(outputDir)")
    print("\nGenerated files:")
    print("  - kokoro_bella_solo.wav (100% Bella)")
    print("  - kokoro_sarah_solo.wav (100% Sarah)")
    print("  - kokoro_bella_sarah_mix.wav (50/50)")
    print("  - kokoro_bella70_sarah30.wav (70/30)")
    print("  - kokoro_bella30_sarah70.wav (30/70)")
    
    GPU.clearCache()
}

// MARK: - Voice Matching Test

func runVoiceMatchingTest() async throws {
    print("\n=== Kokoro Voice Matching Test ===")
    print("Finding optimal voice blends for reference audio files\n")
    
    // Reference audio files to test
    let referenceFiles = [
        "\(kokoroProjectRoot)/AudioTool/Docs/burunow_short.wav",
        "\(kokoroProjectRoot)/AudioTool/Docs/reference.wav",
        "\(kokoroProjectRoot)/AudioTool/Docs/watson_short.wav",
        "\(kokoroProjectRoot)/AudioTool/voice_match_output/dark_knight_short.wav"
    ]
    
    // Verify files exist
    for file in referenceFiles {
        guard FileManager.default.fileExists(atPath: file) else {
            print("❌ File not found: \(file)")
            return
        }
    }
    print("✓ All reference files found\n")
    
    // Output directory
    let outputDir = FileManager.default.currentDirectoryPath + "/voice_match_output"
    try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    
    // Get model path - use standard HF cache (~/.cache/huggingface/hub/)
    print("Loading Kokoro TTS model...")
    let modelPath: URL
    
    if let cachedPath = ModelDownloader.shared.localPath(for: "mlx-community/Kokoro-82M-bf16") {
        modelPath = cachedPath
        print("✓ Using cached model at: \(modelPath.path)")
    } else {
        // Download to standard HF cache (avoids ~/Documents permission issues)
        print("Downloading Kokoro model to ~/.cache/huggingface/hub/...")
        let downloader = ModelDownloader.shared
        modelPath = try await downloader.downloadAndGetPath(
            repo: "mlx-community/Kokoro-82M-bf16",
            matching: ["**/*.safetensors", "config.json"]
        ) { progress in
            print("Downloading: \(Int(progress.fractionCompleted * 100))%")
        }
        print("✓ Downloaded model to: \(modelPath.path)")
    }
    
    let voicesDir = modelPath.appendingPathComponent("voices")
    
    // Create TTS provider
    let tts = KokoroTTSProvider(modelPath: modelPath, language: .americanEnglish)
    try await tts.load()
    try await tts.loadVoices(from: voicesDir)
    print("✓ Loaded \(tts.availableVoices.count) voices: \(tts.availableVoices.sorted().joined(separator: ", "))\n")
    
    // Create speaker embedding provider
    print("Loading speaker embedding model (FluidAudio WeSpeaker)...")
    let embeddingProvider = AudioToolFluidAudio.SpeakerEmbeddingProvider()
    try await embeddingProvider.load()
    print("✓ Embedding model ready\n")
    
    // Create voice matcher
    let matcher = KokoroVoiceMatcher(topK: 5)
    
    // Step 1: Precompute embeddings for all Kokoro voices
    print("=== Step 1: Precomputing Voice Embeddings ===")
    print("This synthesizes calibration text with each voice and extracts speaker embeddings.")
    print("Computing embeddings for \(tts.availableVoices.count) voices...\n")
    
    let precomputeStart = Date()
    let embeddingTable = try await matcher.precomputeEmbeddings(
        tts: tts,
        extractEmbedding: { audio in
            // Resample from 24kHz (Kokoro output) to 16kHz (embedding input)
            let audio16k = resampleTo16kHz(audio, from: 24000)
            return try await embeddingProvider.extractEmbedding(audio16k)
        }
    )
    let precomputeTime = Date().timeIntervalSince(precomputeStart)
    print("✓ Precomputed \(embeddingTable.count) voice embeddings in \(String(format: "%.1f", precomputeTime))s")
    
    // Save embedding table for future use
    let tablePath = URL(fileURLWithPath: outputDir).appendingPathComponent("voice_embeddings.json")
    try embeddingTable.save(to: tablePath)
    print("✓ Saved embedding table to: \(tablePath.path)\n")
    
    // Step 2: Match each reference file
    print("=== Step 2: Matching Reference Audio ===\n")
    
    let testText = "Hello! This is a test of voice matching. The system found the best blend of Kokoro voices to approximate my speaking style."
    
    for refPath in referenceFiles {
        let filename = URL(fileURLWithPath: refPath).deletingPathExtension().lastPathComponent
        print("--- Processing: \(filename) ---")
        
        // Extract embedding from reference audio
        let matchStart = Date()
        let refEmbedding = try await embeddingProvider.extractEmbedding(from: URL(fileURLWithPath: refPath))
        
        // Match voice
        let result = try await matcher.matchVoice(
            referenceAudio: [], // Unused, we use the embedding directly
            embeddingTable: embeddingTable,
            extractEmbedding: { _ in refEmbedding } // Return pre-extracted embedding
        )
        let matchTime = Date().timeIntervalSince(matchStart)
        
        // Print results
        print("Match completed in \(String(format: "%.0f", matchTime * 1000))ms")
        print("Similarity: \(String(format: "%.1f%%", result.similarity * 100))")
        print("Voice blend:")
        for (voice, weight) in result.weights {
            let bar = String(repeating: "█", count: Int(weight * 20))
            print("  \(voice.padding(toLength: 12, withPad: " ", startingAt: 0)) \(String(format: "%5.1f%%", weight * 100)) \(bar)")
        }
        
        // Generate audio with matched voice
        print("\nSynthesizing with matched blend...")
        let synthStart = Date()
        let blendedVoice = try await tts.blendedVoice(from: result)
        let audio = try await tts.synthesize(testText, voiceEmbedding: blendedVoice)
        let synthTime = Date().timeIntervalSince(synthStart)
        print("Synthesis: \(String(format: "%.2f", audio.duration))s audio in \(String(format: "%.2f", synthTime))s (RTF: \(String(format: "%.1fx", audio.duration / synthTime)))")
        
        // Save output
        let outputPath = "\(outputDir)/matched_\(filename).wav"
        let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: outputPath)
        print("✓ Saved: \(outputPath)\n")
    }
    // Step 3: Gender-aware matching (demo for watson_short)
    print("=== Step 3: Gender-Aware Matching (watson_short) ===\n")
    
    let watsonPath = "\(kokoroProjectRoot)/AudioTool/Docs/watson_short.wav"
    let watsonEmbedding = try await embeddingProvider.extractEmbedding(from: URL(fileURLWithPath: watsonPath))
    
    // Filter to female voices only
    let femaleTable = embeddingTable.filtered(by: .female)
    print("Filtered to \(femaleTable.count) female voices (from \(embeddingTable.count) total)\n")
    
    // Match against female voices only
    let matchStart = Date()
    let femaleResult = try await matcher.matchVoice(
        referenceAudio: [],
        embeddingTable: femaleTable,
        extractEmbedding: { _ in watsonEmbedding }
    )
    let matchTime = Date().timeIntervalSince(matchStart)
    
    print("Gender-filtered match completed in \(String(format: "%.0f", matchTime * 1000))ms")
    print("Similarity: \(String(format: "%.1f%%", femaleResult.similarity * 100))")
    print("Voice blend (female only):")
    for (voice, weight) in femaleResult.weights {
        let bar = String(repeating: "█", count: Int(weight * 20))
        print("  \(voice.padding(toLength: 12, withPad: " ", startingAt: 0)) \(String(format: "%5.1f%%", weight * 100)) \(bar)")
    }
    
    // Generate audio with female-filtered blend
    print("\nSynthesizing with female voice blend...")
    let synthStart = Date()
    let femaleBlend = try await tts.blendedVoice(from: femaleResult)
    let femaleAudio = try await tts.synthesize(testText, voiceEmbedding: femaleBlend)
    let synthTime = Date().timeIntervalSince(synthStart)
    print("Synthesis: \(String(format: "%.2f", femaleAudio.duration))s audio in \(String(format: "%.2f", synthTime))s (RTF: \(String(format: "%.1fx", femaleAudio.duration / synthTime)))")
    
    let femalePath = "\(outputDir)/matched_watson_female.wav"
    let saver2 = AudioSaver(config: .init(sampleRate: Double(femaleAudio.sampleRate)))
    try saver2.save(MLXArray(femaleAudio.samples), to: femalePath)
    print("✓ Saved: \(femalePath)\n")
    
    // Step 4: Language + Gender filtered matching tests
    print("=== Step 4: Language + Gender Filtered Matching ===\n")
    
    let testCases: [(name: String, path: String, gender: VoiceGender, language: VoiceLanguage)] = [
        // With gender filter
        ("watson_british", "\(kokoroProjectRoot)/AudioTool/Docs/watson_short.wav", .female, .british),
        ("watson_english", "\(kokoroProjectRoot)/AudioTool/Docs/watson_short.wav", .female, .english),
        ("dark_knight_british", "\(kokoroProjectRoot)/AudioTool/voice_match_output/dark_knight_short.wav", .male, .british),
        ("dark_knight_english", "\(kokoroProjectRoot)/AudioTool/voice_match_output/dark_knight_short.wav", .male, .english),
        // WITHOUT gender filter - does embedding naturally pick correct gender?
        ("watson_any_british", "\(kokoroProjectRoot)/AudioTool/Docs/watson_short.wav", .any, .british),
        ("watson_any_english", "\(kokoroProjectRoot)/AudioTool/Docs/watson_short.wav", .any, .english),
        ("dark_knight_any_british", "\(kokoroProjectRoot)/AudioTool/voice_match_output/dark_knight_short.wav", .any, .british),
        ("dark_knight_any_english", "\(kokoroProjectRoot)/AudioTool/voice_match_output/dark_knight_short.wav", .any, .english),
        // COMPLETELY UNFILTERED - all 54 voices, any language
        ("watson_unfiltered", "\(kokoroProjectRoot)/AudioTool/Docs/watson_short.wav", .any, .any),
        ("dark_knight_unfiltered", "\(kokoroProjectRoot)/AudioTool/voice_match_output/dark_knight_short.wav", .any, .any),
    ]
    
    for testCase in testCases {
        print("--- \(testCase.name) (\(testCase.gender), \(testCase.language)) ---")
        
        // Extract embedding
        let embedding = try await embeddingProvider.extractEmbedding(from: URL(fileURLWithPath: testCase.path))
        
        // Filter by gender and language
        let filteredTable = embeddingTable.filtered(by: testCase.gender, language: testCase.language)
        print("Filtered to \(filteredTable.count) voices (\(testCase.gender), \(testCase.language))")
        
        // Match
        let result = try await matcher.matchVoice(
            referenceAudio: [],
            embeddingTable: filteredTable,
            extractEmbedding: { _ in embedding }
        )
        
        print("Similarity: \(String(format: "%.1f%%", result.similarity * 100))")
        print("Voice blend:")
        for (voice, weight) in result.weights {
            let bar = String(repeating: "█", count: Int(weight * 20))
            print("  \(voice.padding(toLength: 12, withPad: " ", startingAt: 0)) \(String(format: "%5.1f%%", weight * 100)) \(bar)")
        }
        
        // Generate and save audio
        print("\nSynthesizing...")
        let blend = try await tts.blendedVoice(from: result)
        let audio = try await tts.synthesize(testText, voiceEmbedding: blend)
        
        let savePath = "\(outputDir)/matched_\(testCase.name).wav"
        let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: savePath)
        print("✓ Saved: \(savePath)\n")
    }
    
    // Step 5: Generate pure single-voice samples for comparison
    print("=== Step 5: Pure Voice Samples (Chinese voices for comparison) ===\n")
    
    // Use English TTS with the Chinese voice embeddings directly
    let pureVoices = ["zf_xiaoxiao", "zf_xiaoni", "zf_xiaoyi"]
    let pureTestText = "The quick brown fox jumps over the lazy dog. This voice has a unique quality."
    
    for voiceId in pureVoices {
        guard tts.availableVoices.contains(voiceId) else {
            print("Voice \(voiceId) not available in English TTS, skipping")
            continue
        }
        
        print("Generating pure \(voiceId)...")
        // Synthesize directly with the voice (TTS loaded all 54 voices)
        let audio = try await tts.synthesize(pureTestText, voice: voiceId)
        
        let savePath = "\(outputDir)/pure_\(voiceId).wav"
        let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: savePath)
        print("✓ Saved: \(savePath)\n")
    }
    
    // Summary
    print("=== Voice Matching Complete ===")
    print("Output directory: \(outputDir)")
    if let files = try? FileManager.default.contentsOfDirectory(atPath: outputDir) {
        print("\nGenerated files:")
        for file in files.sorted() where file.hasSuffix(".wav") || file.hasSuffix(".json") {
            print("  - \(file)")
        }
    }
    
    GPU.clearCache()
}

// Helper: Resample audio from source rate to 16kHz
private func resampleTo16kHz(_ samples: [Float], from sourceRate: Int) -> [Float] {
    guard sourceRate != 16000 else { return samples }
    
    let ratio = Float(16000) / Float(sourceRate)
    let outputLength = Int(Float(samples.count) * ratio)
    guard outputLength > 0 else { return [] }
    
    var output = [Float](repeating: 0, count: outputLength)
    
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

