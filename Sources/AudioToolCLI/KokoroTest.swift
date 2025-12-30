//
//  KokoroTest.swift
//  ClearVoice
//
//  Test for Kokoro TTS integration with multilingual support
//

import Foundation
import ClearVoice
import ClearVoiceCore
import ClearVoiceTTS
import AudioUtils
import MLX

// MARK: - Kokoro TTS Test

func runKokoroTest() async throws {
    print("\n=== Kokoro TTS Multilingual Test ===")
    print("Using precision: .bf16 → mlx-community/Kokoro-82M-bf16")
    
    // Output directory
    let outputDir = FileManager.default.currentDirectoryPath + "/tts_output"
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
        try provider.loadVoices(from: voicesDir)
        
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
    
    // Output directory
    let outputDir = FileManager.default.currentDirectoryPath + "/tts_output"
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
    try provider.loadVoices(from: voicesDir)
    
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
    
    let mixedVoice = try provider.mixVoices([
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
    let mix70_30 = try provider.mixVoices([
        (name: "af_bella", weight: 0.7),
        (name: "af_sarah", weight: 0.3)
    ])
    audio = try await provider.synthesize(testText, voiceEmbedding: mix70_30)
    try saver.save(MLXArray(audio.samples), to: "\(outputDir)/kokoro_bella70_sarah30.wav")
    print("[70/30] ✓ Saved: kokoro_bella70_sarah30.wav")
    
    // 30/70 Bella/Sarah
    let mix30_70 = try provider.mixVoices([
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
