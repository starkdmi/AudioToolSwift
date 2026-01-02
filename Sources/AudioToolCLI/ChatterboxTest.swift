//
//  ChatterboxTest.swift
//  ClearVoice
//
//  Test for ChatterBox Multilingual TTS with voice cloning
//

import Foundation
import ClearVoice
import ClearVoiceCore
@preconcurrency import ClearVoiceTTS
@preconcurrency import AudioUtils
@preconcurrency import MLX

// MARK: - ChatterBox TTS Test

/// Run ChatterBox TTS tests with English and Russian
/// Uses watson_short.wav as reference for English
/// Uses burunow_short.wav as reference for Russian
func runChatterboxTest() async throws {
    print("\n=== ChatterBox Multilingual TTS Test ===")
    print("Testing English (gb/en) and Russian with voice cloning\n")
    
    // Output directory
    let outputDir = FileManager.default.currentDirectoryPath + "/chatterbox_output"
    try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    
    // Reference audio files
    let watsonPath = "/path/to/clear_voice_research/Docs/watson_short.wav"
    let burunowPath = "/path/to/clear_voice_research/Docs/burunow_short.wav"
    
    // Verify files exist
    guard FileManager.default.fileExists(atPath: watsonPath) else {
        print("❌ Reference file not found: watson_short.wav")
        return
    }
    guard FileManager.default.fileExists(atPath: burunowPath) else {
        print("❌ Reference file not found: burunow_short.wav")
        return
    }
    print("✓ Reference audio files found\n")
    
    // ========================================
    // DEFAULT VOICE TEST (no reference audio)
    // ========================================
    print("--- Default Voice Test (no reference audio) ---\n")
    
    // Create provider - will use conds.safetensors default voice
    let defaultVoiceProvider = TTSProviders.chatterbox(
        precision: .fp32,
        language: .english,
        useRuAccent: false
    )
    
    print("Loading ChatterBox model...")
    var loadStart = Date()
    try await defaultVoiceProvider.load()
    var loadTime = Date().timeIntervalSince(loadStart)
    print("✓ Model loaded in \(String(format: "%.1f", loadTime))s")
    print("Using default voice from conds.safetensors (no reference audio)\n")
    
    // Test texts with default voice
    let defaultVoiceTexts: [(name: String, text: String)] = [
        ("default_greeting", "Hello, this is the default voice speaking. No reference audio required."),
        ("default_science", "The universe is vast and full of mysteries waiting to be discovered.")
    ]
    
    for (name, text) in defaultVoiceTexts {
        print("Synthesizing: \"\(text.prefix(50))...\"")
        let synthStart = Date()
        let audio = try await defaultVoiceProvider.synthesize(text, voice: "")
        let synthTime = Date().timeIntervalSince(synthStart)
        
        let rtf = audio.duration / synthTime
        print("Duration: \(String(format: "%.2f", audio.duration))s, RTF: \(String(format: "%.2fx", rtf))")
        
        let outputPath = "\(outputDir)/chatterbox_\(name).wav"
        let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: outputPath)
        print("✓ Saved: \(outputPath)\n")
    }
    
    // Clear GPU cache before loading next model
    GPU.clearCache()
    
    // ========================================
    // ENGLISH TEST (with reference audio)
    // ========================================
    print("--- English Test (watson_short.wav reference) ---\n")
    
    // Create English provider
    let englishProvider = TTSProviders.chatterbox(
        precision: .fp32,
        language: .english,
        useRuAccent: false
    )
    
    print("Loading ChatterBox model...")
    loadStart = Date()
    try await englishProvider.load()
    loadTime = Date().timeIntervalSince(loadStart)
    print("✓ Model loaded in \(String(format: "%.1f", loadTime))s\n")
    
    // Set reference audio
    print("Setting reference audio: watson_short.wav")
    try await englishProvider.setReferenceAudio(from: URL(fileURLWithPath: watsonPath))
    print("✓ Reference audio processed\n")
    
    // Test texts
    let englishTexts: [(name: String, text: String)] = [
        ("watson_greeting", "Hello, my name is Watson. It's a pleasure to meet you."),
        ("watson_science", "Science is the poetry of reality, revealing the wonders of the universe."),
        ("watson_question", "What fascinating discoveries will tomorrow bring?")
    ]
    
    for (name, text) in englishTexts {
        print("Synthesizing: \"\(text.prefix(50))...\"")
        let synthStart = Date()
        let audio = try await englishProvider.synthesize(text, voice: "")
        let synthTime = Date().timeIntervalSince(synthStart)
        
        let rtf = audio.duration / synthTime
        print("Duration: \(String(format: "%.2f", audio.duration))s, RTF: \(String(format: "%.2fx", rtf))")
        
        let outputPath = "\(outputDir)/chatterbox_english_\(name).wav"
        let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: outputPath)
        print("✓ Saved: \(outputPath)\n")
    }
    
    // Clear GPU cache
    GPU.clearCache()
    
    // ========================================
    // RUSSIAN TEST
    // ========================================
    print("\n--- Russian Test (burunow_short.wav reference) ---")
    print("Testing with RUAccent stress marking\n")
    
    // RUAccent paths
    let ruaccentModelsDir = URL(fileURLWithPath: "/path/to/clear_voice_research/Models/ruaccent/models/balanced")
    let ruaccentAssetsDir = URL(fileURLWithPath: "/path/to/clear_voice_research/Models/ruaccent/assets/balanced")
    
    // Check RUAccent availability
    let ruaccentAvailable = FileManager.default.fileExists(atPath: ruaccentModelsDir.path) &&
                            FileManager.default.fileExists(atPath: ruaccentAssetsDir.path)
    
    // Create Russian provider
    let russianProvider = TTSProviders.chatterbox(
        precision: .fp32,
        language: .russian,
        useRuAccent: ruaccentAvailable,  // Use RUAccent if available
        convertToStressMarks: true
    )
    
    print("Loading ChatterBox model for Russian...")
    try await russianProvider.load()
    print("✓ Model loaded\n")
    
    // Configure RUAccent if available
    var ruAccent: RUAccentProvider? = nil
    if ruaccentAvailable {
        print("Loading RUAccent pipeline (balanced profile)...")
        do {
            ruAccent = try RUAccentProvider(
                profile: .balanced,
                modelsDir: ruaccentModelsDir,
                assetsDir: ruaccentAssetsDir
            )
            await russianProvider.configureRuAccent(ruAccent!)
            print("✓ RUAccent loaded\n")
        } catch {
            print("⚠️ RUAccent failed to load: \(error)")
            print("Continuing without stress marking...\n")
        }
    } else {
        print("⚠️ RUAccent models not found at:")
        print("   Models: \(ruaccentModelsDir.path)")
        print("   Assets: \(ruaccentAssetsDir.path)")
        print("Continuing without stress marking...\n")
    }
    
    // Set reference audio
    print("Setting reference audio: burunow_short.wav")
    try await russianProvider.setReferenceAudio(from: URL(fileURLWithPath: burunowPath))
    print("✓ Reference audio processed\n")
    
    // Test texts - with and without stress examples
    // The words below have different stress patterns that RUAccent should detect
    let russianTexts: [(name: String, text: String, description: String)] = [
        ("russian_greeting", 
         "Привет, меня зовут Бурунов. Рад познакомиться с вами.",
         "Basic greeting"),
        
        ("russian_stressed_words",
         "Замок на горе был окружён замком. Мука для пирога принесла много муки.",
         "Homographs: зАмок/замОк (castle/lock), мукА/мУка (flour/torment)"),
        
        ("russian_emotions",
         "Как прекрасен этот мир! Но иногда он бывает очень грустным.",
         "Emotional expression"),
        
        ("russian_technical",
         "Нейросеть обрабатывает звуковые данные для синтеза речи.",
         "Technical vocabulary"),
        
        ("russian_poetry",
         "Я вас любил, любовь ещё быть может, в душе моей угасла не совсем.",
         "Pushkin poetry - stress patterns critical")
    ]
    
    for (name, text, description) in russianTexts {
        print("--- \(description) ---")
        print("Original:  \"\(text)\"")
        
        // Show RUAccent processed text if available
        if let ruAccent = ruAccent {
            do {
                let processed = try ruAccent.process(text)
                print("RUAccent:  \"\(processed)\"")
                let withMarks = RUAccentProvider.convertToStressMarks(processed)
                print("Stressed:  \"\(withMarks)\"")
            } catch {
                print("RUAccent error: \(error)")
            }
        } else {
            print("RUAccent:  (not available)")
        }
        
        print("\nSynthesizing...")
        let synthStart = Date()
        let audio = try await russianProvider.synthesize(text, voice: "")
        let synthTime = Date().timeIntervalSince(synthStart)
        
        let rtf = audio.duration / synthTime
        print("Duration: \(String(format: "%.2f", audio.duration))s, RTF: \(String(format: "%.2fx", rtf))")
        
        let outputPath = "\(outputDir)/chatterbox_\(name).wav"
        let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: outputPath)
        print("✓ Saved: \(outputPath)\n")
    }
    
    // ========================================
    // COMPARISON: WITH vs WITHOUT RUAccent
    // ========================================
    if ruaccentAvailable {
        print("\n--- Stress Comparison Test ---")
        print("Generating same text WITH and WITHOUT RUAccent\n")
        
        // Create provider without RUAccent
        let russianNoAccent = TTSProviders.chatterbox(
            precision: .fp32,
            language: .russian,
            useRuAccent: false  // Explicitly disable
        )
        try await russianNoAccent.load()
        try await russianNoAccent.setReferenceAudio(from: URL(fileURLWithPath: burunowPath))
        
        // Test with homographs
        let comparisonText = "Замок на горе был окружён замком. Дорого ли стоит дорога?"
        
        print("Text: \"\(comparisonText)\"")
        
        // WITHOUT RUAccent
        print("\n[Without RUAccent]")
        var audio = try await russianNoAccent.synthesize(comparisonText, voice: "")
        var saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: "\(outputDir)/chatterbox_comparison_no_ruaccent.wav")
        print("✓ Saved: chatterbox_comparison_no_ruaccent.wav")
        
        // WITH RUAccent (using the already configured provider)
        print("\n[With RUAccent]")
        if let ruAccent = ruAccent {
            let processed = try ruAccent.process(comparisonText)
            let stressed = RUAccentProvider.convertToStressMarks(processed)
            print("Processed: \"\(stressed)\"")
        }
        audio = try await russianProvider.synthesize(comparisonText, voice: "")
        saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: "\(outputDir)/chatterbox_comparison_with_ruaccent.wav")
        print("✓ Saved: chatterbox_comparison_with_ruaccent.wav")
    }
    
    // Summary
    print("\n=== ChatterBox Test Complete ===")
    print("Output directory: \(outputDir)")
    if let files = try? FileManager.default.contentsOfDirectory(atPath: outputDir) {
        print("\nGenerated \(files.filter { $0.hasSuffix(".wav") }.count) audio files:")
        for file in files.sorted() where file.hasSuffix(".wav") {
            print("  - \(file)")
        }
    }
    
    GPU.clearCache()
}

// MARK: - Expressive Speech Test

/// Test expressive parameter control
func runChatterboxExpressiveTest() async throws {
    print("\n=== ChatterBox Expressive Speech Test ===")
    print("Testing exaggeration and CFG weight parameters\n")
    
    let outputDir = FileManager.default.currentDirectoryPath + "/chatterbox_output"
    try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    
    let watsonPath = "/path/to/clear_voice_research/Docs/watson_short.wav"
    guard FileManager.default.fileExists(atPath: watsonPath) else {
        print("❌ Reference file not found: watson_short.wav")
        return
    }
    
    let provider = TTSProviders.chatterbox(precision: .fp32, language: .english)
    try await provider.load()
    try await provider.setReferenceAudio(from: URL(fileURLWithPath: watsonPath))
    
    let testText = "This is absolutely incredible! I can't believe how amazing this sounds!"
    
    // Test different parameter combinations
    let paramSets: [(name: String, exaggeration: Float, cfg: Float)] = [
        ("default", 0.5, 0.5),
        ("expressive_dramatic", 0.8, 0.3),
        ("calm_deliberate", 0.3, 0.7),
        ("very_expressive", 1.0, 0.2),
        ("neutral", 0.0, 0.5)
    ]
    
    for (name, exag, cfg) in paramSets {
        print("--- \(name): exaggeration=\(exag), cfgWeight=\(cfg) ---")
        await provider.configure(exaggeration: exag, cfgWeight: cfg)
        
        let audio = try await provider.synthesize(testText, voice: "")
        let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: "\(outputDir)/chatterbox_expressive_\(name).wav")
        print("✓ Saved: chatterbox_expressive_\(name).wav\n")
    }
    
    print("=== Expressive Test Complete ===")
    GPU.clearCache()
}
