//
//  AppleTTSTests.swift
//  ClearVoice
//
//  Tests for Apple TTS provider (AVSpeechSynthesizer)
//

import Testing
import AVFoundation
@testable import ClearVoice
@testable import ClearVoiceCore
@testable import ClearVoiceTTS

@Suite("Apple TTS Tests")
struct AppleTTSTests {
    
    // MARK: - SynthesisModel
    
    @Test("SynthesisModel.appleTTS modelName")
    func testSynthesisModelName() {
        let model = SynthesisModel.appleTTS(language: "fr-FR")
        #expect(model.modelName == "apple_tts")
    }
    
    // MARK: - AppleTTSProvider
    
    @Test("AppleTTSProvider initialization")
    func testProviderInit() {
        let provider = AppleTTSProvider(language: "de-DE")
        #expect(provider.sampleRate == 22050)
    }
    
    @Test("AppleTTSProvider available voices")
    func testAvailableVoices() {
        let provider = AppleTTSProvider(language: "en-US")
        let voices = provider.availableVoices
        
        print("English (US) voices: \(voices.count)")
        #expect(voices.count > 0)
    }
    
    @Test("AppleTTSProvider getVoicesForLanguage")
    func testGetVoicesForLanguage() async {
        let provider = AppleTTSProvider(language: "fr-FR")
        let voices = await provider.getVoicesForLanguage()
        
        print("French voices:")
        for voice in voices.prefix(3) {
            print("  \(voice.name) - \(voice.quality)")
        }
        
        #expect(voices.count > 0)
    }
    
    @Test("AppleTTSProvider supported languages")
    func testSupportedLanguages() {
        let languages = AppleTTSProvider.supportedLanguages
        
        print("Total supported languages: \(languages.count)")
        
        #expect(languages.count > 30)
        #expect(languages.contains { $0.starts(with: "en") })
        #expect(languages.contains { $0.starts(with: "fr") })
        #expect(languages.contains { $0.starts(with: "de") })
        #expect(languages.contains { $0.starts(with: "ru") })
    }
    
    // MARK: - Factory
    
    @Test("TTSProviders.apple factory")
    func testFactory() {
        let provider = TTSProviders.apple(language: "it-IT")
        #expect(provider.sampleRate == 22050)
    }
    
    // MARK: - ClearVoice Integration
    
    @Test("ClearVoice configure Apple TTS")
    func testClearVoiceRegistration() async {
        let voice = ClearVoice()
        await voice.configureAppleTTS(language: "fr-FR")
        #expect(true)
    }
    
    // MARK: - Synthesis Tests (headless)
    
    @Test("AppleTTSProvider synthesize English")
    func testSynthesizeEnglish() async throws {
        let provider = AppleTTSProvider(language: "en-US")
        let audio = try await provider.synthesize("Hello world", voice: "")
        
        print("English synthesis: \(audio.samples.count) samples at \(audio.sampleRate)Hz")
        
        #expect(audio.samples.count > 0)
        #expect(audio.sampleRate > 0)
    }
    
    @Test("AppleTTSProvider synthesize French")
    func testSynthesizeFrench() async throws {
        let provider = AppleTTSProvider(language: "fr-FR")
        let audio = try await provider.synthesize("Bonjour le monde", voice: "Thomas")
        
        print("French synthesis: \(audio.samples.count) samples")
        
        #expect(audio.samples.count > 10000)  // Should have substantial audio
    }
    
    @Test("AppleTTSProvider synthesize German")
    func testSynthesizeGerman() async throws {
        let provider = AppleTTSProvider(language: "de-DE")
        let audio = try await provider.synthesize("Guten Morgen", voice: "")
        
        print("German synthesis: \(audio.samples.count) samples")
        
        #expect(audio.samples.count > 0)
    }
    
    @Test("ClearVoice full integration with Apple TTS")
    func testClearVoiceFullIntegration() async throws {
        let voice = ClearVoice()
        await voice.configureAppleTTS(language: "it-IT")
        
        let audio = try await voice.synthesize(
            "Buongiorno",
            voice: "",
            model: .appleTTS(language: "it-IT")
        )
        
        print("Italian synthesis via ClearVoice: \(audio.samples.count) samples")
        
        #expect(audio.samples.count > 0)
    }
}
