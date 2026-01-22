//
//  StreamingPipelineTests.swift
//  ClearVoiceFluidAudioTests
//
//  Streaming pipeline tests for VAD -> diarization -> enhancement -> USS workflows
//

import XCTest
@testable import ClearVoice
import ClearVoiceCore
import ClearVoiceFluidAudio
import ClearVoiceMLX
import ClearVoiceUSS
import AudioUtils
import MLX
import USSMLXSwift

final class StreamingPipelineTests: XCTestCase {
    
    static let projectRoot: String = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()
    
    private actor ProgressTracker {
        private var progressEvents: [(stage: String, percent: Double)] = []
        private var stageCompleteEvents: [String] = []
        
        func record(event: PipelineEvent) {
            switch event {
            case .progress(let stage, let percent):
                progressEvents.append((stage: stage, percent: percent))
            case .stageComplete(let stage, _):
                stageCompleteEvents.append(stage)
            default:
                break
            }
        }
        
        func snapshot() -> (progressEvents: [(stage: String, percent: Double)], stageCompleteEvents: [String]) {
            (progressEvents, stageCompleteEvents)
        }
    }
    
    // MARK: - Test Fixtures
    
    private func harryPotterURL() throws -> URL {
        if let url = Bundle.module.url(forResource: "harry_potter", withExtension: "wav", subdirectory: "Fixtures") {
            return url
        }
        
        let fallback = URL(fileURLWithPath: "\(Self.projectRoot)/Docs/harry_potter.wav")
        guard FileManager.default.fileExists(atPath: fallback.path) else {
            throw XCTSkip("harry_potter.wav not found in Fixtures or Docs")
        }
        return fallback
    }
    
    private func loadAudio(at url: URL, sampleRate: Int) throws -> AudioBuffer {
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: Double(sampleRate),
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: url)
        eval(audio)
        return AudioBuffer(samples: audio.asArray(Float.self), sampleRate: sampleRate, channels: 1)
    }
    
    private func loadAudioAtRates(url: URL) throws -> (audio16k: AudioBuffer, audio48k: AudioBuffer, audio32k: AudioBuffer) {
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        let audio48k = try loadAudio(at: url, sampleRate: 48000)
        let audio32k = try loadAudio(at: url, sampleRate: 32000)
        return (audio16k, audio48k, audio32k)
    }
    
    private func resample48kTo32k(_ buffer: AudioBuffer) -> AudioBuffer {
        guard !buffer.samples.isEmpty else {
            return AudioBuffer(samples: [], sampleRate: 32000, channels: buffer.channels)
        }
        
        let ratio = 32000.0 / 48000.0
        let newLength = Int(Double(buffer.samples.count) * ratio)
        guard newLength > 0 else {
            return AudioBuffer(samples: [], sampleRate: 32000, channels: buffer.channels)
        }
        
        var resampled = [Float](repeating: 0, count: newLength)
        for i in 0..<newLength {
            let srcIndex = Double(i) / ratio
            let idx0 = min(Int(srcIndex), buffer.samples.count - 1)
            let idx1 = min(idx0 + 1, buffer.samples.count - 1)
            let frac = Float(srcIndex - Double(idx0))
            resampled[i] = buffer.samples[idx0] * (1 - frac) + buffer.samples[idx1] * frac
        }
        
        return AudioBuffer(samples: resampled, sampleRate: 32000, channels: buffer.channels)
    }
    
    private func extractSegment(from audio: AudioBuffer, timeRange: TimeRange) -> AudioBuffer {
        audio.slice(timeRange.start..<timeRange.end)
    }
    
    private func concatenateSegments(from audio: AudioBuffer, ranges: [TimeRange]) -> AudioBuffer {
        guard !ranges.isEmpty else {
            return AudioBuffer(samples: [], sampleRate: audio.sampleRate, channels: audio.channels)
        }
        
        let estimatedSamples = ranges.reduce(0) { $0 + Int($1.duration * Double(audio.sampleRate)) }
        var samples: [Float] = []
        samples.reserveCapacity(max(estimatedSamples, 0))
        
        for range in ranges {
            let segment = extractSegment(from: audio, timeRange: range)
            samples.append(contentsOf: segment.samples)
        }
        
        return AudioBuffer(samples: samples, sampleRate: audio.sampleRate, channels: audio.channels)
    }
    
    private func buildNonSpeechRanges(from speechSegments: [VADSegment], totalDuration: Double) -> [TimeRange] {
        let sorted = speechSegments.sorted { $0.timeRange.start < $1.timeRange.start }
        var ranges: [TimeRange] = []
        var cursor = 0.0
        
        for segment in sorted {
            if segment.timeRange.start > cursor {
                ranges.append(TimeRange(start: cursor, end: segment.timeRange.start))
            }
            cursor = max(cursor, segment.timeRange.end)
        }
        
        if cursor < totalDuration {
            ranges.append(TimeRange(start: cursor, end: totalDuration))
        }
        
        return ranges.filter { $0.duration > 0 }
    }
    
    private func assertSampleRate(
        _ audio: AudioBuffer,
        expected: Int,
        context: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(audio.sampleRate, expected, "\(context) requires \(expected)Hz", file: file, line: line)
    }
    
    private func maxAmplitude(_ audio: AudioBuffer) -> Float {
        audio.samples.map { abs($0) }.max() ?? 0
    }
    
    private func runFullBatchPipeline(
        diarizerName: String,
        diarizer: any DiarizationProvider,
        audio16k: AudioBuffer,
        audio48k: AudioBuffer,
        audio32k: AudioBuffer
    ) async throws {
        print("\n=== Full Batch Pipeline (\(diarizerName)) ===")
        
        assertSampleRate(audio16k, expected: 16000, context: "VAD/Diarization input")
        assertSampleRate(audio48k, expected: 48000, context: "SE input")
        assertSampleRate(audio32k, expected: 32000, context: "USS input")
        
        let duration = audio16k.duration
        print("Audio duration: \(String(format: "%.1f", duration))s")
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        let enhancer = MLXProviders.mossformer2SE48K()
        let uss = USSProviders.speechSeparation()
        defer { Task { await uss.unload() } }
        
        let vadLoadStart = Date()
        try await vad.load()
        let vadLoadTime = Date().timeIntervalSince(vadLoadStart)
        print("VAD loaded in \(String(format: "%.2f", vadLoadTime))s")
        
        let seLoadStart = Date()
        try await enhancer.load()
        let seLoadTime = Date().timeIntervalSince(seLoadStart)
        print("SE 48K loaded in \(String(format: "%.2f", seLoadTime))s")
        
        let ussLoadStart = Date()
        try await uss.load()
        let ussLoadTime = Date().timeIntervalSince(ussLoadStart)
        print("USS loaded in \(String(format: "%.2f", ussLoadTime))s")
        
        let vadStart = Date()
        let speechSegments = try await vad.detect(audio16k)
        let vadTime = Date().timeIntervalSince(vadStart)
        let vadRTF = duration / max(vadTime, 0.001)
        print("VAD detected \(speechSegments.count) segments in \(String(format: "%.2f", vadTime))s (RTF: \(String(format: "%.1f", vadRTF))x)")
        XCTAssertFalse(speechSegments.isEmpty)
        
        let speechRanges = speechSegments.map(\.timeRange)
        let speechDuration = speechRanges.reduce(0.0) { $0 + $1.duration }
        let nonSpeechRanges = buildNonSpeechRanges(from: speechSegments, totalDuration: duration)
        let nonSpeechDuration = nonSpeechRanges.reduce(0.0) { $0 + $1.duration }
        print("Speech duration: \(String(format: "%.1f", speechDuration))s, non-speech: \(String(format: "%.1f", nonSpeechDuration))s")
        XCTAssertFalse(nonSpeechRanges.isEmpty)
        
        let speechAudio48k = concatenateSegments(from: audio48k, ranges: speechRanges)
        let nonSpeechAudio32k = concatenateSegments(from: audio32k, ranges: nonSpeechRanges)
        assertSampleRate(speechAudio48k, expected: 48000, context: "SE input")
        assertSampleRate(nonSpeechAudio32k, expected: 32000, context: "USS non-speech input")
        XCTAssertFalse(speechAudio48k.samples.isEmpty)
        XCTAssertFalse(nonSpeechAudio32k.samples.isEmpty)
        
        let diarizeStart = Date()
        let timeline = try await diarizer.diarize(audio16k, vadHint: speechSegments)
        let diarizeTime = Date().timeIntervalSince(diarizeStart)
        let diarizeRTF = duration / max(diarizeTime, 0.001)
        print("\(diarizerName) diarization completed in \(String(format: "%.2f", diarizeTime))s (RTF: \(String(format: "%.1f", diarizeRTF))x)")
        print("Speakers: \(timeline.speakerCount), segments: \(timeline.segments.count), max overlap: \(timeline.maxOverlappingSpeakers)")
        XCTAssertGreaterThan(timeline.segments.count, 0)
        
        let seStart = Date()
        let seResult = try await enhancer.processWithBackground(speechAudio48k)
        let seTime = Date().timeIntervalSince(seStart)
        let seRTF = speechAudio48k.duration / max(seTime, 0.001)
        print("SE processed \(String(format: "%.1f", speechAudio48k.duration))s in \(String(format: "%.2f", seTime))s (RTF: \(String(format: "%.1f", seRTF))x)")
        assertSampleRate(seResult.enhanced, expected: 48000, context: "SE output")
        assertSampleRate(seResult.background, expected: 48000, context: "SE background")
        
        let background32k = resample48kTo32k(seResult.background)
        assertSampleRate(background32k, expected: 32000, context: "USS background input")
        XCTAssertFalse(background32k.samples.isEmpty)
        
        let musicNonSpeechStart = Date()
        let musicNonSpeech = try await uss.process(nonSpeechAudio32k, type: .music)
        let musicNonSpeechTime = Date().timeIntervalSince(musicNonSpeechStart)
        let musicNonSpeechRTF = nonSpeechAudio32k.duration / max(musicNonSpeechTime, 0.001)
        print("USS non-speech music in \(String(format: "%.2f", musicNonSpeechTime))s (RTF: \(String(format: "%.1f", musicNonSpeechRTF))x)")
        
        let animalNonSpeechStart = Date()
        let animalNonSpeech = try await uss.process(nonSpeechAudio32k, type: .animal)
        let animalNonSpeechTime = Date().timeIntervalSince(animalNonSpeechStart)
        let animalNonSpeechRTF = nonSpeechAudio32k.duration / max(animalNonSpeechTime, 0.001)
        print("USS non-speech animal in \(String(format: "%.2f", animalNonSpeechTime))s (RTF: \(String(format: "%.1f", animalNonSpeechRTF))x)")
        
        let musicBackgroundStart = Date()
        let musicBackground = try await uss.process(background32k, type: .music)
        let musicBackgroundTime = Date().timeIntervalSince(musicBackgroundStart)
        let musicBackgroundRTF = background32k.duration / max(musicBackgroundTime, 0.001)
        print("USS background music in \(String(format: "%.2f", musicBackgroundTime))s (RTF: \(String(format: "%.1f", musicBackgroundRTF))x)")
        
        let animalBackgroundStart = Date()
        let animalBackground = try await uss.process(background32k, type: .animal)
        let animalBackgroundTime = Date().timeIntervalSince(animalBackgroundStart)
        let animalBackgroundRTF = background32k.duration / max(animalBackgroundTime, 0.001)
        print("USS background animal in \(String(format: "%.2f", animalBackgroundTime))s (RTF: \(String(format: "%.1f", animalBackgroundRTF))x)")
        
        XCTAssertEqual(musicNonSpeech.sampleRate, 32000)
        XCTAssertEqual(animalNonSpeech.sampleRate, 32000)
        XCTAssertEqual(musicBackground.sampleRate, 32000)
        XCTAssertEqual(animalBackground.sampleRate, 32000)
        
        XCTAssertGreaterThan(musicNonSpeech.samples.count, 0)
        XCTAssertGreaterThan(animalNonSpeech.samples.count, 0)
        XCTAssertGreaterThan(musicBackground.samples.count, 0)
        XCTAssertGreaterThan(animalBackground.samples.count, 0)
        
        XCTAssertGreaterThan(maxAmplitude(musicNonSpeech), 0)
        XCTAssertGreaterThan(maxAmplitude(animalNonSpeech), 0)
        XCTAssertGreaterThan(maxAmplitude(musicBackground), 0)
        XCTAssertGreaterThan(maxAmplitude(animalBackground), 0)
        
        print("✓ Full batch pipeline (\(diarizerName)) completed")
    }
    
    // MARK: - Full Pipeline Tests (Batch Baseline)
    
    func testFullBatchPipeline_Sortformer_HarryPotter() async throws {
        let url = try harryPotterURL()
        let audio = try loadAudioAtRates(url: url)
        
        let diarizer = FluidAudioProviders.sortformerLowLatency()
        let loadStart = Date()
        try await diarizer.load()
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Sortformer loaded in \(String(format: "%.2f", loadTime))s")
        
        try await runFullBatchPipeline(
            diarizerName: "Sortformer",
            diarizer: diarizer,
            audio16k: audio.audio16k,
            audio48k: audio.audio48k,
            audio32k: audio.audio32k
        )
    }
    
    func testFullBatchPipeline_Pyannote_HarryPotter() async throws {
        let url = try harryPotterURL()
        let audio = try loadAudioAtRates(url: url)
        
        let diarizer = FluidAudioProviders.pyannote()
        let loadStart = Date()
        try await diarizer.load()
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Pyannote loaded in \(String(format: "%.2f", loadTime))s")
        
        try await runFullBatchPipeline(
            diarizerName: "Pyannote",
            diarizer: diarizer,
            audio16k: audio.audio16k,
            audio48k: audio.audio48k,
            audio32k: audio.audio32k
        )
    }
    
    // MARK: - Streaming Tests
    
    func testStreamingVADToSortformer_HarryPotter() async throws {
        print("\n=== Streaming VAD -> Sortformer Test ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        assertSampleRate(audio16k, expected: 16000, context: "Streaming input")
        
        let duration = audio16k.duration
        print("Audio duration: \(String(format: "%.1f", duration))s")
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await vad.load()
        let speechSegments = try await vad.detect(audio16k)
        XCTAssertFalse(speechSegments.isEmpty)
        
        let sortformerStreaming = FluidAudioProviders.sortformerLowLatency()
        try await sortformerStreaming.load()
        
        let chunkSize = 4096
        let accumulationTarget = 16000 * 10
        var speechBuffer: [Float] = []
        var processedChunks = 0
        
        for offset in stride(from: 0, to: audio16k.samples.count, by: chunkSize) {
            let end = min(offset + chunkSize, audio16k.samples.count)
            let chunkRange = TimeRange(
                start: Double(offset) / 16000.0,
                end: Double(end) / 16000.0
            )
            let isSpeech = speechSegments.contains { $0.timeRange.overlaps(with: chunkRange) }
            if isSpeech {
                speechBuffer.append(contentsOf: audio16k.samples[offset..<end])
            }
            
            if speechBuffer.count >= accumulationTarget {
                if (try await sortformerStreaming.processChunk(speechBuffer)) != nil {
                    processedChunks += 1
                    print("Processed streaming chunk \(processedChunks)")
                }
                speechBuffer.removeAll(keepingCapacity: true)
            }
        }
        
        if !speechBuffer.isEmpty {
            if (try await sortformerStreaming.processChunk(speechBuffer)) != nil {
                processedChunks += 1
                print("Processed streaming chunk \(processedChunks)")
            }
        }
        
        XCTAssertGreaterThan(processedChunks, 0)
        
        let streamingTimeline = await sortformerStreaming.currentTimeline
        XCTAssertNotNil(streamingTimeline)
        XCTAssertGreaterThan(streamingTimeline?.segments.count ?? 0, 0)
        
        let batchDiarizer = FluidAudioProviders.sortformerLowLatency()
        try await batchDiarizer.load()
        let batchTimeline = try await batchDiarizer.diarize(audio16k)
        
        print("Streaming segments: \(streamingTimeline?.segments.count ?? 0), batch segments: \(batchTimeline.segments.count)")
        print("Streaming speakers: \(streamingTimeline?.speakerCount ?? 0), batch speakers: \(batchTimeline.speakerCount)")
        print("✓ Streaming VAD -> Sortformer test completed")
    }
    
    // MARK: - Multi-Type USS Tests
    
    func testMultiTypeUSS_NonSpeech_HarryPotter() async throws {
        print("\n=== USS Non-Speech Separation (Harry Potter) ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        let audio32k = try loadAudio(at: url, sampleRate: 32000)
        assertSampleRate(audio16k, expected: 16000, context: "VAD input")
        assertSampleRate(audio32k, expected: 32000, context: "USS input")
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await vad.load()
        let speechSegments = try await vad.detect(audio16k)
        let nonSpeechRanges = buildNonSpeechRanges(from: speechSegments, totalDuration: audio16k.duration)
        let nonSpeechAudio = concatenateSegments(from: audio32k, ranges: nonSpeechRanges)
        
        XCTAssertFalse(nonSpeechAudio.samples.isEmpty)
        let nonSpeechDuration = nonSpeechAudio.duration
        print("Non-speech duration: \(String(format: "%.1f", nonSpeechDuration))s")
        
        let uss = USSProviders.speechSeparation()
        defer { Task { await uss.unload() } }
        try await uss.load()
        
        let musicStart = Date()
        let music = try await uss.process(nonSpeechAudio, type: .music)
        let musicTime = Date().timeIntervalSince(musicStart)
        let musicRTF = nonSpeechDuration / max(musicTime, 0.001)
        print("USS music in \(String(format: "%.2f", musicTime))s (RTF: \(String(format: "%.1f", musicRTF))x)")
        
        let animalStart = Date()
        let animal = try await uss.process(nonSpeechAudio, type: .animal)
        let animalTime = Date().timeIntervalSince(animalStart)
        let animalRTF = nonSpeechDuration / max(animalTime, 0.001)
        print("USS animal in \(String(format: "%.2f", animalTime))s (RTF: \(String(format: "%.1f", animalRTF))x)")
        
        XCTAssertEqual(music.sampleRate, 32000)
        XCTAssertEqual(animal.sampleRate, 32000)
        XCTAssertGreaterThan(music.samples.count, 0)
        XCTAssertGreaterThan(animal.samples.count, 0)
        XCTAssertGreaterThan(maxAmplitude(music), 0)
        XCTAssertGreaterThan(maxAmplitude(animal), 0)
        
        print("✓ USS non-speech separation completed")
    }
    
    func testMultiTypeUSS_SEBackground_HarryPotter() async throws {
        print("\n=== USS SE Background Separation (Harry Potter) ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        let audio48k = try loadAudio(at: url, sampleRate: 48000)
        assertSampleRate(audio16k, expected: 16000, context: "VAD input")
        assertSampleRate(audio48k, expected: 48000, context: "SE input")
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await vad.load()
        let speechSegments = try await vad.detect(audio16k)
        XCTAssertFalse(speechSegments.isEmpty)
        
        let speechRanges = speechSegments.map(\.timeRange)
        let speechAudio48k = concatenateSegments(from: audio48k, ranges: speechRanges)
        XCTAssertFalse(speechAudio48k.samples.isEmpty)
        
        let enhancer = MLXProviders.mossformer2SE48K()
        try await enhancer.load()
        
        let seStart = Date()
        let seResult = try await enhancer.processWithBackground(speechAudio48k)
        let seTime = Date().timeIntervalSince(seStart)
        let seRTF = speechAudio48k.duration / max(seTime, 0.001)
        print("SE background extraction in \(String(format: "%.2f", seTime))s (RTF: \(String(format: "%.1f", seRTF))x)")
        
        let background32k = resample48kTo32k(seResult.background)
        assertSampleRate(background32k, expected: 32000, context: "USS background input")
        XCTAssertFalse(background32k.samples.isEmpty)
        
        let uss = USSProviders.speechSeparation()
        defer { Task { await uss.unload() } }
        try await uss.load()
        
        let musicStart = Date()
        let music = try await uss.process(background32k, type: .music)
        let musicTime = Date().timeIntervalSince(musicStart)
        let musicRTF = background32k.duration / max(musicTime, 0.001)
        print("USS background music in \(String(format: "%.2f", musicTime))s (RTF: \(String(format: "%.1f", musicRTF))x)")
        
        let animalStart = Date()
        let animal = try await uss.process(background32k, type: .animal)
        let animalTime = Date().timeIntervalSince(animalStart)
        let animalRTF = background32k.duration / max(animalTime, 0.001)
        print("USS background animal in \(String(format: "%.2f", animalTime))s (RTF: \(String(format: "%.1f", animalRTF))x)")
        
        XCTAssertEqual(music.sampleRate, 32000)
        XCTAssertEqual(animal.sampleRate, 32000)
        XCTAssertGreaterThan(music.samples.count, 0)
        XCTAssertGreaterThan(animal.samples.count, 0)
        XCTAssertGreaterThan(maxAmplitude(music), 0)
        XCTAssertGreaterThan(maxAmplitude(animal), 0)
        
        print("✓ USS SE background separation completed")
    }
    
    func testUSSEmbeddingSwitching_HarryPotter() async throws {
        print("\n=== USS Embedding Switching (Harry Potter) ===")
        
        let url = try harryPotterURL()
        let audio32k = try loadAudio(at: url, sampleRate: 32000)
        assertSampleRate(audio32k, expected: 32000, context: "USS input")
        
        let duration = audio32k.duration
        let uss = USSProviders.speechSeparation()
        defer { Task { await uss.unload() } }
        try await uss.load()
        
        let initialType = await uss.activeEmbeddingType
        XCTAssertEqual(initialType, .speech)
        
        let musicStart = Date()
        let music = try await uss.process(audio32k, type: .music)
        let musicTime = Date().timeIntervalSince(musicStart)
        let musicRTF = duration / max(musicTime, 0.001)
        print("Music separation in \(String(format: "%.2f", musicTime))s (RTF: \(String(format: "%.1f", musicRTF))x)")
        
        let afterMusicType = await uss.activeEmbeddingType
        XCTAssertEqual(afterMusicType, .speech)
        
        let animalStart = Date()
        let animal = try await uss.process(audio32k, type: .animal)
        let animalTime = Date().timeIntervalSince(animalStart)
        let animalRTF = duration / max(animalTime, 0.001)
        print("Animal separation in \(String(format: "%.2f", animalTime))s (RTF: \(String(format: "%.1f", animalRTF))x)")
        
        let afterAnimalType = await uss.activeEmbeddingType
        XCTAssertEqual(afterAnimalType, .speech)
        
        XCTAssertEqual(music.sampleRate, 32000)
        XCTAssertEqual(animal.sampleRate, 32000)
        XCTAssertGreaterThan(music.samples.count, 0)
        XCTAssertGreaterThan(animal.samples.count, 0)
        
        print("✓ USS embedding switching completed")
    }
    
    func testUSSProcessMultiple_HarryPotter() async throws {
        print("\n=== USS processMultiple (Harry Potter) ===")
        
        let url = try harryPotterURL()
        let audio32k = try loadAudio(at: url, sampleRate: 32000)
        assertSampleRate(audio32k, expected: 32000, context: "USS input")
        
        let duration = audio32k.duration
        let uss = USSProviders.speechSeparation()
        defer { Task { await uss.unload() } }
        try await uss.load()
        
        // Use EmbeddingLoader.EmbeddingType for processMultiple
        let types: [EmbeddingLoader.EmbeddingType] = [.music, .animal, .noise]
        let start = Date()
        let results = try await uss.processMultiple(audio32k, types: types)
        let elapsed = Date().timeIntervalSince(start)
        let rtf = duration / max(elapsed, 0.001)
        print("processMultiple in \(String(format: "%.2f", elapsed))s (RTF: \(String(format: "%.1f", rtf))x)")
        
        for type in types {
            guard let result = results[type] else {
                XCTFail("Missing result for \(type.rawValue)")
                continue
            }
            XCTAssertEqual(result.sampleRate, 32000)
            XCTAssertGreaterThan(result.samples.count, 0)
        }
        
        print("✓ USS processMultiple completed")
    }
    
    // MARK: - Diarization Comparison
    
    func testDiarizationComparison_HarryPotter() async throws {
        print("\n=== Diarization Comparison (Harry Potter) ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        assertSampleRate(audio16k, expected: 16000, context: "Diarization input")
        
        let duration = audio16k.duration
        let pyannote = FluidAudioProviders.pyannote()
        let sortformer = FluidAudioProviders.sortformerLowLatency()
        
        let pyannoteLoadStart = Date()
        try await pyannote.load()
        let pyannoteLoadTime = Date().timeIntervalSince(pyannoteLoadStart)
        print("Pyannote loaded in \(String(format: "%.2f", pyannoteLoadTime))s")
        
        let sortformerLoadStart = Date()
        try await sortformer.load()
        let sortformerLoadTime = Date().timeIntervalSince(sortformerLoadStart)
        print("Sortformer loaded in \(String(format: "%.2f", sortformerLoadTime))s")
        
        let pyannoteStart = Date()
        let pyannoteTimeline = try await pyannote.diarize(audio16k)
        let pyannoteTime = Date().timeIntervalSince(pyannoteStart)
        let pyannoteRTF = duration / max(pyannoteTime, 0.001)
        
        let sortformerStart = Date()
        let sortformerTimeline = try await sortformer.diarize(audio16k)
        let sortformerTime = Date().timeIntervalSince(sortformerStart)
        let sortformerRTF = duration / max(sortformerTime, 0.001)
        
        print("Pyannote: \(pyannoteTimeline.speakerCount) speakers, \(pyannoteTimeline.segments.count) segments, RTF \(String(format: "%.1f", pyannoteRTF))x")
        print("Sortformer: \(sortformerTimeline.speakerCount) speakers, \(sortformerTimeline.segments.count) segments, RTF \(String(format: "%.1f", sortformerRTF))x")
        
        XCTAssertGreaterThan(pyannoteTimeline.segments.count, 0)
        XCTAssertGreaterThan(sortformerTimeline.segments.count, 0)
        
        print("✓ Diarization comparison completed")
    }
    
    // MARK: - Performance Benchmark
    
    func testPipelinePerformanceBenchmark_HarryPotter() async throws {
        print("\n=== Pipeline Performance Benchmark (Harry Potter) ===")
        
        let url = try harryPotterURL()
        let audio = try loadAudioAtRates(url: url)
        let duration = audio.audio16k.duration
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        let diarizer = FluidAudioProviders.sortformerLowLatency()
        let enhancer = MLXProviders.mossformer2SE48K()
        let uss = USSProviders.speechSeparation()
        defer { Task { await uss.unload() } }
        
        let vadLoadStart = Date()
        try await vad.load()
        let vadLoadTime = Date().timeIntervalSince(vadLoadStart)
        
        let diarizerLoadStart = Date()
        try await diarizer.load()
        let diarizerLoadTime = Date().timeIntervalSince(diarizerLoadStart)
        
        let enhancerLoadStart = Date()
        try await enhancer.load()
        let enhancerLoadTime = Date().timeIntervalSince(enhancerLoadStart)
        
        let ussLoadStart = Date()
        try await uss.load()
        let ussLoadTime = Date().timeIntervalSince(ussLoadStart)
        
        print("Load times - VAD: \(String(format: "%.2f", vadLoadTime))s, Diarization: \(String(format: "%.2f", diarizerLoadTime))s, SE: \(String(format: "%.2f", enhancerLoadTime))s, USS: \(String(format: "%.2f", ussLoadTime))s")
        
        let vadStart = Date()
        let speechSegments = try await vad.detect(audio.audio16k)
        let vadTime = Date().timeIntervalSince(vadStart)
        let vadRTF = duration / max(vadTime, 0.001)
        
        let speechRanges = speechSegments.map(\.timeRange)
        let speechDuration = speechRanges.reduce(0.0) { $0 + $1.duration }
        let nonSpeechRanges = buildNonSpeechRanges(from: speechSegments, totalDuration: duration)
        let nonSpeechDuration = nonSpeechRanges.reduce(0.0) { $0 + $1.duration }
        let speechAudio48k = concatenateSegments(from: audio.audio48k, ranges: speechRanges)
        let nonSpeechAudio32k = concatenateSegments(from: audio.audio32k, ranges: nonSpeechRanges)
        
        let diarizeStart = Date()
        let timeline = try await diarizer.diarize(audio.audio16k)
        let diarizeTime = Date().timeIntervalSince(diarizeStart)
        let diarizeRTF = duration / max(diarizeTime, 0.001)
        
        let seStart = Date()
        let seResult = try await enhancer.processWithBackground(speechAudio48k)
        let seTime = Date().timeIntervalSince(seStart)
        let seRTF = speechDuration / max(seTime, 0.001)
        
        let background32k = resample48kTo32k(seResult.background)
        
        let ussNonSpeechStart = Date()
        _ = try await uss.process(nonSpeechAudio32k, type: .music)
        let ussNonSpeechTime = Date().timeIntervalSince(ussNonSpeechStart)
        let ussNonSpeechRTF = nonSpeechDuration / max(ussNonSpeechTime, 0.001)
        
        let ussBackgroundStart = Date()
        _ = try await uss.process(background32k, type: .music)
        let ussBackgroundTime = Date().timeIntervalSince(ussBackgroundStart)
        let ussBackgroundRTF = background32k.duration / max(ussBackgroundTime, 0.001)
        
        let totalTime = vadTime + diarizeTime + seTime + ussNonSpeechTime + ussBackgroundTime
        let totalRTF = duration / max(totalTime, 0.001)
        
        print("RTF - VAD: \(String(format: "%.1f", vadRTF))x, Diarization: \(String(format: "%.1f", diarizeRTF))x, SE: \(String(format: "%.1f", seRTF))x")
        print("RTF - USS non-speech: \(String(format: "%.1f", ussNonSpeechRTF))x, USS background: \(String(format: "%.1f", ussBackgroundRTF))x")
        print("Total pipeline time: \(String(format: "%.2f", totalTime))s (RTF: \(String(format: "%.1f", totalRTF))x)")
        print("Speakers: \(timeline.speakerCount), segments: \(timeline.segments.count)")
        
        XCTAssertFalse(speechSegments.isEmpty)
        XCTAssertGreaterThan(timeline.segments.count, 0)
        
        print("✓ Pipeline benchmark completed")
    }
    
    // MARK: - Progress Reporting
    
    func testProgressReportingDuringPipeline() async throws {
        print("\n=== Pipeline Progress Reporting Test ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        assertSampleRate(audio16k, expected: 16000, context: "Pipeline input")
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        let diarizer = FluidAudioProviders.sortformerLowLatency()
        let enhancer = MLXProviders.mossformer2SE48K()
        
        try await vad.load()
        try await diarizer.load()
        try await enhancer.load()
        
        let voice = ClearVoice(
            configuration: .default,
            vad: vad,
            diarization: diarizer,
            enhancer: (.mossformerSE48k, enhancer)
        )
        
        let tracker = ProgressTracker()
        _ = try await voice.pipeline()
            .detect(.silero)
            .diarize()
            .enhance(.mossformerSE48k)
            .onEvent { event in
                await tracker.record(event: event)
            }
            .process(audio: audio16k)
        
        let snapshot = await tracker.snapshot()
        let progressEvents = snapshot.progressEvents
        let stageCompleteEvents = snapshot.stageCompleteEvents
        
        let vadProgress = progressEvents.filter { $0.stage == "vad" }
        let diarizationProgress = progressEvents.filter { $0.stage == "diarization" }
        let enhancementProgress = progressEvents.filter { $0.stage == "enhancement" }
        
        print("Progress events collected:")
        
        // VAD progress - with chunked processing should have intermediate events
        if vadProgress.count <= 2 {
            print("  VAD: \(vadProgress.count) events (batch mode)")
        } else {
            print("  VAD: \(vadProgress.count) intermediate progress events")
        }
        
        // Diarization progress - Sortformer with streaming should have intermediate events
        if diarizationProgress.count <= 2 {
            print("  Diarization: \(diarizationProgress.count) events (batch mode)")
        } else {
            print("  Diarization: \(diarizationProgress.count) intermediate progress events")
        }
        
        print("  Enhancement: \(enhancementProgress.count) events")
        
        XCTAssertTrue(vadProgress.contains { $0.percent == 0 }, "VAD should emit 0% progress")
        XCTAssertTrue(vadProgress.contains { $0.percent == 100 }, "VAD should emit 100% progress")
        
        XCTAssertTrue(diarizationProgress.contains { $0.percent == 0 }, "Diarization should emit 0% progress")
        XCTAssertTrue(diarizationProgress.contains { $0.percent == 100 }, "Diarization should emit 100% progress")
        
        // Enhancement with StreamableOutput should have intermediate progress (more than just 0% and 100%)
        // If only 2 events, provider may not conform to StreamableOutput - still valid but note it
        if enhancementProgress.count <= 2 {
            print("  Note: Enhancement emitted \(enhancementProgress.count) events (batch mode, no streaming)")
        } else {
            print("  Enhancement streaming: \(enhancementProgress.count) intermediate progress events")
        }
        XCTAssertGreaterThanOrEqual(enhancementProgress.count, 2, "Enhancement should emit at least start/end progress")
        
        for stage in ["vad", "diarization", "enhancement"] {
            let stageProgress = progressEvents.filter { $0.stage == stage }.map { $0.percent }
            for index in 1..<stageProgress.count {
                XCTAssertGreaterThanOrEqual(stageProgress[index], stageProgress[index - 1], "Progress should be monotonic for \(stage)")
            }
        }
        
        XCTAssertTrue(stageCompleteEvents.contains("vad"), "stageComplete should be emitted for vad")
        XCTAssertTrue(stageCompleteEvents.contains("diarization"), "stageComplete should be emitted for diarization")
        XCTAssertTrue(stageCompleteEvents.contains("enhancement"), "stageComplete should be emitted for enhancement")
        
        print("✓ Progress reporting test completed")
    }
    
    // MARK: - Quality Comparison Tests
    
    private actor ProgressCounter {
        private var count = 0
        func increment() { count += 1 }
        func value() -> Int { count }
    }
    
    /// Test that VAD quality is identical between batch and progress-aware modes
    /// VAD progress-aware still uses segmentSpeech() for final result, so should be identical
    func testVADQualityBatchVsProgressAware() async throws {
        print("\n=== VAD Quality: Batch vs Progress-Aware ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await vad.load()
        
        // Batch mode (no progress callback)
        let batchSegments = try await vad.detect(audio16k)
        
        // Progress-aware mode (with callback)
        let counter = ProgressCounter()
        let progressCallback: ProgressCallback = { _ in
            await counter.increment()
        }
        let progressSegments = try await vad.detect(audio16k, onProgress: progressCallback)
        let progressCount = await counter.value()
        
        print("Batch mode: \(batchSegments.count) segments")
        print("Progress mode: \(progressSegments.count) segments (\(progressCount) progress events)")
        
        // Compare segment counts
        XCTAssertEqual(batchSegments.count, progressSegments.count, "VAD segment count should be identical")
        
        // Compare each segment's time ranges (should be exactly equal)
        for (idx, (batch, progress)) in zip(batchSegments, progressSegments).enumerated() {
            XCTAssertEqual(batch.timeRange.start, progress.timeRange.start, accuracy: 0.001,
                          "Segment \(idx) start time should match")
            XCTAssertEqual(batch.timeRange.end, progress.timeRange.end, accuracy: 0.001,
                          "Segment \(idx) end time should match")
        }
        
        print("✓ VAD quality is IDENTICAL between batch and progress-aware modes")
    }
    
    /// Test Sortformer diarization quality between batch (processComplete) and streaming (processSamples) modes
    /// Streaming mode may have different results due to limited context at chunk boundaries
    func testDiarizationQualityBatchVsProgressAware() async throws {
        print("\n=== Diarization Quality: Batch vs Progress-Aware (Streaming) ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        
        // Create two separate diarizer instances to avoid state contamination
        let diarizerBatch = FluidAudioProviders.sortformerLowLatency()
        let diarizerProgress = FluidAudioProviders.sortformerLowLatency()
        
        try await diarizerBatch.load()
        try await diarizerProgress.load()
        
        // Batch mode - uses processComplete() internally
        let batchTimeline = try await diarizerBatch.diarize(audio16k)
        
        // Progress-aware mode - uses processSamples() loop internally
        let counter = ProgressCounter()
        let progressCallback: ProgressCallback = { _ in
            await counter.increment()
        }
        let progressTimeline = try await diarizerProgress.diarize(audio16k, onProgress: progressCallback)
        let progressCount = await counter.value()
        
        print("Batch mode: \(batchTimeline.segments.count) segments")
        print("Progress mode: \(progressTimeline.segments.count) segments (\(progressCount) progress events)")
        
        // Calculate total speech duration for each mode
        let batchDuration = batchTimeline.segments.reduce(0.0) { $0 + $1.timeRange.duration }
        let progressDuration = progressTimeline.segments.reduce(0.0) { $0 + $1.timeRange.duration }
        
        print("Batch total speech: \(String(format: "%.2f", batchDuration))s")
        print("Progress total speech: \(String(format: "%.2f", progressDuration))s")
        
        // Compare speaker distribution
        let batchSpeakers = Set(batchTimeline.segments.map { $0.speakerID })
        let progressSpeakers = Set(progressTimeline.segments.map { $0.speakerID })
        
        print("Batch speakers: \(batchSpeakers.map { $0.id }.joined(separator: ", "))")
        print("Progress speakers: \(progressSpeakers.map { $0.id }.joined(separator: ", "))")
        
        // Quality assessment - streaming should be reasonably close to batch
        let segmentCountDiff = abs(batchTimeline.segments.count - progressTimeline.segments.count)
        let durationDiff = abs(batchDuration - progressDuration)
        let durationDiffPercent = (durationDiff / max(batchDuration, 0.001)) * 100
        
        print("\nQuality Comparison:")
        print("  Segment count difference: \(segmentCountDiff)")
        print("  Duration difference: \(String(format: "%.2f", durationDiff))s (\(String(format: "%.1f", durationDiffPercent))%)")
        
        // Warn if there's significant difference (>20% duration diff)
        if durationDiffPercent > 20 {
            print("  ⚠️ WARNING: Significant quality difference detected!")
            print("  Consider using batch mode for quality-critical applications")
        } else if durationDiffPercent > 10 {
            print("  ⚠️ Note: Moderate quality difference (\(String(format: "%.1f", durationDiffPercent))%)")
        } else {
            print("  ✓ Quality difference is acceptable (<10%)")
        }
        
        // Don't fail on quality difference, but document it
        // The test passes as long as both methods produce valid results
        XCTAssertGreaterThan(batchTimeline.segments.count, 0, "Batch mode should detect speakers")
        XCTAssertGreaterThan(progressTimeline.segments.count, 0, "Progress mode should detect speakers")
        
        print("✓ Diarization quality comparison completed")
    }
    
    // MARK: - Speaker Separation Tests
    
    /// Test speaker separation on overlapping speech
    /// Uses WHAMR model for 2-speaker separation (best for noisy environments)
    func testSpeakerSeparation_Overlap_HarryPotter() async throws {
        print("\n=== Speaker Separation on Overlapping Speech ===")
        
        let url = try harryPotterURL()
        let audio8k = try loadAudio(at: url, sampleRate: 8000)  // WHAMR expects 8kHz
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        
        // First, diarize to find overlapping regions
        let diarizer = FluidAudioProviders.sortformerLowLatency()
        try await diarizer.load()
        
        let timeline = try await diarizer.diarize(audio16k)
        print("Speakers detected: \(timeline.speakerCount)")
        print("Max overlapping: \(timeline.maxOverlappingSpeakers)")
        
        let overlappingRanges = timeline.overlappingRanges()
        print("Overlapping regions: \(overlappingRanges.count)")
        
        if overlappingRanges.isEmpty {
            print("No overlapping speech detected - skipping separation test")
            throw XCTSkip("No overlapping speech in test audio")
        }
        
        // Get first overlap region
        let overlapRange = overlappingRanges[0]
        print("Testing overlap at \(String(format: "%.2f", overlapRange.start))s - \(String(format: "%.2f", overlapRange.end))s")
        
        let overlapAudio = audio8k.slice(overlapRange.start..<overlapRange.end)
        print("Overlap duration: \(String(format: "%.2f", overlapAudio.duration))s")
        
        // Skip if overlap is too short
        guard overlapAudio.duration >= 0.5 else {
            print("Overlap too short for separation")
            throw XCTSkip("Overlap region too short")
        }
        
        // Separate using WHAMR model (best for noisy)
        let separator = MLXProviders.mossformer2SS(model: .twoSpeakerWHAMR)
        let loadStart = Date()
        try await separator.load()
        let loadTime = Date().timeIntervalSince(loadStart)
        print("WHAMR separator loaded in \(String(format: "%.2f", loadTime))s")
        
        let separateStart = Date()
        let tracks = try await separator.separate(overlapAudio, speakers: 2)
        let separateTime = Date().timeIntervalSince(separateStart)
        let rtf = overlapAudio.duration / max(separateTime, 0.001)
        print("Separated in \(String(format: "%.2f", separateTime))s (RTF: \(String(format: "%.1f", rtf))x)")
        
        XCTAssertEqual(tracks.count, 2, "Should produce 2 speaker tracks")
        for (idx, track) in tracks.enumerated() {
            XCTAssertGreaterThan(track.samples.count, 0, "Track \(idx) should have samples")
            let maxAmp = maxAmplitude(track)
            print("Track \(idx): \(track.samples.count) samples, max amp: \(String(format: "%.4f", maxAmp))")
            XCTAssertGreaterThan(maxAmp, 0.01, "Track \(idx) should have meaningful audio")
        }
        
        print("✓ Speaker separation completed")
    }
    
    /// Actor to collect progress events safely
    private actor SeparationProgressCollector {
        private var events: [Double] = []
        
        func record(_ percent: Double) {
            events.append(percent)
        }
        
        func snapshot() -> [Double] { events }
    }
    
    /// Test speaker separation progress reporting
    func testSpeakerSeparationProgressReporting() async throws {
        print("\n=== Speaker Separation Progress Reporting ===")
        
        let url = try harryPotterURL()
        let audio8k = try loadAudio(at: url, sampleRate: 8000)
        
        // Use first 10 seconds of audio (enough for chunking to trigger)
        let testDuration = min(10.0, audio8k.duration)
        let testAudio = audio8k.slice(0.0..<testDuration)
        print("Test audio: \(String(format: "%.1f", testDuration))s")
        
        let separator = MLXProviders.mossformer2SS(model: .twoSpeakerWHAMR)
        try await separator.load()
        
        // Collect progress events using actor
        let collector = SeparationProgressCollector()
        let progressCallback: ProgressCallback = { percent in
            await collector.record(percent)
        }
        
        let tracks = try await separator.separate(testAudio, speakers: 2, onProgress: progressCallback)
        
        let progressEvents = await collector.snapshot()
        print("Progress events: \(progressEvents.count)")
        if !progressEvents.isEmpty {
            print("First: \(String(format: "%.1f", progressEvents.first!))%, Last: \(String(format: "%.1f", progressEvents.last!))%")
        }
        
        XCTAssertEqual(tracks.count, 2, "Should produce 2 speaker tracks")
        XCTAssertGreaterThan(progressEvents.count, 0, "Should emit progress events")
        XCTAssertTrue(progressEvents.contains { $0 >= 100.0 }, "Should reach 100%")
        
        // Verify monotonic progress
        for i in 1..<progressEvents.count {
            XCTAssertGreaterThanOrEqual(progressEvents[i], progressEvents[i-1], 
                "Progress should be monotonically increasing")
        }
        
        print("✓ Progress reporting test completed")
    }
    
    /// Test conditional speaker separation based on overlap count
    func testConditionalSpeakerSeparation_HarryPotter() async throws {
        print("\n=== Conditional Speaker Separation Pipeline ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        let diarizer = FluidAudioProviders.sortformerLowLatency()
        
        try await vad.load()
        try await diarizer.load()
        
        let voice = ClearVoice(
            configuration: .default,
            vad: vad,
            diarization: diarizer
        )
        
        // Register separator for 2-speaker WHAMR
        let separator = MLXProviders.mossformer2SS(model: .twoSpeakerWHAMR)
        try await separator.load()
        await voice.register(separator: separator, for: .mossformerWhamr)
        
        let tracker = ProgressTracker()
        let result = try await voice.pipeline()
            .detect(.silero)
            .diarize()
            .separateOverlappingSpeakers(useOriginal: true)
            .onEvent { event in
                await tracker.record(event: event)
            }
            .process(audio: audio16k)
        
        let snapshot = await tracker.snapshot()
        
        print("VAD segments: \(result.analysis?.speechSegments.count ?? 0)")
        print("Speakers: \(result.analysis?.speakers.speakerCount ?? 0)")
        print("Max overlap: \(result.analysis?.speakers.maxOverlappingSpeakers ?? 0)")
        
        if let tracks = result.separatedTracks {
            print("Separated tracks: \(tracks.count)")
            for (idx, track) in tracks.enumerated() {
                print("  Track \(idx): \(String(format: "%.2f", track.duration))s")
            }
        } else {
            print("No separation performed (overlap count may be < 2 or >= 4)")
        }
        
        // Verify progress events
        let separationProgress = snapshot.progressEvents.filter { $0.stage == "separation" }
        print("Separation progress events: \(separationProgress.count)")
        
        print("✓ Conditional separation pipeline completed")
    }
    
    // MARK: - Parallel Transcription + Diarization Tests
    
    /// Test parallel execution of transcription and diarization with merge
    /// This is the recommended pattern for speaker-attributed transcription
    func testParallelTranscriptionDiarization_Sortformer_HarryPotter() async throws {
        print("\n=== Parallel Transcription + Diarization (Sortformer) ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        assertSampleRate(audio16k, expected: 16000, context: "Pipeline input")
        
        let duration = audio16k.duration
        print("Audio duration: \(String(format: "%.1f", duration))s")
        
        // Initialize providers
        let diarizer = FluidAudioProviders.sortformerLowLatency()
        let transcriber = FluidAudioProviders.parakeetTranscriber()
        
        // Load models in parallel
        let loadStart = Date()
        async let diarizerLoad: Void = diarizer.load()
        async let transcriberLoad: Void = transcriber.load()
        try await diarizerLoad
        try await transcriberLoad
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Models loaded in \(String(format: "%.2f", loadTime))s (parallel)")
        
        // Create ClearVoice instance with both providers
        let voice = ClearVoice(
            configuration: .default,
            diarization: diarizer,
            transcriber: (TranscriptionModel.parakeet, transcriber)
        )
        
        // Track events
        let tracker = ProgressTracker()
        
        // Run parallel pipeline: transcription + diarization, then merge
        let pipelineStart = Date()
        let result = try await voice.pipeline()
            .parallel {[
                // Branch 1: Transcribe
                PipelineBuilder().transcribe(TranscriptionModel.parakeet),
                // Branch 2: Diarize
                PipelineBuilder().diarize()
            ]}
            .mergeTranscriptionWithDiarization()
            .onEvent { event in
                await tracker.record(event: event)
            }
            .process(audio: audio16k)
        let pipelineTime = Date().timeIntervalSince(pipelineStart)
        let pipelineRTF = duration / max(pipelineTime, 0.001)
        
        print("Pipeline completed in \(String(format: "%.2f", pipelineTime))s (RTF: \(String(format: "%.1f", pipelineRTF))x)")
        
        // Verify results
        XCTAssertNotNil(result.transcription, "Should have transcription")
        XCTAssertNotNil(result.analysis, "Should have diarization")
        XCTAssertNotNil(result.diarizedTranscription, "Should have merged result")
        
        if let diarizedTranscription = result.diarizedTranscription {
            print("\n--- Diarized Transcription ---")
            print("Total segments: \(diarizedTranscription.segments.count)")
            print("Unique speakers: \(diarizedTranscription.speakerCount)")
            print("Language: \(diarizedTranscription.language ?? "unknown")")
            
            let overlapSegments = diarizedTranscription.uncertainSegments()
            print("Overlap segments: \(overlapSegments.count)")
            
            // Print first 10 segments
            print("\nFirst 10 segments:")
            for (idx, segment) in diarizedTranscription.segments.prefix(10).enumerated() {
                let speaker = segment.speakerID?.id ?? "?"
                let overlapMarker = segment.isOverlapRegion ? " [OVERLAP]" : ""
                let confidence = String(format: "%.0f%%", segment.attributionConfidence * 100)
                print("  [\(idx)] \(speaker) (\(confidence)): \"\(segment.text)\"\(overlapMarker)")
            }
        }
        
        // Print diarization summary
        if let analysis = result.analysis {
            print("\n--- Diarization Summary ---")
            print("Speakers: \(analysis.speakers.speakerCount)")
            print("Segments: \(analysis.speakers.segments.count)")
            print("Max overlap: \(analysis.speakers.maxOverlappingSpeakers)")
        }
        
        print("\n✓ Parallel transcription + diarization completed")
    }
    
    /// Test parallel transcription + diarization with Pyannote
    func testParallelTranscriptionDiarization_Pyannote_HarryPotter() async throws {
        print("\n=== Parallel Transcription + Diarization (Pyannote) ===")
        
        let url = try harryPotterURL()
        let audio16k = try loadAudio(at: url, sampleRate: 16000)
        assertSampleRate(audio16k, expected: 16000, context: "Pipeline input")
        
        let duration = audio16k.duration
        print("Audio duration: \(String(format: "%.1f", duration))s")
        
        // Initialize providers
        let diarizer = FluidAudioProviders.pyannote()
        let transcriber = FluidAudioProviders.parakeetTranscriber()
        
        // Load models in parallel
        let loadStart = Date()
        async let diarizerLoad: Void = diarizer.load()
        async let transcriberLoad: Void = transcriber.load()
        try await diarizerLoad
        try await transcriberLoad
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Models loaded in \(String(format: "%.2f", loadTime))s (parallel)")
        
        // Create ClearVoice instance
        let voice = ClearVoice(
            configuration: .default,
            diarization: diarizer,
            transcriber: (TranscriptionModel.parakeet, transcriber)
        )
        
        // Run pipeline
        let pipelineStart = Date()
        let result = try await voice.pipeline()
            .parallel {[
                PipelineBuilder().transcribe(TranscriptionModel.parakeet),
                PipelineBuilder().diarize()
            ]}
            .mergeTranscriptionWithDiarization()
            .process(audio: audio16k)
        let pipelineTime = Date().timeIntervalSince(pipelineStart)
        let pipelineRTF = duration / max(pipelineTime, 0.001)
        
        print("Pipeline completed in \(String(format: "%.2f", pipelineTime))s (RTF: \(String(format: "%.1f", pipelineRTF))x)")
        
        // Verify results
        XCTAssertNotNil(result.transcription)
        XCTAssertNotNil(result.analysis)
        XCTAssertNotNil(result.diarizedTranscription)
        
        if let diarizedTranscription = result.diarizedTranscription {
            print("Segments: \(diarizedTranscription.segments.count)")
            print("Speakers: \(diarizedTranscription.speakerCount)")
            print("Overlaps: \(diarizedTranscription.uncertainSegments().count)")
        }
        
        print("✓ Parallel transcription + diarization (Pyannote) completed")
    }
    
    /// Test full pipeline: VAD → Parallel(Transcription + Diarization) → Merge → SE
    func testFullPipelineWithTranscription_Sortformer_HarryPotter() async throws {
        print("\n=== Full Pipeline with Transcription (Sortformer) ===")
        
        let url = try harryPotterURL()
        let audio = try loadAudioAtRates(url: url)
        
        let duration = audio.audio16k.duration
        print("Audio duration: \(String(format: "%.1f", duration))s")
        
        // Initialize all providers
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        let diarizer = FluidAudioProviders.sortformerLowLatency()
        let transcriber = FluidAudioProviders.parakeetTranscriber()
        let enhancer = MLXProviders.mossformer2SE48K()
        
        // Load all models
        let loadStart = Date()
        try await vad.load()
        try await diarizer.load()
        try await transcriber.load()
        try await enhancer.load()
        let loadTime = Date().timeIntervalSince(loadStart)
        print("All models loaded in \(String(format: "%.2f", loadTime))s")
        
        // Create ClearVoice instance
        let voice = ClearVoice(
            configuration: .default,
            vad: vad,
            diarization: diarizer,
            enhancer: (EnhancementModel.mossformerSE48k, enhancer),
            transcriber: (TranscriptionModel.parakeet, transcriber)
        )
        
        // Run full pipeline
        let pipelineStart = Date()
        let result = try await voice.pipeline()
            .detect(VADModel.silero)
            .parallel {[
                PipelineBuilder().transcribe(TranscriptionModel.parakeet),
                PipelineBuilder().diarize()
            ]}
            .mergeTranscriptionWithDiarization()
            .enhance(EnhancementModel.mossformerSE48k)
            .process(audio: audio.audio16k)
        let pipelineTime = Date().timeIntervalSince(pipelineStart)
        let pipelineRTF = duration / max(pipelineTime, 0.001)
        
        print("Full pipeline completed in \(String(format: "%.2f", pipelineTime))s (RTF: \(String(format: "%.1f", pipelineRTF))x)")
        
        // Verify all results
        XCTAssertNotNil(result.analysis?.speechSegments, "Should have VAD segments")
        XCTAssertNotNil(result.transcription, "Should have transcription")
        XCTAssertNotNil(result.analysis?.speakers, "Should have diarization")
        XCTAssertNotNil(result.diarizedTranscription, "Should have diarized transcription")
        XCTAssertNotNil(result.audio, "Should have enhanced audio")
        
        print("VAD segments: \(result.analysis?.speechSegments.count ?? 0)")
        print("Speakers: \(result.analysis?.speakers.speakerCount ?? 0)")
        print("Transcript segments: \(result.diarizedTranscription?.segments.count ?? 0)")
        print("Enhanced audio: \(String(format: "%.1f", result.audio?.duration ?? 0))s")
        
        print("✓ Full pipeline with transcription completed")
    }
}
