//
//  main.swift
//  AudioToolCLI
//
//  CLI tool for testing AudioTool MLX providers with chunking
//  
//  Build: xcodebuild build -scheme audio-tool -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
//  Run: .build/DerivedData/Build/Products/Release/audio-tool --model frcrn --input test.wav --output enhanced.wav
//

import Foundation
import AudioTool
import AudioToolCore
import AudioToolMLX
#if canImport(AudioToolSpeech)
import AudioToolSpeech
#endif
import MLX
import AudioUtils

// Parse arguments
var model = "frcrn"
var inputPath = ""
var outputPath = ""
var weightsPath: String? = nil

let args = CommandLine.arguments

var i = 1
while i < args.count {
    switch args[i] {
    case "--model", "-m":
        if i + 1 < args.count {
            model = args[i + 1]
            i += 2
        } else { i += 1 }
    case "--input", "-i":
        if i + 1 < args.count {
            inputPath = args[i + 1]
            i += 2
        } else { i += 1 }
    case "--output", "-o":
        if i + 1 < args.count {
            outputPath = args[i + 1]
            i += 2
        } else { i += 1 }
    case "--weights", "-w":
        if i + 1 < args.count {
            weightsPath = args[i + 1]
            i += 2
        } else { i += 1 }
    case "--help", "-h":
        printUsage()
        exit(0)
    default:
        i += 1
    }
}

func printUsage() {
    print("""
    audio-tool - AudioToolSwift command line interface
    
    Usage: audio-tool --model <model> --input <path> [--output <path>] [--weights <path>]
    
    Models:
      frcrn          FRCRN SE 16K (4s/25%/discard-edges)
      mossformer2_se MossFormer2 SE 48K (4s/25%/discard-edges)
      demucs         Demucs vocals separation (7.8s/25%/triangular)
    
    Options:
      -m, --model    Model to use (default: frcrn)
      -i, --input    Input audio path (required)
      -o, --output   Output audio path (default: input_chunked.wav)
      -w, --weights  Weights path/directory
      -h, --help     Show this help
    
    Example:
      audio-tool -m frcrn -i test.wav -o enhanced.wav
      audio-tool -m demucs -i song.wav -o vocals.wav

    Weights download automatically from HuggingFace on first use.
    --weights is only needed to point at a local override.
    """)
}

// Validate input. Every remaining subcommand transforms an input file; the TTS
// and diarization scratch commands that used to be exempt here were removed
// along with their hardcoded paths into a machine that no longer exists.
guard !inputPath.isEmpty else {
    print("Error: Input path required (--input)")
    printUsage()
    exit(1)
}

// Default output path
if outputPath.isEmpty {
    let url = URL(fileURLWithPath: inputPath)
    let name = url.deletingPathExtension().lastPathComponent
    outputPath = url.deletingLastPathComponent().appendingPathComponent("\(name)_chunked.wav").path
}

// MARK: - Weights Resolution

/// Failures a run can report to the shell.
///
/// Every one of these used to be `print(...)` followed by `return`, which meant the
/// tool announced a failure and then exited 0 - a script could not tell a completed
/// enhancement from a missing model.
enum CLIError: LocalizedError {
    case weightsNotFound(String)
    case unknownVariant(String, available: String)
    case noTranscriptionLocale

    var errorDescription: String? {
        switch self {
        case .weightsNotFound(let path):
            return "Weights not found at: \(path)"
        case .unknownVariant(let variant, let available):
            return "Unknown variant '\(variant)'. Available: \(available)"
        case .noTranscriptionLocale:
            return "No speech-recognition locale is available on this system"
        }
    }
}

/// Validate an explicit `--weights` path, if one was given.
///
/// `nil` means "let the provider fetch it", which is what the help text has always
/// promised. Several runners instead defaulted to a path relative to the working
/// directory - `../Models/frcrn_se_mlx_swift/Weights/...`, a sibling of a checkout
/// that no installed copy of this tool has - and others hand-assembled a path into
/// `~/.cache/huggingface`, bypassing the downloader that knows how to populate it.
func explicitWeights(_ path: String?, isDirectory: Bool = false) throws -> String? {
    guard let path else { return nil }
    var directory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &directory),
          directory.boolValue == isDirectory else {
        throw CLIError.weightsNotFound(path)
    }
    return path
}

/// FRCRN from an explicit path, or downloading on first use.
func makeFRCRN(weightsPath: String?) throws -> FRCRNSE16KProvider {
    if let path = try explicitWeights(weightsPath) {
        return FRCRNSE16KProvider(weightsPath: path)
    }
    return FRCRNSE16KProvider()
}

// MARK: - FRCRN with Chunking

func runFRCRN(inputPath: String, outputPath: String, weightsPath: String?) async throws {
    print("\n=== FRCRN SE 16K with Chunking ===")
    print("Chunking: 4s chunks, 25% overlap, discard-edges")
    
    // Create provider - downloads from HuggingFace unless --weights points elsewhere
    let provider = try makeFRCRN(weightsPath: weightsPath)

    // Load model
    print("Loading model\(weightsPath.map { " from: \($0)" } ?? " (downloading if needed)")")
    try await provider.load()
    print("Model ready")
    
    // Load audio
    print("Loading audio from: \(inputPath)")
    let loaderConfig = AudioLoader.Configuration(targetSampleRate: 16000)
    let loader = AudioLoader(config: loaderConfig)
    let audio = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
    eval(audio)
    let samples = audio.asArray(Float.self)
    
    let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
    print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
    
    // Process (chunking is auto-enabled)
    print("Processing with chunking...")
    let startTime = Date()
    let output = try await provider.process(input)
    let processingTime = Date().timeIntervalSince(startTime)
    
    let rtf = input.duration / processingTime
    print("Output: \(output.samples.count) samples (\(String(format: "%.2f", output.duration))s)")
    print("RTF: \(String(format: "%.2f", rtf))x")
    
    // Save
    let saverConfig = AudioSaver.Configuration(sampleRate: 16000)
    let saver = AudioSaver(config: saverConfig)
    try saver.save(MLXArray(output.samples), to: outputPath)
    print("✓ Saved: \(outputPath)")
}

// MARK: - FRCRN with Background Extraction

func runFRCRNWithBackground(inputPath: String, outputPath: String, weightsPath: String?) async throws {
    print("\n=== FRCRN SE 16K with Background Extraction ===")
    
    // Create provider - downloads from HuggingFace unless --weights points elsewhere
    let provider = try makeFRCRN(weightsPath: weightsPath)

    // Load model
    print("Loading model\(weightsPath.map { " from: \($0)" } ?? " (downloading if needed)")")
    try await provider.load()
    print("Model ready")
    
    // Load audio
    print("Loading audio from: \(inputPath)")
    let loaderConfig = AudioLoader.Configuration(targetSampleRate: 16000)
    let loader = AudioLoader(config: loaderConfig)
    let audio = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
    eval(audio)
    let samples = audio.asArray(Float.self)
    
    let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
    print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
    
    // Process with background extraction (time-domain subtraction)
    print("Processing with background extraction...")
    let startTime = Date()
    let result = try await provider.processWithBackground(input)
    let processingTime = Date().timeIntervalSince(startTime)
    
    let rtf = input.duration / processingTime
    print("Enhanced: \(result.enhanced.samples.count) samples, max: \(String(format: "%.4f", result.enhanced.samples.max() ?? 0))")
    print("Background: \(result.background.samples.count) samples, max: \(String(format: "%.4f", result.background.samples.max() ?? 0))")
    print("RTF: \(String(format: "%.2f", rtf))x")
    
    // Save both
    let saverConfig = AudioSaver.Configuration(sampleRate: 16000)
    let saver = AudioSaver(config: saverConfig)
    
    let basePath = (outputPath as NSString).deletingPathExtension
    let enhancedPath = basePath + "_enhanced.wav"
    let backgroundPath = basePath + "_background.wav"
    
    try saver.save(MLXArray(result.enhanced.samples), to: enhancedPath)
    try saver.save(MLXArray(result.background.samples), to: backgroundPath)
    print("✓ Saved: \(enhancedPath)")
    print("✓ Saved: \(backgroundPath)")
}

// MARK: - MossFormer2 SE with Chunking

func runMossFormer2SE(inputPath: String, outputPath: String, weightsPath: String?) async throws {
    print("\n=== MossFormer2 SE 48K with Chunking ===")
    print("Chunking: 4s chunks, 25% overlap, discard-edges")

    // `--weights` was accepted and then ignored here, unlike every neighbouring
    // command. That mattered once quantized checkpoints existed: the precision is
    // carried by the file, so pointing at `model_int8.safetensors` is the only way
    // to ask for int8 without a download through the catalog.
    let provider = weightsPath.map { MossFormer2SE48KProvider(weightsPath: $0) }
        ?? MLXProviders.mossformer2SE48K()
    
    // Load model
    print("Loading model (may download from HuggingFace)...")
    try await provider.load()
    print("Model ready")
    
    // Load audio
    print("Loading audio from: \(inputPath)")
    let loaderConfig = AudioLoader.Configuration(targetSampleRate: 48000)
    let loader = AudioLoader(config: loaderConfig)
    let audio = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
    eval(audio)
    let samples = audio.asArray(Float.self)
    
    let input = AudioBuffer(samples: samples, sampleRate: 48000, channels: 1)
    print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
    
    // Process (chunking is auto-enabled for >4s audio)
    print("Processing with chunking...")
    let startTime = Date()
    let output = try await provider.process(input)
    let processingTime = Date().timeIntervalSince(startTime)
    
    let rtf = input.duration / processingTime
    print("Output: \(output.samples.count) samples (\(String(format: "%.2f", output.duration))s)")
    print("RTF: \(String(format: "%.2f", rtf))x")
    
    // Save
    let saverConfig = AudioSaver.Configuration(sampleRate: 48000)
    let saver = AudioSaver(config: saverConfig)
    try saver.save(MLXArray(output.samples), to: outputPath)
    print("✓ Saved: \(outputPath)")
}

// MARK: - MossFormer2 SE with Background Extraction

func runMossFormer2SEWithBackground(inputPath: String, outputPath: String) async throws {
    print("\n=== MossFormer2 SE 48K with Background Extraction ===")
    
    // Create provider (downloads from HuggingFace)
    let provider = MLXProviders.mossformer2SE48K()
    
    // Load model
    print("Loading model (may download from HuggingFace)...")
    try await provider.load()
    print("Model ready")
    
    // Load audio
    print("Loading audio from: \(inputPath)")
    let loaderConfig = AudioLoader.Configuration(targetSampleRate: 48000)
    let loader = AudioLoader(config: loaderConfig)
    let audio = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
    eval(audio)
    let samples = audio.asArray(Float.self)
    
    let input = AudioBuffer(samples: samples, sampleRate: 48000, channels: 1)
    print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
    
    // Process with background extraction
    print("Processing with background extraction...")
    let startTime = Date()
    let result = try await provider.processWithBackground(input)
    let processingTime = Date().timeIntervalSince(startTime)
    
    let rtf = input.duration / processingTime
    print("Enhanced: \(result.enhanced.samples.count) samples, max: \(String(format: "%.4f", result.enhanced.samples.max() ?? 0))")
    print("Background: \(result.background.samples.count) samples, max: \(String(format: "%.4f", result.background.samples.max() ?? 0))")
    print("RTF: \(String(format: "%.2f", rtf))x")
    
    // Save both
    let saverConfig = AudioSaver.Configuration(sampleRate: 48000)
    let saver = AudioSaver(config: saverConfig)
    
    let basePath = (outputPath as NSString).deletingPathExtension
    let enhancedPath = basePath + "_enhanced.wav"
    let backgroundPath = basePath + "_background.wav"
    
    try saver.save(MLXArray(result.enhanced.samples), to: enhancedPath)
    try saver.save(MLXArray(result.background.samples), to: backgroundPath)
    print("✓ Saved: \(enhancedPath)")
    print("✓ Saved: \(backgroundPath)")
}

// MARK: - Demucs with Chunking

func runDemucs(inputPath: String, outputPath: String, weightsPath: String?) async throws {
    print("\n=== Demucs Vocals Separation with Chunking ===")
    print("Chunking: 7.8s chunks, 25% overlap, triangular blending")
    
    // Create provider - downloads from HuggingFace unless --weights points elsewhere
    let provider: DemucsProvider
    if let directory = try explicitWeights(weightsPath, isDirectory: true) {
        provider = DemucsProvider(weightsDirectory: directory)
    } else {
        provider = DemucsProvider()
    }

    // Load vocals model - only this stem's weights are fetched
    print("Loading vocals model\(weightsPath.map { " from: \($0)" } ?? " (downloading if needed)")")
    try await provider.load(stem: .vocals)
    print("Model ready")
    
    // Load audio (Demucs uses 44100Hz)
    print("Loading audio from: \(inputPath)")
    let loaderConfig = AudioLoader.Configuration(targetSampleRate: 44100)
    let loader = AudioLoader(config: loaderConfig)
    let audio = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
    eval(audio)
    let samples = audio.asArray(Float.self)
    
    let input = AudioBuffer(samples: samples, sampleRate: 44100, channels: 1)
    print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
    
    // Process (chunking is auto-enabled for >7.8s audio)
    print("Processing with chunking...")
    let startTime = Date()
    let output = try await provider.separate(input, stem: .vocals)
    let processingTime = Date().timeIntervalSince(startTime)
    
    let rtf = input.duration / processingTime
    print("Output: \(output.samples.count) samples (\(String(format: "%.2f", output.duration))s)")
    print("RTF: \(String(format: "%.2f", rtf))x")
    
    // Save
    let saverConfig = AudioSaver.Configuration(sampleRate: 44100)
    let saver = AudioSaver(config: saverConfig)
    try saver.save(MLXArray(output.samples), to: outputPath)
    print("✓ Saved: \(outputPath)")
}

// MARK: - MossFormer2 SS (Speaker Separation)

func runMossFormer2SS(inputPath: String, outputPath: String, variant: String) async throws {
    let modelType: MossFormer2SSProvider.Model
    switch variant.lowercased() {
    case "2spk", "2speaker": modelType = .twoSpeaker
    case "3spk", "3speaker": modelType = .threeSpeaker
    case "whamr": modelType = .twoSpeakerWHAMR
    default:
        throw CLIError.unknownVariant(variant, available: "2spk, 3spk, whamr")
    }
    
    print("\n=== MossFormer2 SS \(modelType.rawValue) with Chunking ===")
    print("Chunking: 4s chunks, 25% overlap, triangular blending")
    
    // The provider fetches and caches its own weights. This used to reach into
    // ~/.cache/huggingface, pick whichever snapshot directory sorted first, and tell
    // the user to run `huggingface-cli` when it found nothing.
    let provider = MossFormer2SSProvider(model: modelType)

    print("Loading model (downloading if needed)...")
    try await provider.load()
    print("Model ready")
    
    // Load audio - IMPORTANT: normalizationMode: .none to preserve original volume
    print("Loading audio from: \(inputPath)")
    let loaderConfig = AudioLoader.Configuration(
        targetSampleRate: Double(modelType.sampleRate),
        normalizationMode: .none
    )
    let loader = AudioLoader(config: loaderConfig)
    let audio = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
    eval(audio)
    let samples = audio.asArray(Float.self)
    
    let input = AudioBuffer(samples: samples, sampleRate: modelType.sampleRate, channels: 1)
    print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
    
    print("Processing with chunking...")
    let startTime = Date()
    let outputs = try await provider.separate(input)
    let processingTime = Date().timeIntervalSince(startTime)
    
    let rtf = input.duration / processingTime
    print("Output: \(outputs.count) speakers, \(outputs.first?.samples.count ?? 0) samples each")
    print("RTF: \(String(format: "%.2f", rtf))x")
    
    // Save each speaker (provider already normalizes output to peak 1.0)
    let saverConfig = AudioSaver.Configuration(sampleRate: Double(modelType.sampleRate))
    let saver = AudioSaver(config: saverConfig)
    
    let baseOutput = URL(fileURLWithPath: outputPath).deletingPathExtension().path
    for (i, output) in outputs.enumerated() {
        let speakerPath = "\(baseOutput)_speaker\(i + 1).wav"
        try saver.save(MLXArray(output.samples), to: speakerPath)
        print("✓ Saved: \(speakerPath)")
    }
}

// MARK: - MossFormer2 SR (Super Resolution)

func runMossFormer2SR(inputPath: String, outputPath: String) async throws {
    print("\n=== MossFormer2 SR 48K with Chunking ===")
    print("Chunking: 4s chunks, 50% overlap, Hann window")
    print("Super-resolution: 16kHz → 48kHz")
    
    // The provider fetches and caches its own weights and config.
    let provider = MossFormer2SR48KProvider()

    print("Loading model (downloading if needed)...")
    try await provider.load()
    print("Model ready")
    
    // Load audio at 16kHz (input sample rate for SR)
    print("Loading audio from: \(inputPath)")
    let loaderConfig = AudioLoader.Configuration(
        targetSampleRate: 16000,
        normalizationMode: .none
    )
    let loader = AudioLoader(config: loaderConfig)
    let audio = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
    eval(audio)
    let samples = audio.asArray(Float.self)
    
    let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
    print("Input: \(input.samples.count) samples at 16kHz (\(String(format: "%.2f", input.duration))s)")
    
    print("Processing with chunking (upsampling to 48kHz)...")
    let startTime = Date()
    let output = try await provider.process(input)
    let processingTime = Date().timeIntervalSince(startTime)
    
    let rtf = input.duration / processingTime
    print("Output: \(output.samples.count) samples at 48kHz (\(String(format: "%.2f", output.duration))s)")
    print("RTF: \(String(format: "%.2f", rtf))x")
    
    // Save at 48kHz
    let saverConfig = AudioSaver.Configuration(sampleRate: 48000)
    let saver = AudioSaver(config: saverConfig)
    try saver.save(MLXArray(output.samples), to: outputPath)
    print("✓ Saved: \(outputPath)")
}

// MARK: - Apple Speech Transcription

#if canImport(AudioToolSpeech)
@available(iOS 26.0, macOS 26.0, *)
func runTranscribe(inputPath: String) async throws {
    print("\n=== Apple Speech Transcription ===")
    
    // Check supported locales
    let locales = await AppleSpeechTranscriber.supportedLocales()
    print("Supported locales: \(locales.count)")
    print("Available: \(locales.map { $0.identifier }.joined(separator: ", "))")
    
    // Prefer en_GB (UK) since that's what user has installed for Dictation
    // Fall back to en_US, then any English, then first available
    let localeToUse: Locale
    
    if let enGB = locales.first(where: { $0.identifier == "en_GB" || $0.identifier == "en-GB" }) {
        localeToUse = enGB
        print("Using UK English: \(enGB.identifier)")
    } else if let enUS = locales.first(where: { $0.identifier == "en_US" || $0.identifier == "en-US" }) {
        localeToUse = enUS
        print("Using US English: \(enUS.identifier)")
    } else if let enLocale = locales.first(where: { $0.identifier.starts(with: "en") }) {
        localeToUse = enLocale
        print("Using English locale: \(enLocale.identifier)")
    } else if let firstLocale = locales.first {
        localeToUse = firstLocale
        print("No English locale, using: \(firstLocale.identifier)")
    } else {
        throw CLIError.noTranscriptionLocale
    }
    
    // Load audio at 16kHz
    print("Loading audio from: \(inputPath)")
    let loaderConfig = AudioLoader.Configuration(targetSampleRate: 16000, normalizationMode: .none)
    let loader = AudioLoader(config: loaderConfig)
    let audio = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
    eval(audio)
    let samples = audio.asArray(Float.self)
    
    let audioBuffer = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
    print("Audio: \(String(format: "%.2f", audioBuffer.duration))s at 16kHz")
    
    // Create and load transcriber
    let transcriber = AppleSpeechTranscriber(locale: localeToUse)
    print("Loading speech model...")
    try await transcriber.load()
    print("✓ Model loaded")
    
    // Transcribe with timeout
    print("Transcribing...")
    let startTime = Date()
    
    let result = try await transcriber.transcribe(audioBuffer)
    
    let elapsed = Date().timeIntervalSince(startTime)
    
    print("\n========================================")
    print("TRANSCRIPTION RESULT")
    print("========================================")
    print("Text: \(result.text)")
    print("Language: \(result.language ?? "unknown")")
    print("Segments: \(result.segments.count)")
    print("Time: \(String(format: "%.2f", elapsed))s")
    print("========================================\n")
}
#endif

// MARK: - Streaming Verification Test

/// Test streaming vs non-streaming output quality
/// Verifies that processStream() produces identical results to process()
func runStreamingVerification(inputPath: String, outputPath: String, weightsPath: String?) async throws {
    print("\n=== Streaming Verification Test ===")
    print("Comparing process() vs processStream() output quality\n")
    
    // Create provider - downloads from HuggingFace unless --weights points elsewhere
    let provider = try makeFRCRN(weightsPath: weightsPath)

    // Load model
    print("Loading FRCRN model...")
    try await provider.load()
    print("Model ready\n")
    
    // Load audio
    print("Loading audio from: \(inputPath)")
    let loaderConfig = AudioLoader.Configuration(targetSampleRate: 16000)
    let loader = AudioLoader(config: loaderConfig)
    let audio = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
    eval(audio)
    let samples = audio.asArray(Float.self)
    
    let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
    print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)\n")
    
    // 1. Non-streaming (batch) processing
    print("--- Non-Streaming (Batch) ---")
    let batchStart = Date()
    let batchOutput = try await provider.process(input)
    let batchTime = Date().timeIntervalSince(batchStart)
    print("Batch output: \(batchOutput.samples.count) samples")
    print("Batch time: \(String(format: "%.2f", batchTime))s\n")
    
    // 2. Streaming processing
    print("--- Streaming ---")
    let streamStart = Date()
    var streamedChunks: [AudioBuffer] = []
    var chunkCount = 0
    
    for try await chunk in provider.processStream(input) {
        chunkCount += 1
        let samples = chunk.samples.count
        print("  Chunk \(chunkCount): \(samples) samples")
        streamedChunks.append(chunk)
    }
    
    let streamTime = Date().timeIntervalSince(streamStart)
    
    // Combine streamed chunks
    let streamedSamples = streamedChunks.flatMap { $0.samples }
    print("Streamed chunks: \(chunkCount)")
    print("Streamed total: \(streamedSamples.count) samples")
    print("Stream time: \(String(format: "%.2f", streamTime))s\n")
    
    // 3. Compare outputs
    print("--- Quality Comparison ---")
    let minLen = min(batchOutput.samples.count, streamedSamples.count)
    let batchPrefix = Array(batchOutput.samples.prefix(minLen))
    let streamPrefix = Array(streamedSamples.prefix(minLen))
    
    // Calculate differences
    var maxDiff: Float = 0
    var sumDiff: Float = 0
    var sumSquaredDiff: Float = 0
    
    for i in 0..<minLen {
        let diff = abs(batchPrefix[i] - streamPrefix[i])
        maxDiff = max(maxDiff, diff)
        sumDiff += diff
        sumSquaredDiff += diff * diff
    }
    
    let meanDiff = sumDiff / Float(minLen)
    let rmsDiff = sqrt(sumSquaredDiff / Float(minLen))
    
    print("Sample counts - Batch: \(batchOutput.samples.count), Stream: \(streamedSamples.count)")
    print("Max difference: \(String(format: "%.6f", maxDiff))")
    print("Mean difference: \(String(format: "%.6f", meanDiff))")
    print("RMS difference: \(String(format: "%.6f", rmsDiff))")
    
    // Quality assessment
    let isNearIdentical = maxDiff < 0.001
    let isAcceptable = maxDiff < 0.01
    
    print("\nQuality Assessment:")
    if isNearIdentical {
        print("✅ NEAR-IDENTICAL: Streaming output matches batch output (max diff < 0.001)")
    } else if isAcceptable {
        print("⚠️ ACCEPTABLE: Small differences detected (max diff < 0.01)")
    } else {
        print("❌ SIGNIFICANT DIFFERENCES: Review implementation (max diff >= 0.01)")
    }
    
    // Save both outputs for manual comparison
    let saverConfig = AudioSaver.Configuration(sampleRate: 16000)
    let saver = AudioSaver(config: saverConfig)
    
    let basePath = (outputPath as NSString).deletingPathExtension
    let batchPath = basePath + "_batch.wav"
    let streamPath = basePath + "_stream.wav"
    
    try saver.save(MLXArray(batchOutput.samples), to: batchPath)
    try saver.save(MLXArray(streamedSamples), to: streamPath)
    print("\n✓ Saved batch output: \(batchPath)")
    print("✓ Saved stream output: \(streamPath)")
    
    // Print timing comparison
    print("\n--- Timing Comparison ---")
    let rtfBatch = input.duration / batchTime
    let rtfStream = input.duration / streamTime
    print("Batch RTF: \(String(format: "%.2fx", rtfBatch))")
    print("Stream RTF: \(String(format: "%.2fx", rtfStream))")
    
    // First chunk latency (streaming benefit)
    if let firstChunkSamples = streamedChunks.first?.samples.count {
        let firstChunkDuration = Double(firstChunkSamples) / 16000.0
        print("First chunk duration: \(String(format: "%.2f", firstChunkDuration))s (streaming enables playback after first chunk)")
    }
}

// MARK: - Main Execution

func printUsageDetailed() {
    print("""
    audio-tool - AudioToolSwift command line interface
    
    Usage: audio-tool --model <model> --input <path> [--output <path>] [--weights <path>]
    
    Models:
      frcrn           FRCRN SE 16K (4s/25%/discard-edges)
      mossformer2_se  MossFormer2 SE 48K (4s/25%/discard-edges)
      demucs          Demucs vocals separation (7.8s/25%/triangular)
      ss_2spk         MossFormer2 SS 2-speaker (4s/25%/triangular)
      ss_3spk         MossFormer2 SS 3-speaker (4s/25%/triangular)
      ss_whamr        MossFormer2 SS WHAMR (4s/25%/triangular)
    
    Options:
      -m, --model    Model to use (default: frcrn)
      -i, --input    Input audio path (required)
      -o, --output   Output audio path (default: input_chunked.wav)
      -w, --weights  Weights path/directory
      -h, --help     Show this help
    
    Example:
      audio-tool -m frcrn -i test.wav -o enhanced.wav
      audio-tool -m demucs -i song.wav -o vocals.wav
      audio-tool -m ss_2spk -i mixed.wav -o separated.wav

    Weights download automatically from HuggingFace on first use.
    --weights is only needed to point at a local override.
    """)
}

Task {
    do {
        switch model.lowercased() {
        case "frcrn":
            try await runFRCRN(inputPath: inputPath, outputPath: outputPath, weightsPath: weightsPath)
        case "frcrn-bg", "frcrnbg", "frcrn_bg":
            try await runFRCRNWithBackground(inputPath: inputPath, outputPath: outputPath, weightsPath: weightsPath)
        case "mossformer2_se", "mossformer2se", "se48k":
            try await runMossFormer2SE(inputPath: inputPath, outputPath: outputPath, weightsPath: weightsPath)
        case "se48k-bg", "se48kbg", "mossformer2_se_bg":
            try await runMossFormer2SEWithBackground(inputPath: inputPath, outputPath: outputPath)
        case "demucs", "vocals":
            try await runDemucs(inputPath: inputPath, outputPath: outputPath, weightsPath: weightsPath)
        case "ss_2spk", "ss2spk", "2spk":
            try await runMossFormer2SS(inputPath: inputPath, outputPath: outputPath, variant: "2spk")
        case "ss_3spk", "ss3spk", "3spk":
            try await runMossFormer2SS(inputPath: inputPath, outputPath: outputPath, variant: "3spk")
        case "ss_whamr", "sswhamr", "whamr":
            try await runMossFormer2SS(inputPath: inputPath, outputPath: outputPath, variant: "whamr")
        case "sr48k", "sr", "mossformer2_sr":
            try await runMossFormer2SR(inputPath: inputPath, outputPath: outputPath)
        #if canImport(AudioToolSpeech)
        case "transcribe", "speech", "stt":
            if #available(macOS 26.0, *) {
                try await runTranscribe(inputPath: inputPath)
            } else {
                print("Error: Transcription requires macOS 26.0+")
                exit(1)
            }
        #endif
        case "streaming_verify", "stream_test", "verify_stream":
            try await runStreamingVerification(inputPath: inputPath, outputPath: outputPath, weightsPath: weightsPath)
        default:
            print("Unknown model: \(model)")
            print("Available: frcrn, frcrn-bg, se48k, se48k-bg, demucs, ss_2spk, ss_3spk, ss_whamr, sr48k, streaming_verify, transcribe")
            exit(1)
        }
        exit(0)
    } catch {
        print("Error: \(error)")
        exit(1)
    }
}

// Keep running until Task completes
RunLoop.main.run()
