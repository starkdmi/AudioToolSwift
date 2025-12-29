//
//  main.swift
//  Generate
//
//  CLI tool for testing ClearVoice MLX providers with chunking
//  
//  Build: xcodebuild build -scheme Generate -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
//  Run: .build/DerivedData/Build/Products/Release/Generate --model frcrn --input test.wav --output enhanced.wav
//

import Foundation
import ClearVoice
import ClearVoiceCore
import ClearVoiceMLX
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
    ClearVoice Generate - MLX Provider CLI with Chunking
    
    Usage: Generate --model <model> --input <path> [--output <path>] [--weights <path>]
    
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
      Generate -m frcrn -i test.wav -o enhanced.wav -w Weights/frcrn_se_16k.safetensors
      Generate -m demucs -i song.wav -o vocals.wav -w ../Models/demucs_mlx_swift/Weights
    """)
}

// Validate input (skip for TTS which doesn't need input audio)
guard !inputPath.isEmpty || model.lowercased() == "kokoro" || model.lowercased() == "tts" else {
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

// MARK: - FRCRN with Chunking

func runFRCRN(inputPath: String, outputPath: String, weightsPath: String?) async throws {
    print("\n=== FRCRN SE 16K with Chunking ===")
    print("Chunking: 4s chunks, 25% overlap, discard-edges")
    
    // Resolve weights path
    let resolvedWeights = weightsPath ?? "../Models/frcrn_se_mlx_swift/Weights/frcrn_se_16k.safetensors"
    guard FileManager.default.fileExists(atPath: resolvedWeights) else {
        print("Error: Weights not found at: \(resolvedWeights)")
        return
    }
    
    // Create provider
    let provider = FRCRNSE16KProvider(weightsPath: resolvedWeights)
    
    // Load model
    print("Loading model from: \(resolvedWeights)")
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
    
    // Resolve weights path
    let resolvedWeights = weightsPath ?? "../Models/frcrn_se_mlx_swift/Weights/frcrn_se_16k.safetensors"
    guard FileManager.default.fileExists(atPath: resolvedWeights) else {
        print("Error: Weights not found at: \(resolvedWeights)")
        return
    }
    
    // Create provider
    let provider = FRCRNSE16KProvider(weightsPath: resolvedWeights)
    
    // Load model
    print("Loading model from: \(resolvedWeights)")
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

func runMossFormer2SE(inputPath: String, outputPath: String) async throws {
    print("\n=== MossFormer2 SE 48K with Chunking ===")
    print("Chunking: 4s chunks, 25% overlap, discard-edges")
    
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
    
    // Resolve weights directory
    let resolvedWeights = weightsPath ?? "../Models/demucs_mlx_swift/Weights"
    guard FileManager.default.fileExists(atPath: resolvedWeights) else {
        print("Error: Weights directory not found at: \(resolvedWeights)")
        return
    }
    
    // Create provider
    let provider = DemucsProvider(weightsDirectory: resolvedWeights)
    
    // Load vocals model
    print("Loading vocals model from: \(resolvedWeights)")
    try await provider.load(source: .vocals)
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
    let output = try await provider.separate(input, source: .vocals)
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
        print("Unknown SS variant: \(variant). Available: 2spk, 3spk, whamr")
        return
    }
    
    print("\n=== MossFormer2 SS \(modelType.rawValue) with Chunking ===")
    print("Chunking: 4s chunks, 25% overlap, triangular blending")
    
    // Find weights in HuggingFace cache - use huggingFaceRepo property
    // Convert "starkdmi/Model_Name" to "models--starkdmi--Model_Name"
    let repoName = modelType.huggingFaceRepo.replacingOccurrences(of: "/", with: "--")
    let hfCache = NSHomeDirectory() + "/.cache/huggingface/hub/models--\(repoName)"
    let snapshotsDir = hfCache + "/snapshots"
    
    guard let snapshots = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir),
          let firstSnapshot = snapshots.first(where: { !$0.hasPrefix(".") }) else {
        print("Error: Weights not found in HuggingFace cache at: \(hfCache)")
        print("Run: huggingface-cli download \(modelType.huggingFaceRepo)")
        return
    }
    
    let weightsPath = snapshotsDir + "/" + firstSnapshot + "/model_fp32.safetensors"
    guard FileManager.default.fileExists(atPath: weightsPath) else {
        print("Error: model_fp32.safetensors not found at: \(weightsPath)")
        return
    }
    
    let provider = MossFormer2SSProvider(model: modelType, weightsPath: weightsPath)
    
    print("Loading model...")
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
    let outputs = try await provider.separate(input, speakers: modelType.numSpeakers)
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
    
    // Find weights in HuggingFace cache
    let hfCache = NSHomeDirectory() + "/.cache/huggingface/hub/models--starkdmi--MossFormer2_SR_48K_MLX"
    let snapshotsDir = hfCache + "/snapshots"
    
    guard let snapshots = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir),
          let firstSnapshot = snapshots.first(where: { !$0.hasPrefix(".") }) else {
        print("Error: Weights not found in HuggingFace cache")
        print("Run: huggingface-cli download starkdmi/MossFormer2_SR_48K_MLX")
        return
    }
    
    let weightsPath = snapshotsDir + "/" + firstSnapshot + "/model_fp32.safetensors"
    let configPath = snapshotsDir + "/" + firstSnapshot + "/config.json"
    
    guard FileManager.default.fileExists(atPath: weightsPath),
          FileManager.default.fileExists(atPath: configPath) else {
        print("Error: model_fp32.safetensors or config.json not found")
        return
    }
    
    let provider = MossFormer2SR48KProvider(weightsPath: weightsPath, configPath: configPath)
    
    print("Loading model...")
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

// MARK: - Main Execution

func printUsageDetailed() {
    print("""
    ClearVoice Generate - MLX Provider CLI with Chunking
    
    Usage: Generate --model <model> --input <path> [--output <path>] [--weights <path>]
    
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
      Generate -m frcrn -i test.wav -o enhanced.wav -w Weights/frcrn_se_16k.safetensors
      Generate -m demucs -i song.wav -o vocals.wav -w ../Models/demucs_mlx_swift/Weights
      Generate -m ss_2spk -i mixed.wav -o separated.wav
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
            try await runMossFormer2SE(inputPath: inputPath, outputPath: outputPath)
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
        case "kokoro", "tts":
            try await runKokoroTest()
        default:
            print("Unknown model: \(model)")
            print("Available: frcrn, frcrn-bg, se48k, se48k-bg, demucs, ss_2spk, ss_3spk, ss_whamr, sr48k, kokoro")
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
