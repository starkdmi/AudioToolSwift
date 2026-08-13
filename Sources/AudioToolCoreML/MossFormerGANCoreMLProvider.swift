//
//  MossFormerGANCoreMLProvider.swift
//  AudioToolCoreML
//
//  CoreML-based MossFormer GAN Speech Enhancement (16kHz)
//  Uses MLX for STFT/ISTFT, CoreML for model inference
//

import Foundation
@preconcurrency import CoreML
import Accelerate
import AudioTool
import AudioToolCore
import MLX
import MLXNN

// MARK: - Segment Stitching

/// How consecutive Core ML invocations are stitched back together.
///
/// Every case keeps each invocation at exactly 25500 samples. That is not a tuning
/// choice: 25500 samples is 256 STFT frames, which is the `group_size=256` the
/// inter-path attention was trained with (`generator.py:542`), and the traced graph
/// collapses grouping to a single group of whatever length it is handed. At 256 it
/// reproduces the original architecture exactly; at any other length it does not,
/// which is why 512 frames measured 0.9445 and 101 frames 0.9218 against 256's
/// 0.9894. So the segment length is fixed and only the stitching varies.
public enum CoreMLGANStitching: String, Sendable, CaseIterable {

    /// Consecutive segments, hard-concatenated. One inference per 1.594 s.
    ///
    /// The default, and the right one. Each segment is inferred with no knowledge
    /// of its neighbours (the inter-path FSMN alone reaches +/-19 frames, about
    /// 119 ms) and the reconstructions are butted together, so the joins are not
    /// mathematically continuous: 3-4 of 18 boundaries on a 30 s clip carry a step,
    /// the worst a single-sample jump of 40% of signal RMS.
    ///
    /// That reads worse than it is. A one-sample step scores enormously on any
    /// derivative measure and carries almost no energy. Against `discardEdges25`
    /// the whole output differs by -32 dB, mostly below 500 Hz, and only 4.7% of
    /// that difference falls within +/-50 ms of a boundary - which is less than the
    /// 6.0% of the file those windows occupy, so the residue is not seam-related at
    /// all. Blind A/B on 6 s and 30 s clips: no audible difference, at the
    /// transitions or overall. See Docs/chunking_research.md section 2d.
    case none

    /// 25% overlap, centre of each segment kept, edges discarded.
    ///
    /// The policy the original PyTorch decode path uses for long audio
    /// (`helpers.py`: `stride = window * 0.75`, then give up `(window - stride) / 2`
    /// at each side). Discards rather than blends, so there are no crossfade
    /// weights to normalise and no way to reintroduce the `sum(w**2)` bug that made
    /// overlap look bad in the first place. Removes the step discontinuities; costs
    /// 33% more inference (measured 3.85x -> 3.08x real time).
    ///
    /// Opt in only if you want the mathematically clean reconstruction or hit
    /// content where it matters. It is not the default because it is not audible:
    /// see the note on ``none``.
    case discardEdges25

    /// Fraction of the segment advanced between invocations.
    var strideFraction: Double {
        switch self {
        case .none: return 1.0
        case .discardEdges25: return 0.75
        }
    }
}

// MARK: - MossFormer GAN CoreML Provider

/// CoreML MossFormer GAN Speech Enhancement (16kHz)
/// Uses Accelerate for STFT/ISTFT and CoreML for model inference
public actor MossFormerGANCoreMLProvider: SpeechEnhancer {
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    public nonisolated let minChunkSize: Int = 3200   // 0.2s at 16kHz
    public nonisolated let recommendedChunkSize: Int = 25500  // 1.594s at 16kHz (256 frames)
    
    // Audio parameters (must match model training)
    private let nFFT: Int = 400
    private let hopLength: Int = 100
    private let winLength: Int = 400
    private let powerCompress: Float = 0.3
    
    // Segment duration: 256 frames = 25500 samples = 1.594s
    private let segmentSamples: Int = 25500
    
    private var model: MLModel?
    private let modelPath: String?
    private let precision: CoreMLGANPrecision
    private let computeUnits: MLComputeUnits
    private let stitching: CoreMLGANStitching

    /// Where the compiled packages come from when no path is given.
    public static let repo = ModelRepository.mossFormerGANSE16KCoreML

    /// The conversions the repository publishes. Core ML fixes precision when the
    /// package is compiled, so these are two files rather than two ways of running
    /// one - see ``CoreMLGANPrecision``.
    public static let supportedPrecisions = CoreMLGANPrecision.allCases

    /// One load at a time; see ``ModelLoadGate``.
    private let loadGate = ModelLoadGate()
    
    // MLX STFT window
    private let mlxWindow: MLXArray
    
    /// - Parameters:
    ///   - modelPath: An `.mlpackage` (or compiled `.mlmodelc`) to load instead of
    ///     the published one. `nil` resolves `precision` through the cache and then
    ///     HuggingFace, like every other provider here.
    ///   - precision: Which conversion to fetch when `modelPath` is `nil`. FP16 by
    ///     default: it is faster than the fp32 package and uses 5.5x less memory,
    ///     the clearest precision choice of any model in this package.
    ///   - computeUnits: Core ML compute units.
    ///   - stitching: How consecutive segments are joined for audio longer than
    ///     1.594 s. See ``CoreMLGANStitching``. `.none` is the cheaper mode and is
    ///     what the published parity fixtures were generated against.
    public init(
        modelPath: String? = nil,
        precision: CoreMLGANPrecision = .fp16,
        computeUnits: MLComputeUnits = .cpuAndGPU,
        stitching: CoreMLGANStitching = .none
    ) {
        self.modelPath = modelPath
        self.precision = precision
        self.computeUnits = computeUnits
        self.stitching = stitching

        // Create periodic Hann window for MLX STFT
        self.mlxWindow = createPeriodicHannWindow(length: winLength)
    }
    
    /// Load CoreML model (auto-compiles .mlpackage if needed)
    ///
    /// Concurrent calls share one load; see ``ModelLoadGate``. Compiling an
    /// `.mlpackage` twice at once is wasted work at best, and both compilations
    /// write to the same temporary location.
    public func load() async throws {
        try await loadGate.run { [self] in try await performLoad() }
    }

    private func performLoad() async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let packageURL = try await resolvePackageURL()
        try Task.checkCancellation()

        // If it's an .mlpackage, compile it first
        let modelURL = packageURL.pathExtension == "mlpackage"
            ? try await Self.compiledModel(for: packageURL)
            : packageURL

        let candidate = try await MLModel.load(contentsOf: modelURL, configuration: config)
        // An `unload()` that landed during compilation cancels this load; publishing
        // here anyway would resurrect the provider behind it.
        try Task.checkCancellation()
        model = candidate
    }

    /// Explicit path if given, otherwise the cache, otherwise HuggingFace.
    ///
    /// The same three-step resolve the MLX providers do. This one took a path and
    /// nothing else until the packages were published, which is why the benchmark
    /// needed `--coreml-gan` to run the case at all.
    private func resolvePackageURL() async throws -> URL {
        if let modelPath { return URL(fileURLWithPath: modelPath) }

        if let cached = ModelDownloader.shared.localPath(
            for: Self.repo,
            matching: ModelFiles.mossFormerGANCoreMLRequired(precision)
        ) {
            return cached.appendingPathComponent(precision.packageName)
        }

        let directory = try await ModelDownloader.shared.downloadAndGetPath(
            repo: Self.repo,
            matching: ModelFiles.mossFormerGANCoreML(precision)
        )
        return directory.appendingPathComponent(precision.packageName)
    }

    /// The compiled `.mlmodelc` for `package`, compiled once and reused after.
    ///
    /// `MLModel.compileModel(at:)` writes into a temporary directory that the system
    /// is free to clear, so compiling on every `load()` both repeated the work and
    /// produced a result with no defined lifetime. That was tolerable while the only
    /// source was a local checkout; once loading normally means loading a downloaded
    /// package, recompiling per launch is the wrong default.
    ///
    /// Keyed on the size and modification date of the package's `model.mlmodel`, so
    /// a re-pin to a different revision compiles afresh rather than serving the old
    /// graph. Superseded entries are left in place: they only accumulate when those
    /// bytes change, which for a pinned download means a deliberate re-pin, and
    /// deleting a directory this code did not create is the more expensive mistake.
    private static func compiledModel(for package: URL) async throws -> URL {
        let fileManager = FileManager.default
        let source = package.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")
        let attributes = try? fileManager.attributesOfItem(atPath: source.path)
        let size = (attributes?[.size] as? NSNumber)?.intValue ?? 0
        let modified = (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let key = "\(package.deletingPathExtension().lastPathComponent)-\(size)-\(Int(modified))"

        let directory = try compiledModelsDirectory()
        let destination = directory.appendingPathComponent("\(key).mlmodelc")
        if fileManager.fileExists(atPath: destination.path) { return destination }

        let compiled = try await MLModel.compileModel(at: package)
        do {
            try fileManager.moveItem(at: compiled, to: destination)
            return destination
        } catch {
            // Lost a race with another process, or the cache is not writable. An
            // already-present destination is as good as the one just compiled;
            // otherwise this session uses the temporary copy and compiles again next
            // launch, which is the previous behaviour rather than a new failure.
            return fileManager.fileExists(atPath: destination.path) ? destination : compiled
        }
    }

    /// Where compiled models are kept. Application Support rather than Caches: a
    /// purge between launches would be silently recovered from, but only by paying
    /// the compile again on a path that is otherwise offline.
    private static func compiledModelsDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = base
            .appendingPathComponent("AudioTool", isDirectory: true)
            .appendingPathComponent("CompiledModels", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormerGAN_CoreML")
        }
        try validateInputFormat(input)
        
        // Process in segments for long audio
        if input.samples.count <= segmentSamples {
            return try await processSegment(input.samples, model: model)
        }
        
        switch stitching {
        case .none:
            return try await processWithChunking(input.samples, model: model)
        case .discardEdges25:
            return try await processWithChunkingDiscardEdges(input.samples, model: model)
        }
    }
    
    // MARK: - Background Extraction
    
    /// Result containing both enhanced speech and background/residual audio
    public struct EnhancedWithBackground: Sendable {
        public let enhanced: AudioBuffer
        public let background: AudioBuffer
    }
    
    /// Process audio and extract both enhanced speech and background track
    /// Uses mask-based spectral extraction for high-quality background separation
    public func processWithBackground(_ input: AudioBuffer) async throws -> EnhancedWithBackground {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormerGAN_CoreML")
        }
        try validateInputFormat(input)
        
        // Process in segments for long audio
        if input.samples.count <= segmentSamples {
            return try await processSegmentWithBackground(input.samples, model: model)
        }
        
        switch stitching {
        case .none:
            return try await processWithChunkingAndBackground(input.samples, model: model)
        case .discardEdges25:
            return try await processWithChunkingDiscardEdgesAndBackground(input.samples, model: model)
        }
    }
    
    /// Process a single segment with background extraction using MLX
    private func processSegmentWithBackground(_ samples: [Float], model: MLModel) async throws -> EnhancedWithBackground {
        // Pad to segment size if needed
        var paddedSamples = samples
        let trimStart: Int
        
        if samples.count < segmentSamples {
            let needPadding = segmentSamples - samples.count
            paddedSamples = [Float](repeating: 0, count: needPadding) + samples
            trimStart = needPadding
        } else {
            trimStart = 0
        }
        
        // Convert to MLXArray [1, samples]
        let inputMLX = MLXArray(paddedSamples).expandedDimensions(axis: 0).asType(.float32)
        
        // Normalize using MLX
        let inputLen = MLXArray(Float(paddedSamples.count))
        let sumSquares = sum(inputMLX * inputMLX, axis: 1, keepDims: true)
        let normFactor = sqrt(inputLen / (sumSquares + 1e-9))
        let normedAudio = inputMLX * normFactor
        
        // MLX STFT -> [batch, freq, time] (keep noisy for background extraction)
        let (noisyReal, noisyImag) = mlxSTFT(
            normedAudio,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true
        )
        
        // Power compress for model input
        let (realC, imagC) = mlxPowerCompress(real: noisyReal, imag: noisyImag)
        
        // Prepare for CoreML: [1, 2, T, F]
        let coremlInput = try prepareForCoreMLFromMLX(real: realC, imag: imagC)
        
        // CoreML inference
        let prediction = try await model.prediction(from: coremlInput)
        
        // Parse output -> [batch, freq, time]
        let (enhancedReal, enhancedImag) = try parseModelOutputToMLX(prediction)
        
        // Power uncompress for enhanced
        let (realUC, imagUC) = mlxPowerUncompress(real: enhancedReal, imag: enhancedImag)
        
        // Compute magnitude mask and extract background using MLX
        let (backgroundReal, backgroundImag) = mlxComputeBackgroundSpectrogram(
            noisyReal: noisyReal,
            noisyImag: noisyImag,
            enhancedReal: realUC,
            enhancedImag: imagUC
        )
        
        // MLX ISTFT for enhanced
        let enhancedRecon = mlxISTFT(
            realPart: realUC,
            imagPart: imagUC,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true,
            audioLength: paddedSamples.count
        )
        
        // MLX ISTFT for background
        let backgroundRecon = mlxISTFT(
            realPart: backgroundReal,
            imagPart: backgroundImag,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true,
            audioLength: paddedSamples.count
        )
        
        // De-normalize both
        let enhancedOut = (enhancedRecon / normFactor).squeezed(axis: 0)
        let backgroundOut = (backgroundRecon / normFactor).squeezed(axis: 0)
        eval(enhancedOut, backgroundOut)
        
        // Convert to [Float] and trim
        let enhancedSamples = Array(enhancedOut.asArray(Float.self).suffix(from: trimStart))
        let backgroundSamples = Array(backgroundOut.asArray(Float.self).suffix(from: trimStart))
        
        return EnhancedWithBackground(
            enhanced: AudioBuffer(samples: enhancedSamples, sampleRate: sampleRate, channels: 1),
            background: AudioBuffer(samples: backgroundSamples, sampleRate: sampleRate, channels: 1)
        )
    }
    
    /// Compute background spectrogram using inverse magnitude mask (MLX version)
    private func mlxComputeBackgroundSpectrogram(
        noisyReal: MLXArray,
        noisyImag: MLXArray,
        enhancedReal: MLXArray,
        enhancedImag: MLXArray
    ) -> (MLXArray, MLXArray) {
        // Compute magnitudes
        let noisyMag = sqrt(noisyReal * noisyReal + noisyImag * noisyImag + 1e-9)
        let enhancedMag = sqrt(enhancedReal * enhancedReal + enhancedImag * enhancedImag + 1e-9)
        
        // Compute soft mask: enhanced / noisy (clipped to [0, 1])
        let mask = clip(enhancedMag / (noisyMag + 1e-9), min: MLXArray(0.0), max: MLXArray(1.0))
        
        // Background = noisy * (1 - mask)
        let inverseMask = 1.0 - mask
        let backgroundReal = noisyReal * inverseMask
        let backgroundImag = noisyImag * inverseMask
        
        return (backgroundReal, backgroundImag)
    }
    
    /// Process long audio in segments with background extraction
    private func processWithChunkingAndBackground(_ samples: [Float], model: MLModel) async throws -> EnhancedWithBackground {
        let numSegments = Int(ceil(Double(samples.count) / Double(segmentSamples)))
        var enhancedSegments: [Float] = []
        var backgroundSegments: [Float] = []
        
        for idx in 0..<numSegments {
            try Task.checkCancellation()
            let start = idx * segmentSamples
            let end = min(start + segmentSamples, samples.count)
            let actualLen = end - start
            
            // For short final segment, use context from previous audio
            var segment: [Float]
            let trimAmount: Int
            
            if actualLen < segmentSamples {
                let needExtra = segmentSamples - actualLen
                let contextStart = max(0, start - needExtra)
                segment = Array(samples[contextStart..<end])
                
                if segment.count < segmentSamples {
                    let padAmt = segmentSamples - segment.count
                    segment = [Float](repeating: 0, count: padAmt) + segment
                    trimAmount = padAmt + (segment.count - actualLen - padAmt)
                } else {
                    trimAmount = segment.count - actualLen
                }
            } else {
                segment = Array(samples[start..<end])
                trimAmount = 0
            }
            
            let result = try await processSegmentWithBackground(segment, model: model)
            
            // Trim to actual portion
            if trimAmount > 0 && trimAmount < result.enhanced.samples.count {
                enhancedSegments.append(contentsOf: result.enhanced.samples.suffix(result.enhanced.samples.count - trimAmount).prefix(actualLen))
                backgroundSegments.append(contentsOf: result.background.samples.suffix(result.background.samples.count - trimAmount).prefix(actualLen))
            } else {
                enhancedSegments.append(contentsOf: result.enhanced.samples.prefix(actualLen))
                backgroundSegments.append(contentsOf: result.background.samples.prefix(actualLen))
            }
        }
        
        return EnhancedWithBackground(
            enhanced: AudioBuffer(samples: enhancedSegments, sampleRate: sampleRate, channels: 1),
            background: AudioBuffer(samples: backgroundSegments, sampleRate: sampleRate, channels: 1)
        )
    }
    
    /// Process a single segment (up to 1.594s) using MLX STFT/ISTFT
    private func processSegment(_ samples: [Float], model: MLModel) async throws -> AudioBuffer {
        // Pad to segment size if needed
        var paddedSamples = samples
        let trimStart: Int
        
        if samples.count < segmentSamples {
            let needPadding = segmentSamples - samples.count
            paddedSamples = [Float](repeating: 0, count: needPadding) + samples
            trimStart = needPadding
        } else {
            trimStart = 0
        }
        
        // Convert to MLXArray [1, samples]
        let inputMLX = MLXArray(paddedSamples).expandedDimensions(axis: 0).asType(.float32)
        
        // Normalize using MLX
        let inputLen = MLXArray(Float(paddedSamples.count))
        let sumSquares = sum(inputMLX * inputMLX, axis: 1, keepDims: true)
        let normFactor = sqrt(inputLen / (sumSquares + 1e-9))
        let normedAudio = inputMLX * normFactor
        
        // MLX STFT -> [batch, freq, time]
        let (stftReal, stftImag) = mlxSTFT(
            normedAudio,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true
        )
        
        // Power compress using MLX
        let (realC, imagC) = mlxPowerCompress(real: stftReal, imag: stftImag)
        
        // Prepare for CoreML: [1, 2, T, F] (transpose from [1, F, T])
        let coremlInput = try prepareForCoreMLFromMLX(real: realC, imag: imagC)
        
        // CoreML inference
        let prediction = try await model.prediction(from: coremlInput)
        
        // Parse output -> [batch, freq, time]
        let (enhancedReal, enhancedImag) = try parseModelOutputToMLX(prediction)
        
        // Power uncompress using MLX
        let (realUC, imagUC) = mlxPowerUncompress(real: enhancedReal, imag: enhancedImag)
        
        // MLX ISTFT -> [batch, samples]
        let reconstructed = mlxISTFT(
            realPart: realUC,
            imagPart: imagUC,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true,
            audioLength: paddedSamples.count
        )
        
        // De-normalize
        let output = (reconstructed / normFactor).squeezed(axis: 0)
        eval(output)
        
        // Convert to [Float] and trim
        let outputSamples = output.asArray(Float.self)
        let finalSamples = Array(outputSamples.suffix(from: trimStart))
        
        return AudioBuffer(samples: finalSamples, sampleRate: sampleRate, channels: 1)
    }
    
    /// Process long audio in segments (no overlap per model recommendation)
    private func processWithChunking(_ samples: [Float], model: MLModel) async throws -> AudioBuffer {
        let numSegments = Int(ceil(Double(samples.count) / Double(segmentSamples)))
        var enhancedSegments: [Float] = []
        
        for idx in 0..<numSegments {
            try Task.checkCancellation()
            let start = idx * segmentSamples
            let end = min(start + segmentSamples, samples.count)
            let actualLen = end - start
            
            // For short final segment, use context from previous audio
            var segment: [Float]
            let trimAmount: Int
            
            if actualLen < segmentSamples {
                let needExtra = segmentSamples - actualLen
                let contextStart = max(0, start - needExtra)
                segment = Array(samples[contextStart..<end])
                
                if segment.count < segmentSamples {
                    // Still short - pad with zeros at start
                    let padAmt = segmentSamples - segment.count
                    segment = [Float](repeating: 0, count: padAmt) + segment
                    trimAmount = padAmt + (segment.count - actualLen - padAmt)
                } else {
                    trimAmount = segment.count - actualLen
                }
            } else {
                segment = Array(samples[start..<end])
                trimAmount = 0
            }
            
            let result = try await processSegment(segment, model: model)
            
            // Trim to actual portion
            if trimAmount > 0 && trimAmount < result.samples.count {
                enhancedSegments.append(contentsOf: result.samples.suffix(result.samples.count - trimAmount).prefix(actualLen))
            } else {
                enhancedSegments.append(contentsOf: result.samples.prefix(actualLen))
            }
        }
        
        return AudioBuffer(samples: enhancedSegments, sampleRate: sampleRate, channels: 1)
    }
    
    // MARK: - Discard-Edges Stitching

    /// One planned inference: which window to run, and which slice of its output
    /// survives into the result.
    private struct EdgePlan {
        /// Offset of the 25500-sample window into the input.
        let start: Int
        /// First index of the segment output that is kept.
        let keepFrom: Int
        /// One past the last index kept.
        let keepTo: Int
    }

    /// Windows advance by 75% of the segment, and each contributes only its centre.
    ///
    /// `giveUp` is `(segment - stride) / 2`, so the kept spans abut: chunk `i` starts
    /// one sample before chunk `i-1` ends, which the write order resolves. Integer
    /// division makes `keep` one sample wider than `stride`; that single-sample
    /// double-write is exactly what the PyTorch reference does and is why the spans
    /// cannot leave a gap.
    ///
    /// The first window keeps its leading edge and the last keeps its trailing edge:
    /// there is no neighbour out there to supply those samples, and they carry no
    /// seam. The final window is end-aligned rather than zero-padded, so the tail is
    /// inferred with real audio around it.
    private func edgePlans(totalSamples: Int) -> [EdgePlan] {
        precondition(totalSamples > segmentSamples, "caller routes shorter input to the single-segment path")

        let stride = Int(Double(segmentSamples) * stitching.strideFraction)
        let giveUp = (segmentSamples - stride) / 2

        var starts: [Int] = []
        var pos = 0
        while pos + segmentSamples <= totalSamples {
            starts.append(pos)
            pos += stride
        }
        if let last = starts.last, last + segmentSamples < totalSamples {
            starts.append(totalSamples - segmentSamples)
        }

        return starts.enumerated().map { index, start in
            EdgePlan(
                start: start,
                keepFrom: index == 0 ? 0 : giveUp,
                keepTo: index == starts.count - 1 ? segmentSamples : segmentSamples - giveUp
            )
        }
    }

    private func processWithChunkingDiscardEdges(
        _ samples: [Float], model: MLModel
    ) async throws -> AudioBuffer {
        var output = [Float](repeating: 0, count: samples.count)

        for plan in edgePlans(totalSamples: samples.count) {
            try Task.checkCancellation()
            let segment = Array(samples[plan.start..<(plan.start + segmentSamples)])
            let result = try await processSegment(segment, model: model)
            Self.copyKept(result.samples, plan, into: &output)
        }

        return AudioBuffer(samples: output, sampleRate: sampleRate, channels: 1)
    }

    private func processWithChunkingDiscardEdgesAndBackground(
        _ samples: [Float], model: MLModel
    ) async throws -> EnhancedWithBackground {
        var enhanced = [Float](repeating: 0, count: samples.count)
        var background = [Float](repeating: 0, count: samples.count)

        for plan in edgePlans(totalSamples: samples.count) {
            try Task.checkCancellation()
            let segment = Array(samples[plan.start..<(plan.start + segmentSamples)])
            let result = try await processSegmentWithBackground(segment, model: model)
            Self.copyKept(result.enhanced.samples, plan, into: &enhanced)
            Self.copyKept(result.background.samples, plan, into: &background)
        }

        return EnhancedWithBackground(
            enhanced: AudioBuffer(samples: enhanced, sampleRate: sampleRate, channels: 1),
            background: AudioBuffer(samples: background, sampleRate: sampleRate, channels: 1)
        )
    }

    /// Copies `plan`'s kept span from a segment output into the destination, clamped
    /// to both buffers.
    private static func copyKept(_ segment: [Float], _ plan: EdgePlan, into output: inout [Float]) {
        let keepTo = min(plan.keepTo, segment.count)
        guard plan.keepFrom < keepTo else { return }

        let destination = plan.start + plan.keepFrom
        let count = min(keepTo - plan.keepFrom, output.count - destination)
        guard count > 0 else { return }

        output.replaceSubrange(
            destination..<(destination + count),
            with: segment[plan.keepFrom..<(plan.keepFrom + count)]
        )
    }

    // MARK: - MLX Power Compression
    
    /// Power compress spectrogram using MLX (matches Python exactly)
    private func mlxPowerCompress(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let mag = sqrt(real * real + imag * imag + 1e-8)
        let phase = atan2(imag, real)
        let magC = pow(mag, MLXArray(powerCompress))
        return (magC * cos(phase), magC * sin(phase))
    }
    
    /// Power uncompress spectrogram using MLX
    private func mlxPowerUncompress(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let mag = sqrt(real * real + imag * imag + 1e-8)
        let phase = atan2(imag, real)
        let magUC = pow(mag, MLXArray(1.0 / powerCompress))
        return (magUC * cos(phase), magUC * sin(phase))
    }
    
    // MARK: - MLX CoreML I/O
    
    /// Prepare MLXArray spectrogram for CoreML input [1, 2, T, F]
    private func prepareForCoreMLFromMLX(real: MLXArray, imag: MLXArray) throws -> MLFeatureProvider {
        // real/imag are [batch=1, F, T], need [1, 2, T, F]
        let realT = real.transposed(0, 2, 1)  // [1, T, F]
        let imagT = imag.transposed(0, 2, 1)  // [1, T, F]
        
        eval(realT, imagT)
        
        let T = realT.shape[1]
        let F = realT.shape[2]
        
        // Stack as [1, 2, T, F]
        let spec = MLX.stacked([realT, imagT], axis: 1)  // [1, 2, T, F]
        eval(spec)
        
        // Convert to MLMultiArray
        let shape: [NSNumber] = [1, 2, NSNumber(value: T), NSNumber(value: F)]
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Copy data using buffer pointer (much faster than element-by-element)
        let flatData = spec.flattened().asArray(Float.self)
        let dataPointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        flatData.withUnsafeBufferPointer { srcPtr in
            dataPointer.update(from: srcPtr.baseAddress!, count: flatData.count)
        }
        
        return try MLDictionaryFeatureProvider(dictionary: ["spectrogram": multiArray])
    }
    
    /// Parse CoreML output to MLXArray [batch, freq, time]
    private func parseModelOutputToMLX(
        _ output: MLFeatureProvider
    ) throws -> (MLXArray, MLXArray) {
        // The published .mlpackage does not name its output `enhanced_spectrogram`
        // - the converter left it as `var_9026`. Requiring the friendly name makes
        // every prediction throw, so accept a sole unnamed output and only refuse
        // when the choice is genuinely ambiguous.
        let feature: MLFeatureValue
        if let named = output.featureValue(for: "enhanced_spectrogram") {
            feature = named
        } else if output.featureNames.count == 1,
                  let only = output.featureNames.first,
                  let sole = output.featureValue(for: only) {
            feature = sole
        } else {
            throw AudioToolError.incompatibleModelVersion(
                expected: "one CoreML output, or one named enhanced_spectrogram",
                found: output.featureNames.sorted().joined(separator: ", ")
            )
        }
        guard let multiArray = feature.multiArrayValue else {
            throw AudioToolError.incompatibleModelVersion(
                expected: "the enhanced spectrogram as MLMultiArray",
                found: "feature type \(feature.type.rawValue)"
            )
        }
        // Float16 as well as Float32, because precision on this backend is baked
        // into the compiled `.mlpackage` rather than chosen at load: the FP16
        // conversion of this graph returns `MLMultiArrayDataTypeFloat16` (65552),
        // and requiring Float32 rejected the model outright with a message about a
        // raw type code. The model was fine; only this reader was not.
        guard multiArray.dataType == .float32 || multiArray.dataType == .float16 else {
            throw AudioToolError.incompatibleModelVersion(
                expected: "float32 or float16 enhanced_spectrogram",
                found: "MLMultiArray data type \(multiArray.dataType.rawValue)"
            )
        }

        let shape = multiArray.shape.map(\.intValue)
        guard shape.count == 4,
              shape[0] == 1,
              shape[1] == 2,
              shape[2] > 0,
              shape[3] == nFFT / 2 + 1 else {
            throw AudioToolError.incompatibleModelVersion(
                expected: "enhanced_spectrogram shape [1, 2, T, \(nFFT / 2 + 1)] with T > 0",
                found: "shape \(shape)"
            )
        }
        
        // Output is [1, 2, T, F]
        let timeFrames = shape[2]
        let bins = shape[3]

        // Read through the array's own strides rather than assuming the backing
        // buffer is packed.
        //
        // It is not. CoreML returns this model's output as [1, 2, 256, 201] with
        // strides [106496, 53248, 208, 1] - the innermost dimension is 201 wide but
        // rows sit 208 floats apart, padded up to a 16-float boundary. A flat copy
        // of 2*T*F floats therefore reads every row after the first 7 floats
        // further out of place than the last, and 256 rows in the spectrogram is
        // sheared beyond recognition. That is what produced ganse_enhanced.wav.
        //
        // Python never hits this: coremltools unpacks MLMultiArray into a proper
        // C-contiguous ndarray before predict() returns, which is why run.py needs
        // no equivalent and its outputs were always correct. Swift's dataPointer is
        // the raw backing store, padding included.
        //
        // Measured on one segment, same prediction read both ways: flat copy gives
        // -1.1 dB against the MLX Python reference, this gives 129.3 dB. The
        // padding is present under every compute unit, so this path never worked.
        //
        // The fast path is kept for the case where CoreML does hand back packed
        // data, which is what the shape implies and what a smaller F would give.
        let strides = multiArray.strides.map(\.intValue)
        guard strides.count == 4, strides.allSatisfy({ $0 > 0 }) else {
            throw AudioToolError.incompatibleModelVersion(
                expected: "four positive enhanced_spectrogram strides",
                found: "strides \(strides)"
            )
        }
        let totalCount = 2 * timeFrames * bins

        // One de-striding routine over both element types. Generic rather than
        // duplicated: the padding described above is a property of how CoreML lays
        // the array out, not of the element width, so an fp16 reader that did not
        // repeat the stride handling would reproduce the sheared-spectrogram bug
        // this comment exists to explain - and would look correct until someone
        // measured it.
        func gather<T: BinaryFloatingPoint>(_ type: T.Type) -> [Float] {
            var flat = [Float](repeating: 0, count: totalCount)
            let src = multiArray.dataPointer.assumingMemoryBound(to: T.self)
            flat.withUnsafeMutableBufferPointer { dstPtr in
                guard let destination = dstPtr.baseAddress else { return }
                if strides == [totalCount, timeFrames * bins, bins, 1] {
                    for i in 0..<totalCount { destination[i] = Float(src[i]) }
                    return
                }
                for part in 0..<2 {
                    for frame in 0..<timeFrames {
                        let row = part * strides[1] + frame * strides[2]
                        let offset = (part * timeFrames + frame) * bins
                        for bin in 0..<bins {
                            destination[offset + bin] = Float(src[row + bin * strides[3]])
                        }
                    }
                }
            }
            return flat
        }

        let flatData = multiArray.dataType == .float16
            ? gather(Float16.self)
            : gather(Float.self)

        let mlxData = MLXArray(flatData).reshaped([1, 2, timeFrames, bins])
        
        // Extract real/imag and transpose to [1, F, T]
        let realT = mlxData[0..., 0, 0..., 0...]  // [1, T, F]
        let imagT = mlxData[0..., 1, 0..., 0...]  // [1, T, F]
        
        let real = realT.transposed(0, 2, 1)  // [1, F, T]
        let imag = imagT.transposed(0, 2, 1)  // [1, F, T]
        
        eval(real, imag)
        return (real, imag)
    }
    
    // MARK: - Streaming
    
    public nonisolated func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                for await chunk in input {
                    do {
                        try Task.checkCancellation()
                        let processed = try await process(chunk)
                        if case .terminated = continuation.yield(processed) { return }
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }
    
    public func reset() async {}
    
    // MARK: - Output Streaming
    
    /// Process audio and stream output chunks as they're ready.
    /// Uses no overlap - each segment is processed independently.
    public nonisolated func processStream(_ input: AudioBuffer) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    try await self.processStreamImpl(input, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }
    
    /// Internal implementation for streaming - runs within actor context
    private func processStreamImpl(_ input: AudioBuffer, continuation: AsyncThrowingStream<AudioBuffer, Error>.Continuation) async throws {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormerGAN_CoreML")
        }
        try validateInputFormat(input)
        
        let samples = input.samples
        let totalLength = samples.count
        
        // For short audio, just yield single result
        if totalLength <= segmentSamples {
            let result = try await processSegment(samples, model: model)
            if case .terminated = continuation.yield(result) { throw CancellationError() }
            continuation.finish()
            return
        }
        
        // Process in segments (no overlap per model recommendation)
        let numSegments = Int(ceil(Double(totalLength) / Double(segmentSamples)))
        
        for idx in 0..<numSegments {
            try Task.checkCancellation()
            let start = idx * segmentSamples
            let end = min(start + segmentSamples, totalLength)
            let actualLen = end - start
            
            // For short final segment, use context from previous audio
            var segment: [Float]
            let trimAmount: Int
            
            if actualLen < segmentSamples {
                let needExtra = segmentSamples - actualLen
                let contextStart = max(0, start - needExtra)
                segment = Array(samples[contextStart..<end])
                
                if segment.count < segmentSamples {
                    let padAmt = segmentSamples - segment.count
                    segment = [Float](repeating: 0, count: padAmt) + segment
                    trimAmount = padAmt + (segment.count - actualLen - padAmt)
                } else {
                    trimAmount = segment.count - actualLen
                }
            } else {
                segment = Array(samples[start..<end])
                trimAmount = 0
            }
            
            let result = try await processSegment(segment, model: model)
            
            // Trim to actual portion
            var outputSamples: [Float]
            if trimAmount > 0 && trimAmount < result.samples.count {
                outputSamples = Array(result.samples.suffix(result.samples.count - trimAmount).prefix(actualLen))
            } else {
                outputSamples = Array(result.samples.prefix(actualLen))
            }
            
            let chunkBuffer = AudioBuffer(
                samples: outputSamples,
                sampleRate: sampleRate,
                channels: 1
            )
            if case .terminated = continuation.yield(chunkBuffer) { throw CancellationError() }
        }
        
        continuation.finish()
    }
}

// MARK: - StreamableOutput Conformance

extension MossFormerGANCoreMLProvider: StreamableOutput {}

// MARK: - ManagedModel

extension MossFormerGANCoreMLProvider: ManagedModel {
    public nonisolated var modelId: String { "mossformer_gan_se_16k" }

    /// ~30 MB: CoreML FP16, and the ANE holds much of it outside our accounting.
    /// Measured peak 1631 MiB, 30 s at 16 kHz on an M1 Pro, against the 30 MB
    /// this declared for a 13 MB `.mlpackage`.
    ///
    /// Nearly all of it is outside CoreML: `mlxPeakDuringRunBytes` is 5 MiB, because
    /// the STFT/ISTFT and the segment stitching either side of the model run in MLX
    /// and the model itself is small. An estimate drawn from the `.mlpackage` was
    /// never going to describe this provider.
    public nonisolated var estimatedMemoryBytes: Int { 1_700_000_000 }

    public func checkIfLoaded() async -> Bool { model != nil }

    public func unload() async {
        // Cancel first: a load mid-compile must not publish over this unload.
        // Bracketed: the gate stays shut until this method's state reset is
        // done, so a concurrent load cannot publish a model into the gap and
        // have it wiped by the lines below.
        let teardown = await loadGate.beginTeardown()
        defer { loadGate.endTeardown(teardown) }
        model = nil
    }
}
