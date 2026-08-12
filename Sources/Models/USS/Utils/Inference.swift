import MLX
import MLXNN
import AVFoundation
import AudioUtils

/// USS inference pipeline
public class USSInference {

    /// MLX's cache is process-global. Trimming after every two-second segment
    /// serializes unrelated inference and defeats allocator reuse, so long-running
    /// USS jobs only trim periodically once cached allocations are substantial.
    private static let cacheTrimInterval = 8
    /// A quarter of `mlxCacheLimitBytes`, scaled with it. See the note there.
    private static let cacheTrimThresholdBytes = 128 * 1024 * 1024

    /// The interval is a politeness measure, not a bound - past this, trim at the
    /// next segment regardless of where the interval falls.
    private static let cacheTrimHardCeilingBytes = 4 * cacheTrimThresholdBytes

    /// MLX's default cache ceiling is the memory limit, which is itself 1.5x the
    /// device's recommended working set - so without an explicit cap a segmented
    /// job's resident size tracks the machine rather than the work.
    ///
    /// 512 MB, down from 3 GB. Measured across a 24x sweep of this value, on both a
    /// direct forward pass and a chunked one, throughput did not move outside noise
    /// - so the larger cache was retaining memory the host could not use and buying
    /// nothing. See `MLXCachePolicy.cacheLimitBytes` for the numbers.
    ///
    /// Process-global, and mirrored by `MLXCachePolicy.cacheLimitBytes` because
    /// `USSMLXSwift` sits below `AudioToolMLX` and the two cannot share a constant.
    /// Change both together.
    static let mlxCacheLimitBytes = 512 * 1024 * 1024

    /// Ceiling on total MLX allocation, mirroring `MLXCachePolicy.memoryLimitBytes`.
    ///
    /// Applied here as well as there because a host that links only `AudioToolUSS`
    /// never runs `MLXCachePolicy`, and would otherwise keep MLX's default of 1.5x
    /// the recommended working set with nothing applying backpressure.
    ///
    /// Backpressure rather than a wall: `relaxed` defaults to true, so past the
    /// limit MLX waits on scheduled work and then allocates anyway.
    static let mlxMemoryLimitBytes: Int = {
        let info = GPU.deviceInfo()
        var limit = info.memorySize / 5 * 3
        let recommended = Int(info.maxRecommendedWorkingSetSize)
        if recommended > 0 {
            limit = min(limit, recommended)
        }
        return max(limit, 2 * 1024 * 1024 * 1024)
    }()


    private let model: ResUNet30
//    private let stft: STFT
//    private let istft: ISTFT
    private let sampleRate: Int
    private let segmentDuration: Float
    private var isPrewarmed: Bool = false
    private let segmentBatchSize: Int
    
    public init(
        model: ResUNet30,
        sampleRate: Int = 32000,
        segmentDuration: Float = 2.0,
        compile: Bool = true,
        segmentBatchSize: Int = 1  // Default to 1 based on performance testing
    ) {
        precondition(sampleRate > 0, "Sample rate must be positive")
        precondition(segmentDuration > 0, "Segment duration must be positive")
        precondition(segmentBatchSize > 0, "Segment batch size must be positive")
        self.model = model
        self.sampleRate = sampleRate
        self.segmentDuration = segmentDuration
        self.segmentBatchSize = segmentBatchSize

        // Applied before any inference, not on the first trim - a short job that
        // never reaches a trim boundary should still run under the cap.
        Self.applyMLXCacheLimit()

        // Enable or disable compilation
        model.setCompile(compile)
        
        // Warm the exact normalization entry used by this model and segment
        // length. The cache is keyed by the ISTFT window's identity, so a
        // separately-created Hann window cannot warm the inference path.
        let segmentSamples = Int(segmentDuration * Float(sampleRate))
        model.prewarmNormalization(forSignalLength: segmentSamples)
    }
    
    /// Prewarm the model with dummy input to compile the graph
    public func prewarm(conditioning: MLXArray) {
        if isPrewarmed { return }
        
        let segmentSamples = Int(segmentDuration * Float(sampleRate))
        let dummyMixture = MLXArray.zeros([1, 1, segmentSamples])
        
        let inputs: [String: MLXArray] = [
            "mixture": dummyMixture,
            "condition": conditioning
        ]
        
        // Warm up with multiple calls to ensure JIT compilation is complete
        for i in 0..<2 {
            let outputs = model(inputs)
            guard let waveform = outputs["waveform"] else {
                preconditionFailure("USS model prewarm did not produce a waveform")
            }
            // Evaluating an input does not compile the model graph. Force the
            // actual output graph and materialize it before declaring readiness.
            eval(waveform)
            _ = waveform.asArray(Float.self)
            if i == 0 {
                // Clear cache after first run to ensure clean state
                GPU.clearCache()
            }
        }
        
        isPrewarmed = true
    }
    
    /// Perform source separation on audio
    public func separate(
        audio: MLXArray,
        conditioning: MLXArray,
        compile: Bool = false
    ) -> MLXArray {
        // Prewarm model if needed
        if !isPrewarmed {
            prewarm(conditioning: conditioning)
        }
        
        // Calculate segment parameters
        let segmentSamples = Int(segmentDuration * Float(sampleRate))
        let audioLength = audio.shape[1]
        
        // Process full audio if it's short enough
        if audioLength <= segmentSamples {
            return processSingleSegment(audio: audio, conditioning: conditioning, compile: compile)
        }
        
        // Non-overlapping segments, matching the Python reference.
        return processSegmentedSimple(
            audio: audio,
            conditioning: conditioning,
            segmentSamples: segmentSamples,
            compile: compile
        )
    }
    
    /// Process a single audio segment
    private func processSingleSegment(
        audio: MLXArray,
        conditioning: MLXArray,
        compile: Bool
    ) -> MLXArray {        
        // Add channel dimension if needed (batch, samples) -> (batch, channels, samples)
        let mixture = audio.ndim == 2 ? audio.expandedDimensions(axis: 1) : audio
        
        // Prepare model inputs
        let inputs: [String: MLXArray] = [
            "mixture": mixture,
            "condition": conditioning
        ]
        
        // Run model
        let outputs = model(inputs)
        
        // Extract waveform output
        guard let waveform = outputs["waveform"] else {
            fatalError("Model did not return waveform output")
        }
        
        return waveform
    }
    
    /// Process multiple segments as a batch
    private func processBatchSegments(
        segments: [MLXArray],
        conditioning: MLXArray,
        compile: Bool
    ) -> [MLXArray] {
        // Stack segments into batch
        let batchSize = segments.count
        var batchedSegments: [MLXArray] = []
        
        for segment in segments {
            // Add channel dimension if needed
            let mixture = segment.ndim == 2 ? segment.expandedDimensions(axis: 1) : segment
            batchedSegments.append(mixture)
        }
        
        // Stack into single batch tensor
        let batchedMixture = MLX.concatenated(batchedSegments, axis: 0)
        
        // Expand conditioning for batch
        var batchedCondition = conditioning
        if batchedCondition.shape[0] == 1 && batchSize > 1 {
            batchedCondition = repeated(batchedCondition, count: batchSize, axis: 0)
        }
        
        // Prepare model inputs
        let inputs: [String: MLXArray] = [
            "mixture": batchedMixture,
            "condition": batchedCondition
        ]
        
        // Run model on entire batch
        let outputs = model(inputs)
        
        // Extract waveform output
        guard let waveform = outputs["waveform"] else {
            fatalError("Model did not return waveform output")
        }
        
        // Split batch results - keep same format as processSingleSegment
        var results: [MLXArray] = []
        for i in 0..<batchSize {
            // Each result should be [batch=1, channels=1, samples]
            results.append(waveform[i].expandedDimensions(axis: 0))
        }
        
        return results
    }
    
    /// Process audio in simple non-overlapping segments (matches Python implementation)
    private func processSegmentedSimple(
        audio: MLXArray,
        conditioning: MLXArray,
        segmentSamples: Int,
        compile: Bool
    ) -> MLXArray {
        let audioLength = audio.shape[1]
        let numSegments = Int(ceil(Float(audioLength) / Float(segmentSamples)))
        
        var output = [Float](repeating: 0, count: audioLength)
        
        // Use configurable batch size
        let batchSegments = segmentBatchSize
        let numBatches = (numSegments + batchSegments - 1) / batchSegments
        
        for batchIdx in 0..<numBatches {
            let startSeg = batchIdx * batchSegments
            let endSeg = min(startSeg + batchSegments, numSegments)
            // let batchSize = endSeg - startSeg
            
            if batchSegments == 1 {
                // For batch size 1, process sequentially (matches Python)
                for i in startSeg..<endSeg {
                    let start = i * segmentSamples
                    let end = min(start + segmentSamples, audioLength)
                    let actualLength = end - start
                    var segment = audio[0..., start..<end]
                    if actualLength < segmentSamples {
                        segment = MLX.padded(
                            segment,
                            widths: [IntOrPair(0), IntOrPair([0, segmentSamples - actualLength])]
                        )
                    }
                    
                    let separated = processSingleSegment(
                        audio: segment,
                        conditioning: conditioning,
                        compile: compile
                    )
                    
                    eval(separated)
                    let samples = separated[0, 0, 0..<actualLength].asArray(Float.self)
                    output.replaceSubrange(start..<end, with: samples)
                    Self.trimCacheIfNeeded(afterUnit: i + 1)
                }
            } else {
                // For batch size > 1, use true batch processing
                // Collect segments for batch processing
                var batchSegments: [MLXArray] = []
                var batchRanges: [Range<Int>] = []
                for i in startSeg..<endSeg {
                    let start = i * segmentSamples
                    let end = min(start + segmentSamples, audioLength)
                    var segment = audio[0..., start..<end]
                    if end - start < segmentSamples {
                        segment = MLX.padded(
                            segment,
                            widths: [IntOrPair(0), IntOrPair([0, segmentSamples - (end - start)])]
                        )
                    }
                    batchSegments.append(segment)
                    batchRanges.append(start..<end)
                }
                
                // Process entire batch at once
                let batchResults = processBatchSegments(
                    segments: batchSegments,
                    conditioning: conditioning,
                    compile: compile
                )
                
                eval(batchResults)
                for (result, range) in zip(batchResults, batchRanges) {
                    let samples = result[0, 0, 0..<range.count].asArray(Float.self)
                    output.replaceSubrange(range, with: samples)
                }
                Self.trimCacheIfNeeded(afterUnit: batchIdx + 1)
            }
        }

        return MLXArray(output).reshaped([1, 1, audioLength])
    }

    private static func trimCacheIfNeeded(afterUnit completedUnits: Int) {
        applyMLXCacheLimit()
        guard completedUnits > 0 else { return }
        guard GPU.cacheMemory >= cacheTrimHardCeilingBytes else {
            guard completedUnits.isMultiple(of: cacheTrimInterval),
                  GPU.cacheMemory >= cacheTrimThresholdBytes else {
                return
            }
            GPU.clearCache()
            return
        }
        GPU.clearCache()
    }

    /// Idempotent. See `mlxCacheLimitBytes` and `mlxMemoryLimitBytes`.
    static func applyMLXCacheLimit() {
        _ = limitsApplied
    }

    private static let limitsApplied: Void = {
        GPU.set(cacheLimit: mlxCacheLimitBytes)
        GPU.set(memoryLimit: mlxMemoryLimitBytes)
    }()
    
    
    /// Process multiple embeddings for the same audio.
    ///
    /// - Parameters:
    ///   - audio: Input audio as MLXArray
    ///   - conditionings: Array of conditioning embeddings
    /// - Returns: Array of separated audio, one per conditioning
    public func separateMultipleEmbeddings(
        audio: MLXArray,
        conditionings: [MLXArray]
    ) -> [MLXArray] {
        // Prewarm model if needed with first conditioning
        if !isPrewarmed && !conditionings.isEmpty {
            prewarm(conditioning: conditionings[0])
        }
        
        // Process each embedding
        var results: [MLXArray] = []
        for conditioning in conditionings {
            let result = separate(
                audio: audio,
                conditioning: conditioning,
                compile: true
            )
            results.append(result)
        }
        
        return results
    }
    
    /// Process a single segment with multiple embeddings
    private func processSegmentWithMultipleEmbeddings(
        segment: MLXArray,
        conditionings: [MLXArray]
    ) -> [MLXArray] {
        // The USS model doesn't support batching across different conditioning vectors
        // Process each embedding separately but efficiently
        var results: [MLXArray] = []
        
        for conditioning in conditionings {
            // Process this segment with this conditioning
            let result = processSingleSegment(
                audio: segment,
                conditioning: conditioning,
                compile: true
            )
            results.append(result)
        }
        
        // Force evaluation of all results at once for better GPU utilization
        eval(results)
        
        return results
    }
}

// `runUSS` lived here: a file-in, file-out convenience that loaded audio, weights
// and a conditioning vector from paths and wrote a wav. Nothing in the package
// called it, and the only thing it still needed from this module was the embedding
// file loader, which is gone - the seven conditioning vectors are class lists in
// `SoundEmbedding` now. `USSMLXProvider` is the supported entry point.
