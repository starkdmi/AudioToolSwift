import MLX
import MLXNN
import AVFoundation
import AudioUtils

/// USS inference pipeline
public class USSInference {
    
    private let model: ResUNet30
//    private let stft: STFT
//    private let istft: ISTFT
    private let sampleRate: Int
    private let segmentDuration: Float
    private let hopLength: Float
    private var isPrewarmed: Bool = false
    private let segmentBatchSize: Int
    
    public init(
        model: ResUNet30,
        sampleRate: Int = 32000,
        segmentDuration: Float = 2.0,
        hopLength: Float = 0.5,
        compile: Bool = true,
        segmentBatchSize: Int = 1  // Default to 1 based on performance testing
    ) {
        self.model = model
        self.sampleRate = sampleRate
        self.segmentDuration = segmentDuration
        self.hopLength = hopLength
        self.segmentBatchSize = segmentBatchSize
        
        // Enable or disable compilation
        model.setCompile(compile)
        
        // Pre-warm STFT normalization buffer cache
        prewarmSTFTCache()
    }
    
    /// Prewarm the model with dummy input to compile the graph
    public func prewarm(conditioning: MLXArray) {
        if isPrewarmed { return }
        
        print("[Inference] Prewarming compiled model...")
        
        // Optimization: Set GPU cache limit for better memory management
        // This helps with repeated allocations during segmented processing
        GPU.set(cacheLimit: 3 * 1024 * 1024 * 1024) // 3GB cache
        
        let segmentSamples = Int(segmentDuration * Float(sampleRate))
        let dummyMixture = MLXArray.zeros([1, 1, segmentSamples])
        
        let inputs: [String: MLXArray] = [
            "mixture": dummyMixture,
            "condition": conditioning
        ]
        
        // Warm up with multiple calls to ensure JIT compilation is complete
        for i in 0..<2 {
            let _ = model(inputs)
            eval(dummyMixture) // Force evaluation
            if i == 0 {
                // Clear cache after first run to ensure clean state
                GPU.clearCache()
            }
        }
        
        isPrewarmed = true
        print("[Inference] Model prewarmed successfully")
    }
    
    /// Perform source separation on audio
    public func separate(
        audio: MLXArray,
        conditioning: MLXArray,
        compile: Bool = false,
        useSimpleSegmentation: Bool = true  // Default to Python-compatible segmentation
    ) -> MLXArray {
        // Prewarm model if needed
        if !isPrewarmed {
            prewarm(conditioning: conditioning)
        }
        
        // Calculate segment parameters
        let segmentSamples = Int(segmentDuration * Float(sampleRate))
        let hopSamples = Int(hopLength * Float(sampleRate))
        let audioLength = audio.shape[1]
        
        // Process full audio if it's short enough
        if audioLength <= segmentSamples {
            return processSingleSegment(audio: audio, conditioning: conditioning, compile: compile)
        }
        
        // Process in segments for longer audio
        if useSimpleSegmentation {
            // Use simple non-overlapping segmentation (matches Python)
            return processSegmentedSimple(
                audio: audio,
                conditioning: conditioning,
                segmentSamples: segmentSamples,
                compile: compile
            )
        } else {
            // Use overlap-add segmentation (original Swift implementation)
            return processSegmented(
                audio: audio,
                conditioning: conditioning,
                segmentSamples: segmentSamples,
                hopSamples: hopSamples,
                compile: compile
            )
        }
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
            print("[Inference] ERROR: Missing waveform output! Available outputs: \(outputs.keys)")
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
            print("[Inference] ERROR: Missing waveform output! Available outputs: \(outputs.keys)")
            fatalError("Model did not return waveform output")
        }
        
        // Split batch results - keep same format as processSingleSegment
        var results: [MLXArray] = []
        for i in 0..<batchSize {
            // Each result should be [batch=1, channels=1, samples]
            results.append(waveform[i, 0..., 0...].expandedDimensions(axis: 0).expandedDimensions(axis: 0))
        }
        
        return results
    }
    
    /// Process audio in segments with overlap
    private func processSegmented(
        audio: MLXArray,
        conditioning: MLXArray,
        segmentSamples: Int,
        hopSamples: Int,
        compile: Bool
    ) -> MLXArray {
        let audioLength = audio.shape[1]
        let numSegments = (audioLength - segmentSamples) / hopSamples + 1
        
        var separatedSegments: [MLXArray] = []
        var weights: [MLXArray] = []
        
        // Create window for overlap-add
        let window = createTriangularWindow(segmentSamples)
        
        for i in 0..<numSegments {
            let start = i * hopSamples
            let end = min(start + segmentSamples, audioLength)
            let actualLength = end - start
            
            // Extract segment
            var segment = audio[0..., start..<end]
            
            // Pad if needed
            if actualLength < segmentSamples {
                let padAmount = segmentSamples - actualLength
                segment = MLX.padded(segment, widths: [IntOrPair(0), IntOrPair([0, padAmount])])
            }
            
            // Process segment
            let separated = processSingleSegment(
                audio: segment,
                conditioning: conditioning,
                compile: compile
            )
            
            // Handle channel dimension - separated is [batch, channels, samples]
            // Extract first channel to get [batch, samples]
            let separatedMono = separated[0..., 0, 0...]
            
            // Apply window
            let windowedSeparated = separatedMono * window.reshaped([1, -1])
            
            // Store segment and weight
            if actualLength < segmentSamples {
                separatedSegments.append(windowedSeparated[0..., 0..<actualLength])
                weights.append(window[0..<actualLength])
            } else {
                separatedSegments.append(windowedSeparated)
                weights.append(window)
            }
        }
        
        // Overlap-add reconstruction
        var outputSeparated = MLXArray.zeros([1, audioLength])
        let outputWeights = MLXArray.zeros([audioLength])
        
        for (i, segment) in separatedSegments.enumerated() {
            let start = i * hopSamples
            let length = segment.shape[1]
            let end = start + length
            
            outputSeparated[0..., start..<end] = outputSeparated[0..., start..<end] + segment
            outputWeights[start..<end] = outputWeights[start..<end] + weights[i]
        }
        
        // Normalize by weights
        let eps: Float = 1e-8
        outputSeparated = outputSeparated / MLX.maximum(outputWeights.expandedDimensions(axis: 0), eps)
        
        // Add channel dimension back to match expected output format [batch, channels, samples]
        return outputSeparated.expandedDimensions(axis: 1)
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
        let totalSamples = numSegments * segmentSamples
        
        print("[Inference] Using optimized segmentation: \(numSegments) segments of \(segmentSamples) samples")
        
        // Pad entire audio upfront (like Python)
        var paddedAudio = audio
        if audioLength < totalSamples {
            let padding = totalSamples - audioLength
            paddedAudio = MLX.padded(audio, widths: [IntOrPair(0), IntOrPair([0, padding])])
            print("[Inference] Padded audio from \(audioLength) to \(totalSamples) samples")
        }
        
        // For now, skip asStrided optimization due to shape issues
        // Will use direct slicing which is still efficient with MLX
        
        // Process all segments at once if possible
        var separatedSegments: [MLXArray] = []
        
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
                    let end = start + segmentSamples
                    let segment = paddedAudio[0..., start..<end]
                    
                    print("[Inference] Processing segment \(i + 1)/\(numSegments): [\(start):\(end)]")
                    
                    let separated = processSingleSegment(
                        audio: segment,
                        conditioning: conditioning,
                        compile: compile
                    )
                    
                    // No explicit eval needed - happens automatically
                    separatedSegments.append(separated)
                }
            } else {
                // For batch size > 1, use true batch processing
                print("[Inference] Processing segments \(startSeg + 1)-\(endSeg)/\(numSegments) as batch")
                
                // Collect segments for batch processing
                var batchSegments: [MLXArray] = []
                for i in startSeg..<endSeg {
                    let start = i * segmentSamples
                    let end = start + segmentSamples
                    let segment = paddedAudio[0..., start..<end]
                    batchSegments.append(segment)
                }
                
                // Process entire batch at once
                let batchResults = processBatchSegments(
                    segments: batchSegments,
                    conditioning: conditioning,
                    compile: compile
                )
                
                // Add results
                separatedSegments.append(contentsOf: batchResults)
                
                // Single evaluation for entire batch
                eval(batchResults)
            }
        }
        
        // Concatenate all segments along the samples dimension
        let concatenated = MLX.concatenated(separatedSegments, axis: 2)  // axis=2 is samples dimension
        
        // Trim to original audio length
        return concatenated[0..., 0..., 0..<audioLength]
    }
    
    /// Create triangular window for overlap-add
    private func createTriangularWindow(_ length: Int) -> MLXArray {
        let half = length / 2
        let firstHalf = MLX.linspace(0, 1, count: half)
        let secondHalf = MLX.linspace(1, 0, count: length - half)
        return MLX.concatenated([firstHalf, secondHalf])
    }
    
    /// Process multiple embeddings for the same audio.
    ///
    /// **Performance Note**: Benchmarks show this method is ~2x slower than calling
    /// `separate()` sequentially for each embedding due to logging overhead.
    /// For best performance, call `separate()` directly in a loop instead.
    ///
    /// ```swift
    /// // Recommended (faster):
    /// for conditioning in conditionings {
    ///     let result = inference.separate(audio: audio, conditioning: conditioning)
    /// }
    ///
    /// // Not recommended (slower due to logging overhead):
    /// let results = inference.separateMultipleEmbeddings(audio: audio, conditionings: conditionings)
    /// ```
    ///
    /// - Parameters:
    ///   - audio: Input audio as MLXArray
    ///   - conditionings: Array of conditioning embeddings
    ///   - useSimpleSegmentation: Use simple non-overlapping segmentation (default: true)
    /// - Returns: Array of separated audio, one per conditioning
    public func separateMultipleEmbeddings(
        audio: MLXArray,
        conditionings: [MLXArray],
        useSimpleSegmentation: Bool = true
    ) -> [MLXArray] {
        // Prewarm model if needed with first conditioning
        if !isPrewarmed && !conditionings.isEmpty {
            prewarm(conditioning: conditionings[0])
        }
        
        print("[Inference] Processing \(conditionings.count) embeddings sequentially")
        
        // Process each embedding
        var results: [MLXArray] = []
        for (idx, conditioning) in conditionings.enumerated() {
            print("[Inference] Processing embedding \(idx + 1)/\(conditionings.count)")
            let result = separate(
                audio: audio,
                conditioning: conditioning,
                compile: true,
                useSimpleSegmentation: useSimpleSegmentation
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

/// Run USS inference on audio file
public func runUSS(
    audioPath: String,
    modelPath: String,
    embeddingType: EmbeddingLoader.EmbeddingType,
    embeddingsDir: String,
    outputDir: String,
    compile: Bool = false
) throws {
    // Load audio
    // let audioLoader = AudioLoader()
    // let audio = try audioLoader.loadAudio(from: audioPath)
    // AudioUtils
    let config = AudioLoader.Configuration(
         targetSampleRate: 32000,
         maxFileSize: .max, // Skip file size validation
         maxDuration: .greatestFiniteMagnitude,
         enableFloat16: false,
         normalizationMode: .none,
         resamplingMethod: .avAudioConverter(
             algorithm: AVSampleRateConverterAlgorithm_Mastering,
             quality: .max
         )
    )
    let audioLoader = AudioLoader(config: config)
    let audio = try audioLoader.loadMono(from: URL(fileURLWithPath: audioPath)).reshaped([1, -1])

    // Load model
    let model = ResUNet30()
    try WeightLoader.loadWeights(model: model, from: modelPath)
    // MLX Swift doesn't expose training setter, weights are loaded in eval mode
    
    // Load embedding
    let conditioning = try EmbeddingLoader.loadEmbedding(type: embeddingType, from: embeddingsDir)
    
    // Create inference pipeline
    let inference = USSInference(model: model, compile: compile, segmentBatchSize: 1)
    
    // Run separation
    print("Running source separation...")
    let separated = inference.separate(
        audio: audio,
        conditioning: conditioning,
        compile: compile,
        useSimpleSegmentation: true  // Use Python-compatible segmentation
    )
    
    // Save results
    let audioSaver = AudioSaver()
    let baseName = URL(fileURLWithPath: audioPath).deletingPathExtension().lastPathComponent
    
    // Handle different output shapes
    let outputAudio: MLXArray
    if separated.ndim == 3 {
        // (batch, channels, samples) -> extract first batch and channel
        outputAudio = separated[0, 0, 0...]
    } else if separated.ndim == 2 {
        // (batch, samples) -> extract first batch
        outputAudio = separated[0, 0...]
    } else {
        // Just use as is
        outputAudio = separated
    }
    try audioSaver.save(outputAudio, to: "\(outputDir)/\(baseName)_\(embeddingType.rawValue).wav")
    
    print("Separation complete! Result saved to \(outputDir)")
}
