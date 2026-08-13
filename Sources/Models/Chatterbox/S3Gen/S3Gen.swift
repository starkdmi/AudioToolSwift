// Copyright © 2025
// S3Gen - Token-to-Waveform decoder using flow matching and HiFi-GAN
// Pure MLX port integrating all S3Gen components

import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioUtils

// MARK: - Constants

public let S3GEN_SR = 24000
public let S3_SR = 16000
public let SPEECH_VOCAB_SIZE = 6561

/// Speech tokens per second. One token decodes to `S3GEN_SR / S3_TOKEN_RATE` = 960 samples.
public let S3_TOKEN_RATE = 25

// MARK: - S3Token2Mel

/// S3Gen CFM decoder: speech tokens -> mel spectrograms
public class S3Token2Mel: Module {
    @ModuleInfo public var speaker_encoder: CAMPPlus
    @ModuleInfo(key: "flow") public var flow: CausalMaskedDiffWithXvec
    
    public override init() {
        self._speaker_encoder = ModuleInfo(wrappedValue: CAMPPlus())
        self._flow = ModuleInfo(wrappedValue: CausalMaskedDiffWithXvec())
        
        super.init()
    }
    
    public func embed_ref(
        ref_wav: MLXArray,
        ref_sr: Int,
        ref_speech_tokens: MLXArray,
        ref_speech_token_lens: MLXArray
    ) -> [String: MLXArray] {
        var wav = ref_wav
        if wav.ndim == 1 {
            wav = wav.expandedDimensions(axis: 0)
        }

        func resampleBatch(_ audio: MLXArray, origSR: Int, targetSR: Int) -> MLXArray {
            if origSR == targetSR { return audio }
            if audio.shape[0] == 1 {
                let resampled = resampleAudioPolyphase(audio[0], origSR: origSR, targetSR: targetSR)
                return resampled.expandedDimensions(axis: 0)
            }
            var resampled: [MLXArray] = []
            resampled.reserveCapacity(audio.shape[0])
            for i in 0..<audio.shape[0] {
                resampled.append(resampleAudioPolyphase(audio[i], origSR: origSR, targetSR: targetSR))
            }
            let maxLen = resampled.map { $0.shape[0] }.max() ?? 0
            let padded = resampled.map { sample -> MLXArray in
                if sample.shape[0] < maxLen {
                    let pad = MLXArray.zeros([maxLen - sample.shape[0]], dtype: sample.dtype)
                    return concatenated([sample, pad], axis: 0)
                }
                return sample
            }
            return stacked(padded, axis: 0)
        }

        let wav24 = resampleBatch(wav, origSR: ref_sr, targetSR: S3GEN_SR)
        let wav16 = resampleBatch(wav, origSR: ref_sr, targetSR: S3_SR)

        let refMels24: MLXArray
        do {
            refMels24 = try melSpectrogram(
                wav24,
                nFFT: 1920,
                numMels: 80,
                samplingRate: S3GEN_SR,
                hopSize: 480,
                winSize: 1920,
                fmin: 0,
                fmax: 8000,
                center: false
            )
        } catch {
            refMels24 = MLXArray.zeros([wav24.shape[0], 80, 1])
        }
        var promptFeat = refMels24.transposed(0, 2, 1)
        let xVector = speaker_encoder.inference(wav16)

        var promptToken = ref_speech_tokens
        var actualTokenLen = promptToken.shape[1]
        let expectedTokenLen = promptFeat.shape[1] / 2
        if actualTokenLen != expectedTokenLen {
            if actualTokenLen < expectedTokenLen {
                let expectedMelLen = 2 * actualTokenLen
                promptFeat = promptFeat[0..., 0..<expectedMelLen, 0...]
            } else {
                promptToken = promptToken[0..., 0..<expectedTokenLen]
                actualTokenLen = expectedTokenLen
            }
        }

        let promptTokenLen = MLXArray([Int32(actualTokenLen)])
        let promptFeatLen = MLXArray([Int32(promptFeat.shape[1])])

        return [
            "prompt_token": promptToken,
            "prompt_token_len": promptTokenLen,
            "prompt_feat": promptFeat,
            "prompt_feat_len": promptFeatLen,
            "embedding": xVector
        ]
    }
    
    public func callAsFunction(
        speechTokens: MLXArray,
        refDict: [String: MLXArray],
        finalize: Bool = false
    ) -> MLXArray {
        var tokens = speechTokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        
        let tokenLen = MLXArray([Int32(tokens.shape[1])])
        
        // Get reference data with defaults
        let promptToken = refDict["prompt_token"] ?? MLXArray.zeros([1, 1], dtype: .int32)
        let promptTokenLen = refDict["prompt_token_len"] ?? MLXArray([1])
        let promptFeat = refDict["prompt_feat"] ?? MLX.zeros([1, 1, 80])
        let promptFeatLen = refDict["prompt_feat_len"] ?? MLXArray([1])
        let embedding = refDict["embedding"] ?? MLX.zeros([1, 80])
        
        // Use flow decoder for mel generation
        let (mels, _) = flow.inference(
            token: tokens.asType(Int32.self),
            tokenLen: tokenLen,
            promptToken: promptToken.asType(Int32.self),
            promptTokenLen: promptTokenLen,
            promptFeat: promptFeat,
            promptFeatLen: promptFeatLen,
            embedding: embedding,
            finalize: finalize
        )
        
        return mels
    }
}

// MARK: - S3Token2Wav

/// Full S3Gen decoder: token-to-mel (CFM) + mel-to-waveform (HiFi-GAN)
public class S3Token2Wav: Module {
    @ModuleInfo(key: "flow") public var flowEncoder: S3Token2Mel
    @ModuleInfo public var mel2wav: HiFTGenerator
    
    @ParameterInfo public var trim_fade: MLXArray
    
    public override init() {
        self._flowEncoder = ModuleInfo(wrappedValue: S3Token2Mel())
        self._mel2wav = ModuleInfo(wrappedValue: HiFTGenerator())
        
        let nTrim = S3GEN_SR / 50
        let trimFadeCos = (cos(MLX.linspace(Float.pi, 0, count: nTrim)) + 1) / 2
        self._trim_fade.wrappedValue = concatenated([zeros([nTrim]), trimFadeCos], axis: 0)
        
        super.init()
    }
    
    public func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]
        let currWeights = Dictionary(uniqueKeysWithValues: parameters().flattened())

        var speakerWeights: [String: MLXArray] = [:]
        var otherWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("speaker_encoder.") {
                speakerWeights[String(key.dropFirst("speaker_encoder.".count))] = value
            } else {
                otherWeights[key] = value
            }
        }

        if !speakerWeights.isEmpty {
            let sanitizedSpeaker = flowEncoder.speaker_encoder.sanitize(speakerWeights)
            for (key, value) in sanitizedSpeaker {
                newWeights["flowEncoder.speaker_encoder.\(key)"] = value
            }
        }

        for (key, value) in otherWeights {
            var newKey = key

            if newKey.hasPrefix("flow.") {
                newKey = "flowEncoder.\(newKey)"
            }
            var newValue = value
            
            if key.contains("num_batches_tracked") { continue }
            
            // === Encoder block key transformations (flow encoder) ===
            newKey = transformEncoderKey(newKey)

            // === Decoder block key transformations ===
            // Map pretrained nested index structure to Swift named properties
            newKey = sanitizeDecoderKey(newKey)
            
            // === Tensor transformations ===
            if newValue.ndim == 3, newKey.contains("weight") {
                if let expected = currWeights[newKey], expected.shape != newValue.shape {
                    if newKey.contains(".ups.") {
                        newValue = newValue.transposed(1, 2, 0)
                    } else {
                        newValue = newValue.swappedAxes(1, 2)
                    }
                }
            }
            
            newWeights[newKey] = newValue
        }
        
        return newWeights
    }
    
    /// Transform decoder block keys from pretrained format to Swift module format
    private func sanitizeDecoderKey(_ key: String) -> String {
        var newKey = key
        
        // DownBlock: down_blocks.X.0 = resnet, .X.1.Y = transformer_Y, .X.2 = downsample
        newKey = transformBlockKey(newKey, blockType: "down_blocks", resnetIdx: 0, transformerIdx: 1, downsampleIdx: 2)
        
        // MidBlock: mid_blocks.X.0 = resnet, .X.1.Y = transformer_Y
        newKey = transformMidBlockKey(newKey)
        
        // UpBlock: same structure as DownBlock
        newKey = transformBlockKey(newKey, blockType: "up_blocks", resnetIdx: 0, transformerIdx: 1, downsampleIdx: 2)
        
        // Attention key transformations: attn1.to_q → attn.query_proj
        newKey = newKey.replacingOccurrences(of: ".attn1.to_q.", with: ".attn.query_proj.")
        newKey = newKey.replacingOccurrences(of: ".attn1.to_k.", with: ".attn.key_proj.")
        newKey = newKey.replacingOccurrences(of: ".attn1.to_v.", with: ".attn.value_proj.")
        newKey = newKey.replacingOccurrences(of: ".attn1.to_out.0.", with: ".attn.out_proj.")
        
        // FeedForward key transformations: ff.net.0.proj → ff.layers.0
        newKey = newKey.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.layers.0.")
        newKey = newKey.replacingOccurrences(of: ".ff.net.2.", with: ".ff.layers.1.")
        newKey = newKey.replacingOccurrences(of: ".ff.layers.0.", with: ".ff.layers.layer_0.")
        newKey = newKey.replacingOccurrences(of: ".ff.layers.1.", with: ".ff.layers.layer_1.")
        
        // CausalBlock1D: block.0 → conv.conv, block.2 → norm
        newKey = newKey.replacingOccurrences(of: ".block1.block.0.", with: ".block1.conv.conv.")
        newKey = newKey.replacingOccurrences(of: ".block1.block.2.", with: ".block1.norm.")
        newKey = newKey.replacingOccurrences(of: ".block2.block.0.", with: ".block2.conv.conv.")
        newKey = newKey.replacingOccurrences(of: ".block2.block.2.", with: ".block2.norm.")
        
        // Resnet mlp.1 → mlp_linear
        newKey = newKey.replacingOccurrences(of: ".mlp.1.", with: ".mlp_linear.")
        
        return newKey
    }

    /// Transform encoder keys (encoders.X -> encoders_X, up_encoders.X -> up_encoders_X)
    private func transformEncoderKey(_ key: String) -> String {
        var newKey = key

        let encoderPattern = "\\.encoders\\.(\\d+)\\."
        if let regex = try? NSRegularExpression(pattern: encoderPattern, options: []),
           let match = regex.firstMatch(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey)) {
            if let blockNumRange = Range(match.range(at: 1), in: newKey) {
                let blockNum = String(newKey[blockNumRange])
                newKey = regex.stringByReplacingMatches(
                    in: newKey,
                    options: [],
                    range: NSRange(newKey.startIndex..., in: newKey),
                    withTemplate: ".encoders_\(blockNum)."
                )
            }
        }

        let upEncoderPattern = "\\.up_encoders\\.(\\d+)\\."
        if let regex = try? NSRegularExpression(pattern: upEncoderPattern, options: []),
           let match = regex.firstMatch(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey)) {
            if let blockNumRange = Range(match.range(at: 1), in: newKey) {
                let blockNum = String(newKey[blockNumRange])
                newKey = regex.stringByReplacingMatches(
                    in: newKey,
                    options: [],
                    range: NSRange(newKey.startIndex..., in: newKey),
                    withTemplate: ".up_encoders_\(blockNum)."
                )
            }
        }

        return newKey
    }
    
    /// Transform down_blocks/up_blocks keys
    private func transformBlockKey(_ key: String, blockType: String, resnetIdx: Int, transformerIdx: Int, downsampleIdx: Int) -> String {
        var newKey = key
        
        // Pattern: <blockType>.X.0 → <blockType>_X.resnet
        let resnetPattern = "\\.\(blockType)\\.(\\d+)\\.0\\."
        if let regex = try? NSRegularExpression(pattern: resnetPattern, options: []),
           let match = regex.firstMatch(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey)) {
            if let blockNumRange = Range(match.range(at: 1), in: newKey) {
                let blockNum = String(newKey[blockNumRange])
                newKey = regex.stringByReplacingMatches(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey), withTemplate: ".\(blockType)_\(blockNum).resnet.")
            }
        }
        
        // Pattern: <blockType>.X.1.Y → <blockType>_X.transformer_Y
        let transformerPattern = "\\.\(blockType)\\.(\\d+)\\.1\\.(\\d+)\\."
        if let regex = try? NSRegularExpression(pattern: transformerPattern, options: []),
           let match = regex.firstMatch(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey)) {
            if let blockNumRange = Range(match.range(at: 1), in: newKey),
               let transNumRange = Range(match.range(at: 2), in: newKey) {
                let blockNum = String(newKey[blockNumRange])
                let transNum = String(newKey[transNumRange])
                newKey = regex.stringByReplacingMatches(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey), withTemplate: ".\(blockType)_\(blockNum).transformer_\(transNum).")
            }
        }
        
        // Pattern: <blockType>.X.2 → <blockType>_X.downsample.conv
        let downsamplePattern = "\\.\(blockType)\\.(\\d+)\\.2\\."
        if let regex = try? NSRegularExpression(pattern: downsamplePattern, options: []),
           let match = regex.firstMatch(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey)) {
            if let blockNumRange = Range(match.range(at: 1), in: newKey) {
                let blockNum = String(newKey[blockNumRange])
                newKey = regex.stringByReplacingMatches(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey), withTemplate: ".\(blockType)_\(blockNum).downsample.conv.")
            }
        }
        
        return newKey
    }
    
    /// Transform mid_blocks keys
    private func transformMidBlockKey(_ key: String) -> String {
        var newKey = key
        
        // Pattern: mid_blocks.X.0 → mid_blocks_X.resnet
        let resnetPattern = "\\.mid_blocks\\.(\\d+)\\.0\\."
        if let regex = try? NSRegularExpression(pattern: resnetPattern, options: []),
           let match = regex.firstMatch(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey)) {
            if let blockNumRange = Range(match.range(at: 1), in: newKey) {
                let blockNum = String(newKey[blockNumRange])
                newKey = regex.stringByReplacingMatches(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey), withTemplate: ".mid_blocks_\(blockNum).resnet.")
            }
        }
        
        // Pattern: mid_blocks.X.1.Y → mid_blocks_X.transformer_Y
        let transformerPattern = "\\.mid_blocks\\.(\\d+)\\.1\\.(\\d+)\\."
        if let regex = try? NSRegularExpression(pattern: transformerPattern, options: []),
           let match = regex.firstMatch(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey)) {
            if let blockNumRange = Range(match.range(at: 1), in: newKey),
               let transNumRange = Range(match.range(at: 2), in: newKey) {
                let blockNum = String(newKey[blockNumRange])
                let transNum = String(newKey[transNumRange])
                newKey = regex.stringByReplacingMatches(in: newKey, options: [], range: NSRange(newKey.startIndex..., in: newKey), withTemplate: ".mid_blocks_\(blockNum).transformer_\(transNum).")
            }
        }
        
        return newKey
    }
    
    public func embed_ref(
        ref_wav: MLXArray,
        ref_sr: Int,
        ref_speech_tokens: MLXArray,
        ref_speech_token_lens: MLXArray
    ) -> [String: MLXArray] {
        return flowEncoder.embed_ref(
            ref_wav: ref_wav,
            ref_sr: ref_sr,
            ref_speech_tokens: ref_speech_tokens,
            ref_speech_token_lens: ref_speech_token_lens
        )
    }
    
    public func callAsFunction(
        speechTokens: MLXArray,
        refDict: [String: MLXArray],
        finalize: Bool = false
    ) -> MLXArray {
        let outputMels = flowEncoder(speechTokens: speechTokens, refDict: refDict, finalize: finalize)
        let (outputWavs, _) = mel2wav.inference(speechFeat: outputMels)
        
        let fadeLen = trim_fade.shape[0]
        if outputWavs.shape[outputWavs.ndim - 1] >= fadeLen {
            var faded = outputWavs
            faded[0..., 0..<fadeLen] = faded[0..., 0..<fadeLen] * trim_fade
            return faded
        }

        return outputWavs
    }
    
    public func flow_inference(
        speechTokens: MLXArray,
        refDict: [String: MLXArray],
        finalize: Bool = false
    ) -> MLXArray {
        return flowEncoder(speechTokens: speechTokens, refDict: refDict, finalize: finalize)
    }
    
    public func hift_inference(speechFeat: MLXArray, cacheSource: MLXArray? = nil) -> (MLXArray, MLXArray) {
        return mel2wav.inference(speechFeat: speechFeat, cacheSource: cacheSource)
    }
}
