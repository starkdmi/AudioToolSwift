// Copyright © 2025
// CausalMaskedDiffWithXvec - Flow matching wrapper for S3Gen tokens-to-mel
// Wrapper combining encoder, CFM decoder, and speaker conditioning

import Foundation
import MLX
import MLXNN

private func l2Norm(_ x: MLXArray, axis: Int, keepDims: Bool) -> MLXArray {
    MLX.sqrt(MLX.sum(x * x, axis: axis, keepDims: keepDims))
}

/// Causal masked diffusion model with speaker embeddings for streaming TTS
public class CausalMaskedDiffWithXvec: Module {
    public let inputSize: Int
    public let outputSize: Int
    public let vocabSize: Int
    public let tokenMelRatio: Int
    public let preLookaheadLen: Int
    public let nTimesteps: Int

    @ModuleInfo(key: "input_embedding") public var input_embedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") public var spk_embed_affine_layer: Linear
    @ModuleInfo public var encoder: UpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") public var encoder_proj: Linear
    @ModuleInfo public var decoder: CausalConditionalCFM

    public init(
        inputSize: Int = 512,
        outputSize: Int = 80,
        spkEmbedDim: Int = 192,
        vocabSize: Int = 6561,
        tokenMelRatio: Int = 2,
        preLookaheadLen: Int = 3,
        nTimesteps: Int = 10
    ) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.vocabSize = vocabSize
        self.tokenMelRatio = tokenMelRatio
        self.preLookaheadLen = preLookaheadLen
        self.nTimesteps = nTimesteps

        self._input_embedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: vocabSize, dimensions: inputSize))
        self._spk_embed_affine_layer = ModuleInfo(wrappedValue: Linear(spkEmbedDim, outputSize))
        self._encoder = ModuleInfo(wrappedValue: UpsampleConformerEncoder(inputSize: inputSize, outputSize: inputSize))
        self._encoder_proj = ModuleInfo(wrappedValue: Linear(inputSize, outputSize))

        let estimator = ConditionalDecoder(
            inChannels: 320,
            outChannels: 80,
            causal: true
        )
        self._decoder = ModuleInfo(wrappedValue: CausalConditionalCFM(
            inChannels: 320,
            cfmParams: CFMParams(),
            nSpks: 1,
            spkEmbDim: outputSize,
            estimator: estimator
        ))

        super.init()
    }

    /// Inference for mel generation from speech tokens
    public func inference(
        token: MLXArray,
        tokenLen: MLXArray,
        promptToken: MLXArray,
        promptTokenLen: MLXArray,
        promptFeat: MLXArray,
        promptFeatLen: MLXArray,
        embedding: MLXArray,
        finalize: Bool,
        nTimesteps: Int? = nil,
        streaming: Bool = false
    ) -> (MLXArray, MLXArray?) {
        var tokens = token
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        var spkEmb = embedding
        let norm = l2Norm(spkEmb, axis: 1, keepDims: true)
        spkEmb = spkEmb / (norm + 1e-12)
        spkEmb = spk_embed_affine_layer(spkEmb)

        tokens = concatenated([promptToken, tokens], axis: 1)
        let tokenLenCombined = promptTokenLen + tokenLen

        let batchSize = tokenLenCombined.shape[0]
        let maxLen = tokens.shape[1]
        let seqRange = MLXArray(stride(from: 0, to: maxLen, by: 1))
        let seqRangeExpand = broadcastTo(seqRange.expandedDimensions(axis: 0), [batchSize, maxLen])
        let seqLengthExpand = tokenLenCombined.expandedDimensions(axis: -1)
        var mask = (seqRangeExpand .< seqLengthExpand)
        mask = mask.expandedDimensions(axis: -1).asType(spkEmb.dtype)

        let numEmbeddings = input_embedding.weight.shape[0]
        let maxToken = MLXArray(Int32(numEmbeddings - 1))
        let clipped = MLX.minimum(MLX.maximum(tokens, MLXArray(Int32(0))), maxToken)
        let tokenIds = clipped.asType(Int32.self)
        var h = input_embedding(tokenIds) * mask

        let (encoded, hMask) = encoder(h, tokenLenCombined, streaming: streaming)
        h = encoded

        if !finalize {
            let trimLen = h.shape[1] - preLookaheadLen * tokenMelRatio
            h = h[0..., 0..<trimLen, 0...]
        }

        let melLen1 = promptFeat.shape[1]
        let melLen2 = h.shape[1] - melLen1
        h = encoder_proj(h)

        let totalLen = melLen1 + melLen2
        var conds = MLX.zeros([1, totalLen, outputSize], dtype: h.dtype)
        if melLen1 > 0 {
            conds[0..., 0..<melLen1, 0...] = promptFeat
        }
        conds = conds.transposed(0, 2, 1)

        let decoderMask: MLXArray
        if hMask.shape[2] == h.shape[1] {
            decoderMask = hMask.asType(h.dtype)
        } else {
            decoderMask = MLX.ones([1, 1, totalLen], dtype: h.dtype)
        }

        let mu = h.transposed(0, 2, 1)
        let (feat, _) = decoder(
            mu: mu,
            mask: decoderMask,
            nTimesteps: nTimesteps ?? self.nTimesteps,
            temperature: 1.0,
            spks: spkEmb,
            cond: conds,
            streaming: streaming,
            noisedMels: nil
        )

        let newFeat = feat[0..., 0..., melLen1...]
        return (newFeat, nil)
    }

    public func inferenceDebug(
        token: MLXArray,
        tokenLen: MLXArray,
        promptToken: MLXArray,
        promptTokenLen: MLXArray,
        promptFeat: MLXArray,
        promptFeatLen: MLXArray,
        embedding: MLXArray,
        finalize: Bool,
        nTimesteps: Int? = nil,
        streaming: Bool = false
    ) -> (MLXArray, [String: MLXArray]) {
        var tokens = token
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        var spkEmb = embedding
        let norm = l2Norm(spkEmb, axis: 1, keepDims: true)
        spkEmb = spkEmb / (norm + 1e-12)
        spkEmb = spk_embed_affine_layer(spkEmb)

        tokens = concatenated([promptToken, tokens], axis: 1)
        let tokenLenCombined = promptTokenLen + tokenLen

        let batchSize = tokenLenCombined.shape[0]
        let maxLen = tokens.shape[1]
        let seqRange = MLXArray(stride(from: 0, to: maxLen, by: 1))
        let seqRangeExpand = broadcastTo(seqRange.expandedDimensions(axis: 0), [batchSize, maxLen])
        let seqLengthExpand = tokenLenCombined.expandedDimensions(axis: -1)
        var mask = (seqRangeExpand .< seqLengthExpand)
        mask = mask.expandedDimensions(axis: -1).asType(spkEmb.dtype)

        let numEmbeddings = input_embedding.weight.shape[0]
        let maxToken = MLXArray(Int32(numEmbeddings - 1))
        let clipped = MLX.minimum(MLX.maximum(tokens, MLXArray(Int32(0))), maxToken)
        let tokenIds = clipped.asType(Int32.self)
        let tokenEmbed = input_embedding(tokenIds) * mask

        let (encoded, hMask, encDebug) = encoder.callAsFunctionDebug(
            tokenEmbed,
            tokenLenCombined,
            context: nil,
            decodingChunkSize: 0,
            numDecodingLeftChunks: -1,
            streaming: streaming
        )
        var h = encoded

        if !finalize {
            let trimLen = h.shape[1] - preLookaheadLen * tokenMelRatio
            h = h[0..., 0..<trimLen, 0...]
        }

        let melLen1 = promptFeat.shape[1]
        let melLen2 = h.shape[1] - melLen1
        let hProj = encoder_proj(h)

        let totalLen = melLen1 + melLen2
        var conds = MLX.zeros([1, totalLen, outputSize], dtype: hProj.dtype)
        if melLen1 > 0 {
            conds[0..., 0..<melLen1, 0...] = promptFeat
        }
        conds = conds.transposed(0, 2, 1)

        let decoderMask: MLXArray
        if hMask.shape[2] == hProj.shape[1] {
            decoderMask = hMask.asType(hProj.dtype)
        } else {
            decoderMask = MLX.ones([1, 1, totalLen], dtype: hProj.dtype)
        }

        let mu = hProj.transposed(0, 2, 1)
        let (feat, decoderDebug) = decoder.callAsFunctionDebug(
            mu: mu,
            mask: decoderMask,
            nTimesteps: nTimesteps ?? self.nTimesteps,
            temperature: 1.0,
            spks: spkEmb,
            cond: conds,
            streaming: streaming,
            noisedMels: nil
        )

        let newFeat = feat[0..., 0..., melLen1...]
        var debugOutputs: [String: MLXArray] = [
            "flow_spk_emb": spkEmb,
            "flow_mask": mask,
            "flow_token_embed": tokenEmbed,
            "flow_encoder_out": encoded,
            "flow_encoder_mask": hMask,
            "flow_encoder_proj": hProj,
            "flow_conds": conds,
            "flow_decoder_mask": decoderMask,
            "flow_mu": mu,
            "flow_decoder_out_full": feat,
            "flow_decoder_out": newFeat,
        ]
        debugOutputs.merge(decoderDebug, uniquingKeysWith: { current, _ in current })
        for (key, value) in encDebug {
            debugOutputs[key] = value
        }
        return (newFeat, debugOutputs)
    }
}
