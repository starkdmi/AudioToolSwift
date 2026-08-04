// Copyright © 2025
// UpsampleConformerEncoder - Conformer encoder with upsampling for S3Gen
// Ported from Python MLX

import Foundation
import MLX
import MLXNN

// MARK: - Upsample 1D

/// 1D upsampling layer with convolution
public class Upsample1D: Module {
    let channels: Int
    let outChannels: Int
    let stride: Int

    @ModuleInfo(key: "conv") public var conv: Conv1d

    public init(channels: Int, outChannels: Int, stride: Int = 2) {
        self.channels = channels
        self.outChannels = outChannels
        self.stride = stride
        self._conv = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: channels,
            outputChannels: outChannels,
            kernelSize: stride * 2 + 1,
            stride: 1,
            padding: 0
        ))
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, _ inputLengths: MLXArray) -> (MLXArray, MLXArray) {
        var outputs = MLX.repeat(inputs, count: stride, axis: 2)
        outputs = padded(outputs, widths: [[0, 0], [0, 0], [stride * 2, 0]])
        outputs = outputs.transposed(0, 2, 1)
        outputs = conv(outputs)
        outputs = outputs.transposed(0, 2, 1)
        return (outputs, inputLengths * stride)
    }
}

// MARK: - Pre-Lookahead Layer

/// Pre-lookahead layer for causal processing
public class PreLookaheadLayer: Module {
    let channels: Int
    let preLookaheadLen: Int

    @ModuleInfo(key: "conv1") public var conv1: Conv1d
    @ModuleInfo(key: "conv2") public var conv2: Conv1d

    public init(channels: Int, preLookaheadLen: Int = 1) {
        self.channels = channels
        self.preLookaheadLen = preLookaheadLen
        self._conv1 = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: preLookaheadLen + 1,
            stride: 1,
            padding: 0
        ))
        self._conv2 = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            stride: 1,
            padding: 0
        ))
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, context: MLXArray? = nil) -> MLXArray {
        var outputs = inputs

        if let context = context, context.shape[1] > 0 {
            outputs = concatenated([outputs, context], axis: 1)
            let remaining = preLookaheadLen - context.shape[1]
            if remaining > 0 {
                outputs = padded(outputs, widths: [[0, 0], [0, remaining], [0, 0]])
            }
        } else {
            outputs = padded(outputs, widths: [[0, 0], [0, preLookaheadLen], [0, 0]])
        }

        outputs = leakyRelu(conv1(outputs))
        outputs = padded(outputs, widths: [[0, 0], [2, 0], [0, 0]])
        outputs = conv2(outputs)
        return outputs + inputs
    }
}

// MARK: - Mask helpers

private func makePadMask(_ lengths: MLXArray, maxLen: Int) -> MLXArray {
    let batchSize = lengths.shape[0]
    let seqRange = MLXArray(stride(from: 0, to: maxLen, by: 1))
    let seqRangeExpand = broadcastTo(seqRange.expandedDimensions(axis: 0), [batchSize, maxLen])
    let seqLengthExpand = lengths.expandedDimensions(axis: -1)
    return seqRangeExpand .>= seqLengthExpand
}

private func subsequentChunkMask(size: Int, chunkSize: Int, numLeftChunks: Int = -1) -> MLXArray {
    let posIdx = MLXArray(stride(from: 0, to: size, by: 1))
    let blockValue = ((posIdx / chunkSize) + 1) * chunkSize
    let ret = expandDims(posIdx, axis: 0) .< expandDims(blockValue, axis: 1)
    return ret
}

private func addOptionalChunkMask(
    xs: MLXArray,
    masks: MLXArray,
    useDynamicChunk: Bool,
    useDynamicLeftChunk: Bool,
    decodingChunkSize: Int,
    staticChunkSize: Int,
    numDecodingLeftChunks: Int
) -> MLXArray {
    var chunkMasks: MLXArray
    if useDynamicChunk {
        let maxLen = xs.shape[1]
        let chunkSize: Int
        let numLeftChunks: Int
        if decodingChunkSize < 0 {
            chunkSize = maxLen
            numLeftChunks = -1
        } else if decodingChunkSize > 0 {
            chunkSize = decodingChunkSize
            numLeftChunks = numDecodingLeftChunks
        } else {
            chunkSize = maxLen
            numLeftChunks = -1
        }

        chunkMasks = subsequentChunkMask(size: xs.shape[1], chunkSize: chunkSize, numLeftChunks: numLeftChunks)
        chunkMasks = expandDims(chunkMasks, axis: 0)
        chunkMasks = logicalAnd(masks, chunkMasks)
    } else if staticChunkSize > 0 {
        let numLeftChunks = numDecodingLeftChunks
        chunkMasks = subsequentChunkMask(size: xs.shape[1], chunkSize: staticChunkSize, numLeftChunks: numLeftChunks)
        chunkMasks = expandDims(chunkMasks, axis: 0)
        chunkMasks = logicalAnd(masks, chunkMasks)
    } else {
        chunkMasks = masks
    }

    let maskSums = MLX.sum(chunkMasks, axis: -1)
    if MLX.any(maskSums .== 0).item(Bool.self) {
        let expanded = expandDims(maskSums .== 0, axis: -1)
        chunkMasks = MLX.where(expanded, MLX.ones(like: chunkMasks), chunkMasks)
    }

    return chunkMasks
}

// MARK: - Upsample Conformer Encoder

/// Conformer encoder with upsampling for speech synthesis
public class UpsampleConformerEncoder: Module {
    let outputDim: Int
    let numEncoders: Int
    let numUpEncoders: Int
    let staticChunkSize: Int
    let useDynamicChunk: Bool
    let useDynamicLeftChunk: Bool
    let normalizeBefore: Bool

    @ModuleInfo(key: "embed") public var embed: LinearNoSubsampling
    @ModuleInfo(key: "up_embed") public var up_embed: LinearNoSubsampling
    @ModuleInfo(key: "after_norm") public var after_norm: ManualLayerNorm
    @ModuleInfo(key: "pre_lookahead_layer") public var pre_lookahead_layer: PreLookaheadLayer
    @ModuleInfo(key: "up_layer") public var up_layer: Upsample1D

    @ModuleInfo(key: "encoders_0") public var encoders_0: ConformerEncoderLayer
    @ModuleInfo(key: "encoders_1") public var encoders_1: ConformerEncoderLayer
    @ModuleInfo(key: "encoders_2") public var encoders_2: ConformerEncoderLayer
    @ModuleInfo(key: "encoders_3") public var encoders_3: ConformerEncoderLayer
    @ModuleInfo(key: "encoders_4") public var encoders_4: ConformerEncoderLayer
    @ModuleInfo(key: "encoders_5") public var encoders_5: ConformerEncoderLayer

    @ModuleInfo(key: "up_encoders_0") public var up_encoders_0: ConformerEncoderLayer
    @ModuleInfo(key: "up_encoders_1") public var up_encoders_1: ConformerEncoderLayer
    @ModuleInfo(key: "up_encoders_2") public var up_encoders_2: ConformerEncoderLayer
    @ModuleInfo(key: "up_encoders_3") public var up_encoders_3: ConformerEncoderLayer

    public init(
        inputSize: Int = 512,
        outputSize: Int = 512,
        attentionHeads: Int = 8,
        linearUnits: Int = 2048,
        numBlocks: Int = 6,
        numUpBlocks: Int = 4,
        dropoutRate: Float = 0.1,
        positionalDropoutRate: Float = 0.1,
        attentionDropoutRate: Float = 0.1,
        inputLayer: String = "linear",
        posEncLayerType: String = "rel_pos_espnet",
        normalizeBefore: Bool = true,
        staticChunkSize: Int = 0,
        useDynamicChunk: Bool = false,
        useDynamicLeftChunk: Bool = false,
        positionwiseConvKernelSize: Int = 1,
        macaronStyle: Bool = false,
        selfattentionLayerType: String = "rel_selfattn",
        activationType: String = "swish",
        useCnnModule: Bool = false,
        cnnModuleKernel: Int = 15,
        causal: Bool = false,
        cnnModuleNorm: String = "batch_norm",
        keyBias: Bool = true,
        preLookaheadLen: Int = 3,
        upsampleStride: Int = 2
    ) {
        self.outputDim = outputSize
        self.numEncoders = numBlocks
        self.numUpEncoders = numUpBlocks
        self.staticChunkSize = staticChunkSize
        self.useDynamicChunk = useDynamicChunk
        self.useDynamicLeftChunk = useDynamicLeftChunk
        self.normalizeBefore = normalizeBefore

        let posEnc = EspnetRelPositionalEncoding(dModel: outputSize, dropoutRate: positionalDropoutRate)
        let upPosEnc = EspnetRelPositionalEncoding(dModel: outputSize, dropoutRate: positionalDropoutRate)

        self._embed = ModuleInfo(wrappedValue: LinearNoSubsampling(inputSize: inputSize, outputSize: outputSize, dropoutRate: dropoutRate, posEnc: posEnc))
        self._up_embed = ModuleInfo(wrappedValue: LinearNoSubsampling(inputSize: inputSize, outputSize: outputSize, dropoutRate: dropoutRate, posEnc: upPosEnc))
        self._after_norm = ModuleInfo(wrappedValue: ManualLayerNorm(dimensions: outputSize, eps: 1e-5))
        self._pre_lookahead_layer = ModuleInfo(wrappedValue: PreLookaheadLayer(channels: outputSize, preLookaheadLen: preLookaheadLen))
        self._up_layer = ModuleInfo(wrappedValue: Upsample1D(channels: outputSize, outChannels: outputSize, stride: upsampleStride))

        func makeEncoder() -> ConformerEncoderLayer {
            let attn = RelPositionMultiHeadedAttention(nHead: attentionHeads, nFeat: outputSize, dropoutRate: attentionDropoutRate, keyBias: keyBias)
            let ff = PositionwiseFeedForward(inputDim: outputSize, hiddenDim: linearUnits, dropoutRate: dropoutRate)
            return ConformerEncoderLayer(size: outputSize, selfAttn: attn, feedForward: ff, dropoutRate: dropoutRate, normalizeBefore: normalizeBefore)
        }

        self._encoders_0 = ModuleInfo(wrappedValue: makeEncoder())
        self._encoders_1 = ModuleInfo(wrappedValue: makeEncoder())
        self._encoders_2 = ModuleInfo(wrappedValue: makeEncoder())
        self._encoders_3 = ModuleInfo(wrappedValue: makeEncoder())
        self._encoders_4 = ModuleInfo(wrappedValue: makeEncoder())
        self._encoders_5 = ModuleInfo(wrappedValue: makeEncoder())

        self._up_encoders_0 = ModuleInfo(wrappedValue: makeEncoder())
        self._up_encoders_1 = ModuleInfo(wrappedValue: makeEncoder())
        self._up_encoders_2 = ModuleInfo(wrappedValue: makeEncoder())
        self._up_encoders_3 = ModuleInfo(wrappedValue: makeEncoder())

        super.init()
    }

    public func outputSize() -> Int {
        return outputDim
    }

    public func callAsFunction(
        _ xs: MLXArray,
        _ xsLens: MLXArray,
        context: MLXArray? = nil,
        decodingChunkSize: Int = 0,
        numDecodingLeftChunks: Int = -1,
        streaming: Bool = false
    ) -> (MLXArray, MLXArray) {
        var masks = MLX.logicalNot(makePadMask(xsLens, maxLen: xs.shape[1]))
        masks = expandDims(masks, axis: 1)

        var (x, posEmb, masksOut) = embed(xs, masks: masks)
        var embeddedContext: MLXArray? = nil
        if let context = context, context.shape[1] > 0 {
            let contextMasks = MLX.ones([1, 1, context.shape[1]])
            let (ctx, _, _) = embed(context, masks: contextMasks, offset: xs.shape[1])
            embeddedContext = ctx
        }

        var maskPad = masksOut
        let effectiveChunkSize = streaming ? staticChunkSize : 0
        var chunkMasks = addOptionalChunkMask(
            xs: x,
            masks: masksOut,
            useDynamicChunk: useDynamicChunk,
            useDynamicLeftChunk: useDynamicLeftChunk,
            decodingChunkSize: decodingChunkSize,
            staticChunkSize: effectiveChunkSize,
            numDecodingLeftChunks: numDecodingLeftChunks
        )

        x = pre_lookahead_layer(x, context: embeddedContext)
        x = forwardLayers(x, chunkMasks, posEmb, maskPad)

        x = x.transposed(0, 2, 1)
        let (xUp, newLens) = up_layer(x, xsLens)
        x = xUp.transposed(0, 2, 1)

        let newT = x.shape[1]
        masks = MLX.logicalNot(makePadMask(newLens, maxLen: newT))
        masks = expandDims(masks, axis: 1)

        (x, posEmb, masksOut) = up_embed(x, masks: masks)
        maskPad = masksOut

        let effectiveUpChunkSize = effectiveChunkSize * up_layer.stride
        chunkMasks = addOptionalChunkMask(
            xs: x,
            masks: masksOut,
            useDynamicChunk: useDynamicChunk,
            useDynamicLeftChunk: useDynamicLeftChunk,
            decodingChunkSize: decodingChunkSize,
            staticChunkSize: effectiveUpChunkSize,
            numDecodingLeftChunks: numDecodingLeftChunks
        )

        x = forwardUpLayers(x, chunkMasks, posEmb, maskPad)
        if normalizeBefore {
            x = after_norm(x)
        }

        return (x, masksOut)
    }

    public func callAsFunctionDebug(
        _ xs: MLXArray,
        _ xsLens: MLXArray,
        context: MLXArray? = nil,
        decodingChunkSize: Int = 0,
        numDecodingLeftChunks: Int = -1,
        streaming: Bool = false
    ) -> (MLXArray, MLXArray, [String: MLXArray]) {
        var debug: [String: MLXArray] = [:]
        var masks = MLX.logicalNot(makePadMask(xsLens, maxLen: xs.shape[1]))
        masks = expandDims(masks, axis: 1)
        debug["enc_input_mask"] = masks

        let embedLinear = embed.linear(xs)
        let embedNorm = embed.norm(embedLinear)
        let embedMean = MLX.mean(embedLinear, axis: -1, keepDims: true)
        let embedVar = MLX.variance(embedLinear, axis: -1, keepDims: true)
        let embedCentered = embedLinear - embedMean
        let embedInvStd = MLX.rsqrt(embedVar + MLXArray(embed.norm.eps))
        var embedNormManual = embedCentered * embedInvStd
        embedNormManual = embedNormManual * embed.norm.weight
        debug["enc_embed_norm_weight"] = embed.norm.weight
        embedNormManual = embedNormManual + embed.norm.bias
        debug["enc_embed_norm_bias"] = embed.norm.bias
        debug["enc_embed_mean"] = embedMean
        debug["enc_embed_var"] = embedVar
        debug["enc_embed_centered"] = embedCentered
        debug["enc_embed_inv_std"] = embedInvStd
        let (embedScaled, embedPosEmb) = embed.pos_enc(embedNorm, offset: 0)
        var x = embedScaled
        var posEmb = embedPosEmb
        var masksOut = masks
        debug["enc_embed_linear"] = embedLinear
        debug["enc_embed_norm"] = embedNorm
        debug["enc_embed_norm_manual"] = embedNormManual
        debug["enc_embed_out"] = x
        debug["enc_pos_emb"] = posEmb

        var embeddedContext: MLXArray? = nil
        if let context = context, context.shape[1] > 0 {
            let contextMasks = MLX.ones([1, 1, context.shape[1]])
            let (ctx, _, _) = embed(context, masks: contextMasks, offset: xs.shape[1])
            embeddedContext = ctx
        }

        let maskPad = masksOut
        let effectiveChunkSize = streaming ? staticChunkSize : 0
        var chunkMasks = addOptionalChunkMask(
            xs: x,
            masks: masksOut,
            useDynamicChunk: useDynamicChunk,
            useDynamicLeftChunk: useDynamicLeftChunk,
            decodingChunkSize: decodingChunkSize,
            staticChunkSize: effectiveChunkSize,
            numDecodingLeftChunks: numDecodingLeftChunks
        )
        debug["enc_chunk_mask"] = chunkMasks

        var preOut = x
        if let context = embeddedContext, context.shape[1] > 0 {
            preOut = concatenated([preOut, context], axis: 1)
            let remaining = pre_lookahead_layer.preLookaheadLen - context.shape[1]
            if remaining > 0 {
                preOut = padded(preOut, widths: [[0, 0], [0, remaining], [0, 0]])
            }
        } else {
            preOut = padded(preOut, widths: [[0, 0], [0, pre_lookahead_layer.preLookaheadLen], [0, 0]])
        }
        debug["enc_pre_lookahead_pad"] = preOut
        let preConv1 = pre_lookahead_layer.conv1(preOut)
        debug["enc_pre_lookahead_conv1"] = preConv1
        debug["enc_pre_lookahead_conv1_weight"] = pre_lookahead_layer.conv1.weight
        if let conv1Bias = pre_lookahead_layer.conv1.bias {
            debug["enc_pre_lookahead_conv1_bias"] = conv1Bias
        }
        let preRelu = leakyRelu(preConv1)
        debug["enc_pre_lookahead_relu"] = preRelu
        let prePad2 = padded(preRelu, widths: [[0, 0], [2, 0], [0, 0]])
        debug["enc_pre_lookahead_pad2"] = prePad2
        let preConv2 = pre_lookahead_layer.conv2(prePad2)
        debug["enc_pre_lookahead_conv2"] = preConv2
        debug["enc_pre_lookahead_conv2_weight"] = pre_lookahead_layer.conv2.weight
        if let conv2Bias = pre_lookahead_layer.conv2.bias {
            debug["enc_pre_lookahead_conv2_bias"] = conv2Bias
        }
        x = preConv2 + x
        debug["enc_pre_lookahead"] = x

        var out = x
        let encoders = [encoders_0, encoders_1, encoders_2, encoders_3, encoders_4, encoders_5]
        for (idx, layer) in encoders.enumerated() {
            if idx == 0 {
                let (xLayer, layerDebug) = layer.callAsFunctionDebug(out, chunkMasks, posEmb, maskPad)
                out = xLayer
                debug["enc_layer0_out"] = out
                if let attnRaw = layerDebug["attn_raw"] {
                    debug["enc_layer0_attn_raw"] = attnRaw
                }
                if let attnResid = layerDebug["attn_resid"] {
                    debug["enc_layer0_attn_resid"] = attnResid
                }
                if let ffRaw = layerDebug["ff_raw"] {
                    debug["enc_layer0_ff_raw"] = ffRaw
                }
                if let ffResid = layerDebug["ff_resid"] {
                    debug["enc_layer0_ff_resid"] = ffResid
                }
                for (key, value) in layerDebug where key.hasPrefix("attn_") {
                    debug["enc_layer0_\(key)"] = value
                }
                debug["enc_layer0_pos_bias_u"] = layer.self_attn.pos_bias_u
                debug["enc_layer0_pos_bias_v"] = layer.self_attn.pos_bias_v
                debug["enc_layer0_linear_q_weight"] = layer.self_attn.linear_q.weight
                debug["enc_layer0_linear_pos_weight"] = layer.self_attn.linear_pos.weight
                debug["enc_layer0_linear_out_weight"] = layer.self_attn.linear_out.weight
                if let outBias = layer.self_attn.linear_out.bias {
                    debug["enc_layer0_linear_out_bias"] = outBias
                }
            } else {
                let (xLayer, _, _, _) = layer(out, chunkMasks, posEmb, maskPad)
                out = xLayer
            }
            if idx == encoders.count - 1 {
                debug["enc_layer_last_out"] = out
            }
        }
        x = out

        x = x.transposed(0, 2, 1)
        let upInput = x
        let (xUp, newLens) = up_layer(x, xsLens)
        x = xUp.transposed(0, 2, 1)
        debug["enc_up_layer_out"] = x
        let upRepeat = MLX.repeat(upInput, count: up_layer.stride, axis: 2)
        let upPad = padded(upRepeat, widths: [[0, 0], [0, 0], [up_layer.stride * 2, 0]])
        let upConvIn = upPad.transposed(0, 2, 1)
        let upConvOut = up_layer.conv(upConvIn)
        debug["enc_up_repeat"] = upRepeat
        debug["enc_up_pad"] = upPad
        debug["enc_up_conv_in"] = upConvIn
        debug["enc_up_conv_out"] = upConvOut

        let newT = x.shape[1]
        masks = MLX.logicalNot(makePadMask(newLens, maxLen: newT))
        masks = expandDims(masks, axis: 1)

        (x, posEmb, masksOut) = up_embed(x, masks: masks)
        debug["enc_up_embed_out"] = x
        debug["enc_up_pos_emb"] = posEmb

        let effectiveUpChunkSize = effectiveChunkSize * up_layer.stride
        chunkMasks = addOptionalChunkMask(
            xs: x,
            masks: masksOut,
            useDynamicChunk: useDynamicChunk,
            useDynamicLeftChunk: useDynamicLeftChunk,
            decodingChunkSize: decodingChunkSize,
            staticChunkSize: effectiveUpChunkSize,
            numDecodingLeftChunks: numDecodingLeftChunks
        )
        debug["enc_up_chunk_mask"] = chunkMasks

        out = x
        let upEncoders = [up_encoders_0, up_encoders_1, up_encoders_2, up_encoders_3]
        for (idx, layer) in upEncoders.enumerated() {
            if idx == 0 {
                let (xLayer, layerDebug) = layer.callAsFunctionDebug(out, chunkMasks, posEmb, masksOut)
                out = xLayer
                debug["enc_up_layer0_out"] = out
                if let attnRaw = layerDebug["attn_raw"] {
                    debug["enc_up_layer0_attn_raw"] = attnRaw
                }
                if let attnResid = layerDebug["attn_resid"] {
                    debug["enc_up_layer0_attn_resid"] = attnResid
                }
                if let ffRaw = layerDebug["ff_raw"] {
                    debug["enc_up_layer0_ff_raw"] = ffRaw
                }
                if let ffResid = layerDebug["ff_resid"] {
                    debug["enc_up_layer0_ff_resid"] = ffResid
                }
                for (key, value) in layerDebug where key.hasPrefix("attn_") {
                    debug["enc_up_layer0_\(key)"] = value
                }
            } else {
                let (xLayer, _, _, _) = layer(out, chunkMasks, posEmb, masksOut)
                out = xLayer
            }
            if idx == upEncoders.count - 1 {
                debug["enc_up_layer_last_out"] = out
            }
        }
        x = out

        if normalizeBefore {
            x = after_norm(x)
        }
        debug["enc_after_norm"] = x
        debug["enc_output_mask"] = masksOut

        return (x, masksOut, debug)
    }

    private func forwardLayers(_ xs: MLXArray, _ chunkMasks: MLXArray, _ posEmb: MLXArray, _ maskPad: MLXArray) -> MLXArray {
        var out = xs
        for layer in [encoders_0, encoders_1, encoders_2, encoders_3, encoders_4, encoders_5] {
            let (x, _, _, _) = layer(out, chunkMasks, posEmb, maskPad)
            out = x
        }
        return out
    }

    private func forwardUpLayers(_ xs: MLXArray, _ chunkMasks: MLXArray, _ posEmb: MLXArray, _ maskPad: MLXArray) -> MLXArray {
        var out = xs
        for layer in [up_encoders_0, up_encoders_1, up_encoders_2, up_encoders_3] {
            let (x, _, _, _) = layer(out, chunkMasks, posEmb, maskPad)
            out = x
        }
        return out
    }
}
