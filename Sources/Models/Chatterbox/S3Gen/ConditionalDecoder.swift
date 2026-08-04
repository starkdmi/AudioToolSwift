// Copyright © 2025
// ConditionalDecoder - U-Net decoder for flow matching (full port)
// Matches Python chatterbox/s3gen/decoder.py structure

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Mask helpers

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

        var chunkMasks = subsequentChunkMask(size: xs.shape[1], chunkSize: chunkSize, numLeftChunks: numLeftChunks)
        chunkMasks = expandDims(chunkMasks, axis: 0)
        return logicalAnd(masks, chunkMasks)
    } else if staticChunkSize > 0 {
        let numLeftChunks = numDecodingLeftChunks
        var chunkMasks = subsequentChunkMask(size: xs.shape[1], chunkSize: staticChunkSize, numLeftChunks: numLeftChunks)
        chunkMasks = expandDims(chunkMasks, axis: 0)
        return logicalAnd(masks, chunkMasks)
    }

    return broadcastTo(masks, [masks.shape[0], xs.shape[1], xs.shape[1]])
}

// MARK: - Block Wrappers

/// Container for down block components
public class DownBlock: Module {
    @ModuleInfo public var resnet: CausalResnetBlock1D
    @ModuleInfo(key: "transformer_0") public var transformer_0: BasicTransformerBlock
    @ModuleInfo(key: "transformer_1") public var transformer_1: BasicTransformerBlock
    @ModuleInfo(key: "transformer_2") public var transformer_2: BasicTransformerBlock
    @ModuleInfo(key: "transformer_3") public var transformer_3: BasicTransformerBlock
    @ModuleInfo public var downsample: CausalConv1d

    public let n_transformer: Int = 4

    public init(
        inChannels: Int,
        outChannels: Int,
        timeEmbedDim: Int,
        numHeads: Int = 8,
        attentionHeadDim: Int = 64,
        nBlocks: Int = 4
    ) {
        self._resnet = ModuleInfo(wrappedValue: CausalResnetBlock1D(dim: inChannels, dimOut: outChannels, timeEmbDim: timeEmbedDim))
        self._transformer_0 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: outChannels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._transformer_1 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: outChannels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._transformer_2 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: outChannels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._transformer_3 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: outChannels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._downsample = ModuleInfo(wrappedValue: CausalConv1d(inChannels: outChannels, outChannels: outChannels, kernelSize: 3))
        super.init()
    }

    public var transformerBlocks: [BasicTransformerBlock] {
        [transformer_0, transformer_1, transformer_2, transformer_3]
    }
}

/// Container for mid block components
public class MidBlock: Module {
    @ModuleInfo public var resnet: CausalResnetBlock1D
    @ModuleInfo(key: "transformer_0") public var transformer_0: BasicTransformerBlock
    @ModuleInfo(key: "transformer_1") public var transformer_1: BasicTransformerBlock
    @ModuleInfo(key: "transformer_2") public var transformer_2: BasicTransformerBlock
    @ModuleInfo(key: "transformer_3") public var transformer_3: BasicTransformerBlock

    public let n_transformer: Int = 4

    public init(
        channels: Int,
        timeEmbedDim: Int,
        numHeads: Int = 8,
        attentionHeadDim: Int = 64
    ) {
        self._resnet = ModuleInfo(wrappedValue: CausalResnetBlock1D(dim: channels, dimOut: channels, timeEmbDim: timeEmbedDim))
        self._transformer_0 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: channels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._transformer_1 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: channels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._transformer_2 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: channels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._transformer_3 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: channels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        super.init()
    }

    public var transformerBlocks: [BasicTransformerBlock] {
        [transformer_0, transformer_1, transformer_2, transformer_3]
    }
}

/// Container for up block components
public class UpBlock: Module {
    @ModuleInfo public var resnet: CausalResnetBlock1D
    @ModuleInfo(key: "transformer_0") public var transformer_0: BasicTransformerBlock
    @ModuleInfo(key: "transformer_1") public var transformer_1: BasicTransformerBlock
    @ModuleInfo(key: "transformer_2") public var transformer_2: BasicTransformerBlock
    @ModuleInfo(key: "transformer_3") public var transformer_3: BasicTransformerBlock
    @ModuleInfo public var upsample: CausalConv1d

    public let n_transformer: Int = 4

    public init(
        inChannels: Int,
        outChannels: Int,
        timeEmbedDim: Int,
        numHeads: Int = 8,
        attentionHeadDim: Int = 64
    ) {
        self._resnet = ModuleInfo(wrappedValue: CausalResnetBlock1D(dim: inChannels, dimOut: outChannels, timeEmbDim: timeEmbedDim))
        self._transformer_0 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: outChannels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._transformer_1 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: outChannels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._transformer_2 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: outChannels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._transformer_3 = ModuleInfo(wrappedValue: BasicTransformerBlock(dim: outChannels, numAttentionHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._upsample = ModuleInfo(wrappedValue: CausalConv1d(inChannels: outChannels, outChannels: outChannels, kernelSize: 3))
        super.init()
    }

    public var transformerBlocks: [BasicTransformerBlock] {
        [transformer_0, transformer_1, transformer_2, transformer_3]
    }
}

// MARK: - Conditional Decoder

/// Conditional U-Net decoder for flow matching
public class ConditionalDecoder: Module {
    public let in_channels: Int
    public let out_channels: Int
    public let causal: Bool
    public let static_chunk_size: Int
    public let num_decoding_left_chunks: Int

    @ModuleInfo(key: "time_embeddings") public var time_embeddings: SinusoidalPosEmb
    @ModuleInfo(key: "time_mlp") public var time_mlp: TimestepEmbedding

    @ModuleInfo(key: "down_blocks_0") public var down_blocks_0: DownBlock

    @ModuleInfo(key: "mid_blocks_0") public var mid_blocks_0: MidBlock
    @ModuleInfo(key: "mid_blocks_1") public var mid_blocks_1: MidBlock
    @ModuleInfo(key: "mid_blocks_2") public var mid_blocks_2: MidBlock
    @ModuleInfo(key: "mid_blocks_3") public var mid_blocks_3: MidBlock
    @ModuleInfo(key: "mid_blocks_4") public var mid_blocks_4: MidBlock
    @ModuleInfo(key: "mid_blocks_5") public var mid_blocks_5: MidBlock
    @ModuleInfo(key: "mid_blocks_6") public var mid_blocks_6: MidBlock
    @ModuleInfo(key: "mid_blocks_7") public var mid_blocks_7: MidBlock
    @ModuleInfo(key: "mid_blocks_8") public var mid_blocks_8: MidBlock
    @ModuleInfo(key: "mid_blocks_9") public var mid_blocks_9: MidBlock
    @ModuleInfo(key: "mid_blocks_10") public var mid_blocks_10: MidBlock
    @ModuleInfo(key: "mid_blocks_11") public var mid_blocks_11: MidBlock

    @ModuleInfo(key: "up_blocks_0") public var up_blocks_0: UpBlock

    @ModuleInfo(key: "final_block") public var final_block: CausalBlock1D
    @ModuleInfo(key: "final_proj") public var final_proj: Conv1d

    let channels: Int
    let timeEmbedDim: Int

    public init(
        inChannels: Int = 320,
        outChannels: Int = 80,
        causal: Bool = true,
        channels: Int = 256,
        dropout: Float = 0.0,
        attentionHeadDim: Int = 64,
        nBlocks: Int = 4,
        numMidBlocks: Int = 12,
        numHeads: Int = 8,
        actFn: String = "gelu",
        staticChunkSize: Int = 50,
        numDecodingLeftChunks: Int = 2
    ) {
        self.in_channels = inChannels
        self.out_channels = outChannels
        self.causal = causal
        self.channels = channels
        self.timeEmbedDim = channels * 4
        self.static_chunk_size = staticChunkSize
        self.num_decoding_left_chunks = numDecodingLeftChunks

        self._time_embeddings = ModuleInfo(wrappedValue: SinusoidalPosEmb(dim: inChannels))
        self._time_mlp = ModuleInfo(wrappedValue: TimestepEmbedding(inChannels: inChannels, timeEmbedDim: timeEmbedDim, actFn: "silu"))

        self._down_blocks_0 = ModuleInfo(wrappedValue: DownBlock(inChannels: inChannels, outChannels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))

        self._mid_blocks_0 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_1 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_2 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_3 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_4 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_5 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_6 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_7 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_8 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_9 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_10 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))
        self._mid_blocks_11 = ModuleInfo(wrappedValue: MidBlock(channels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))

        self._up_blocks_0 = ModuleInfo(wrappedValue: UpBlock(inChannels: channels * 2, outChannels: channels, timeEmbedDim: timeEmbedDim, numHeads: numHeads, attentionHeadDim: attentionHeadDim))

        self._final_block = ModuleInfo(wrappedValue: CausalBlock1D(dim: channels, dimOut: channels))
        self._final_proj = ModuleInfo(wrappedValue: Conv1d(inputChannels: channels, outputChannels: outChannels, kernelSize: 1))

        super.init()
    }

    public func callAsFunction(
        x: MLXArray,
        mask: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil,
        streaming: Bool = false
    ) -> MLXArray {
        var tEmb = time_embeddings(t)
        tEmb = time_mlp(tEmb)

        var h = concatenated([x, mu], axis: 1)
        if let spks = spks {
            let spksExpanded = broadcastTo(spks.expandedDimensions(axis: -1), [spks.shape[0], spks.shape[1], h.shape[2]])
            h = concatenated([h, spksExpanded], axis: 1)
        }
        if let cond = cond {
            h = concatenated([h, cond], axis: 1)
        }

        var hiddens: [MLXArray] = []
        var masks: [MLXArray] = [mask]

        for downBlock in [down_blocks_0] {
            let maskDown = masks[masks.count - 1]
            h = downBlock.resnet(h, mask: maskDown, timeEmb: tEmb)

            var hT = h.transposed(0, 2, 1)
            let maskBool = maskDown .> 0
            let effectiveChunkSize = streaming ? static_chunk_size : 0
            let attnMask = addOptionalChunkMask(
                xs: hT,
                masks: maskBool,
                useDynamicChunk: false,
                useDynamicLeftChunk: false,
                decodingChunkSize: 0,
                staticChunkSize: effectiveChunkSize,
                numDecodingLeftChunks: -1
            )
            let attnBias = maskToBias(attnMask, dtype: hT.dtype)

            for transformer in downBlock.transformerBlocks {
                hT = transformer(hT, attentionMask: attnBias, timestep: tEmb)
            }
            h = hT.transposed(0, 2, 1)

            hiddens.append(h)
            h = downBlock.downsample(h * maskDown)
            let downIdx = MLXArray(stride(from: 0, to: maskDown.shape[2], by: 2))
            masks.append(MLX.take(maskDown, downIdx, axis: 2))
        }

        masks = Array(masks.dropLast())
        let maskMid = masks[masks.count - 1]

        for midBlock in [mid_blocks_0, mid_blocks_1, mid_blocks_2, mid_blocks_3, mid_blocks_4, mid_blocks_5, mid_blocks_6, mid_blocks_7, mid_blocks_8, mid_blocks_9, mid_blocks_10, mid_blocks_11] {
            h = midBlock.resnet(h, mask: maskMid, timeEmb: tEmb)
            var hT = h.transposed(0, 2, 1)

            let maskBool = maskMid .> 0
            let effectiveChunkSize = streaming ? static_chunk_size : 0
            let attnMask = addOptionalChunkMask(
                xs: hT,
                masks: maskBool,
                useDynamicChunk: false,
                useDynamicLeftChunk: false,
                decodingChunkSize: 0,
                staticChunkSize: effectiveChunkSize,
                numDecodingLeftChunks: -1
            )
            let attnBias = maskToBias(attnMask, dtype: hT.dtype)

            for transformer in midBlock.transformerBlocks {
                hT = transformer(hT, attentionMask: attnBias, timestep: tEmb)
            }
            h = hT.transposed(0, 2, 1)
        }

        var finalMask = maskMid
        for upBlock in [up_blocks_0] {
            let maskUp = masks.removeLast()
            finalMask = maskUp
            let skip = hiddens.removeLast()
            h = concatenated([h[0..., 0..., 0..<skip.shape[2]], skip], axis: 1)
            h = upBlock.resnet(h, mask: maskUp, timeEmb: tEmb)

            var hT = h.transposed(0, 2, 1)
            let maskBool = maskUp .> 0
            let effectiveChunkSize = streaming ? static_chunk_size : 0
            let attnMask = addOptionalChunkMask(
                xs: hT,
                masks: maskBool,
                useDynamicChunk: false,
                useDynamicLeftChunk: false,
                decodingChunkSize: 0,
                staticChunkSize: effectiveChunkSize,
                numDecodingLeftChunks: -1
            )
            let attnBias = maskToBias(attnMask, dtype: hT.dtype)

            for transformer in upBlock.transformerBlocks {
                hT = transformer(hT, attentionMask: attnBias, timestep: tEmb)
            }
            h = hT.transposed(0, 2, 1)

            h = upBlock.upsample(h * maskUp)
        }

        h = final_block(h, mask: finalMask)
        var hProj = (h * finalMask).transposed(0, 2, 1)
        hProj = final_proj(hProj)
        hProj = hProj.transposed(0, 2, 1)
        return hProj * finalMask
    }
}
