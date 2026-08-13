// Copyright © 2025
// ConformerEncoder - Conformer encoder layer for S3Gen

import Foundation
import MLX
import MLXNN

// MARK: - Positional Encoding

/// Relative positional encoding module (ESPnet implementation)
public class EspnetRelPositionalEncoding: Module {
    let dModel: Int
    let xscale: Float
    let dropoutRate: Float
    let maxLen: Int
    public var pe: MLXArray

    public init(dModel: Int, dropoutRate: Float = 0.1, maxLen: Int = 5000) {
        self.dModel = dModel
        self.xscale = sqrt(Float(dModel))
        self.dropoutRate = dropoutRate
        self.maxLen = maxLen
        self.pe = MLXArray.zeros([1, 0, dModel])
        super.init()
        extendPE(size: maxLen)
    }

    private func extendPE(size: Int) {
        if pe.shape[1] >= size * 2 - 1 {
            return
        }

        let position = MLXArray(stride(from: 0, to: size, by: 1)).asType(.float32)
            .expandedDimensions(axis: 1)
        let divTerm = MLX.exp(
            MLXArray(stride(from: 0, to: dModel, by: 2)).asType(.float32)
                * (-(Foundation.log(10000.0) / Float(dModel)))
        )

        let pePositiveSin = MLX.sin(position * divTerm)
        let pePositiveCos = MLX.cos(position * divTerm)
        let pePositive = stack([pePositiveSin, pePositiveCos], axis: -1).reshaped([size, dModel])

        let peNegativeSin = MLX.sin(-position * divTerm)
        let peNegativeCos = MLX.cos(-position * divTerm)
        let peNegative = stack([peNegativeSin, peNegativeCos], axis: -1).reshaped([size, dModel])

        let reverseIdx = MLXArray(stride(from: size - 1, through: 0, by: -1))
        let pePositiveRev = MLX.take(pePositive, reverseIdx, axis: 0).expandedDimensions(axis: 0)
        let peNegativeSlice = peNegative[1...].expandedDimensions(axis: 0)

        pe = concatenated([pePositiveRev, peNegativeSlice], axis: 1)
    }

    public func positionEncoding(size: Int, offset: Int = 0) -> MLXArray {
        let center = pe.shape[1] / 2
        let start = center - size + 1
        let end = center + size
        return pe[0..., start..<end, 0...]
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        extendPE(size: x.shape[1])
        let scaled = x * xscale
        let posEmb = positionEncoding(size: x.shape[1], offset: offset)
        return (scaled, posEmb)
    }
}

// MARK: - Linear No Subsampling

/// Linear input embedding without subsampling
public class LinearNoSubsampling: Module {
    @ModuleInfo(key: "linear") public var linear: Linear
    @ModuleInfo(key: "norm") public var norm: ManualLayerNorm
    @ModuleInfo(key: "pos_enc") public var pos_enc: EspnetRelPositionalEncoding

    public init(inputSize: Int, outputSize: Int, dropoutRate: Float, posEnc: EspnetRelPositionalEncoding) {
        self._linear = ModuleInfo(wrappedValue: Linear(inputSize, outputSize))
        self._norm = ModuleInfo(wrappedValue: ManualLayerNorm(dimensions: outputSize, eps: 1e-5))
        self._pos_enc = ModuleInfo(wrappedValue: posEnc)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, masks: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray, MLXArray) {
        var out = linear(x)
        out = norm(out)
        let (scaled, posEmb) = pos_enc(out, offset: offset)
        return (scaled, posEmb, masks)
    }
}

// MARK: - Positionwise Feed Forward

/// Position-wise feed-forward module
public class PositionwiseFeedForward: Module {
    @ModuleInfo(key: "w_1") public var w_1: Linear
    @ModuleInfo(key: "w_2") public var w_2: Linear
    let activation: (MLXArray) -> MLXArray

    public init(inputDim: Int, hiddenDim: Int, dropoutRate: Float = 0.1, activation: @escaping (MLXArray) -> MLXArray = silu) {
        self._w_1 = ModuleInfo(wrappedValue: Linear(inputDim, hiddenDim))
        self._w_2 = ModuleInfo(wrappedValue: Linear(hiddenDim, inputDim))
        self.activation = activation
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return w_2(activation(w_1(x)))
    }
}

// MARK: - Conformer Encoder Layer

/// Conformer encoder layer - combines attention and feed-forward modules
public class ConformerEncoderLayer: Module {
    let size: Int
    let normalizeBefore: Bool
    let ffScale: Float
    let dropoutRate: Float

    @ModuleInfo(key: "self_attn") public var self_attn: RelPositionMultiHeadedAttention
    @ModuleInfo(key: "feed_forward") public var feed_forward: PositionwiseFeedForward
    @ModuleInfo(key: "norm_ff") public var norm_ff: ManualLayerNorm
    @ModuleInfo(key: "norm_mha") public var norm_mha: ManualLayerNorm

    public init(
        size: Int,
        selfAttn: RelPositionMultiHeadedAttention,
        feedForward: PositionwiseFeedForward,
        feedForwardMacaron: PositionwiseFeedForward? = nil,
        convModule: Module? = nil,
        dropoutRate: Float = 0.1,
        normalizeBefore: Bool = true
    ) {
        self.size = size
        self.normalizeBefore = normalizeBefore
        self.dropoutRate = dropoutRate
        self.ffScale = feedForwardMacaron == nil ? 1.0 : 0.5

        self._self_attn = ModuleInfo(wrappedValue: selfAttn)
        self._feed_forward = ModuleInfo(wrappedValue: feedForward)
        self._norm_ff = ModuleInfo(wrappedValue: ManualLayerNorm(dimensions: size, eps: 1e-12))
        self._norm_mha = ModuleInfo(wrappedValue: ManualLayerNorm(dimensions: size, eps: 1e-12))
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        _ mask: MLXArray,
        _ posEmb: MLXArray,
        _ maskPad: MLXArray? = nil,
        _ attCache: MLXArray? = nil,
        _ cnnCache: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        var out = x

        var residual = out
        if normalizeBefore {
            out = norm_mha(out)
        }
        let (attOut, newAttCache) = self_attn(query: out, key: out, value: out, mask: mask, posEmb: posEmb, cache: attCache)
        out = residual + attOut
        if !normalizeBefore {
            out = norm_mha(out)
        }

        residual = out
        if normalizeBefore {
            out = norm_ff(out)
        }
        let ffOut = feed_forward(out)
        out = residual + ffScale * ffOut
        if !normalizeBefore {
            out = norm_ff(out)
        }

        let emptyCache = MLX.zeros([0, 0, 0])
        return (out, mask, newAttCache, emptyCache)
    }

    public func callAsFunctionDebug(
        _ x: MLXArray,
        _ mask: MLXArray,
        _ posEmb: MLXArray,
        _ maskPad: MLXArray? = nil
    ) -> (MLXArray, [String: MLXArray]) {
        var debug: [String: MLXArray] = [:]
        var out = x

        var residual = out
        var attIn = out
        if normalizeBefore {
            attIn = norm_mha(attIn)
        }
        let attOut: MLXArray
        if let relAttn = self_attn as? RelPositionMultiHeadedAttention {
            let (attOutDebug, attnInfo) = relAttn.debugCall(query: attIn, key: attIn, value: attIn, mask: mask, posEmb: posEmb, cache: nil)
            attOut = attOutDebug
            for (key, value) in attnInfo {
                debug["attn_\(key)"] = value
            }
        } else {
            let (attOutBase, _) = self_attn(query: attIn, key: attIn, value: attIn, mask: mask, posEmb: posEmb, cache: nil)
            attOut = attOutBase
        }
        debug["attn_raw"] = attOut
        out = residual + attOut
        debug["attn_resid"] = out
        if !normalizeBefore {
            out = norm_mha(out)
        }

        residual = out
        var ffIn = out
        if normalizeBefore {
            ffIn = norm_ff(ffIn)
        }
        let ffOut = feed_forward(ffIn)
        debug["ff_raw"] = ffOut
        out = residual + ffScale * ffOut
        debug["ff_resid"] = out
        if !normalizeBefore {
            out = norm_ff(out)
        }

        return (out, debug)
    }
}
