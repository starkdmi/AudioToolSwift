// Copyright © 2025
// ConformerAttention - Multi-headed attention for Conformer encoder

import Foundation
import MLX
import MLXNN
import MLXFast
import MLXRandom

// MARK: - Multi-Headed Attention

/// Multi-Head Attention layer for Conformer encoder
public class MultiHeadedAttention: Module {
    let d_k: Int
    let h: Int
    let dropoutRate: Float

    @ModuleInfo(key: "linear_q") public var linear_q: Linear
    @ModuleInfo(key: "linear_k") public var linear_k: Linear
    @ModuleInfo(key: "linear_v") public var linear_v: Linear
    @ModuleInfo(key: "linear_out") public var linear_out: Linear

    public init(nHead: Int, nFeat: Int, dropoutRate: Float = 0.1, keyBias: Bool = true) {
        precondition(nFeat % nHead == 0, "nFeat must be divisible by nHead")
        self.d_k = nFeat / nHead
        self.h = nHead
        self.dropoutRate = dropoutRate

        self._linear_q = ModuleInfo(wrappedValue: Linear(nFeat, nFeat))
        self._linear_k = ModuleInfo(wrappedValue: Linear(nFeat, nFeat, bias: keyBias))
        self._linear_v = ModuleInfo(wrappedValue: Linear(nFeat, nFeat))
        self._linear_out = ModuleInfo(wrappedValue: Linear(nFeat, nFeat))
        super.init()
    }

    public func forwardQKV(query: MLXArray, key: MLXArray, value: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let nBatch = query.shape[0]
        var q = linear_q(query).reshaped(nBatch, -1, h, d_k)
        var k = linear_k(key).reshaped(nBatch, -1, h, d_k)
        var v = linear_v(value).reshaped(nBatch, -1, h, d_k)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        return (q, k, v)
    }

    public func forwardAttention(value: MLXArray, scores: MLXArray, mask: MLXArray?) -> MLXArray {
        let nBatch = value.shape[0]
        var attn: MLXArray

        if let mask = mask, mask.shape[2] > 0 {
            var maskExpanded = expandDims(mask, axis: 1)
            maskExpanded = maskExpanded[0..., 0..., 0..., 0..<scores.shape[scores.ndim - 1]]
            let maskedScores = MLX.where(maskExpanded .== 0, MLXArray(-Float.infinity), scores)
            attn = softmax(maskedScores, axis: -1)
            attn = MLX.where(maskExpanded .== 0, MLXArray(0.0), attn)
        } else {
            attn = softmax(scores, axis: -1)
        }

        var x = matmul(attn, value)
        x = x.transposed(0, 2, 1, 3)
        x = x.reshaped(nBatch, -1, h * d_k)

        return linear_out(x)
    }

    public func callAsFunction(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        mask: MLXArray? = nil,
        posEmb: MLXArray? = nil,
        cache: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        var (q, k, v) = forwardQKV(query: query, key: key, value: value)

        if let cache = cache, cache.shape[0] > 0 {
            let split = cache.split(parts: 2, axis: -1)
            let keyCache = split[0]
            let valueCache = split[1]
            k = concatenated([keyCache, k], axis: 2)
            v = concatenated([valueCache, v], axis: 2)
        }

        let newCache = concatenated([k, v], axis: -1)
        let scores = matmul(q, k.swappedAxes(-2, -1)) / sqrt(Float(d_k))
        return (forwardAttention(value: v, scores: scores, mask: mask), newCache)
    }
}

// MARK: - Relative Position Multi-Headed Attention

/// Multi-Head Attention with relative positional encoding
public class RelPositionMultiHeadedAttention: MultiHeadedAttention {
    @ModuleInfo(key: "linear_pos") public var linear_pos: Linear
    @ParameterInfo(key: "pos_bias_u") public var pos_bias_u: MLXArray
    @ParameterInfo(key: "pos_bias_v") public var pos_bias_v: MLXArray

    public override init(nHead: Int, nFeat: Int, dropoutRate: Float = 0.1, keyBias: Bool = true) {
        self._linear_pos = ModuleInfo(wrappedValue: Linear(nFeat, nFeat, bias: false))

        let dK = nFeat / nHead
        let initScale = sqrt(6.0 / Float(nHead + dK))
        self._pos_bias_u.wrappedValue = MLXRandom.uniform(low: -1.0, high: 1.0, [nHead, dK]) * initScale
        self._pos_bias_v.wrappedValue = MLXRandom.uniform(low: -1.0, high: 1.0, [nHead, dK]) * initScale
        super.init(nHead: nHead, nFeat: nFeat, dropoutRate: dropoutRate, keyBias: keyBias)
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let zeroPad = MLX.zeros([x.shape[0], x.shape[1], x.shape[2], 1])
        var xPadded = concatenated([zeroPad, x], axis: -1)
        xPadded = xPadded.reshaped(x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2])
        let shifted = xPadded[0..., 0..., 1..., 0...].reshaped(x.shape)
        return shifted[0..., 0..., 0..., 0..<(x.shape[x.ndim - 1] / 2 + 1)]
    }

    public override func callAsFunction(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        mask: MLXArray? = nil,
        posEmb: MLXArray? = nil,
        cache: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        var (q, k, v) = forwardQKV(query: query, key: key, value: value)
        q = q.transposed(0, 2, 1, 3)

        if let cache = cache, cache.shape[0] > 0 {
            let split = cache.split(parts: 2, axis: -1)
            k = concatenated([split[0], k], axis: 2)
            v = concatenated([split[1], v], axis: 2)
        }

        let newCache = concatenated([k, v], axis: -1)

        guard let posEmb = posEmb else {
            let scores = matmul(q.transposed(0, 2, 1, 3), k.swappedAxes(-2, -1)) / sqrt(Float(d_k))
            return (forwardAttention(value: v, scores: scores, mask: mask), newCache)
        }

        let nBatchPos = posEmb.shape[0]
        var p = linear_pos(posEmb).reshaped(nBatchPos, -1, h, d_k)
        p = p.transposed(0, 2, 1, 3)

        let qWithBiasU = (q + pos_bias_u).transposed(0, 2, 1, 3)
        let qWithBiasV = (q + pos_bias_v).transposed(0, 2, 1, 3)

        let matrixAC = matmul(qWithBiasU, k.swappedAxes(-2, -1))
        var matrixBD = matmul(qWithBiasV, p.swappedAxes(-2, -1))

        if matrixAC.shape != matrixBD.shape {
            matrixBD = relShift(matrixBD)
        }

        let scores = (matrixAC + matrixBD) / sqrt(Float(d_k))
        return (forwardAttention(value: v, scores: scores, mask: mask), newCache)
    }

    public func debugCall(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        mask: MLXArray? = nil,
        posEmb: MLXArray? = nil,
        cache: MLXArray? = nil
    ) -> (MLXArray, [String: MLXArray]) {
        var debug: [String: MLXArray] = [:]
        var (q, k, v) = forwardQKV(query: query, key: key, value: value)
        q = q.transposed(0, 2, 1, 3)
        debug["q"] = q
        debug["k"] = k
        debug["v"] = v

        if let cache = cache, cache.shape[0] > 0 {
            let split = cache.split(parts: 2, axis: -1)
            k = concatenated([split[0], k], axis: 2)
            v = concatenated([split[1], v], axis: 2)
        }

        guard let posEmb = posEmb else {
            let scores = matmul(q.transposed(0, 2, 1, 3), k.swappedAxes(-2, -1)) / sqrt(Float(d_k))
            let out = forwardAttention(value: v, scores: scores, mask: mask)
            debug["scores"] = scores
            return (out, debug)
        }

        let nBatchPos = posEmb.shape[0]
        var p = linear_pos(posEmb).reshaped(nBatchPos, -1, h, d_k)
        p = p.transposed(0, 2, 1, 3)
        debug["p"] = p

        let qWithBiasU = (q + pos_bias_u).transposed(0, 2, 1, 3)
        let qWithBiasV = (q + pos_bias_v).transposed(0, 2, 1, 3)

        let matrixAC = matmul(qWithBiasU, k.swappedAxes(-2, -1))
        var matrixBD = matmul(qWithBiasV, p.swappedAxes(-2, -1))
        debug["matrix_ac"] = matrixAC
        debug["matrix_bd_pre"] = matrixBD

        if matrixAC.shape != matrixBD.shape {
            matrixBD = relShift(matrixBD)
        }
        debug["matrix_bd"] = matrixBD

        let scores = (matrixAC + matrixBD) / sqrt(Float(d_k))
        debug["scores"] = scores

        var attnWeights: MLXArray
        if let mask = mask, mask.shape[2] > 0 {
            var maskExpanded = expandDims(mask, axis: 1)
            maskExpanded = maskExpanded[0..., 0..., 0..., 0..<scores.shape[scores.ndim - 1]]
            let maskedScores = MLX.where(maskExpanded .== 0, MLXArray(-Float.infinity), scores)
            attnWeights = softmax(maskedScores, axis: -1)
            attnWeights = MLX.where(maskExpanded .== 0, MLXArray(0.0), attnWeights)
        } else {
            attnWeights = softmax(scores, axis: -1)
        }
        debug["attn_weights"] = attnWeights

        var attnOut = matmul(attnWeights, v)
        debug["attn_out"] = attnOut

        let nBatch = v.shape[0]
        attnOut = attnOut.transposed(0, 2, 1, 3)
        attnOut = attnOut.reshaped(nBatch, -1, h * d_k)
        let projected = linear_out(attnOut)
        debug["attn_projected"] = projected
        return (projected, debug)
    }
}
