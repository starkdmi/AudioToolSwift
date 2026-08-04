import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

final class AttentionQKV: Module {
    let n_heads: Int
    let head_dim: Int
    let scale: Float

    init(nHeads: Int, headDim: Int, scale: Float? = nil) {
        self.n_heads = nHeads
        self.head_dim = headDim
        self.scale = scale ?? Float(Foundation.pow(Double(headDim), -0.5))
        super.init()
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray?) -> MLXArray {
        let qh = splitHeads(q)
        let kh = splitHeads(k)
        let vh = splitHeads(v)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask = mask {
            maskMode = .array(mask)
        } else {
            maskMode = .none
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: qh,
            keys: kh,
            values: vh,
            scale: scale,
            mask: maskMode
        )

        return combineHeads(out)
    }

    private func splitHeads(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let reshaped = x.reshaped(B, T, n_heads, head_dim)
        return reshaped.transposed(0, 2, 1, 3)
    }

    private func combineHeads(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(2)
        let transposed = x.transposed(0, 2, 1, 3)
        return transposed.reshaped(B, T, -1)
    }
}

final class AttentionBlock: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var to_q: Linear
    @ModuleInfo var to_k: Linear
    @ModuleInfo var to_v: Linear
    @ModuleInfo var proj_out: Linear

    private let attention: AttentionQKV

    init(channels: Int, numHeads: Int = 1) {
        self._norm.wrappedValue = LayerNorm(dimensions: channels)
        self._to_q.wrappedValue = Linear(channels, channels)
        self._to_k.wrappedValue = Linear(channels, channels)
        self._to_v.wrappedValue = Linear(channels, channels)
        self._proj_out.wrappedValue = Linear(channels, channels)
        self.attention = AttentionQKV(nHeads: numHeads, headDim: channels / numHeads)
        super.init()
    }

    func callAsFunction(_ x1: MLXArray, _ x2: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let x1Norm = norm(x1)
        let x2Norm = norm(x2)

        let q = to_q(x1Norm)
        let k = to_k(x2Norm)
        let v = to_v(x2Norm)

        let h = attention(q, k, v, mask: mask)
        return x1 + proj_out(h)
    }

    func debugForward(_ x1: MLXArray, _ x2: MLXArray, prefix: String) -> [String: MLXArray] {
        let x1Norm = norm(x1)
        let x2Norm = norm(x2)

        let q = to_q(x1Norm)
        let k = to_k(x2Norm)
        let v = to_v(x2Norm)

        let attnOut = attention(q, k, v, mask: nil)
        let proj = proj_out(attnOut)
        let out = x1 + proj
        return [
            "\(prefix)_x1_norm": x1Norm,
            "\(prefix)_x2_norm": x2Norm,
            "\(prefix)_q": q,
            "\(prefix)_k": k,
            "\(prefix)_v": v,
            "\(prefix)_attn": attnOut,
            "\(prefix)_proj": proj,
            "\(prefix)_out": out,
        ]
    }
}

public final class Perceiver: Module {
    @ModuleInfo var pre_attention_query: MLXArray
    @ModuleInfo var attn: AttentionBlock

    private let query_tokens: Int
    private let query_size: Int

    public init(
        preAttentionQueryToken: Int = 32,
        preAttentionQuerySize: Int = 1024,
        embeddingDim: Int = 1024,
        numAttnHeads: Int = 4
    ) {
        self.query_tokens = preAttentionQueryToken
        self.query_size = preAttentionQuerySize

        let variance = Foundation.sqrt(3.0)
            * Foundation.sqrt(2.0 / Double(preAttentionQueryToken + preAttentionQueryToken))
        self._pre_attention_query.wrappedValue = MLXRandom.uniform(
            low: -Float(variance),
            high: Float(variance),
            [1, preAttentionQueryToken, preAttentionQuerySize]
        )

        self._attn.wrappedValue = AttentionBlock(channels: embeddingDim, numHeads: numAttnHeads)
        super.init()
    }

    public func callAsFunction(_ h: MLXArray) -> MLXArray {
        let B = h.dim(0)
        let query = broadcastTo(pre_attention_query, [B, query_tokens, query_size])

        let preAtt = attn(query, h)
        return attn(preAtt, preAtt)
    }

    func debugQuery() -> MLXArray {
        pre_attention_query
    }

    func debugForward(_ h: MLXArray) -> [String: MLXArray] {
        let B = h.dim(0)
        let query = broadcastTo(pre_attention_query, [B, query_tokens, query_size])
        var outputs = attn.debugForward(query, h, prefix: "t3_perceiver_cross")
        guard let preAtt = outputs["t3_perceiver_cross_out"] else {
            return outputs
        }
        let selfOutputs = attn.debugForward(preAtt, preAtt, prefix: "t3_perceiver_self")
        outputs.merge(selfOutputs, uniquingKeysWith: { current, _ in current })
        if let out = outputs["t3_perceiver_self_out"] {
            outputs["t3_perceiver_out"] = out
        }
        return outputs
    }
}
