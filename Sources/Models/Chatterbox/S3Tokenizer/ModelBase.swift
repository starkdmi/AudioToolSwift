import Foundation
import MLX
import MLXFast
import MLXNN

class MultiHeadAttention: Module {
    let n_head: Int

    @ModuleInfo var query: Linear
    @ModuleInfo var key: Linear
    @ModuleInfo var value: Linear
    @ModuleInfo var out: Linear

    init(n_state: Int, n_head: Int) {
        self.n_head = n_head
        self._query.wrappedValue = Linear(n_state, n_state)
        self._key.wrappedValue = Linear(n_state, n_state, bias: false)
        self._value.wrappedValue = Linear(n_state, n_state)
        self._out.wrappedValue = Linear(n_state, n_state)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> (MLXArray, MLXArray?) {
        let q = query(x)
        let k = key(x)
        let v = value(x)
        let (wv, qk) = qkv_attention(q: q, k: k, v: v, mask: mask)
        return (out(wv), qk)
    }

    func qkv_attention(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray?) -> (MLXArray, MLXArray?) {
        let B = q.dim(0)
        let T = q.dim(1)
        let D = q.dim(2)
        let scale = Float(Foundation.pow(Double(D / n_head), -0.25))

        var qh = q.reshaped(B, T, n_head, -1).transposed(0, 2, 1, 3) * scale
        var kh = k.reshaped(B, T, n_head, -1).transposed(0, 2, 1, 3) * scale
        let vh = v.reshaped(B, T, n_head, -1).transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask = mask {
            maskMode = .array(mask)
        } else {
            maskMode = .none
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: qh,
            keys: kh,
            values: vh,
            scale: 1.0,
            mask: maskMode
        )
        let merged = output.transposed(0, 2, 1, 3).reshaped(B, T, D)
        return (merged, nil)
    }
}

class ResidualAttentionBlock: Module {
    @ModuleInfo var attn: MultiHeadAttention
    @ModuleInfo var attn_ln: LayerNorm
    @ModuleInfo var mlp: Sequential
    @ModuleInfo var mlp_ln: LayerNorm

    init(n_state: Int, n_head: Int) {
        self._attn.wrappedValue = MultiHeadAttention(n_state: n_state, n_head: n_head)
        self._attn_ln.wrappedValue = LayerNorm(dimensions: n_state)

        let n_mlp = n_state * 4
        self._mlp.wrappedValue = Sequential([
            Linear(n_state, n_mlp),
            GELU(),
            Linear(n_mlp, n_state)
        ])
        self._mlp_ln.wrappedValue = LayerNorm(dimensions: n_state)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = x + attn(attn_ln(x), mask: mask).0
        out = out + mlp(mlp_ln(out))
        return out
    }
}
