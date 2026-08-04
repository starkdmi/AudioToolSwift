import Foundation
import MLX
import MLXFast
import MLXNN

public protocol RoPELayer: AnyObject {
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray
}

public final class StandardRoPE: Module, RoPELayer {
    private let dims: Int
    private let traditional: Bool
    private let base: Float
    private let scale: Float

    public init(dims: Int, traditional: Bool, base: Float, scale: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: base,
            scale: scale,
            offset: offset
        )
    }
}

private final class Attention: Module {
    let args: LlamaConfig
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    var rope: RoPELayer

    init(_ args: LlamaConfig) {
        self.args = args

        let dim = args.hidden_size
        let heads = args.num_attention_heads
        let kvHeads = args.num_key_value_heads
        let headDim = args.head_dim

        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attention_bias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attention_bias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attention_bias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attention_bias)

        self.rope = StandardRoPE(
            dims: headDim,
            traditional: args.rope_traditional,
            base: args.rope_theta
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        let heads = args.num_attention_heads
        let kvHeads = args.num_key_value_heads

        queries = queries.reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache = cache {
            queries = rope.callAsFunction(queries, offset: cache.offset)
            keys = rope.callAsFunction(keys, offset: cache.offset)
        } else {
            queries = rope.callAsFunction(queries, offset: 0)
            keys = rope.callAsFunction(keys, offset: 0)
        }

        let output: MLXArray
        if let cache = cache {
            output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
        } else {
            output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask
            )
        }

        let merged = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(merged)
    }
}

private final class MLP: Module {
    let args: LlamaConfig

    @ModuleInfo(key: "gate_proj") var gate_proj: Linear
    @ModuleInfo(key: "down_proj") var down_proj: Linear
    @ModuleInfo(key: "up_proj") var up_proj: Linear

    init(_ args: LlamaConfig) {
        self.args = args
        let dim = args.hidden_size
        let hiddenDim = args.intermediate_size

        self._gate_proj.wrappedValue = Linear(dim, hiddenDim, bias: args.mlp_bias)
        self._down_proj.wrappedValue = Linear(hiddenDim, dim, bias: args.mlp_bias)
        self._up_proj.wrappedValue = Linear(dim, hiddenDim, bias: args.mlp_bias)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down_proj(MLXNN.silu(gate_proj(x)) * up_proj(x))
    }
}

private final class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var input_layernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var post_attention_layernorm: RMSNorm

    init(_ args: LlamaConfig) {
        self._attention.wrappedValue = Attention(args)
        self._mlp.wrappedValue = MLP(args)
        self._input_layernorm.wrappedValue = RMSNorm(
            dimensions: args.hidden_size,
            eps: args.rms_norm_eps
        )
        self._post_attention_layernorm.wrappedValue = RMSNorm(
            dimensions: args.hidden_size,
            eps: args.rms_norm_eps
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var r = attention(input_layernorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(post_attention_layernorm(h))
        return h + r
    }
}

final class LlamaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embed_tokens: Embedding

    private let layers: [TransformerBlock]
    let norm: RMSNorm

    init(_ args: LlamaConfig) {
        self._embed_tokens.wrappedValue = Embedding(
            embeddingCount: args.vocab_size,
            dimensions: args.hidden_size
        )
        self.layers = (0 ..< args.num_hidden_layers).map { _ in TransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hidden_size, eps: args.rms_norm_eps)
        super.init()
    }

    func callAsFunction(
        inputs: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache]?,
        input_embeddings: MLXArray?
    ) -> MLXArray {
        let h: MLXArray
        if let input_embeddings = input_embeddings {
            h = input_embeddings
        } else if let inputs = inputs {
            h = embed_tokens(inputs)
        } else {
            fatalError("Either inputs or input_embeddings must be provided")
        }

        var hidden = h
        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[i])
        }
        return norm(hidden)
    }

    func replaceRoPE(_ rope: RoPELayer) {
        for layer in layers {
            layer.attention.rope = rope
        }
    }

    func debugLayer0(
        input_embeddings: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> [String: MLXArray] {
        let layer = layers[0]
        let normed = layer.input_layernorm(input_embeddings)
        let B = input_embeddings.dim(0)
        let L = input_embeddings.dim(1)
        let heads = layer.attention.args.num_attention_heads
        let kvHeads = layer.attention.args.num_key_value_heads

        let qProj = layer.attention.wq(normed)
        let kProj = layer.attention.wk(normed)
        let vProj = layer.attention.wv(normed)

        let q = qProj.reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let k = kProj.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let v = vProj.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let qRope = layer.attention.rope.callAsFunction(q, offset: 0)
        let kRope = layer.attention.rope.callAsFunction(k, offset: 0)

        let ropeTestOffset = 3
        let ropeTestInput = MLXArray(
            stride(from: 0, to: layer.attention.args.head_dim * 2, by: 1)
        ).asType(.float32).reshaped([1, 1, 2, layer.attention.args.head_dim])
        let ropeTestOutput = layer.attention.rope.callAsFunction(
            ropeTestInput,
            offset: ropeTestOffset
        )

        let attn = layer.attention(normed, mask: mask, cache: nil)
        let resid = input_embeddings + attn
        let mlpOut = layer.mlp(layer.post_attention_layernorm(resid))
        let out = resid + mlpOut
        return [
            "llama_layer0_input": input_embeddings,
            "llama_layer0_norm": normed,
            "llama_layer0_q_proj": qProj,
            "llama_layer0_k_proj": kProj,
            "llama_layer0_v_proj": vProj,
            "llama_layer0_q": q,
            "llama_layer0_k": k,
            "llama_layer0_v": v,
            "llama_layer0_q_rope": qRope,
            "llama_layer0_k_rope": kRope,
            "llama_rope_test_input": ropeTestInput,
            "llama_rope_test_output": ropeTestOutput,
            "llama_rope_test_offset": MLXArray([Int32(ropeTestOffset)]),
            "llama_layer0_attn": attn,
            "llama_layer0_resid": resid,
            "llama_layer0_mlp": mlpOut,
            "llama_layer0_out": out,
        ]
    }
}

public final class LlamaModel: Module {
    @ModuleInfo(key: "model") var model: LlamaModelInner

    public init(_ args: LlamaConfig) {
        self._model.wrappedValue = LlamaModelInner(args)
        super.init()
    }

    public func callAsFunction(
        inputs: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache]?,
        input_embeddings: MLXArray?
    ) -> MLXArray {
        model(inputs: inputs, mask: mask, cache: cache, input_embeddings: input_embeddings)
    }

    public func replaceRoPE(_ rope: RoPELayer) {
        model.replaceRoPE(rope)
    }

    public func debugLayer0(
        input_embeddings: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> [String: MLXArray] {
        model.debugLayer0(input_embeddings: input_embeddings, mask: mask)
    }
}
