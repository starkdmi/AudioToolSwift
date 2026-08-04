// Copyright © 2025
// MatchaTransformer - Transformer blocks for flow matching decoder
// Pure MLX port of Python MLX matcha/transformer.py

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - DiffusersAttention

/// Attention module matching diffusers.models.attention_processor.Attention
public class DiffusersAttention: Module {
    let heads: Int
    let dim_head: Int
    let inner_dim: Int
    let scale: Float
    
    @ModuleInfo(key: "query_proj") public var query_proj: Linear
    @ModuleInfo(key: "key_proj") public var key_proj: Linear
    @ModuleInfo(key: "value_proj") public var value_proj: Linear
    @ModuleInfo(key: "out_proj") public var out_proj: Linear
    
    public init(queryDim: Int, heads: Int = 8, dimHead: Int = 64, qkvBias: Bool = false, outBias: Bool = true) {
        self.heads = heads
        self.dim_head = dimHead
        self.inner_dim = heads * dimHead
        self.scale = pow(Float(dimHead), -0.5)
        
        self._query_proj = ModuleInfo(wrappedValue: Linear(queryDim, inner_dim, bias: qkvBias))
        self._key_proj = ModuleInfo(wrappedValue: Linear(queryDim, inner_dim, bias: qkvBias))
        self._value_proj = ModuleInfo(wrappedValue: Linear(queryDim, inner_dim, bias: qkvBias))
        self._out_proj = ModuleInfo(wrappedValue: Linear(inner_dim, queryDim, bias: outBias))
        
        super.init()
    }
    
    public func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let (B, T, _) = (hiddenStates.shape[0], hiddenStates.shape[1], hiddenStates.shape[2])
        
        // Project to q, k, v
        var q = query_proj(hiddenStates)
        var k = key_proj(hiddenStates)
        var v = value_proj(hiddenStates)
        
        // Reshape to (B, heads, T, dim_head)
        q = q.reshaped(B, T, heads, dim_head).transposed(0, 2, 1, 3)
        k = k.reshaped(B, T, heads, dim_head).transposed(0, 2, 1, 3)
        v = v.reshaped(B, T, heads, dim_head).transposed(0, 2, 1, 3)
        
        // Handle attention mask
        var out: MLXArray
        if let mask = attentionMask {
            // Check if it's additive bias (contains large negative values)
            // Additive bias - compute attention manually
            var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale  // (B, heads, T, T)
            
            // Add bias (broadcast across heads)
            if mask.ndim == 3 {
                scores = scores + mask.expandedDimensions(axis: 1)
            } else {
                scores = scores + mask
            }
            
            let weights = softmax(scores, axis: -1)
            out = matmul(weights, v)
        } else {
            // No mask - use fast path
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: scale, mask: .none
            )
        }
        
        // Reshape back: (B, heads, T, dim_head) -> (B, T, inner_dim)
        out = out.transposed(0, 2, 1, 3).reshaped(B, T, inner_dim)
        
        // Output projection
        out = out_proj(out)
        
        return out
    }
}

// MARK: - FeedForward

/// Feed-forward network with GELU activation
public class FeedForward: Module {
    @ModuleInfo public var layers: LayerList
    
    public init(dim: Int, mult: Int) {
        let innerDim = mult
        self._layers = ModuleInfo(wrappedValue: LayerList([
            Linear(dim, innerDim),
            Linear(innerDim, dim)
        ]))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = layers[0](x)
        h = gelu(h)
        h = layers[1](h)
        return h
    }
}

// MARK: - BasicTransformerBlock

/// Basic transformer block for decoder
public class BasicTransformerBlock: Module {
    @ModuleInfo public var norm1: LayerNorm
    @ModuleInfo public var norm3: LayerNorm
    @ModuleInfo public var attn: DiffusersAttention
    @ModuleInfo public var ff: FeedForward
    let activation_fn: String
    
    public init(
        dim: Int,
        numAttentionHeads: Int,
        attentionHeadDim: Int,
        dropout: Float = 0.0,
        activationFn: String = "gelu"
    ) {
        self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim))
        self._norm3 = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim))
        self._attn = ModuleInfo(wrappedValue: DiffusersAttention(
            queryDim: dim,
            heads: numAttentionHeads,
            dimHead: attentionHeadDim,
            qkvBias: false,
            outBias: true
        ))
        self._ff = ModuleInfo(wrappedValue: FeedForward(dim: dim, mult: dim * 4))
        self.activation_fn = activationFn
        super.init()
    }
    
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        timestep: MLXArray? = nil
    ) -> MLXArray {
        var h = hiddenStates

        // Self-attention
        let normed1 = norm1(h)
        let attnOut = attn(normed1, attentionMask: attentionMask)
        h = h + attnOut
        
        // Feed-forward
        let normed3 = norm3(h)
        let ffOut = ff(normed3)
        h = h + ffOut
        
        return h
    }
}
