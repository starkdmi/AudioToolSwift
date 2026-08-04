import MLX
import MLXNN
import Foundation

// MARK: - Positional Embeddings

/// Create 1D sinusoidal embedding
/// - Parameters:
///   - length: Sequence length
///   - dim: Embedding dimension (must be even)
///   - shift: Position shift
///   - maxPeriod: Maximum period
/// - Returns: Embedding array of shape (length, 1, dim)
public func createSinEmbedding(length: Int, dim: Int, shift: Int = 0, maxPeriod: Float = 10000) -> MLXArray {
    guard dim % 2 == 0 else {
        fatalError("Dim must be even: \(dim)")
    }
    
    let pos = (MLXArray(0..<length) + MLXArray(Float(shift))).reshaped([-1, 1, 1])
    let halfDim = dim / 2
    let adim = MLXArray(0..<halfDim).reshaped([1, 1, -1])
    
    // phase: (Length, 1, HalfDim)
    let phase = pos / MLX.pow(MLXArray(maxPeriod), adim / MLXArray(Float(halfDim - 1)))
    
    // (Length, 1, Dim)
    return MLX.concatenated([MLX.cos(phase), MLX.sin(phase)], axis: -1)
}

/// Create 2D sinusoidal embedding for freq/time
/// - Parameters:
///   - dModel: Embedding dimension (must be divisible by 4)
///   - height: Height (freq bins)
///   - width: Width (time frames)
///   - maxPeriod: Maximum period
/// - Returns: Embedding array of shape (1, height, width, dModel)
public func create2DSinEmbedding(dModel: Int, height: Int, width: Int, maxPeriod: Float = 10000) -> MLXArray {
    guard dModel % 4 == 0 else {
        fatalError("Cannot use 2D sin/cos encoding with dim not divisible by 4: \(dModel)")
    }
    
    let dModelHalf = dModel / 2
    let divTerm = MLX.exp(MLXArray(stride(from: 0, to: dModelHalf, by: 2).map { Float($0) }) * MLXArray(-log(maxPeriod) / Float(dModelHalf)))
    
    let posW = MLXArray(0..<width).reshaped([-1, 1])
    let posH = MLXArray(0..<height).reshaped([-1, 1])
    
    let phaseW = posW * divTerm.expandedDimensions(axis: 0)
    let sinW = MLX.sin(phaseW)
    let cosW = MLX.cos(phaseW)
    
    var embW = MLX.stacked([sinW, cosW], axis: -1).reshaped([width, dModelHalf])
    embW = embW.transposed(1, 0)
    embW = MLX.broadcast(embW.expandedDimensions(axis: 1), to: [dModelHalf, height, width])
    
    let phaseH = posH * divTerm.expandedDimensions(axis: 0)
    let sinH = MLX.sin(phaseH)
    let cosH = MLX.cos(phaseH)
    
    var embH = MLX.stacked([sinH, cosH], axis: -1).reshaped([height, dModelHalf])
    embH = embH.transposed(1, 0)
    embH = MLX.broadcast(embH.expandedDimensions(axis: 2), to: [dModelHalf, height, width])
    
    var pe = MLX.concatenated([embW, embH], axis: 0)
    // Return (1, H, W, C) for channel last usage
    pe = pe.expandedDimensions(axis: 0).transposed(0, 2, 3, 1)
    return pe
}

// MARK: - MyGroupNorm

/// GroupNorm wrapper with 1 group (LayerNorm-like)
public class MyGroupNorm: Module, UnaryLayer {
    @ModuleInfo var norm: GroupNorm
    
    public init(channels: Int) {
        self.norm = GroupNorm(groupCount: 1, dimensions: channels, pytorchCompatible: true)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return norm(x)
    }
}

// MARK: - MyTransformerEncoderLayer

/// Self-attention transformer encoder layer
public class MyTransformerEncoderLayer: Module {
    @ModuleInfo var self_attn: MultiHeadAttention
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    @ModuleInfo var norm1: UnaryLayer
    @ModuleInfo var norm2: UnaryLayer
    @ModuleInfo var norm_out: UnaryLayer?
    @ModuleInfo var gamma_1: UnaryLayer
    @ModuleInfo var gamma_2: UnaryLayer
    
    let norm_first: Bool
    let use_norm_out: Bool
    let sparse: Bool
    let mask_type: String
    let sparse_attn_window: Int
    var src_mask: MLXArray?
    
    public init(
        dModel: Int,
        nhead: Int,
        dimFeedforward: Int = 2048,
        dropout: Float = 0.1,
        normFirst: Bool = false,
        layerScale: Bool = false,
        initValues: Float = 1e-4,
        groupNorm: Bool = false,
        useNormOut: Bool = false,
        sparse: Bool = false,
        maskType: String = "diag",
        sparseAttnWindow: Int = 400,
        globalWindow: Int = 50,
        maskRandomSeed: Int = 42,
        sparsity: Float = 0.95,
        autoSparsity: Bool = false
    ) {
        self.norm_first = normFirst
        self.use_norm_out = useNormOut
        self.sparse = sparse
        self.mask_type = maskType
        self.sparse_attn_window = sparseAttnWindow
        self.src_mask = nil
        
        self.self_attn = MultiHeadAttention(dimensions: dModel, numHeads: nhead, bias: true)
        self.linear1 = Linear(dModel, dimFeedforward)
        self.linear2 = Linear(dimFeedforward, dModel)
        
        if groupNorm {
            self.norm1 = MyGroupNorm(channels: dModel)
            self.norm2 = MyGroupNorm(channels: dModel)
        } else {
            self.norm1 = LayerNorm(dimensions: dModel)
            self.norm2 = LayerNorm(dimensions: dModel)
        }
        
        if normFirst && useNormOut {
            self.norm_out = MyGroupNorm(channels: dModel)
        }
        
        if layerScale {
            self.gamma_1 = LayerScale(channels: dModel, init_value: initValues)
            self.gamma_2 = LayerScale(channels: dModel, init_value: initValues)
        } else {
            self.gamma_1 = Identity()
            self.gamma_2 = Identity()
        }
        
        super.init()
    }
    
    func generateMask(length: Int) -> MLXArray? {
        if mask_type == "diag" {
            let indices = MLXArray(0..<length)
            let diff = MLX.abs(indices.expandedDimensions(axis: 1) - indices.expandedDimensions(axis: 0))
            let mask = diff .<= MLXArray(sparse_attn_window)
            // Convert to additive mask: 0 for True, -inf for False
            return MLX.where(mask, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        }
        return nil
    }
    
    public func callAsFunction(_ src: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var attnMask = mask
        
        if sparse && attnMask == nil {
            if src_mask == nil || src_mask!.shape[0] != src.shape[1] {
                src_mask = generateMask(length: src.shape[1])
            }
            attnMask = src_mask
        }
        
        var output = src
        
        if norm_first {
            let xNorm = norm1(output)
            let attnOut = self_attn(xNorm, keys: xNorm, values: xNorm, mask: attnMask)
            output = output + gamma_1(attnOut)
            
            let xNorm2 = norm2(output)
            let ffOut = linear2(GELU()(linear1(xNorm2)))
            output = output + gamma_2(ffOut)
            
            if let normOut = norm_out {
                output = normOut(output)
            }
        } else {
            let attnOut = self_attn(output, keys: output, values: output, mask: attnMask)
            output = norm1(output + gamma_1(attnOut))
            let ffOut = linear2(GELU()(linear1(output)))
            output = norm2(output + gamma_2(ffOut))
        }
        
        return output
    }
}

// MARK: - CrossTransformerEncoderLayer

/// Cross-attention transformer encoder layer
public class CrossTransformerEncoderLayer: Module {
    @ModuleInfo var cross_attn: MultiHeadAttention
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    @ModuleInfo var norm1: UnaryLayer
    @ModuleInfo var norm2: UnaryLayer
    @ModuleInfo var norm3: UnaryLayer
    @ModuleInfo var norm_out: UnaryLayer?
    @ModuleInfo var gamma_1: UnaryLayer
    @ModuleInfo var gamma_2: UnaryLayer
    
    let norm_first: Bool
    let use_norm_out: Bool
    
    public init(
        dModel: Int,
        nhead: Int,
        dimFeedforward: Int = 2048,
        dropout: Float = 0.1,
        normFirst: Bool = false,
        layerScale: Bool = false,
        initValues: Float = 1e-4,
        groupNorm: Bool = false,
        useNormOut: Bool = false
    ) {
        self.norm_first = normFirst
        self.use_norm_out = useNormOut
        
        self.cross_attn = MultiHeadAttention(dimensions: dModel, numHeads: nhead, bias: true)
        self.linear1 = Linear(dModel, dimFeedforward)
        self.linear2 = Linear(dimFeedforward, dModel)
        
        if groupNorm {
            self.norm1 = MyGroupNorm(channels: dModel)
            self.norm2 = MyGroupNorm(channels: dModel)
            self.norm3 = MyGroupNorm(channels: dModel)
        } else {
            self.norm1 = LayerNorm(dimensions: dModel)
            self.norm2 = LayerNorm(dimensions: dModel)
            self.norm3 = LayerNorm(dimensions: dModel)
        }
        
        if normFirst && useNormOut {
            self.norm_out = MyGroupNorm(channels: dModel)
        }
        
        if layerScale {
            self.gamma_1 = LayerScale(channels: dModel, init_value: initValues)
            self.gamma_2 = LayerScale(channels: dModel, init_value: initValues)
        } else {
            self.gamma_1 = Identity()
            self.gamma_2 = Identity()
        }
        
        super.init()
    }
    
    public func callAsFunction(_ q: MLXArray, _ k: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var output: MLXArray
        
        if norm_first {
            let qNorm = norm1(q)
            let kNorm = norm2(k)
            let attnOut = cross_attn(qNorm, keys: kNorm, values: kNorm, mask: mask)
            var x = q + gamma_1(attnOut)
            let xNorm = norm3(x)
            let ffOut = linear2(GELU()(linear1(xNorm)))
            x = x + gamma_2(ffOut)
            if let normOut = norm_out {
                x = normOut(x)
            }
            output = x
        } else {
            let attnOut = cross_attn(q, keys: k, values: k, mask: mask)
            var x = norm1(q + gamma_1(attnOut))
            let ffOut = linear2(GELU()(linear1(x)))
            x = norm2(x + gamma_2(ffOut))
            output = x
        }
        
        return output
    }
}

// MARK: - CrossTransformerEncoder

/// Cross-transformer encoder for processing freq and time branches
public class CrossTransformerEncoder: Module {
    let rescale: Float
    let max_period: Float
    let weight_pos_embed: Float
    let classic_parity: Int
    let num_layers: Int
    
    @ModuleInfo var norm_in: UnaryLayer
    @ModuleInfo var norm_in_t: UnaryLayer
    
    // Flattened layer arrays - up to 5 layers each
    @ModuleInfo var layers_0: Module?
    @ModuleInfo var layers_1: Module?
    @ModuleInfo var layers_2: Module?
    @ModuleInfo var layers_3: Module?
    @ModuleInfo var layers_4: Module?
    
    @ModuleInfo var layers_t_0: Module?
    @ModuleInfo var layers_t_1: Module?
    @ModuleInfo var layers_t_2: Module?
    @ModuleInfo var layers_t_3: Module?
    @ModuleInfo var layers_t_4: Module?
    
    // Cache for positional embeddings
    var posEmbCache: ((Int, Int, Int), MLXArray)?
    var posEmb1DCache: ((Int, Int), MLXArray)?
    
    public init(
        dim: Int,
        emb: String = "sin",
        hiddenScale: Float = 4.0,
        numHeads: Int = 8,
        numLayers: Int = 5,
        crossFirst: Bool = false,
        dropout: Float = 0.0,
        maxPositions: Int = 10000,
        normIn: Bool = true,
        normInGroup: Bool = false,
        groupNorm: Bool = false,
        normFirst: Bool = true,
        normOut: Bool = true,
        maxPeriod: Float = 10000.0,
        weightDecay: Float = 0.0,
        lr: Float? = nil,
        layerScale: Bool = true,
        gelu: Bool = true,
        weightPosEmbed: Float = 1.0,
        rescale: Float = 0.1
    ) {
        self.rescale = rescale
        self.max_period = maxPeriod
        self.weight_pos_embed = weightPosEmbed
        self.classic_parity = crossFirst ? 1 : 0
        self.num_layers = numLayers
        
        let hiddenDim = Int(Float(dim) * hiddenScale)
        
        // Norm in
        if normIn {
            if normInGroup {
                self.norm_in = MyGroupNorm(channels: dim)
                self.norm_in_t = MyGroupNorm(channels: dim)
            } else {
                self.norm_in = LayerNorm(dimensions: dim)
                self.norm_in_t = LayerNorm(dimensions: dim)
            }
        } else {
            self.norm_in = Identity()
            self.norm_in_t = Identity()
        }
        
        // Common kwargs for layers
        func createSelfAttnLayer() -> MyTransformerEncoderLayer {
            return MyTransformerEncoderLayer(
                dModel: dim,
                nhead: numHeads,
                dimFeedforward: hiddenDim,
                dropout: dropout,
                normFirst: normFirst,
                layerScale: layerScale,
                groupNorm: groupNorm,
                useNormOut: normOut
            )
        }
        
        func createCrossAttnLayer() -> CrossTransformerEncoderLayer {
            return CrossTransformerEncoderLayer(
                dModel: dim,
                nhead: numHeads,
                dimFeedforward: hiddenDim,
                dropout: dropout,
                normFirst: normFirst,
                layerScale: layerScale,
                groupNorm: groupNorm,
                useNormOut: normOut
            )
        }
        
        // Create layers based on parity
        if numLayers > 0 {
            if 0 % 2 == classic_parity {
                self.layers_0 = createSelfAttnLayer()
                self.layers_t_0 = createSelfAttnLayer()
            } else {
                self.layers_0 = createCrossAttnLayer()
                self.layers_t_0 = createCrossAttnLayer()
            }
        }
        
        if numLayers > 1 {
            if 1 % 2 == classic_parity {
                self.layers_1 = createSelfAttnLayer()
                self.layers_t_1 = createSelfAttnLayer()
            } else {
                self.layers_1 = createCrossAttnLayer()
                self.layers_t_1 = createCrossAttnLayer()
            }
        }
        
        if numLayers > 2 {
            if 2 % 2 == classic_parity {
                self.layers_2 = createSelfAttnLayer()
                self.layers_t_2 = createSelfAttnLayer()
            } else {
                self.layers_2 = createCrossAttnLayer()
                self.layers_t_2 = createCrossAttnLayer()
            }
        }
        
        if numLayers > 3 {
            if 3 % 2 == classic_parity {
                self.layers_3 = createSelfAttnLayer()
                self.layers_t_3 = createSelfAttnLayer()
            } else {
                self.layers_3 = createCrossAttnLayer()
                self.layers_t_3 = createCrossAttnLayer()
            }
        }
        
        if numLayers > 4 {
            if 4 % 2 == classic_parity {
                self.layers_4 = createSelfAttnLayer()
                self.layers_t_4 = createSelfAttnLayer()
            } else {
                self.layers_4 = createCrossAttnLayer()
                self.layers_t_4 = createCrossAttnLayer()
            }
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, _ xt: MLXArray) -> (MLXArray, MLXArray) {
        // x: (Batch, Fr, T1, C) <- Channel Last
        let B = x.shape[0]
        let Fr = x.shape[1]
        let T1 = x.shape[2]
        let C = x.shape[3]
        
        // Pos Emb 2D - use cache if dimensions match
        let posEmb2DFlat: MLXArray
        if let cache = posEmbCache, cache.0 == (C, Fr, T1) {
            posEmb2DFlat = cache.1
        } else {
            let posEmb2D = create2DSinEmbedding(dModel: C, height: Fr, width: T1, maxPeriod: max_period)
            let flattened = posEmb2D.transposed(0, 2, 1, 3).reshaped([1, T1 * Fr, C])
            posEmbCache = ((C, Fr, T1), flattened)
            posEmb2DFlat = flattened
        }
        
        // Flatten x: (B, T1*Fr, C)
        var xFlat = x.transposed(0, 2, 1, 3).reshaped([B, T1 * Fr, C])
        xFlat = norm_in(xFlat)
        xFlat = xFlat + weight_pos_embed * posEmb2DFlat
        
        // Process xt
        let T2 = xt.shape[1]
        let Cxt = xt.shape[2]
        
        // Pos Emb 1D
        let posEmb: MLXArray
        if let cache = posEmb1DCache, cache.0 == (T2, Cxt) {
            posEmb = cache.1
        } else {
            let emb = createSinEmbedding(length: T2, dim: Cxt, maxPeriod: max_period)
            let reshaped = emb.reshaped([1, T2, Cxt])
            posEmb1DCache = ((T2, Cxt), reshaped)
            posEmb = reshaped
        }
        
        var xtFlat = norm_in_t(xt)
        xtFlat = xtFlat + weight_pos_embed * posEmb
        

        
        // Process through layers
        func processLayer(idx: Int, xIn: MLXArray, xtIn: MLXArray) -> (MLXArray, MLXArray) {
            let layer: Module?
            let layerT: Module?
            
            switch idx {
            case 0: layer = layers_0; layerT = layers_t_0
            case 1: layer = layers_1; layerT = layers_t_1
            case 2: layer = layers_2; layerT = layers_t_2
            case 3: layer = layers_3; layerT = layers_t_3
            case 4: layer = layers_4; layerT = layers_t_4
            default: return (xIn, xtIn)
            }
            
            guard let l1 = layer, let l2 = layerT else { return (xIn, xtIn) }
            
            let res: (MLXArray, MLXArray)
            if idx % 2 == classic_parity {
                // Self-attention
                let selfAttn1 = l1 as! MyTransformerEncoderLayer
                let selfAttn2 = l2 as! MyTransformerEncoderLayer
                res = (selfAttn1(xIn), selfAttn2(xtIn))
            } else {
                // Cross-attention
                let crossAttn1 = l1 as! CrossTransformerEncoderLayer
                let crossAttn2 = l2 as! CrossTransformerEncoderLayer
                let oldX = xIn
                res = (crossAttn1(xIn, xtIn), crossAttn2(xtIn, oldX))
            }
            

            
            return res
        }
        
        for idx in 0..<num_layers {
            (xFlat, xtFlat) = processLayer(idx: idx, xIn: xFlat, xtIn: xtFlat)
        }
        
        // Reshape back
        let xOut = xFlat.reshaped([B, T1, Fr, C]).transposed(0, 2, 1, 3)
        let xtOut = xtFlat
        
        return (xOut, xtOut)
    }
}
