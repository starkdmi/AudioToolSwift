import Foundation
import MLX
import MLXFast

public protocol KVCache: AnyObject {
    var offset: Int { get set }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
}

open class BaseKVCache: KVCache {
    public var offset: Int = 0

    public init() {}

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("BaseKVCache.update must be overridden")
    }
}

public final class KVCacheSimple: BaseKVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    public var step: Int = 256

    public override init() {
        super.init()
    }
    
    /// Initialize with pre-allocated size to avoid dynamic growth during generation.
    /// This reduces allocation overhead in autoregressive loops.
    public convenience init(maxSize: Int) {
        self.init()
        self.step = max(256, maxSize)  // Use maxSize as step to pre-allocate full size
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        let needsReset: Bool
        if let currentKeys = self.keys {
            needsReset = (previous + keys.dim(2)) > currentKeys.dim(2)
        } else {
            needsReset = true
        }

        if needsReset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                var trimmedKeys = currentKeys
                var trimmedValues = currentValues
                if previous % step != 0 {
                    trimmedKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    trimmedValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = MLX.concatenated([trimmedKeys, newK], axis: 2)
                self.values = MLX.concatenated([trimmedValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        offset += keys.dim(2)

        self.keys?[.ellipsis, previous ..< offset, 0...] = keys
        self.values?[.ellipsis, previous ..< offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<offset, 0...]

        return (returnedKeys, returnedValues)
    }
}

public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache = cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }

    let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: cachedKeys,
        values: cachedValues,
        scale: scale,
        mask: mask
    )
}
