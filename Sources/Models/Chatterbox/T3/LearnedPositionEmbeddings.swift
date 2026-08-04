import MLX
import MLXNN
import MLXRandom

public final class LearnedPositionEmbeddings: Module {
    @ModuleInfo var emb: Embedding

    public init(seqLen: Int, modelDim: Int, initScale: Float = 0.02) {
        let weight = MLXRandom.normal([seqLen, modelDim], scale: initScale)
        self._emb.wrappedValue = Embedding(weight: weight)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let sl = x.dim(1)
        return emb(MLXArray(stride(from: 0, to: sl, by: 1)))
    }

    public func getFixedEmbedding(_ idx: MLXArray) -> MLXArray {
        var indices = idx
        if indices.ndim == 1 {
            indices = expandDims(indices, axis: 0)
        }
        return emb(indices)
    }

    public func getFixedEmbedding(_ idx: Int) -> MLXArray {
        let indices = MLXArray([idx], [1, 1])
        return emb(indices)
    }
}
