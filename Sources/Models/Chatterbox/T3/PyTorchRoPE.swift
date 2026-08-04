import Foundation
import MLX
import MLXFast
import MLXNN

func computeLlama3InvFreq(
    dims: Int,
    base: Float,
    factor: Float,
    lowFreqFactor: Float,
    highFreqFactor: Float,
    oldContextLen: Int
) -> MLXArray {
    return Stream.withNewDefaultStream(device: .cpu) {
        let arange = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float64)
        let baseArray = MLXArray(Double(base))
        let invFreq = 1.0 / MLX.pow(baseArray, arange / Double(dims))

        let lowFreqWavelen = Double(oldContextLen) / Double(lowFreqFactor)
        let highFreqWavelen = Double(oldContextLen) / Double(highFreqFactor)

        let wavelen = (2.0 * Double.pi) / invFreq
        var smooth = (Double(oldContextLen) / wavelen - Double(lowFreqFactor))
            / Double(highFreqFactor - lowFreqFactor)
        smooth = MLX.clip(smooth, min: 0.0, max: 1.0)

        let scaled = invFreq / Double(factor)
        let interpolated = (1.0 - smooth) * scaled + smooth * invFreq

        let newInvFreq = MLX.where(
            wavelen .> MLXArray(lowFreqWavelen),
            scaled,
            MLX.where(wavelen .< MLXArray(highFreqWavelen), invFreq, interpolated)
        )

        return newInvFreq.asType(.float32)
    }
}

public final class PyTorchCompatibleRoPE: Module, RoPELayer {
    private let dims: Int
    private let maxSeqLen: Int
    private let cosTable: MLXArray
    private let sinTable: MLXArray

    public init(
        dims: Int,
        base: Float = 500000.0,
        maxSeqLen: Int = 8192,
        factor: Float = 8.0,
        lowFreqFactor: Float = 1.0,
        highFreqFactor: Float = 4.0,
        oldContextLen: Int = 8192
    ) {
        self.dims = dims
        self.maxSeqLen = maxSeqLen

        let invFreq = computeLlama3InvFreq(
            dims: dims,
            base: base,
            factor: factor,
            lowFreqFactor: lowFreqFactor,
            highFreqFactor: highFreqFactor,
            oldContextLen: oldContextLen
        )

        let (cosTable, sinTable) = Stream.withNewDefaultStream(device: .cpu) {
            let positions = MLXArray(stride(from: 0, to: maxSeqLen, by: 1)).asType(.float64)
            let invFreqF64 = invFreq.asType(.float64)
            let freqs = MLX.outer(positions, invFreqF64)
            let emb = MLX.concatenated([freqs, freqs], axis: -1)
            let cosTable = MLX.cos(emb).asType(.float32)
            let sinTable = MLX.sin(emb).asType(.float32)
            return (cosTable, sinTable)
        }

        self.cosTable = cosTable
        self.sinTable = sinTable

        super.init()
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = dims / 2
        let left = x[.ellipsis, ..<half]
        let right = x[.ellipsis, half...]
        return MLX.concatenated([-right, left], axis: -1)
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let length = x.dim(-2)
        if offset + length > maxSeqLen {
            fatalError("RoPE length \(offset + length) exceeds maxSeqLen \(maxSeqLen)")
        }

        var cos = cosTable[offset ..< offset + length, 0...]
        var sin = sinTable[offset ..< offset + length, 0...]
        cos = expandDims(expandDims(cos, axis: 0), axis: 0)
        sin = expandDims(expandDims(sin, axis: 0), axis: 0)

        return x * cos + rotateHalf(x) * sin
    }
}
