// Copyright © 2025
// ConvRNNF0Predictor - pitch predictor for HiFTGenerator
// Pure MLX Swift port of python_mlx/chatterbox/s3gen/f0_predictor.py

import MLX
import MLXNN

private func elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    MLX.where(x .>= 0, x, alpha * (MLX.exp(x) - 1))
}

public class ConvRNNF0Predictor: Module {
    @ModuleInfo public var condnet: [Conv1d]
    @ModuleInfo public var classifier: Linear

    public init(
        numClass: Int = 1,
        inChannels: Int = 80,
        condChannels: Int = 512
    ) {
        self._condnet = ModuleInfo(wrappedValue: [
            Conv1d(inputChannels: inChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
            Conv1d(inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
            Conv1d(inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
            Conv1d(inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
            Conv1d(inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
        ])
        self._classifier = ModuleInfo(wrappedValue: Linear(condChannels, numClass))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.swappedAxes(1, 2)
        for conv in condnet {
            out = elu(conv(out))
        }
        out = classifier(out)
        out = out.squeezed(axis: -1)
        return abs(out)
    }
}
