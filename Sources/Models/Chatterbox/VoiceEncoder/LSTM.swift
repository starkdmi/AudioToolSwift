import MLX
import MLXNN

final class LSTMLayer: Module {
    @ParameterInfo(key: "Wx") var wx: MLXArray
    @ParameterInfo(key: "Wh") var wh: MLXArray
    @ParameterInfo var bias: MLXArray

    let inputSize: Int
    let hiddenSize: Int

    init(inputSize: Int, hiddenSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self._wx.wrappedValue = MLXArray.zeros([hiddenSize * 4, inputSize])
        self._wh.wrappedValue = MLXArray.zeros([hiddenSize * 4, hiddenSize])
        self._bias.wrappedValue = MLXArray.zeros([hiddenSize * 4])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, h0: MLXArray?, c0: MLXArray?) -> (MLXArray, MLXArray, MLXArray) {
        let B = x.dim(0)
        let T = x.dim(1)
        var h = h0 ?? MLXArray.zeros([B, hiddenSize])
        var c = c0 ?? MLXArray.zeros([B, hiddenSize])

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(T)

        for t in 0..<T {
            let xt = x[0..., t, 0...]
            let gates = xt.matmul(wx.T) + h.matmul(wh.T) + bias
            let i = MLX.sigmoid(gates[0..., 0..<hiddenSize])
            let f = MLX.sigmoid(gates[0..., hiddenSize..<(hiddenSize * 2)])
            let g = MLX.tanh(gates[0..., (hiddenSize * 2)..<(hiddenSize * 3)])
            let o = MLX.sigmoid(gates[0..., (hiddenSize * 3)..<(hiddenSize * 4)])

            c = f * c + i * g
            h = o * MLX.tanh(c)
            outputs.append(h)
        }

        let stacked = stack(outputs, axis: 1)
        return (stacked, h, c)
    }
}

final class StackedLSTM: Module {
    let inputSize: Int
    let hiddenSize: Int
    let numLayers: Int

    @ModuleInfo var layers: [LSTMLayer]

    init(inputSize: Int, hiddenSize: Int, numLayers: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self._layers.wrappedValue = (0..<numLayers).map { idx in
            LSTMLayer(inputSize: idx == 0 ? inputSize : hiddenSize, hiddenSize: hiddenSize)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, hidden: (MLXArray, MLXArray)? = nil) -> (MLXArray, (MLXArray, MLXArray)) {
        var output = x
        var hStates: [MLXArray] = []
        var cStates: [MLXArray] = []

        var h0List: [MLXArray] = []
        var c0List: [MLXArray] = []
        if let hidden = hidden {
            let h0 = hidden.0
            let c0 = hidden.1
            for i in 0..<numLayers {
                h0List.append(h0[i, 0..., 0...])
                c0List.append(c0[i, 0..., 0...])
            }
        } else {
            h0List = Array(repeating: MLXArray.zeros([output.dim(0), hiddenSize]), count: numLayers)
            c0List = Array(repeating: MLXArray.zeros([output.dim(0), hiddenSize]), count: numLayers)
        }

        for (i, layer) in layers.enumerated() {
            let (out, h, c) = layer(output, h0: h0List[i], c0: c0List[i])
            output = out
            hStates.append(h)
            cStates.append(c)
        }

        let hStack = stack(hStates, axis: 0)
        let cStack = stack(cStates, axis: 0)
        return (output, (hStack, cStack))
    }
}
