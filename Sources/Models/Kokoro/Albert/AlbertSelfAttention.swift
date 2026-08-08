//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertSelfAttention: Module {
  let numAttentionHeads: Int
  let attentionHeadSize: Int
  let allHeadSize: Int

  @ModuleInfo var query: Linear
  @ModuleInfo var key: Linear
  @ModuleInfo var value: Linear
  @ModuleInfo var dense: Linear
  @ModuleInfo var layerNorm: LayerNormInference

  init(weights: [String: MLXArray], config: AlbertModelArgs, layerNum: Int, innerGroupNum: Int) throws {
    numAttentionHeads = config.numAttentionHeads
    attentionHeadSize = config.hiddenSize / config.numAttentionHeads
    allHeadSize = numAttentionHeads * attentionHeadSize

    let prefix = "bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention"
    self._query.wrappedValue = Linear(
      weight: try weights.required("\(prefix).query.weight"),
      bias: try weights.required("\(prefix).query.bias")
    )
    self._key.wrappedValue = Linear(
      weight: try weights.required("\(prefix).key.weight"),
      bias: try weights.required("\(prefix).key.bias")
    )
    self._value.wrappedValue = Linear(
      weight: try weights.required("\(prefix).value.weight"),
      bias: try weights.required("\(prefix).value.bias")
    )
    self._dense.wrappedValue = Linear(
      weight: try weights.required("\(prefix).dense.weight"),
      bias: try weights.required("\(prefix).dense.bias")
    )

    let weightName = "\(prefix).LayerNorm.weight"
    let biasName = "\(prefix).LayerNorm.bias"
    let layerNormWeights = try weights.required(weightName)
    let layerNormBiases = try weights.required(biasName)

    guard layerNormWeights.count == config.hiddenSize else {
      throw KokoroModelLoadingError.invalidWeightShape(
        name: weightName,
        expected: config.hiddenSize,
        found: layerNormWeights.count
      )
    }
    guard layerNormBiases.count == config.hiddenSize else {
      throw KokoroModelLoadingError.invalidWeightShape(
        name: biasName,
        expected: config.hiddenSize,
        found: layerNormBiases.count
      )
    }

    // Use LayerNormInference which accepts weights directly in init
    self._layerNorm.wrappedValue = LayerNormInference(weight: layerNormWeights, bias: layerNormBiases, eps: config.layerNormEps)
  }

  func transposeForScores(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    var newShape: [Int] = []

    for i in 0 ..< (shape.count - 1) {
      newShape.append(shape[i])
    }

    newShape.append(numAttentionHeads)
    newShape.append(attentionHeadSize)

    let reshaped = x.reshaped(newShape)
    return reshaped.transposed(0, 2, 1, 3)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    let mixedQueryLayer = query(hiddenStates)
    let mixedKeyLayer = key(hiddenStates)
    let mixedValueLayer = value(hiddenStates)

    let queryLayer = transposeForScores(mixedQueryLayer)
    let keyLayer = transposeForScores(mixedKeyLayer)
    let valueLayer = transposeForScores(mixedValueLayer)

    let keyLayerTransposed = keyLayer.transposed(0, 1, 3, 2)
    var attentionScores = MLX.matmul(queryLayer, keyLayerTransposed)
    attentionScores = attentionScores / sqrt(Float(attentionHeadSize))

    if let attentionMask = attentionMask {
      attentionScores = attentionScores + attentionMask
    }

    let attentionProbs = MLX.softmax(attentionScores, axis: -1)

    var contextLayer = MLX.matmul(attentionProbs, valueLayer)
    contextLayer = contextLayer.transposed(0, 2, 1, 3)

    var newContextLayerShape: [Int] = []
    let shape = contextLayer.shape

    for i in 0 ..< (shape.count - 2) {
      newContextLayerShape.append(shape[i])
    }

    newContextLayerShape.append(allHeadSize)

    contextLayer = contextLayer.reshaped(newContextLayerShape)
    contextLayer = dense(contextLayer)
    contextLayer = layerNorm(contextLayer + hiddenStates)

    return contextLayer
  }
}
