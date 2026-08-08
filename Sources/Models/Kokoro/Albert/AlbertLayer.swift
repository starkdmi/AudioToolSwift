//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertLayer: Module {
  @ModuleInfo var attention: AlbertSelfAttention
  @ModuleInfo var fullLayerLayerNorm: LayerNormInference
  @ModuleInfo var ffn: Linear
  @ModuleInfo var ffnOutput: Linear

  init(weights: [String: MLXArray], config: AlbertModelArgs, layerNum: Int, innerGroupNum: Int) throws {
    let prefix = "bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum)"
    self._attention.wrappedValue = try AlbertSelfAttention(
      weights: weights,
      config: config,
      layerNum: layerNum,
      innerGroupNum: innerGroupNum
    )
    self._ffn.wrappedValue = Linear(
      weight: try weights.required("\(prefix).ffn.weight"),
      bias: try weights.required("\(prefix).ffn.bias")
    )
    self._ffnOutput.wrappedValue = Linear(
      weight: try weights.required("\(prefix).ffn_output.weight"),
      bias: try weights.required("\(prefix).ffn_output.bias")
    )

    let weightName = "\(prefix).full_layer_layer_norm.weight"
    let biasName = "\(prefix).full_layer_layer_norm.bias"
    let fullLayerLayerNormWeights = try weights.required(weightName)
    let fullLayerLayerNormBiases = try weights.required(biasName)

    guard fullLayerLayerNormWeights.count == config.hiddenSize else {
      throw KokoroModelLoadingError.invalidWeightShape(
        name: weightName,
        expected: config.hiddenSize,
        found: fullLayerLayerNormWeights.count
      )
    }
    guard fullLayerLayerNormBiases.count == config.hiddenSize else {
      throw KokoroModelLoadingError.invalidWeightShape(
        name: biasName,
        expected: config.hiddenSize,
        found: fullLayerLayerNormBiases.count
      )
    }

    // Use LayerNormInference which accepts weights directly in init
    self._fullLayerLayerNorm.wrappedValue = LayerNormInference(weight: fullLayerLayerNormWeights, bias: fullLayerLayerNormBiases, eps: config.layerNormEps)
  }

  func ffChunk(_ attentionOutput: MLXArray) -> MLXArray {
    var ffnOutputArray = ffn(attentionOutput)
    ffnOutputArray = MLXNN.gelu(ffnOutputArray)
    ffnOutputArray = ffnOutput(ffnOutputArray)
    return ffnOutputArray
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    let attentionOutput = attention(hiddenStates, attentionMask: attentionMask)
    let ffnOutput = ffChunk(attentionOutput)
    let output = fullLayerLayerNorm(ffnOutput + attentionOutput)
    return output
  }
}
