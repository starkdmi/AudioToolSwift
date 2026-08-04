import Foundation
import MLX
import MLXNN

private func l2Norm(_ x: MLXArray, axis: Int, keepDims: Bool) -> MLXArray {
    MLX.sqrt(MLX.sum(x * x, axis: axis, keepDims: keepDims))
}

private func safeL2Normalize(_ x: MLXArray, axis: Int, eps: Float = 1e-8) -> MLXArray {
    let norm = l2Norm(x, axis: axis, keepDims: true)
    let safeNorm = MLX.maximum(norm, MLXArray(eps))
    return x / safeNorm
}

public func getNumWins(nFrames: Int, step: Int, minCoverage: Float, hp: VoiceEncConfig) -> (Int, Int) {
    precondition(nFrames > 0)
    let winSize = hp.ve_partial_frames
    let diff = max(nFrames - winSize + step, 0)
    let nWins = diff / step
    let remainder = diff % step
    var totalWins = nWins
    if nWins == 0 || Float(remainder + (winSize - step)) / Float(winSize) >= minCoverage {
        totalWins += 1
    }
    let targetN = winSize + step * (totalWins - 1)
    return (totalWins, targetN)
}

public func getFrameStep(overlap: Float, rate: Float?, hp: VoiceEncConfig) -> Int {
    precondition(overlap >= 0 && overlap < 1)
    let frameStep: Int
    if let rate = rate {
        frameStep = Int(round((Float(hp.sample_rate) / rate) / Float(hp.ve_partial_frames)))
    } else {
        frameStep = Int(round(Float(hp.ve_partial_frames) * (1.0 - overlap)))
    }
    precondition(frameStep > 0 && frameStep <= hp.ve_partial_frames)
    return frameStep
}

public final class VoiceEncoder: Module {
    public let hp: VoiceEncConfig

    @ModuleInfo var lstm: StackedLSTM
    @ModuleInfo var proj: Linear
    @ModuleInfo var similarity_weight: MLXArray
    @ModuleInfo var similarity_bias: MLXArray

    public init(_ hp: VoiceEncConfig = VoiceEncConfig()) {
        self.hp = hp
        self._lstm.wrappedValue = StackedLSTM(inputSize: hp.num_mels, hiddenSize: hp.ve_hidden_size, numLayers: 3)
        self._proj.wrappedValue = Linear(hp.ve_hidden_size, hp.speaker_embed_size)
        self._similarity_weight.wrappedValue = MLXArray([10.0])
        self._similarity_bias.wrappedValue = MLXArray([-5.0])
        super.init()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]
        var biasIH: [Int: MLXArray] = [:]
        var biasHH: [Int: MLXArray] = [:]

        for (key, value) in weights {
            if key.contains("lstm.") && (key.contains("weight_ih") || key.contains("weight_hh") || key.contains("bias_ih") || key.contains("bias_hh")) {
                if let (weightType, layerIdx) = parseLSTMLayerKey(key) {
                    switch weightType {
                    case "weight_ih":
                        newWeights["lstm.layers.\(layerIdx).Wx"] = value
                    case "weight_hh":
                        newWeights["lstm.layers.\(layerIdx).Wh"] = value
                    case "bias_ih":
                        biasIH[layerIdx] = value
                    case "bias_hh":
                        biasHH[layerIdx] = value
                    default:
                        break
                    }
                }
            } else {
                newWeights[key] = value
            }
        }

        for (layerIdx, ih) in biasIH {
            if let hh = biasHH[layerIdx] {
                newWeights["lstm.layers.\(layerIdx).bias"] = ih + hh
            }
        }

        return newWeights
    }

    public func callAsFunction(_ mels: MLXArray) -> MLXArray {
        if hp.normalized_mels {
            let minVal = mels.min().item(Float.self)
            let maxVal = mels.max().item(Float.self)
            if minVal < 0 || maxVal > 1 {
                fatalError("Mels outside [0, 1]. Min=\(minVal), Max=\(maxVal)")
            }
        }

        let (_, (hN, _)) = lstm(mels)
        let finalHidden = hN[hN.dim(0) - 1, 0..., 0...]
        var rawEmbeds = proj(finalHidden)
        if hp.ve_final_relu {
            rawEmbeds = MLXNN.relu(rawEmbeds)
        }

        return safeL2Normalize(rawEmbeds, axis: 1)
    }

    public func inference(
        _ mels: MLXArray,
        melLens: [Int],
        overlap: Float = 0.5,
        rate: Float? = nil,
        minCoverage: Float = 0.8,
        batchSize: Int? = nil
    ) -> MLXArray {
        let frameStep = getFrameStep(overlap: overlap, rate: rate, hp: hp)
        var nPartialsList: [Int] = []
        var targetLens: [Int] = []
        for length in melLens {
            let (nPartials, target) = getNumWins(nFrames: length, step: frameStep, minCoverage: minCoverage, hp: hp)
            nPartialsList.append(nPartials)
            targetLens.append(target)
        }

        var mels = mels
        if let maxTarget = targetLens.max(), maxTarget > mels.dim(1) {
            let padLen = maxTarget - mels.dim(1)
            let pad = MLXArray.zeros([mels.dim(0), padLen, hp.num_mels])
            mels = MLX.concatenated([mels, pad], axis: 1)
        }

        var partialList: [MLXArray] = []
        for i in 0..<mels.dim(0) {
            let mel = mels[i, 0..., 0...]
            let nPartials = nPartialsList[i]
            if nPartials <= 0 {
                continue
            }
            let partialStarts = MLXArray(stride(from: 0, to: nPartials, by: 1)) * frameStep
            let frameOffsets = MLXArray(stride(from: 0, to: hp.ve_partial_frames, by: 1))
            let indices = expandDims(partialStarts, axis: 1) + expandDims(frameOffsets, axis: 0)
            let flat = indices.flattened()
            let melPartials = MLX.take(mel, flat, axis: 0).reshaped([nPartials, hp.ve_partial_frames, mel.dim(1)])
            partialList.append(melPartials)
        }

        let partials = MLX.concatenated(partialList, axis: 0)
        let totalPartials = partials.dim(0)
        let partialEmbeds: MLXArray
        if batchSize == nil || batchSize! >= totalPartials {
            partialEmbeds = callAsFunction(partials)
        } else {
            var embeds: [MLXArray] = []
            var start = 0
            while start < totalPartials {
                let end = min(start + batchSize!, totalPartials)
                embeds.append(callAsFunction(partials[start..<end, 0..., 0...]))
                start = end
            }
            partialEmbeds = MLX.concatenated(embeds, axis: 0)
        }

        var slices: [Int] = [0]
        for n in nPartialsList {
            slices.append(slices.last! + n)
        }

        var rawEmbeds: [MLXArray] = []
        for i in 0..<nPartialsList.count {
            let start = slices[i]
            let end = slices[i + 1]
            let segment = partialEmbeds[start..<end, 0...]
            rawEmbeds.append(MLX.mean(segment, axis: 0))
        }
        let stacked = stack(rawEmbeds, axis: 0)
        return safeL2Normalize(stacked, axis: 1)
    }

    public static func uttToSpkEmbed(_ uttEmbeds: MLXArray) -> MLXArray {
        precondition(uttEmbeds.ndim == 2)
        let mean = MLX.mean(uttEmbeds, axis: 0)
        let norm = l2Norm(mean, axis: 0, keepDims: false)
        return mean / norm
    }

    public static func voiceSimilarity(_ embedsX: MLXArray, _ embedsY: MLXArray) -> Float {
        var x = embedsX
        var y = embedsY
        if x.ndim != 1 {
            x = uttToSpkEmbed(x)
        }
        if y.ndim != 1 {
            y = uttToSpkEmbed(y)
        }
        return (x * y).sum().item(Float.self)
    }

    public func embedsFromMels(
        _ mels: [MLXArray],
        melLens: [Int]? = nil,
        asSpk: Bool = false,
        batchSize: Int = 32,
        overlap: Float = 0.5,
        rate: Float? = nil,
        minCoverage: Float = 0.8
    ) -> MLXArray {
        var melLens = melLens
        var melStack: MLXArray
        if melLens == nil {
            melLens = mels.map { $0.dim(0) }
            let maxLen = melLens!.max() ?? 0
            var padded: [MLXArray] = []
            for mel in mels {
                var mel = mel
                if mel.dim(0) < maxLen {
                    let pad = MLXArray.zeros([maxLen - mel.dim(0), mel.dim(1)], dtype: mel.dtype)
                    mel = MLX.concatenated([mel, pad], axis: 0)
                }
                padded.append(mel)
            }
            melStack = stack(padded, axis: 0)
        } else {
            melStack = stack(mels, axis: 0)
        }

        let uttEmbeds = inference(
            melStack,
            melLens: melLens ?? [],
            overlap: overlap,
            rate: rate,
            minCoverage: minCoverage,
            batchSize: batchSize
        )
        return asSpk ? VoiceEncoder.uttToSpkEmbed(uttEmbeds) : uttEmbeds
    }

    public func embedsFromWavs(
        _ wavs: [MLXArray],
        sampleRate: Int,
        asSpk: Bool = false,
        batchSize: Int = 32,
        trimTopDb: Float? = 20.0,
        overlap: Float = 0.5,
        rate: Float? = nil,
        minCoverage: Float = 0.8
    ) -> MLXArray {
        var processed: [MLXArray] = wavs

        if sampleRate != hp.sample_rate {
            processed = processed.map { resampleAudioPolyphase($0, origSR: sampleRate, targetSR: hp.sample_rate) }
        }

        if let trimTopDb = trimTopDb {
            processed = processed.map { trimSilenceLibrosa($0, topDb: trimTopDb) }
        }

        let mels = processed.map { wav -> MLXArray in
            let mel = melspectrogram(wav, hp: hp)
            return mel.T
        }

        return embedsFromMels(
            mels,
            melLens: nil,
            asSpk: asSpk,
            batchSize: batchSize,
            overlap: overlap,
            rate: rate ?? 1.3,
            minCoverage: minCoverage
        )
    }
}

private func parseLSTMLayerKey(_ key: String) -> (String, Int)? {
    let pattern = "lstm\\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\\d+)"
    guard let regex = try? NSRegularExpression(pattern: pattern) else {
        return nil
    }
    let range = NSRange(key.startIndex..<key.endIndex, in: key)
    guard let match = regex.firstMatch(in: key, range: range) else {
        return nil
    }
    guard let typeRange = Range(match.range(at: 1), in: key),
          let layerRange = Range(match.range(at: 2), in: key) else {
        return nil
    }
    let weightType = String(key[typeRange])
    let layerIdx = Int(key[layerRange]) ?? 0
    return (weightType, layerIdx)
}
