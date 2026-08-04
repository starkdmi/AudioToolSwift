import Foundation
import MLX
import MLXFast
import MLXNN

public struct S3TokenizerConfig {
    public var n_mels: Int = 128
    public var n_audio_ctx: Int = 1500
    public var n_audio_state: Int = 1280
    public var n_audio_head: Int = 20
    public var n_audio_layer: Int = 6
    public var n_codebook_size: Int = Int(Foundation.pow(3.0, 8.0))

    public init() {}
}

private func precomputeFreqsCis(dim: Int, end: Int, theta: Float = 10000.0, scaling: Float? = nil) -> (MLXArray, MLXArray) {
    let freqs = 1.0 / MLX.pow(
        MLXArray(theta),
        MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim)
    )
    var t = MLXArray(stride(from: 0, to: end, by: 1)).asType(.float32)
    if let scaling = scaling {
        t = t * scaling
    }
    let outer = MLX.outer(t, freqs)
    let cosFreqs = MLX.cos(outer)
    let sinFreqs = MLX.sin(outer)
    let cosCat = MLX.concatenated([cosFreqs, cosFreqs], axis: -1)
    let sinCat = MLX.concatenated([sinFreqs, sinFreqs], axis: -1)
    return (cosCat, sinCat)
}

private func applyRotaryEmb(_ xq: MLXArray, _ xk: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
    let cos = expandDims(expandDims(cos, axis: 0), axis: 2)
    let sin = expandDims(expandDims(sin, axis: 0), axis: 2)

    let D = xq.dim(-1)
    let half = D / 2

    let xqLeft = xq[0..., 0..., 0..., 0..<half]
    let xqRight = xq[0..., 0..., 0..., half...]
    let xqRot = MLX.concatenated([-xqRight, xqLeft], axis: -1)

    let xkLeft = xk[0..., 0..., 0..., 0..<half]
    let xkRight = xk[0..., 0..., 0..., half...]
    let xkRot = MLX.concatenated([-xkRight, xkLeft], axis: -1)

    let qOut = xq * cos + xqRot * sin
    let kOut = xk * cos + xkRot * sin

    return (qOut, kOut)
}

final class FSQCodebook: Module {
    @ModuleInfo var project_down: Linear
    let level: Int

    init(dim: Int, level: Int = 3) {
        self.level = level
        self._project_down.wrappedValue = Linear(dim, 8)
        super.init()
    }

    func preprocess(_ x: MLXArray) -> MLXArray {
        x.reshaped([-1, x.dim(-1)])
    }

    func encode(_ x: MLXArray) -> MLXArray {
        let shape0 = x.dim(0)
        let shape1 = x.dim(1)
        var h = preprocess(x)
        h = MLX.tanh(project_down(h).asType(.float32)) * 0.9990000128746033
        h = MLX.round(h) + 1.0

        let powers = MLX.pow(
            MLXArray(Float(level)),
            MLXArray(stride(from: 0, to: Int(pow(2.0, Double(level))), by: 1)).asType(h.dtype)
        )
        let mu = MLX.sum(h * expandDims(powers, axis: 0), axis: -1)
        return mu.reshaped([shape0, shape1]).asType(.int32)
    }
}

final class FSQVectorQuantization: Module {
    @ModuleInfo var fsq_codebook: FSQCodebook
    let codebook_size: Int

    init(dim: Int, codebookSize: Int) {
        self.codebook_size = codebookSize
        self._fsq_codebook.wrappedValue = FSQCodebook(dim: dim, level: 3)
        super.init()
    }

    func encode(_ x: MLXArray) -> MLXArray {
        fsq_codebook.encode(x)
    }
}

final class FSMNMultiHeadAttention: MultiHeadAttention {
    @ModuleInfo var fsmn_block: Conv1d
    let left_padding: Int
    let right_padding: Int

    init(n_state: Int, n_head: Int, kernelSize: Int = 31) {
        self._fsmn_block.wrappedValue = Conv1d(
            inputChannels: n_state,
            outputChannels: n_state,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            groups: n_state,
            bias: false
        )
        self.left_padding = (kernelSize - 1) / 2
        self.right_padding = kernelSize - 1 - left_padding
        super.init(n_state: n_state, n_head: n_head)
    }

    private func forwardFSMN(_ inputs: MLXArray, maskPad: MLXArray?) -> MLXArray {
        let B = inputs.dim(0)
        let T = inputs.dim(1)
        var x = inputs.reshaped(B, T, -1)
        let D = x.dim(2)

        if let maskPad = maskPad, maskPad.dim(2) > 0 {
            x = x * maskPad
        }

        let padLeft = MLXArray.zeros([B, left_padding, D], dtype: x.dtype)
        let padRight = MLXArray.zeros([B, right_padding, D], dtype: x.dtype)
        let padded = MLX.concatenated([padLeft, x, padRight], axis: 1)
        var out = fsmn_block(padded)
        out = out + x

        if let maskPad = maskPad {
            out = out * maskPad
        }
        return out
    }

    func qkv_attention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        mask: MLXArray?,
        maskPad: MLXArray?,
        freqs: (MLXArray, MLXArray)?
    ) -> (MLXArray, MLXArray?, MLXArray) {
        let B = q.dim(0)
        let T = q.dim(1)
        let D = q.dim(2)
        let scale = pow(Float(D / n_head), -0.25)

        var qh = q.reshaped(B, T, n_head, -1)
        var kh = k.reshaped(B, T, n_head, -1)
        let vh = v.reshaped(B, T, n_head, -1)

        if let freqs = freqs {
            let (cos, sin) = freqs
            let (rotQ, rotK) = applyRotaryEmb(qh, kh, cos: cos[..<T], sin: sin[..<T])
            qh = rotQ
            kh = rotK
        }

        let fsmMemory = forwardFSMN(vh, maskPad: maskPad)

        qh = qh.transposed(0, 2, 1, 3) * scale
        kh = kh.transposed(0, 2, 1, 3) * scale
        let vt = vh.transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask = mask {
            maskMode = .array(mask)
        } else {
            maskMode = .none
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: qh,
            keys: kh,
            values: vt,
            scale: 1.0,
            mask: maskMode
        )
        let merged = output.transposed(0, 2, 1, 3).reshaped(B, T, D)
        return (merged, nil, fsmMemory)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        maskPad: MLXArray? = nil,
        freqs: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, MLXArray?) {
        let q = query(x)
        let k = key(x)
        let v = value(x)
        let (wv, qk, fsmMemory) = qkv_attention(q: q, k: k, v: v, mask: mask, maskPad: maskPad, freqs: freqs)
        return (out(wv) + fsmMemory, qk)
    }
}

final class ResidualAttentionBlockV2: Module {
    @ModuleInfo var attn: FSMNMultiHeadAttention
    @ModuleInfo var attn_ln: LayerNorm
    @ModuleInfo var mlp: Sequential
    @ModuleInfo var mlp_ln: LayerNorm

    init(n_state: Int, n_head: Int) {
        self._attn.wrappedValue = FSMNMultiHeadAttention(n_state: n_state, n_head: n_head)
        self._attn_ln.wrappedValue = LayerNorm(dimensions: n_state)

        let n_mlp = n_state * 4
        self._mlp.wrappedValue = Sequential([
            Linear(n_state, n_mlp),
            GELU(),
            Linear(n_mlp, n_state)
        ])
        self._mlp_ln.wrappedValue = LayerNorm(dimensions: n_state)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, maskPad: MLXArray, freqs: (MLXArray, MLXArray)) -> MLXArray {
        let attnOut = attn(attn_ln(x), mask: mask, maskPad: maskPad, freqs: freqs).0
        var out = x + attnOut
        out = out + mlp(mlp_ln(out))
        return out
    }
}

final class AudioEncoderV2: Module {
    let stride: Int

    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo var blocks: [ResidualAttentionBlockV2]

    private let freqs: (MLXArray, MLXArray)

    init(nMels: Int, nState: Int, nHead: Int, nLayer: Int, stride: Int) {
        self.stride = stride
        self._conv1.wrappedValue = Conv1d(
            inputChannels: nMels,
            outputChannels: nState,
            kernelSize: 3,
            stride: stride,
            padding: 1
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: nState,
            outputChannels: nState,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )

        self.freqs = precomputeFreqsCis(dim: 64, end: 1024 * 2)
        self._blocks.wrappedValue = (0..<nLayer).map { _ in ResidualAttentionBlockV2(n_state: nState, n_head: nHead) }

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ xLen: MLXArray) -> (MLXArray, MLXArray) {
        let T = x.dim(-1)
        var mask = makeNonPadMask(xLen, maxLen: T)
        mask = expandDims(mask, axis: 1)

        var x = x.transposed(0, 2, 1)
        var maskT = mask.transposed(0, 2, 1)

        x = conv1(x * maskT)
        x = MLXNN.gelu(x)

        let kernel = 3
        let pad = 1
        var newLen = MLX.floorDivide(xLen + 2 * pad - (kernel - 1) - 1, stride) + 1
        var xSlen = (T + 2 * pad - (kernel - 1) - 1) / stride + 1

        mask = makeNonPadMask(newLen, maxLen: xSlen)
        maskT = expandDims(mask, axis: -1)

        x = conv2(x * maskT)
        x = MLXNN.gelu(x)

        newLen = MLX.floorDivide(newLen + 2 * pad - (kernel - 1) - 1, 2) + 1
        xSlen = (xSlen + 2 * pad - (kernel - 1) - 1) / 2 + 1

        mask = makeNonPadMask(newLen, maxLen: xSlen)
        let maskPad = expandDims(mask, axis: -1)
        let maskBias = expandDims(maskToBias(mask, dtype: x.dtype), axis: 1)

        for block in blocks {
            x = block(x, mask: maskBias, maskPad: maskPad, freqs: freqs)
        }

        return (x, newLen)
    }
}

public final class S3TokenizerV2: Module {
    public let config: S3TokenizerConfig

    @ModuleInfo var encoder: AudioEncoderV2
    @ModuleInfo var quantizer: FSQVectorQuantization

    public init(name: String, config: S3TokenizerConfig = S3TokenizerConfig()) {
        var cfg = config
        if !name.contains("v1") {
            cfg.n_codebook_size = Int(pow(3.0, 8.0))
        }
        self.config = cfg
        self._encoder.wrappedValue = AudioEncoderV2(
            nMels: cfg.n_mels,
            nState: cfg.n_audio_state,
            nHead: cfg.n_audio_head,
            nLayer: cfg.n_audio_layer,
            stride: 2
        )
        self._quantizer.wrappedValue = FSQVectorQuantization(
            dim: cfg.n_audio_state,
            codebookSize: cfg.n_codebook_size
        )
        super.init()
    }

    public func callAsFunction(_ mel: MLXArray, _ melLen: MLXArray) -> (MLXArray, MLXArray) {
        quantize(mel, melLen)
    }

    public func quantize(_ mel: MLXArray, _ melLen: MLXArray) -> (MLXArray, MLXArray) {
        let maxFrames = 3000
        let longMask = melLen .> MLXArray(maxFrames)
        if MLX.any(longMask).item(Bool.self) {
            return quantizeMixedBatch(mel, melLen, longMask, maxFrames: maxFrames)
        }

        let (hidden, codeLen) = encoder(mel, melLen)
        let code = quantizer.encode(hidden)
        return (code, codeLen)
    }

    private func quantizeMixedBatch(
        _ mel: MLXArray,
        _ melLen: MLXArray,
        _ longMask: MLXArray,
        maxFrames: Int
    ) -> (MLXArray, MLXArray) {
        let batchSize = mel.dim(0)
        let sampleRate = 16000
        let hopLength = 160
        let windowSize = 30
        let overlap = 4

        let framesPerWindow = windowSize * sampleRate / hopLength
        let framesPerOverlap = overlap * sampleRate / hopLength
        let framesPerStride = framesPerWindow - framesPerOverlap

        var allSegments: [MLXArray] = []
        var allSegmentLens: [Int] = []
        var segmentInfo: [(batchIdx: Int, isLong: Bool, segmentIdx: Int, totalSegments: Int?)] = []

        for batchIdx in 0..<batchSize {
            let audioMel = mel[batchIdx, 0..., 0...]
            let audioMelLen = melLen[batchIdx].item(Int.self)
            let isLong = longMask[batchIdx].item(Bool.self)

            if !isLong {
                var segment = audioMel[0..., 0..<audioMelLen]
                let segLen = audioMelLen
                if segLen < framesPerWindow {
                    let padSize = framesPerWindow - segLen
                    let pad = MLXArray.zeros([segment.dim(0), padSize], dtype: segment.dtype)
                    segment = MLX.concatenated([segment, pad], axis: 1)
                }
                allSegments.append(segment)
                allSegmentLens.append(segLen)
                segmentInfo.append((batchIdx, false, 0, 1))
            } else {
                var start = 0
                var segIdx = 0
                while start < audioMelLen {
                    let end = min(start + framesPerWindow, audioMelLen)
                    var segment = audioMel[0..., start..<end]
                    let segLen = segment.dim(1)
                    if segLen < framesPerWindow {
                        let padSize = framesPerWindow - segLen
                        let pad = MLXArray.zeros([segment.dim(0), padSize], dtype: segment.dtype)
                        segment = MLX.concatenated([segment, pad], axis: 1)
                    }
                    allSegments.append(segment)
                    allSegmentLens.append(segLen)
                    segmentInfo.append((batchIdx, true, segIdx, nil))
                    segIdx += 1
                    start += framesPerStride
                }
                for i in 0..<segmentInfo.count {
                    if segmentInfo[i].batchIdx == batchIdx && segmentInfo[i].isLong {
                        segmentInfo[i].totalSegments = segIdx
                    }
                }
            }
        }

        if allSegments.isEmpty {
            return (MLXArray.zeros([batchSize, 0], dtype: .int32), MLXArray.zeros([batchSize], dtype: .int32))
        }

        let batchMel = stack(allSegments, axis: 0)
        let batchLens = MLXArray(allSegmentLens)

        let (hidden, codeLen) = encoder(batchMel, batchLens)
        let codes = quantizer.encode(hidden)

        var results: [Int: [Int]] = [:]
        var lengths: [Int: Int] = [:]
        var longSegments: [Int: [[Int]]] = [:]

        for (segIdx, info) in segmentInfo.enumerated() {
            let segLen = codeLen[segIdx].item(Int.self)
            let segmentCode = codes[segIdx, 0..<segLen].asArray(Int.self)

            if !info.isLong {
                results[info.batchIdx] = segmentCode
                lengths[info.batchIdx] = segmentCode.count
            } else {
                if longSegments[info.batchIdx] == nil {
                    longSegments[info.batchIdx] = []
                }
                longSegments[info.batchIdx]?.append(segmentCode)
            }
        }

        for batchIdx in 0..<batchSize {
            if longMask[batchIdx].item(Bool.self) {
                let segments = longSegments[batchIdx] ?? []
                let merged = mergeTokenizedSegments(segments, overlap: overlap, tokenRate: 25)
                results[batchIdx] = merged
                lengths[batchIdx] = merged.count
            }
        }

        let maxLen = lengths.values.max() ?? 0
        var padded = MLXArray.zeros([batchSize, maxLen], dtype: .int32)
        var outLens = MLXArray.zeros([batchSize], dtype: .int32)

        for i in 0..<batchSize {
            let tokens = results[i] ?? []
            if !tokens.isEmpty {
                padded[i, 0..<tokens.count] = MLXArray(tokens)
                outLens[i] = MLXArray(tokens.count)
            }
        }

        return (padded, outLens)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]
        let currWeights = Dictionary(uniqueKeysWithValues: parameters().flattened())

        for (key, value) in weights {
            if key.contains("freqs_cis") || key.contains("_mel_filters") {
                continue
            }
            if key.hasPrefix("onnx::") {
                continue
            }

            var newKey = key
            newKey = newKey.replacingOccurrences(of: "quantizer._codebook.", with: "quantizer.fsq_codebook.")
            newKey = newKey.replacingOccurrences(of: "quantizer.codebook.", with: "quantizer.fsq_codebook.")
            newKey = replaceRegex(newKey, pattern: "\\.mlp\\.(\\d+)\\.", replacement: ".mlp.layers.$1.")

            if (newKey.contains(".conv1.") || newKey.contains(".conv2.") || newKey.contains(".fsmn_block.")),
               newKey.contains("weight"),
               value.ndim == 3
            {
                if let expected = currWeights[newKey], value.shape != expected.shape {
                    newWeights[newKey] = value.swappedAxes(1, 2)
                    continue
                }
            }

            newWeights[newKey] = value
        }

        return newWeights
    }
}

private func replaceRegex(_ input: String, pattern: String, replacement: String) -> String {
    guard let regex = try? NSRegularExpression(pattern: pattern) else {
        return input
    }
    let range = NSRange(input.startIndex..<input.endIndex, in: input)
    return regex.stringByReplacingMatches(in: input, range: range, withTemplate: replacement)
}
