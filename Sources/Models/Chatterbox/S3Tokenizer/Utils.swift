import Foundation
import MLX

public func logMelSpectrogramCompat(_ audio: MLXArray, nMels: Int = 128, padding: Int = 0) -> MLXArray {
    var audio = audio
    var was1d = false
    if audio.ndim == 1 {
        audio = expandDims(audio, axis: 0)
        was1d = true
    }

    if padding > 0 {
        let pad = MLXArray.zeros([audio.dim(0), padding], dtype: audio.dtype)
        audio = MLX.concatenated([audio, pad], axis: -1)
    }

    let nFFT = 400
    let hopLength = 160
    let winLength = 400

    let window = createHannWindowS3(winLength, periodic: true)

    var realParts: [MLXArray] = []
    var imagParts: [MLXArray] = []
    for i in 0..<audio.dim(0) {
        let (real, imag) = stftS3(
            audio[i, 0...],
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: window,
            center: true
        )
        realParts.append(real)
        imagParts.append(imag)
    }

    let stackedReal = stack(realParts, axis: 0)
    let stackedImag = stack(imagParts, axis: 0)
    var magnitudes = stackedReal * stackedReal + stackedImag * stackedImag
    magnitudes = magnitudes[0..., 0..., 0..<max(0, magnitudes.dim(2) - 1)]
    magnitudes = magnitudes.transposed(0, 2, 1)

    let filters = createMelFilterBankS3(
        sr: 16000,
        nFFT: nFFT,
        nMels: nMels,
        fmin: 0.0,
        fmax: 8000.0
    )

    var melSpec = magnitudes.matmul(filters.T)
    melSpec = melSpec.transposed(0, 2, 1)

    var logSpec = MLX.log10(MLX.maximum(melSpec, 1e-10))
    logSpec = MLX.maximum(logSpec, MLX.max(logSpec) - 8.0)
    logSpec = (logSpec + 4.0) / 4.0

    if was1d {
        return logSpec.squeezed(axis: 0)
    }
    return logSpec
}

private func stftS3(
    _ x: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool
) -> (MLXArray, MLXArray) {
    var signal = x
    if center {
        let padAmount = nFFT / 2
        if padAmount > 0 {
            let leftIdx = MLXArray(
                Array(stride(from: padAmount, through: 1, by: -1))
            )
            let rightStart = signal.dim(0) - 2
            let rightEnd = signal.dim(0) - padAmount - 1
            let rightIdx = MLXArray(
                Array(stride(from: rightStart, through: rightEnd, by: -1))
            )
            let left = MLX.take(signal, leftIdx, axis: 0)
            let right = MLX.take(signal, rightIdx, axis: 0)
            signal = MLX.concatenated([left, signal, right], axis: 0)
        }
    }

    let paddedLen = signal.dim(0)
    let numFrames = (paddedLen - winLength) / hopLength + 1
    let frameStarts = MLXArray(Array(stride(from: 0, to: numFrames, by: 1))) * hopLength
    let sampleOffsets = MLXArray(Array(stride(from: 0, to: winLength, by: 1)))
    let indices =
        frameStarts.expandedDimensions(axis: 1) + sampleOffsets.expandedDimensions(axis: 0)
    let flatIndices = indices.flattened().asType(.int32)
    let frames = MLX.take(signal, flatIndices, axis: 0).reshaped([numFrames, winLength])

    var win = window
    if win.dim(0) < nFFT {
        let padSize = nFFT - win.dim(0)
        let pad = MLXArray.zeros([padSize], dtype: win.dtype)
        win = MLX.concatenated([win, pad], axis: 0)
    }

    let windowed = frames * win[..<winLength]
    let fft = MLX.rfft(windowed, n: nFFT, axis: 1)
    return (fft.realPart().transposed(1, 0), fft.imaginaryPart().transposed(1, 0))
}

private func createHannWindowS3(_ winLen: Int, periodic: Bool) -> MLXArray {
    let n = MLXArray(stride(from: 0, to: winLen, by: 1)).asType(.float32)
    if periodic {
        return 0.5 * (1 - MLX.cos(2 * Float.pi * n / Float(winLen)))
    } else {
        return 0.5 * (1 - MLX.cos(2 * Float.pi * n / Float(winLen - 1)))
    }
}

private func hzToMelS3(_ frequencies: MLXArray, htk: Bool = false) -> MLXArray {
    if htk {
        return 2595.0 * MLX.log10(1.0 + frequencies / 700.0)
    }
    let fMin: Float = 0.0
    let fSp: Float = 200.0 / 3.0
    let minLogHz: Float = 1000.0
    let minLogMel = (minLogHz - fMin) / fSp
    let logstep = Float(Foundation.log(6.4) / 27.0)

    var mels = (frequencies - fMin) / fSp
    let logRegion = frequencies .>= MLXArray(minLogHz)
    mels = MLX.where(
        logRegion,
        MLXArray(minLogMel) + MLX.log(frequencies / minLogHz) / logstep,
        mels
    )
    return mels
}

private func melToHzS3(_ mels: MLXArray, htk: Bool = false) -> MLXArray {
    if htk {
        return 700.0 * (MLX.pow(10.0, mels / 2595.0) - 1.0)
    }
    let fMin: Float = 0.0
    let fSp: Float = 200.0 / 3.0
    let minLogHz: Float = 1000.0
    let minLogMel = (minLogHz - fMin) / fSp
    let logstep = Float(log(6.4) / 27.0)

    var freqs = fMin + fSp * mels
    let logRegion = mels .>= MLXArray(minLogMel)
    freqs = MLX.where(
        logRegion,
        MLXArray(minLogHz) * MLX.exp(logstep * (mels - minLogMel)),
        freqs
    )
    return freqs
}

private func createMelFilterBankS3(
    sr: Int,
    nFFT: Int,
    nMels: Int = 128,
    fmin: Float = 0.0,
    fmax: Float? = nil
) -> MLXArray {
    let fmax = fmax ?? Float(sr) / 2.0
    let fftfreqs = MLX.linspace(0, Float(sr) / 2.0, count: 1 + nFFT / 2)

    let minMel = hzToMelS3(MLXArray(fmin)).item(Float.self)
    let maxMel = hzToMelS3(MLXArray(fmax)).item(Float.self)
    let mels = MLX.linspace(minMel, maxMel, count: nMels + 2)
    let melF = melToHzS3(mels)

    let melFLeft = melF[0..<(melF.dim(0) - 1)]
    let melFRight = melF[1..<melF.dim(0)]
    let fdiff = melFRight - melFLeft
    let ramps = expandDims(melF, axis: -1) - expandDims(fftfreqs, axis: 0)
    let fdiffLeft = fdiff[0..<(fdiff.dim(0) - 1)]
    let fdiffRight = fdiff[1..<fdiff.dim(0)]
    let lower = -ramps[0..<(ramps.dim(0) - 2), 0...] / expandDims(fdiffLeft, axis: 1)
    let upper = ramps[2..<ramps.dim(0), 0...] / expandDims(fdiffRight, axis: 1)

    var weights = MLX.maximum(0, MLX.minimum(lower, upper))

    let enorm = 2.0 / (melF[2..<(nMels + 2)] - melF[0..<nMels])
    weights = weights * expandDims(enorm, axis: 1)

    return weights
}

public func makeNonPadMask(_ lengths: MLXArray, maxLen: Int = 0) -> MLXArray {
    let batchSize = lengths.dim(0)
    let maxLen = maxLen > 0 ? maxLen : lengths.max().item(Int.self)
    let seqRange = MLXArray(stride(from: 0, to: maxLen, by: 1))
    let seqRangeExpand = broadcastTo(expandDims(seqRange, axis: 0), [batchSize, maxLen])
    let seqLengthExpand = expandDims(lengths, axis: -1)
    let mask = seqRangeExpand .>= seqLengthExpand
    return MLX.logicalNot(mask)
}

public func maskToBias(_ mask: MLXArray, dtype: MLX.DType = .float32) -> MLXArray {
    var mask = mask.asType(dtype)
    mask = (1.0 - mask) * -1.0e10
    return mask
}

public func padding(_ data: [MLXArray]) -> (MLXArray, MLXArray) {
    let lengths = MLXArray(data.map { $0.dim(1) })
    let maxLen = data.map { $0.dim(1) }.max() ?? 0
    let batchSize = data.count
    let nMels = data[0].dim(0)

    var padded = MLXArray.zeros([batchSize, nMels, maxLen], dtype: data[0].dtype)
    for (i, feat) in data.enumerated() {
        let seqLen = feat.dim(1)
        padded[i, 0..., 0..<seqLen] = feat
    }

    return (padded, lengths)
}

public func mergeTokenizedSegments(_ segments: [[Int]], overlap: Int, tokenRate: Int) -> [Int] {
    var merged: [Int] = []
    let overlapTokens = (overlap / 2) * tokenRate
    for (index, tokens) in segments.enumerated() {
        let left = index == 0 ? 0 : overlapTokens
        let right = index == segments.count - 1 ? tokens.count : max(0, tokens.count - overlapTokens)
        if right > left {
            merged.append(contentsOf: tokens[left..<right])
        }
    }
    return merged
}
