// Copyright © 2025
// CAMPPlus speaker encoder with Kaldi fbank
// Pure MLX Swift port of python_mlx/chatterbox/s3gen/xvector.py

import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioUtils

// MARK: - Kaldi fbank helpers

private let kaldiEpsilon: Float = 1.1920928955078125e-07
private let millisecondsToSeconds: Float = 0.001
private let windowTypePovey = "povey"

private func nextPowerOf2(_ x: Int) -> Int {
    if x == 0 { return 1 }
    var value = 1
    while value < x {
        value <<= 1
    }
    return value
}

private func asStrided2d(_ waveform: MLXArray, m: Int, windowSize: Int, windowShift: Int) -> MLXArray {
    MLX.asStrided(waveform, [m, windowSize], strides: [windowShift, 1])
}

private func getStrided(
    _ waveform: MLXArray,
    windowSize: Int,
    windowShift: Int,
    snipEdges: Bool
) -> MLXArray {
    let numSamples = waveform.shape[0]
    if snipEdges {
        if numSamples < windowSize {
            return MLXArray.zeros([0, 0], dtype: waveform.dtype)
        }
        let m = 1 + (numSamples - windowSize) / windowShift
        return asStrided2d(waveform, m: m, windowSize: windowSize, windowShift: windowShift)
    }

    let reversed = MLX.take(
        waveform,
        MLXArray(Array(stride(from: numSamples - 1, through: 0, by: -1))),
        axis: 0
    )
    let m = (numSamples + (windowShift / 2)) / windowShift
    let pad = windowSize / 2 - windowShift / 2

    var padded = waveform
    if pad > 0 {
        let start = max(reversed.shape[0] - pad, 0)
        let padLeft = reversed[start..<reversed.shape[0]]
        padded = MLX.concatenated([padLeft, waveform, reversed], axis: 0)
    } else {
        let start = max(-pad, 0)
        padded = MLX.concatenated([waveform[start..<waveform.shape[0]], reversed], axis: 0)
    }
    return asStrided2d(padded, m: m, windowSize: windowSize, windowShift: windowShift)
}

private func featureWindowFunction(
    windowType: String,
    windowSize: Int,
    dtype: DType
) -> MLXArray {
    if windowType != windowTypePovey {
        fatalError("Unsupported window type \(windowType)")
    }
    let n = MLXArray(0..<windowSize).asType(dtype)
    let a = 2.0 * Float.pi / Float(windowSize - 1)
    return MLX.pow(0.5 - 0.5 * MLX.cos(a * n), 0.85)
}

private func logEnergy(_ strided: MLXArray, epsilon: MLXArray, energyFloor: Float) -> MLXArray {
    let logEnergy = MLX.log(MLX.maximum(MLX.sum(MLX.square(strided), axis: 1), epsilon))
    if energyFloor == 0.0 {
        return logEnergy
    }
    return MLX.maximum(logEnergy, MLXArray(log(energyFloor), dtype: strided.dtype))
}

private func getWaveformAndWindowProperties(
    _ waveform: MLXArray,
    channel: Int,
    sampleFrequency: Float,
    frameShift: Float,
    frameLength: Float,
    roundToPowerOfTwo: Bool,
    preemphasisCoefficient: Float
) -> (MLXArray, Int, Int, Int) {
    let ch = max(channel, 0)
    precondition(ch < waveform.shape[0], "Invalid channel \(ch)")

    let wave = waveform[ch, 0...]
    let windowShift = Int(sampleFrequency * frameShift * millisecondsToSeconds)
    let windowSize = Int(sampleFrequency * frameLength * millisecondsToSeconds)
    let paddedWindowSize = roundToPowerOfTwo ? nextPowerOf2(windowSize) : windowSize

    precondition(windowShift > 0, "windowShift must be > 0")
    precondition(windowSize >= 2 && windowSize <= wave.shape[0], "Invalid window size")
    precondition(paddedWindowSize % 2 == 0, "padded window size must be even")
    precondition(preemphasisCoefficient >= 0.0 && preemphasisCoefficient <= 1.0, "Invalid preemphasis")
    precondition(sampleFrequency > 0.0, "Invalid sample frequency")

    return (wave, windowShift, windowSize, paddedWindowSize)
}

private func getWindow(
    waveform: MLXArray,
    paddedWindowSize: Int,
    windowSize: Int,
    windowShift: Int,
    windowType: String,
    snipEdges: Bool,
    rawEnergy: Bool,
    energyFloor: Float,
    dither: Float,
    removeDcOffset: Bool,
    preemphasisCoefficient: Float
) -> (MLXArray, MLXArray) {
    let dtype = waveform.dtype
    let epsilon = MLXArray(kaldiEpsilon, dtype: dtype)

    var strided = getStrided(waveform, windowSize: windowSize, windowShift: windowShift, snipEdges: snipEdges)

    if dither != 0.0 {
        let rand = MLXRandom.normal(strided.shape) * dither
        strided = strided + rand
    }

    if removeDcOffset {
        let rowMeans = MLX.mean(strided, axis: 1, keepDims: true)
        strided = strided - rowMeans
    }

    let signalLogEnergy: MLXArray
    if rawEnergy {
        signalLogEnergy = logEnergy(strided, epsilon: epsilon, energyFloor: energyFloor)
    } else {
        signalLogEnergy = MLXArray.zeros([max(1, strided.shape[0])], dtype: dtype)
    }

    if preemphasisCoefficient != 0.0 && strided.shape[1] > 1 {
        let firstCol = strided[0..., 0..<1]
        let shifted = MLX.concatenated([firstCol, strided], axis: 1)
        strided = strided - preemphasisCoefficient * shifted[0..., 0..<(shifted.shape[1] - 1)]
    }

    let window = featureWindowFunction(windowType: windowType, windowSize: windowSize, dtype: dtype)
    strided = strided * window.expandedDimensions(axis: 0)

    if paddedWindowSize > windowSize {
        let padRight = paddedWindowSize - windowSize
        strided = MLX.padded(strided, widths: [IntOrPair(0), IntOrPair([0, padRight])])
    }

    if !rawEnergy {
        return (strided, logEnergy(strided, epsilon: epsilon, energyFloor: energyFloor))
    }
    return (strided, signalLogEnergy)
}

private func melScale(_ freq: MLXArray) -> MLXArray {
    let dtype = freq.dtype
    let one = MLXArray(1.0, dtype: dtype)
    let scale = MLXArray(1127.0, dtype: dtype)
    let denom = MLXArray(700.0, dtype: dtype)
    return scale * MLX.log(one + freq / denom)
}

private func inverseMelScale(_ mel: MLXArray) -> MLXArray {
    let dtype = mel.dtype
    let one = MLXArray(1.0, dtype: dtype)
    let scale = MLXArray(700.0, dtype: dtype)
    let denom = MLXArray(1127.0, dtype: dtype)
    return scale * (MLX.exp(mel / denom) - one)
}

private func getMelBanks(
    numBins: Int,
    windowLengthPadded: Int,
    sampleFreq: Float,
    lowFreq: Float,
    highFreq: Float
) -> (MLXArray, MLXArray) {
    precondition(numBins > 3, "numBins must be > 3")
    precondition(windowLengthPadded % 2 == 0, "windowLengthPadded must be even")

    let numFftBins = windowLengthPadded / 2
    let nyquist = 0.5 * sampleFreq
    var high = highFreq
    if high <= 0.0 {
        high += nyquist
    }

    let fftBinWidth = sampleFreq / Float(windowLengthPadded)
    let melLow = melScale(MLXArray(lowFreq)).item(Float.self)
    let melHigh = melScale(MLXArray(high)).item(Float.self)
    let melDelta = (melHigh - melLow) / Float(numBins + 1)

    let binIdx = MLXArray(0..<numBins).asType(.float32).expandedDimensions(axis: 1)
    let leftMel = MLXArray(melLow) + binIdx * melDelta
    let centerMel = MLXArray(melLow) + (binIdx + 1.0) * melDelta
    let rightMel = MLXArray(melLow) + (binIdx + 2.0) * melDelta

    let centerFreqs = inverseMelScale(centerMel)
    let mel = melScale(fftBinWidth * MLXArray(0..<numFftBins).asType(.float32)).expandedDimensions(axis: 0)

    let upSlope = (mel - leftMel) / (centerMel - leftMel)
    let downSlope = (rightMel - mel) / (rightMel - centerMel)
    let bins = MLX.maximum(zerosLike(upSlope), MLX.minimum(upSlope, downSlope))

    return (bins, centerFreqs)
}

private func kaldiFbank(
    _ audio: MLXArray,
    sampleRate: Int = 16000,
    numMelBins: Int = 80,
    frameLength: Float = 25.0,
    frameShift: Float = 10.0,
    dither: Float = 0.0,
    lowFreq: Float = 20.0,
    highFreq: Float = 0.0
) -> MLXArray {
    var waveform = audio
    if waveform.ndim == 1 {
        waveform = waveform.expandedDimensions(axis: 0)
    }

    let dtype = waveform.dtype
    let (wave, windowShift, windowSize, paddedWindowSize) = getWaveformAndWindowProperties(
        waveform,
        channel: 0,
        sampleFrequency: Float(sampleRate),
        frameShift: frameShift,
        frameLength: frameLength,
        roundToPowerOfTwo: true,
        preemphasisCoefficient: 0.97
    )

    let (strided, _) = getWindow(
        waveform: wave,
        paddedWindowSize: paddedWindowSize,
        windowSize: windowSize,
        windowShift: windowShift,
        windowType: windowTypePovey,
        snipEdges: true,
        rawEnergy: true,
        energyFloor: 1.0,
        dither: dither,
        removeDcOffset: true,
        preemphasisCoefficient: 0.97
    )

    if strided.shape[0] == 0 {
        return MLXArray.zeros([0, numMelBins], dtype: dtype)
    }

    let fft = MLX.rfft(strided, n: paddedWindowSize, axis: -1)
    let spectrum = MLX.square(MLX.abs(fft))
    var (melEnergies, _) = getMelBanks(
        numBins: numMelBins,
        windowLengthPadded: paddedWindowSize,
        sampleFreq: Float(sampleRate),
        lowFreq: lowFreq,
        highFreq: highFreq
    )
    melEnergies = melEnergies.asType(dtype)
    melEnergies = MLX.padded(melEnergies, widths: [IntOrPair(0), IntOrPair([0, 1])])

    var mel = MLX.matmul(spectrum, melEnergies.T)
    mel = MLX.log(MLX.maximum(mel, MLXArray(kaldiEpsilon, dtype: dtype)))
    return mel
}

private func kaldiFbankDebug(
    _ audio: MLXArray,
    sampleRate: Int = 16000,
    numMelBins: Int = 80,
    frameLength: Float = 25.0,
    frameShift: Float = 10.0,
    dither: Float = 0.0,
    lowFreq: Float = 20.0,
    highFreq: Float = 0.0
) -> (MLXArray, [String: MLXArray]) {
    var waveform = audio
    if waveform.ndim == 1 {
        waveform = waveform.expandedDimensions(axis: 0)
    }

    let dtype = waveform.dtype
    let (wave, windowShift, windowSize, paddedWindowSize) = getWaveformAndWindowProperties(
        waveform,
        channel: 0,
        sampleFrequency: Float(sampleRate),
        frameShift: frameShift,
        frameLength: frameLength,
        roundToPowerOfTwo: true,
        preemphasisCoefficient: 0.97
    )

    let (strided, _) = getWindow(
        waveform: wave,
        paddedWindowSize: paddedWindowSize,
        windowSize: windowSize,
        windowShift: windowShift,
        windowType: windowTypePovey,
        snipEdges: true,
        rawEnergy: true,
        energyFloor: 1.0,
        dither: dither,
        removeDcOffset: true,
        preemphasisCoefficient: 0.97
    )

    if strided.shape[0] == 0 {
        let empty = MLXArray.zeros([0, numMelBins], dtype: dtype)
        return (empty, ["speaker_fbank_strided": strided])
    }

    let fft = MLX.rfft(strided, n: paddedWindowSize, axis: -1)
    let spectrum = MLX.square(MLX.abs(fft))
    var (melEnergies, _) = getMelBanks(
        numBins: numMelBins,
        windowLengthPadded: paddedWindowSize,
        sampleFreq: Float(sampleRate),
        lowFreq: lowFreq,
        highFreq: highFreq
    )
    melEnergies = melEnergies.asType(dtype)
    melEnergies = MLX.padded(melEnergies, widths: [IntOrPair(0), IntOrPair([0, 1])])

    let melPre = MLX.matmul(spectrum, melEnergies.T)
    let mel = MLX.log(MLX.maximum(melPre, MLXArray(kaldiEpsilon, dtype: dtype)))
    let debug: [String: MLXArray] = [
        "speaker_fbank_strided": strided,
        "speaker_fbank_fft_real": fft.realPart(),
        "speaker_fbank_fft_imag": fft.imaginaryPart(),
        "speaker_fbank_spectrum": spectrum,
        "speaker_fbank_mel_energies": melEnergies,
        "speaker_fbank_pre_log": melPre,
        "speaker_fbank_pre_norm": mel,
    ]
    return (mel, debug)
}

// MARK: - CAMPPlus modules

public class BatchNorm: Module {
    @ParameterInfo public var weight: MLXArray
    @ParameterInfo public var bias: MLXArray
    @ParameterInfo(key: "running_mean") public var running_mean: MLXArray
    @ParameterInfo(key: "running_var") public var running_var: MLXArray
    let eps: Float

    public init(featureCount: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([featureCount])
        self._bias.wrappedValue = MLXArray.zeros([featureCount])
        self._running_mean.wrappedValue = MLXArray.zeros([featureCount])
        self._running_var.wrappedValue = MLXArray.ones([featureCount])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let axes = Array(0..<(x.ndim - 1))
        let mean: MLXArray
        let variance: MLXArray
        if training {
            mean = MLX.mean(x, axes: axes)
            variance = MLX.variance(x, axes: axes)
        } else {
            mean = running_mean
            variance = running_var
        }
        var out = (x - mean) * rsqrt(variance + eps)
        out = out * weight + bias
        return out
    }
}

public class BatchNormNoAffine: Module {
    @ParameterInfo(key: "running_mean") public var running_mean: MLXArray
    @ParameterInfo(key: "running_var") public var running_var: MLXArray
    let eps: Float

    public init(featureCount: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._running_mean.wrappedValue = MLXArray.zeros([featureCount])
        self._running_var.wrappedValue = MLXArray.ones([featureCount])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let axes = Array(0..<(x.ndim - 1))
        let mean: MLXArray
        let variance: MLXArray
        if training {
            mean = MLX.mean(x, axes: axes)
            variance = MLX.variance(x, axes: axes)
        } else {
            mean = running_mean
            variance = running_var
        }
        return (x - mean) * rsqrt(variance + eps)
    }
}

private final class ReLUModule: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXNN.relu(x)
    }
}

private func applyNonlinear(_ layers: [Module], to input: MLXArray) -> MLXArray {
    var x = input
    for layer in layers {
        if let bn = layer as? BatchNorm {
            x = bn(x)
        } else if let bn = layer as? BatchNormNoAffine {
            x = bn(x)
        } else if let relu = layer as? ReLUModule {
            x = relu(x)
        } else {
            fatalError("Unsupported nonlinear layer \(type(of: layer))")
        }
    }
    return x
}

public class BasicResBlock: Module {
    @ModuleInfo public var conv1: Conv2d
    @ModuleInfo public var bn1: BatchNorm
    @ModuleInfo public var conv2: Conv2d
    @ModuleInfo public var bn2: BatchNorm
    @ModuleInfo public var shortcut: [Module]

    public init(inPlanes: Int, planes: Int, stride: Int = 1) {
        self._conv1 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: inPlanes,
            outputChannels: planes,
            kernelSize: 3,
            stride: [stride, 1],
            padding: 1,
            bias: false
        ))
        self._bn1 = ModuleInfo(wrappedValue: BatchNorm(featureCount: planes))
        self._conv2 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: planes,
            outputChannels: planes,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: false
        ))
        self._bn2 = ModuleInfo(wrappedValue: BatchNorm(featureCount: planes))

        if stride != 1 || inPlanes != planes {
            self._shortcut = ModuleInfo(wrappedValue: [
                Conv2d(
                    inputChannels: inPlanes,
                    outputChannels: planes,
                    kernelSize: 1,
                    stride: [stride, 1],
                    bias: false
                ),
                BatchNorm(featureCount: planes),
            ])
        } else {
            self._shortcut = ModuleInfo(wrappedValue: [])
        }

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = MLXNN.relu(bn1(conv1(x)))
        out = bn2(conv2(out))

        var shortcutValue = x
        for layer in shortcut {
            if let conv = layer as? Conv2d {
                shortcutValue = conv(shortcutValue)
            } else if let bn = layer as? BatchNorm {
                shortcutValue = bn(shortcutValue)
            } else {
                fatalError("Unsupported shortcut layer \(type(of: layer))")
            }
        }

        return MLXNN.relu(out + shortcutValue)
    }
}

public class FCM: Module {
    @ModuleInfo public var conv1: Conv2d
    @ModuleInfo public var bn1: BatchNorm
    @ModuleInfo public var layer1: [BasicResBlock]
    @ModuleInfo public var layer2: [BasicResBlock]
    @ModuleInfo public var conv2: Conv2d
    @ModuleInfo public var bn2: BatchNorm

    public let out_channels: Int

    public init(mChannels: Int = 32, featDim: Int = 80) {
        self._conv1 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: 1,
            outputChannels: mChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: false
        ))
        self._bn1 = ModuleInfo(wrappedValue: BatchNorm(featureCount: mChannels))

        var inPlanes = mChannels
        let makeLayer: (Int, Int) -> [BasicResBlock] = { planes, stride in
            let strides = [stride, 1]
            var blocks: [BasicResBlock] = []
            for s in strides {
                blocks.append(BasicResBlock(inPlanes: inPlanes, planes: planes, stride: s))
                inPlanes = planes
            }
            return blocks
        }

        let layer1Blocks = makeLayer(mChannels, 2)
        self._layer1 = ModuleInfo(wrappedValue: layer1Blocks)
        let layer2Blocks = makeLayer(mChannels, 2)
        self._layer2 = ModuleInfo(wrappedValue: layer2Blocks)

        self._conv2 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: mChannels,
            outputChannels: mChannels,
            kernelSize: 3,
            stride: [2, 1],
            padding: 1,
            bias: false
        ))
        self._bn2 = ModuleInfo(wrappedValue: BatchNorm(featureCount: mChannels))
        self.out_channels = mChannels * (featDim / 8)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.expandedDimensions(axis: -1)
        out = MLXNN.relu(bn1(conv1(out)))

        for layer in layer1 { out = layer(out) }
        for layer in layer2 { out = layer(out) }

        out = MLXNN.relu(bn2(conv2(out)))

        let b = out.shape[0]
        let h = out.shape[1]
        let w = out.shape[2]
        let c = out.shape[3]
        out = out.transposed(0, 3, 1, 2)
        out = out.reshaped([b, c * h, w])
        return out
    }
}

private func makeNonlinear(_ config: String, channels: Int) -> [Module] {
    var layers: [Module] = []
    for name in config.split(separator: "-") {
        switch name {
        case "relu":
            layers.append(ReLUModule())
        case "batchnorm":
            layers.append(BatchNorm(featureCount: channels))
        case "batchnorm_":
            layers.append(BatchNormNoAffine(featureCount: channels))
        default:
            fatalError("Unexpected nonlinear component \(name)")
        }
    }
    return layers
}

private func statisticsPooling(_ x: MLXArray, axis: Int = -1, keepDim: Bool = false) -> MLXArray {
    let mean = MLX.mean(x, axis: axis, keepDims: keepDim)
    let variance = MLX.variance(x, axis: axis, keepDims: keepDim)
    let std = MLX.sqrt(variance + 1e-5)
    if keepDim {
        return MLX.concatenated([mean, std], axis: axis)
    }
    return MLX.concatenated([mean, std], axis: -1)
}

private func conv1dPytorchFormat(_ x: MLXArray, _ conv: Conv1d) -> MLXArray {
    var out = x.swappedAxes(1, 2)
    out = conv(out)
    return out.swappedAxes(1, 2)
}

public class StatsPool: Module {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        statisticsPooling(x)
    }
}

public class TDNNLayer: Module {
    @ModuleInfo public var linear: Conv1d
    @ModuleInfo public var nonlinear: [Module]

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        bias: Bool = false,
        config: String = "batchnorm-relu"
    ) {
        var pad = padding
        if pad < 0 {
            precondition(kernelSize % 2 == 1, "Expected odd kernel size")
            pad = (kernelSize - 1) / 2 * dilation
        }
        self._linear = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: pad,
            dilation: dilation,
            bias: bias
        ))
        self._nonlinear = ModuleInfo(wrappedValue: makeNonlinear(config, channels: outChannels))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.swappedAxes(1, 2)
        out = linear(out)
        out = applyNonlinear(nonlinear, to: out)
        return out.swappedAxes(1, 2)
    }
}

public class CAMLayer: Module {
    @ModuleInfo public var linear_local: Conv1d
    @ModuleInfo public var linear1: Conv1d
    @ModuleInfo public var linear2: Conv1d

    let sineAmp: Float = 0.0

    public init(
        bnChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int,
        dilation: Int,
        bias: Bool,
        reduction: Int = 2
    ) {
        self._linear_local = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: bnChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        ))
        self._linear1 = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: bnChannels,
            outputChannels: bnChannels / reduction,
            kernelSize: 1
        ))
        self._linear2 = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: bnChannels / reduction,
            outputChannels: outChannels,
            kernelSize: 1
        ))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv1dPytorchFormat(x, linear_local)
        let context = MLX.mean(x, axis: -1, keepDims: true) + segPooling(x)
        var attn = conv1dPytorchFormat(context, linear1)
        attn = MLXNN.relu(attn)
        attn = conv1dPytorchFormat(attn, linear2)
        let m = MLX.sigmoid(attn)
        return y * m
    }

    private func segPooling(_ x: MLXArray, segLen: Int = 100, type: String = "avg") -> MLXArray {
        let b = x.shape[0]
        let c = x.shape[1]
        let t = x.shape[2]
        let nSegs = (t + segLen - 1) / segLen
        let padLen = nSegs * segLen - t

        var padded = x
        if padLen > 0 {
            let pad = MLXArray.zeros([b, c, padLen], dtype: x.dtype)
            padded = MLX.concatenated([x, pad], axis: -1)
        }
        let reshaped = padded.reshaped([b, c, nSegs, segLen])
        let seg: MLXArray
        if type == "avg" {
            seg = MLX.mean(reshaped, axis: -1)
        } else if type == "max" {
            seg = MLX.max(reshaped, axis: -1)
        } else {
            fatalError("Invalid segment pooling type")
        }
        let expanded = seg.expandedDimensions(axis: -1)
        let broadcasted = broadcast(expanded, to: [b, c, nSegs, segLen])
        var out = broadcasted.reshaped([b, c, nSegs * segLen])
        if out.shape[2] > t {
            out = out[0..., 0..., 0..<t]
        }
        return out
    }
}

public class CAMDenseTDNNLayer: Module {
    @ModuleInfo public var nonlinear1: [Module]
    @ModuleInfo public var linear1: Conv1d
    @ModuleInfo public var nonlinear2: [Module]
    @ModuleInfo public var cam_layer: CAMLayer

    public init(
        inChannels: Int,
        outChannels: Int,
        bnChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        bias: Bool = false,
        config: String = "batchnorm-relu"
    ) {
        precondition(kernelSize % 2 == 1, "Expected odd kernel size")
        let padding = (kernelSize - 1) / 2 * dilation
        self._nonlinear1 = ModuleInfo(wrappedValue: makeNonlinear(config, channels: inChannels))
        self._linear1 = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: inChannels,
            outputChannels: bnChannels,
            kernelSize: 1,
            bias: false
        ))
        self._nonlinear2 = ModuleInfo(wrappedValue: makeNonlinear(config, channels: bnChannels))
        self._cam_layer = ModuleInfo(wrappedValue: CAMLayer(
            bnChannels: bnChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        ))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.swappedAxes(1, 2)
        out = applyNonlinear(nonlinear1, to: out)
        out = linear1(out)
        out = applyNonlinear(nonlinear2, to: out)
        out = out.swappedAxes(1, 2)
        return cam_layer(out)
    }
}

public class CAMDenseTDNNBlock: Module {
    @ModuleInfo public var layers: [CAMDenseTDNNLayer]

    public init(
        numLayers: Int,
        inChannels: Int,
        outChannels: Int,
        bnChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        bias: Bool = false,
        config: String = "batchnorm-relu"
    ) {
        self._layers = ModuleInfo(wrappedValue: (0..<numLayers).map { idx in
            CAMDenseTDNNLayer(
                inChannels: inChannels + idx * outChannels,
                outChannels: outChannels,
                bnChannels: bnChannels,
                kernelSize: kernelSize,
                stride: stride,
                dilation: dilation,
                bias: bias,
                config: config
            )
        })
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers {
            out = MLX.concatenated([out, layer(out)], axis: 1)
        }
        return out
    }
}

public class TransitLayer: Module {
    @ModuleInfo public var nonlinear: [Module]
    @ModuleInfo public var linear: Conv1d

    public init(
        inChannels: Int,
        outChannels: Int,
        bias: Bool = true,
        config: String = "batchnorm-relu"
    ) {
        self._nonlinear = ModuleInfo(wrappedValue: makeNonlinear(config, channels: inChannels))
        self._linear = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            bias: bias
        ))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.swappedAxes(1, 2)
        out = applyNonlinear(nonlinear, to: out)
        out = linear(out)
        return out.swappedAxes(1, 2)
    }
}

public class DenseLayer: Module {
    @ModuleInfo public var linear: Conv1d
    @ModuleInfo public var nonlinear: [Module]

    public init(
        inChannels: Int,
        outChannels: Int,
        bias: Bool = false,
        config: String = "batchnorm-relu"
    ) {
        self._linear = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            bias: bias
        ))
        self._nonlinear = ModuleInfo(wrappedValue: makeNonlinear(config, channels: outChannels))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.ndim == 2 {
            var out = x.expandedDimensions(axis: 1)
            out = linear(out)
            out = applyNonlinear(nonlinear, to: out)
            return out.squeezed(axis: 1)
        }
        var out = x.swappedAxes(1, 2)
        out = linear(out)
        out = applyNonlinear(nonlinear, to: out)
        return out.swappedAxes(1, 2)
    }
}

public class CAMPPlus: Module {
    @ModuleInfo public var head: FCM
    @ModuleInfo public var tdnn: TDNNLayer
    @ModuleInfo public var blocks: [CAMDenseTDNNBlock]
    @ModuleInfo public var transits: [TransitLayer]
    @ModuleInfo public var out_nonlinear: [Module]
    @ModuleInfo public var stats: StatsPool?
    @ModuleInfo public var dense: DenseLayer?

    public let output_level: String

    public init(
        featDim: Int = 80,
        embeddingSize: Int = 192,
        growthRate: Int = 32,
        bnSize: Int = 4,
        initChannels: Int = 128,
        config: String = "batchnorm-relu",
        outputLevel: String = "segment"
    ) {
        let headModule = FCM(mChannels: 32, featDim: featDim)
        self._head = ModuleInfo(wrappedValue: headModule)
        var channels = headModule.out_channels
        self.output_level = outputLevel

        self._tdnn = ModuleInfo(wrappedValue: TDNNLayer(
            inChannels: channels,
            outChannels: initChannels,
            kernelSize: 5,
            stride: 2,
            padding: -1,
            dilation: 1,
            config: config
        ))
        channels = initChannels

        let blockSettings = [(12, 3, 1), (24, 3, 2), (16, 3, 2)]
        var blockList: [CAMDenseTDNNBlock] = []
        var transitList: [TransitLayer] = []
        for (numLayers, kernelSize, dilation) in blockSettings {
            let block = CAMDenseTDNNBlock(
                numLayers: numLayers,
                inChannels: channels,
                outChannels: growthRate,
                bnChannels: bnSize * growthRate,
                kernelSize: kernelSize,
                dilation: dilation,
                config: config
            )
            blockList.append(block)
            channels = channels + numLayers * growthRate

            let transit = TransitLayer(
                inChannels: channels,
                outChannels: channels / 2,
                bias: false,
                config: config
            )
            transitList.append(transit)
            channels = channels / 2
        }
        self._blocks = ModuleInfo(wrappedValue: blockList)
        self._transits = ModuleInfo(wrappedValue: transitList)
        self._out_nonlinear = ModuleInfo(wrappedValue: makeNonlinear(config, channels: channels))

        if outputLevel == "segment" {
            self._stats = ModuleInfo(wrappedValue: StatsPool())
            self._dense = ModuleInfo(wrappedValue: DenseLayer(
                inChannels: channels * 2,
                outChannels: embeddingSize,
                config: "batchnorm_"
            ))
        } else {
            self._stats = ModuleInfo(wrappedValue: nil)
            self._dense = ModuleInfo(wrappedValue: nil)
        }

        super.init()
    }

    public func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]
        let currWeights = Dictionary(uniqueKeysWithValues: parameters().flattened())

        for (key, value) in weights {
            if key.contains("num_batches_tracked") { continue }
            var newKey = key

            newKey = replaceRegex(newKey, pattern: "xvector\\.block(\\d+)\\.", replacement: { m in
                let idx = (Int(m[1]) ?? 1) - 1
                return "blocks.\(idx)."
            })
            newKey = replaceRegex(newKey, pattern: "xvector\\.transit(\\d+)\\.", replacement: { m in
                let idx = (Int(m[1]) ?? 1) - 1
                return "transits.\(idx)."
            })
            newKey = newKey.replacingOccurrences(of: "xvector.tdnn.", with: "tdnn.")
            newKey = newKey.replacingOccurrences(of: "xvector.dense.", with: "dense.")
            newKey = newKey.replacingOccurrences(of: "xvector.out_nonlinear.", with: "out_nonlinear.")

            newKey = replaceRegex(newKey, pattern: "\\.tdnnd(\\d+)\\.", replacement: { m in
                let idx = (Int(m[1]) ?? 1) - 1
                return ".layers.\(idx)."
            })

            newKey = replaceRegex(newKey, pattern: "\\.nonlinear(\\d+)\\.batchnorm\\.", replacement: { m in
                return ".nonlinear\(m[1]).0."
            })
            newKey = newKey.replacingOccurrences(of: ".nonlinear.batchnorm.", with: ".nonlinear.0.")
            newKey = newKey.replacingOccurrences(of: ".out_nonlinear.batchnorm.", with: ".out_nonlinear.0.")
            if newKey.hasPrefix("out_nonlinear.batchnorm.") {
                newKey = newKey.replacingOccurrences(of: "out_nonlinear.batchnorm.", with: "out_nonlinear.0.")
            }

            var newValue = value
            if newValue.ndim == 4, newKey.contains("weight") {
                if let expected = currWeights[newKey], expected.shape != newValue.shape {
                    newValue = newValue.transposed(0, 2, 3, 1)
                }
            } else if newValue.ndim == 3, newKey.contains("weight") {
                if let expected = currWeights[newKey], expected.shape != newValue.shape {
                    newValue = newValue.swappedAxes(1, 2)
                }
            }
            newWeights[newKey] = newValue
        }
        return newWeights
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.swappedAxes(1, 2)
        out = head(out)
        out = tdnn(out)

        for (block, transit) in zip(blocks, transits) {
            out = block(out)
            out = transit(out)
        }

        out = out.swappedAxes(1, 2)
        out = applyNonlinear(out_nonlinear, to: out)
        out = out.swappedAxes(1, 2)

        if output_level == "segment" {
            if let stats = stats, let dense = dense {
                out = stats(out)
                out = dense(out)
                if out.ndim == 3 && out.shape.last == 1 {
                    out = out.squeezed(axis: -1)
                }
            }
        }
        return out
    }

    public func inference(_ audio: MLXArray) -> MLXArray {
        var audioBatch = audio
        if audioBatch.ndim == 1 {
            audioBatch = audioBatch.expandedDimensions(axis: 0)
        }

        var features: [MLXArray] = []
        for i in 0..<audioBatch.shape[0] {
            var fbank = KaldiFbankCompat.compute(audioBatch[i])
            fbank = fbank - MLX.mean(fbank, axis: 0, keepDims: true)
            features.append(fbank)
        }

        let maxLen = features.map { $0.shape[0] }.max() ?? 0
        var padded: [MLXArray] = []
        padded.reserveCapacity(features.count)
        for f in features {
            if f.shape[0] < maxLen {
                let pad = MLXArray.zeros([maxLen - f.shape[0], f.shape[1]], dtype: f.dtype)
                padded.append(MLX.concatenated([f, pad], axis: 0))
            } else {
                padded.append(f)
            }
        }

        let batchFeatures = stacked(padded, axis: 0)
        return callAsFunction(batchFeatures)
    }

}

private func replaceRegex(_ input: String, pattern: String, replacement: ( [String] ) -> String) -> String {
    guard let regex = try? NSRegularExpression(pattern: pattern) else {
        return input
    }
    let matches = regex.matches(in: input, range: NSRange(input.startIndex..., in: input))
    guard let match = matches.first else {
        return input
    }
    var groups: [String] = []
    for idx in 0..<match.numberOfRanges {
        let range = match.range(at: idx)
        if let swiftRange = Range(range, in: input) {
            groups.append(String(input[swiftRange]))
        } else {
            groups.append("")
        }
    }
    let replacementString = replacement(groups)
    return regex.stringByReplacingMatches(in: input, range: NSRange(input.startIndex..., in: input), withTemplate: replacementString)
}
