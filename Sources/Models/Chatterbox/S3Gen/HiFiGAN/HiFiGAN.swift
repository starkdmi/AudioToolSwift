// Copyright © 2025
// HiFiGAN - Complete HiFi-GAN vocoder for S3Gen
// Based on mlx-audio-swift Kokoro implementation
// Pure MLX Swift port matching Python MLX hifigan.py

import Foundation
import MLX
import MLXNN
import MLXFast
import MLXRandom

// MARK: - Interpolate

/// 1D interpolation (linear or nearest neighbor)
func interpolate(
    input: MLXArray,
    size: [Int]? = nil,
    scaleFactor: [Float]? = nil,
    mode: String = "nearest",
    alignCorners: Bool? = nil
) -> MLXArray {
    let ndim = input.ndim
    if ndim < 3 {
        fatalError("Expected at least 3D input (N, C, D1), got \(ndim)D")
    }
    
    let spatialDims = ndim - 2
    
    if size != nil && scaleFactor != nil {
        fatalError("Only one of size or scaleFactor should be defined")
    } else if size == nil && scaleFactor == nil {
        fatalError("One of size or scaleFactor must be defined")
    }
    
    var outputSize: [Int] = []
    if let scaleFactor = scaleFactor {
        let factors = scaleFactor.count == 1 ? Array(repeating: scaleFactor[0], count: spatialDims) : scaleFactor
        for i in 0..<spatialDims {
            let currSize = max(1, Int(ceil(Float(input.shape[i + 2]) * factors[i])))
            outputSize.append(currSize)
        }
    } else if let size = size {
        outputSize = size.count == 1 ? Array(repeating: size[0], count: spatialDims) : size
    }
    
    if spatialDims == 1 {
        return interpolate1d(input: input, size: outputSize[0], mode: mode, alignCorners: alignCorners)
    } else {
        fatalError("Only 1D interpolation currently supported, got \(spatialDims)D")
    }
}

func interpolate1d(
    input: MLXArray,
    size: Int,
    mode: String = "linear",
    alignCorners: Bool? = nil
) -> MLXArray {
    let shape = input.shape
    let batchSize = shape[0]
    let channels = shape[1]
    let inWidth = shape[2]
    
    let outputSize = max(1, size)
    let inputWidth = max(1, inWidth)
    
    if mode == "nearest" {
        if outputSize == 1 {
            let indices = MLXArray(converting: [0]).asType(.int32)
            return input[0..., 0..., indices]
        } else {
            let scale = Float(inputWidth) / Float(outputSize)
            let indices = floor(MLXArray(0..<outputSize).asType(.float32) * scale).asType(.int32)
            let clippedIndices = clip(indices, min: 0, max: inputWidth - 1)
            return input[0..., 0..., clippedIndices]
        }
    }
    
    // Linear interpolation
    var x: MLXArray
    if alignCorners == true && outputSize > 1 {
        x = MLXArray(0..<outputSize).asType(.float32) * (Float(inputWidth - 1) / Float(outputSize - 1))
    } else {
        if outputSize == 1 {
            x = MLXArray(converting: [0.0]).asType(.float32)
        } else {
            x = MLXArray(0..<outputSize).asType(.float32) * (Float(inputWidth) / Float(outputSize))
            if alignCorners != true {
                x = x + 0.5 * (Float(inputWidth) / Float(outputSize)) - 0.5
            }
        }
    }
    
    if inputWidth == 1 {
        let outputShape = [batchSize, channels, outputSize]
        return broadcast(input, to: outputShape)
    }
    
    let xLow = floor(x).asType(.int32)
    let xHigh = minimum(xLow + 1, MLXArray(inputWidth - 1))
    let xFrac = x - xLow.asType(.float32)
    
    let yLow = input[0..., 0..., xLow]
    let yHigh = input[0..., 0..., xHigh]
    
    let oneMinusXFrac = 1 - xFrac
    let output = yLow * oneMinusXFrac.expandedDimensions(axis: 0).expandedDimensions(axis: 0) +
        yHigh * xFrac.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    
    return output
}

// MARK: - Hanning Window

func hanningWindow(length: Int) -> MLXArray {
    if length == 1 {
        return MLXArray(1.0)
    }
    let n = MLXArray(Array(stride(from: Float(1 - length), to: Float(length), by: 2.0)))
    let factor = Float.pi / Float(length - 1)
    return 0.5 + 0.5 * cos(n * factor)
}

/// Periodic Hann window for STFT
func hannWindowPeriodic(size: Int) -> MLXArray {
    var result: [Float] = []
    for n in 0..<size {
        result.append(0.5 * (1 - cos(2 * Float.pi * Float(n) / Float(size))))
    }
    return MLXArray(result)
}

// MARK: - Unwrap

func unwrap(p: MLXArray) -> MLXArray {
    let period: Float = 2.0 * .pi
    let discont: Float = period / 2.0
    
    let pDiff1 = p[0..., 0..<p.shape[1] - 1]
    let pDiff2 = p[0..., 1..<p.shape[1]]
    
    let pDiff = pDiff2 - pDiff1
    
    let intervalHigh: Float = period / 2.0
    let intervalLow: Float = -intervalHigh
    
    var pDiffMod = pDiff - intervalLow
    pDiffMod = (((pDiffMod % period) + period) % period) + intervalLow
    
    let ddSignArray = MLX.where(pDiff .> 0, intervalHigh, pDiffMod)
    pDiffMod = MLX.where(pDiffMod .== intervalLow, ddSignArray, pDiffMod)
    
    var phCorrect = pDiffMod - pDiff
    phCorrect = MLX.where(abs(pDiff) .< discont, MLXArray(0.0), phCorrect)
    
    return concatenated([p[0..., 0..<1], p[0..., 1...] + phCorrect.cumsum(axis: 1)], axis: 1)
}

// MARK: - STFT Functions

func getWindow(window: Any, winLen: Int, nFft: Int) -> MLXArray {
    var w: MLXArray
    if let windowStr = window as? String {
        if windowStr.lowercased() == "hann" {
            w = hanningWindow(length: winLen + 1)[0..<winLen]
        } else {
            fatalError("Only hanning is supported for window, not \(windowStr)")
        }
    } else if let windowArray = window as? MLXArray {
        w = windowArray
    } else {
        fatalError("Window must be a string or MLXArray")
    }
    
    if w.shape[0] < nFft {
        let padSize = nFft - w.shape[0]
        w = concatenated([w, zeros([padSize])], axis: 0)
    }
    return w
}

func mlxStft(
    x: MLXArray,
    nFft: Int = 800,
    hopLength: Int? = nil,
    winLength: Int? = nil,
    window: Any = "hann",
    center: Bool = true,
    padMode: String = "reflect"
) -> MLXArray {
    let hopLen = hopLength ?? nFft / 4
    let winLen = winLength ?? nFft
    
    let w = getWindow(window: window, winLen: winLen, nFft: nFft)
    
    func pad(_ x: MLXArray, padding: Int, padMode: String = "reflect") -> MLXArray {
        if padMode == "constant" {
            return padded(x, width: [padding, padding])
        } else if padMode == "reflect" {
            let prefix = x[1..<padding + 1][.stride(by: -1)]
            let suffix = x[-(padding + 1)..<(-1)][.stride(by: -1)]
            return concatenated([prefix, x, suffix])
        } else {
            fatalError("Invalid pad mode \(padMode)")
        }
    }
    
    var xArray = x
    
    if center {
        xArray = pad(xArray, padding: nFft / 2, padMode: padMode)
    }
    
    let numFrames = 1 + (xArray.shape[0] - nFft) / hopLen
    if numFrames <= 0 {
        fatalError("Input is too short")
    }
    
    // Create frames manually (MLX doesn't have asStrided)
    var framesList: [MLXArray] = []
    for i in 0..<numFrames {
        let startIdx = i * hopLen
        framesList.append(xArray[startIdx..<(startIdx + nFft)])
    }
    let frames = stacked(framesList, axis: 0)
    
    let spec = MLXFFT.rfft(frames * w)
    return spec.transposed(1, 0)
}

func mlxIstft(
    x: MLXArray,
    hopLength: Int? = nil,
    winLength: Int? = nil,
    window: Any = "hann"
) -> MLXArray {
    let winLen = winLength ?? ((x.shape[1] - 1) * 2)
    let hopLen = hopLength ?? (winLen / 4)
    
    let w = getWindow(window: window, winLen: winLen, nFft: winLen)
    
    let xTransposed = x.transposed(1, 0)
    let t = (xTransposed.shape[0] - 1) * hopLen + winLen
    let windowModLen = 20 / 5
    
    let wSquared = w * w
    
    let totalWsquared = concatenated(Array(repeating: wSquared, count: t / winLen))
    
    
    let output = MLXFFT.irfft(xTransposed, axis: 1) * w
    
    var outputs: [MLXArray] = []
    var windowSums: [MLXArray] = []
    
    for i in 0..<windowModLen {
        let strideStart = i
        var strideElements: [MLXArray] = []
        var idx = strideStart
        while idx < output.shape[0] {
            strideElements.append(output[idx])
            idx += windowModLen
        }
        let outputStride = strideElements.isEmpty ? zeros([1]).reshaped([-1]) : concatenated(strideElements, axis: 0)
        let windowSumArray = totalWsquared[0..<min(outputStride.shape[0], totalWsquared.shape[0])]
        
        outputs.append(concatenated([
            zeros([i * hopLen]),
            outputStride,
            zeros([max(0, t - i * hopLen - outputStride.shape[0])]),
        ]))
        
        windowSums.append(concatenated([
            zeros([i * hopLen]),
            windowSumArray,
            zeros([max(0, t - i * hopLen - windowSumArray.shape[0])]),
        ]))
    }
    
    var reconstructed = outputs[0]
    var windowSum = windowSums[0]
    for i in 1..<windowModLen {
        reconstructed = reconstructed + outputs[i]
        windowSum = windowSum + windowSums[i]
    }
    
    reconstructed =
        reconstructed[winLen / 2..<(reconstructed.shape[0] - winLen / 2)] /
        windowSum[winLen / 2..<(reconstructed.shape[0] - winLen / 2)]
    
    return reconstructed
}

// MARK: - MLXSTFT Class

public class MLXSTFT {
    let filterLength: Int
    let hopLength: Int
    let winLength: Int
    let window: String
    
    public init(filterLength: Int = 800, hopLength: Int = 200, winLength: Int = 800, window: String = "hann") {
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.winLength = winLength
        self.window = window
    }
    
    public func transform(inputData: MLXArray) -> (MLXArray, MLXArray) {
        var audioArray = inputData
        if audioArray.ndim == 1 {
            audioArray = audioArray.expandedDimensions(axis: 0)
        }
        
        var magnitudes: [MLXArray] = []
        var phases: [MLXArray] = []
        
        for batchIdx in 0..<audioArray.shape[0] {
            let stft = mlxStft(
                x: audioArray[batchIdx],
                nFft: filterLength,
                hopLength: hopLength,
                winLength: winLength,
                window: window,
                center: true,
                padMode: "reflect"
            )
            magnitudes.append(abs(stft))
            phases.append(atan2(stft.imaginaryPart(), stft.realPart()))
        }
        
        let magnitudesStacked = stacked(magnitudes, axis: 0)
        let phasesStacked = stacked(phases, axis: 0)
        
        return (magnitudesStacked, phasesStacked)
    }
    
    public func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        var reconstructed: [MLXArray] = []
        
        for batchIdx in 0..<magnitude.shape[0] {
            let phaseCont = unwrap(p: phase[batchIdx])
            let stft = magnitude[batchIdx] * exp(MLXArray(real: 0, imaginary: 1) * phaseCont)
            
            let audio = mlxIstft(
                x: stft,
                hopLength: hopLength,
                winLength: winLength,
                window: window
            )
            reconstructed.append(audio)
        }
        
        let reconstructedStacked = stacked(reconstructed, axis: 0)
        return reconstructedStacked.expandedDimensions(axis: 1)
    }
}

// MARK: - InstanceNorm1d

public class InstanceNorm1d {
    let numFeatures: Int
    let eps: Float
    let affine: Bool
    
    var weight: MLXArray?
    var bias: MLXArray?
    
    public init(numFeatures: Int, eps: Float = 1e-5, affine: Bool = false) {
        self.numFeatures = numFeatures
        self.eps = eps
        self.affine = affine
        
        if affine {
            weight = ones([numFeatures])
            bias = zeros([numFeatures])
        }
    }
    
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // input: (B, C, T) or (C, T)
        let mean = MLX.mean(input, axis: -1, keepDims: true)
        let variance = MLX.variance(input, axis: -1, keepDims: true)
        
        let xNorm = (input - mean) / sqrt(variance + eps)
        
        if affine, let weight = weight, let bias = bias {
            return xNorm * weight.expandedDimensions(axis: -1) + bias.expandedDimensions(axis: -1)
        }
        return xNorm
    }
}

// MARK: - Weight Norm Utilities

func computeNorm(x: MLXArray, p: Int, dim: [Int]? = nil, keepdim: Bool = false) -> MLXArray {
    guard p == 1 || p == 2 else {
        fatalError("Only p-norms with p of 1 or 2 are supported")
    }
    
    let dimensions = dim ?? Array(0..<x.ndim)
    
    if p == 1 {
        return sum(abs(x), axes: dimensions, keepDims: keepdim)
    } else {
        return sqrt(sum(x * x, axes: dimensions, keepDims: keepdim))
    }
}

func weightNorm(weightV: MLXArray, weightG: MLXArray, dim: Int? = nil) -> MLXArray {
    let rank = weightV.shape.count
    
    var axes: [Int]
    if let dim = dim {
        var adjustedDim = dim
        if dim < 0 {
            adjustedDim += rank
        }
        axes = Array(0..<rank)
        if adjustedDim != -1 {
            axes.removeAll(where: { $0 == adjustedDim })
        }
    } else {
        axes = Array(0..<rank)
    }
    
    let normV = computeNorm(x: weightV, p: 2, dim: axes, keepdim: true)
    let normalizedWeight = weightV / (normV + 1e-7)
    return normalizedWeight * weightG
}

// MARK: - ConvWeighted (Weight-normalized Conv1d)

public class ConvWeighted: Module {
    var weightG: MLXArray
    var weightV: MLXArray
    var bias: MLXArray?
    
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int
    
    public init(
        weightG: MLXArray,
        weightV: MLXArray,
        bias: MLXArray?,
        stride: Int = 1,
        padding: Int = 1,
        dilation: Int = 1,
        groups: Int = 1
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weightG = weightG
        self.weightV = weightV
        self.bias = bias
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, conv: (MLXArray, MLXArray, Int, Int, Int, Int, StreamOrDevice) -> MLXArray) -> MLXArray {
        let weight = weightNorm(weightV: weightV, weightG: weightG, dim: 0)
        var b = bias?.reshaped([1, 1, -1])
        
        func applyConv(x: MLXArray, weightToUse: MLXArray) -> MLXArray {
            let result = conv(x, weightToUse, stride, padding, dilation, groups, .default)
            if let b = b {
                return result + b
            }
            return result
        }
        
        if x.shape.last == weight.shape.last || groups > 1 {
            return applyConv(x: x, weightToUse: weight)
        } else {
            return applyConv(x: x, weightToUse: weight.transposed())
        }
    }
}

// MARK: - SineGen

public class SineGen {
    private let sineAmp: Float
    private let noiseStd: Float
    private let harmonicNum: Int
    private let dim: Int
    private let samplingRate: Int
    private let voicedThreshold: Float
    private let useInterpolation: Bool
    private let upsampleScale: Int
    
    public init(
        sampRate: Int,
        harmonicNum: Int = 0,
        sineAmp: Float = 0.1,
        noiseStd: Float = 0.003,
        voicedThreshold: Float = 0,
        useInterpolation: Bool = false,
        upsampleScale: Int = 1
    ) {
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.harmonicNum = harmonicNum
        self.dim = harmonicNum + 1
        self.samplingRate = sampRate
        self.voicedThreshold = voicedThreshold
        self.useInterpolation = useInterpolation
        self.upsampleScale = upsampleScale
    }
    
    private func _f02uv(_ f0: MLXArray) -> MLXArray {
        return (f0 .> voicedThreshold).asType(.float32)
    }
    
    private func linearInterpolate1dToSize(_ x: MLXArray, newSize: Int) -> MLXArray {
        let t = x.shape[x.ndim - 1]
        if newSize == t {
            return x
        }
        let positions = MLX.linspace(0, Float(t - 1), count: newSize)
        let idxLow = floor(positions).asType(.int32)
        let idxHigh = minimum(idxLow + 1, MLXArray(t - 1))
        let weights = positions - idxLow.asType(.float32)
        let lowVals = MLX.take(x, idxLow, axis: x.ndim - 1)
        let highVals = MLX.take(x, idxHigh, axis: x.ndim - 1)
        return lowVals + weights * (highVals - lowVals)
    }

    private func f02SineInterpolation(_ f0Values: MLXArray) -> MLXArray {
        let b = f0Values.shape[0]
        let t = f0Values.shape[1]
        let h = f0Values.shape[2]

        var radValues = (f0Values / Float(samplingRate)) % 1
        var randIni = MLXRandom.uniform(low: 0.0, high: 1.0, [b, h])
        randIni = concatenated([zeros([b, 1]), randIni[0..., 1...]], axis: 1)
        radValues[0..., 0, 0...] = radValues[0..., 0, 0...] + randIni

        let radValuesT = radValues.swappedAxes(1, 2)
        let tDown = max(1, t / max(1, upsampleScale))
        var radDown = linearInterpolate1dToSize(radValuesT, newSize: tDown)
        radDown = radDown.swappedAxes(1, 2)

        var phase = cumsum(radDown, axis: 1) * 2 * Float.pi
        var phaseT = phase.swappedAxes(1, 2) * Float(upsampleScale)
        phaseT = linearInterpolate1dToSize(phaseT, newSize: t)
        let phaseUp = phaseT.swappedAxes(1, 2)
        return sin(phaseUp)
    }
    
    public func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let b = f0.shape[0]
        let h = harmonicNum + 1

        let harmonic = MLXArray(1...h).asType(.float32).reshaped([1, h, 1])
        let fMat = f0 * harmonic / Float(samplingRate)

        let sineWaves: MLXArray
        if useInterpolation {
            let fn = f0.swappedAxes(1, 2) * MLXArray(1...h).asType(.float32)
            var waves = f02SineInterpolation(fn) * sineAmp
            waves = waves.swappedAxes(1, 2)
            sineWaves = waves
        } else {
            var theta = cumsum(fMat, axis: -1) % 1
            theta = theta * 2 * Float.pi
            let phase = MLXRandom.uniform(low: -Float.pi, high: Float.pi, [b, h, 1])
            let mask = MLXArray(0..<h).reshaped([1, h, 1]) .> 0
            let phaseVec = MLX.where(mask, phase, MLXArray(0.0))
            sineWaves = sineAmp * sin(theta + phaseVec)
        }

        let uv = _f02uv(f0)
        let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
        let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)
        let result = sineWaves * uv + noise
        return (result, uv, noise)
    }
}

// MARK: - SourceModuleHnNSF

public class SourceModuleHnNSF: Module {
    private let sineAmp: Float
    private let noiseStd: Float
    private let lSinGen: SineGen
    @ModuleInfo(key: "l_linear") public var l_linear: Linear
    
    public init(
        samplingRate: Int,
        upsampleScale: Int,
        harmonicNum: Int = 0,
        sineAmp: Float = 0.1,
        addNoiseStd: Float = 0.003,
        voicedThreshold: Float = 0,
        useInterpolation: Bool = false
    ) {
        self.sineAmp = sineAmp
        self.noiseStd = addNoiseStd
        
        lSinGen = SineGen(
            sampRate: samplingRate,
            harmonicNum: harmonicNum,
            sineAmp: sineAmp,
            noiseStd: addNoiseStd,
            voicedThreshold: voicedThreshold,
            useInterpolation: useInterpolation,
            upsampleScale: upsampleScale
        )
        self._l_linear = ModuleInfo(wrappedValue: Linear(harmonicNum + 1, 1))
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (sineWavs, uv, _) = lSinGen(x.swappedAxes(1, 2))
        let sineMerge = tanh(l_linear(sineWavs.swappedAxes(1, 2)))
        let uvT = uv.swappedAxes(1, 2)
        
        let noise = MLXRandom.normal(uvT.shape) * (sineAmp / 3)
        
        return (sineMerge, noise, uvT)
    }
}

// MARK: - Snake Activation

public class Snake: Module {
    @ParameterInfo(key: "alpha") public var alpha: MLXArray
    let alphaLogscale: Bool
    
    public init(inFeatures: Int, alpha: Float = 1.0, alphaLogscale: Bool = false) {
        self.alphaLogscale = alphaLogscale
        
        if alphaLogscale {
            self._alpha.wrappedValue = zeros([inFeatures]) * alpha
        } else {
            self._alpha.wrappedValue = ones([inFeatures]) * alpha
        }
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var a = alpha.reshaped([1, -1, 1])
        
        if alphaLogscale {
            a = exp(a)
        }
        
        // Clamp to avoid fp16 overflow
        let minAlpha: Float = 1e-4
        let alphaSign = sign(a)
        let alphaAbs = abs(a)
        var alphaClamped = alphaSign * maximum(alphaAbs, minAlpha)
        alphaClamped = MLX.where(alphaAbs .< 1e-9, minAlpha, alphaClamped)
        
        return x + (1.0 / alphaClamped) * pow(sin(x * a), 2)
    }
}

// MARK: - ResBlock with Snake

public class ResBlock: Module {
    let channels: Int
    @ModuleInfo public var convs1: [Conv1d]
    @ModuleInfo public var convs2: [Conv1d]
    @ModuleInfo public var activations1: [Snake]
    @ModuleInfo public var activations2: [Snake]
    
    public init(channels: Int = 512, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5]) {
        self.channels = channels
        self._convs1 = ModuleInfo(wrappedValue: dilations.map { dilation in
            let padding = (kernelSize * dilation - dilation) / 2
            return Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: padding,
                dilation: dilation
            )
        })
        self._convs2 = ModuleInfo(wrappedValue: dilations.map { _ in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: (kernelSize - 1) / 2
            )
        })
        self._activations1 = ModuleInfo(wrappedValue: dilations.map { _ in
            Snake(inFeatures: channels)
        })
        self._activations2 = ModuleInfo(wrappedValue: dilations.map { _ in
            Snake(inFeatures: channels)
        })
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        
        for i in 0..<convs1.count {
            var xt = activations1[i](result)
            xt = xt.swappedAxes(1, 2)
            xt = convs1[i](xt)
            xt = xt.swappedAxes(1, 2)
            
            xt = activations2[i](xt)
            xt = xt.swappedAxes(1, 2)
            xt = convs2[i](xt)
            xt = xt.swappedAxes(1, 2)
            
            result = xt + result
        }
        return result
    }
}

// MARK: - ReflectionPad1d

public class ReflectionPad1d {
    let padding: (Int, Int)
    
    public init(padding: (Int, Int)) {
        self.padding = padding
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T)
        let (left, right) = padding
        var result = x
        
        if left > 0 {
            let prefix = x[0..., 0..., 1..<left + 1][0..., 0..., .stride(by: -1)]
            result = concatenated([prefix, result], axis: 2)
        }
        if right > 0 {
            let suffix = x[0..., 0..., -(right + 1)..<(-1)][0..., 0..., .stride(by: -1)]
            result = concatenated([result, suffix], axis: 2)
        }
        return result
    }
}

// MARK: - Upsample

public class Upsample {
    let scaleFactor: Float
    
    public init(scaleFactor: Float) {
        self.scaleFactor = scaleFactor
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return interpolate(input: x, scaleFactor: [scaleFactor], mode: "nearest")
    }
}

// MARK: - LeakyReLU Helper

public class LeakyReLU {
    let negativeSlope: Float
    
    public init(negativeSlope: Float = 0.01) {
        self.negativeSlope = negativeSlope
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return leakyRelu(x, negativeSlope: negativeSlope)
    }
}
