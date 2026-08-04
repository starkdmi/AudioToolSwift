// Copyright © 2025
// HiFTGenerator - HiFi-GAN vocoder with Neural Source Filter
// Pure MLX Swift port of python_mlx/chatterbox/s3gen/hifigan.py

import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioUtils

// MARK: - ConvTranspose1d

public class ConvTranspose1d: Module {
    @ParameterInfo public var weight: MLXArray
    @ParameterInfo public var bias: MLXArray
    let stride: Int
    let padding: Int
    let outputPadding: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0
    ) {
        self.stride = stride
        self.padding = padding
        self.outputPadding = outputPadding
        let scale = sqrt(2.0 / Float(inputChannels * kernelSize))
        self._weight.wrappedValue = MLXRandom.normal([outputChannels, kernelSize, inputChannels]) * scale
        self._bias.wrappedValue = MLXArray.zeros([outputChannels])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = convTransposed1d(
            x,
            weight,
            stride: stride,
            padding: padding,
            dilation: 1,
            outputPadding: outputPadding
        )
        out = out + bias.reshaped([1, 1, -1])
        return out
    }
}

// MARK: - HiFTGenerator

public class HiFTGenerator: Module {
    let out_channels: Int = 1
    let nb_harmonics: Int
    let sampling_rate: Int
    let istft_params: [String: Int]
    let lrelu_slope: Float
    let audio_limit: Float

    let num_kernels: Int
    let num_upsamples: Int
    let f0_upsample_scale: Int

    @ModuleInfo(key: "m_source") public var m_source: SourceModuleHnNSF
    @ModuleInfo(key: "conv_pre") public var conv_pre: Conv1d
    @ModuleInfo(key: "conv_post") public var conv_post: Conv1d
    @ModuleInfo public var ups: [ConvTranspose1d]
    @ModuleInfo public var source_downs: [Conv1d]
    @ModuleInfo public var source_resblocks: [ResBlock]
    @ModuleInfo public var resblocks: [ResBlock]
    @ModuleInfo public var f0_predictor: ConvRNNF0Predictor

    @ParameterInfo public var stft_window: MLXArray

    public init(
        inChannels: Int = 80,
        baseChannels: Int = 512,
        nbHarmonics: Int = 8,
        samplingRate: Int = 24000,
        nsfAlpha: Float = 0.1,
        nsfSigma: Float = 0.003,
        nsfVoicedThreshold: Float = 10,
        upsampleRates: [Int] = [8, 5, 3],
        upsampleKernelSizes: [Int] = [16, 11, 7],
        istftParams: [String: Int] = ["n_fft": 16, "hop_len": 4],
        resblockKernelSizes: [Int] = [3, 7, 11],
        resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        sourceResblockKernelSizes: [Int] = [7, 7, 11],
        sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lreluSlope: Float = 0.1,
        audioLimit: Float = 0.99,
        f0Predictor: ConvRNNF0Predictor? = nil,
        useInterpolation: Bool = false
    ) {
        self.nb_harmonics = nbHarmonics
        self.sampling_rate = samplingRate
        self.istft_params = istftParams
        self.lrelu_slope = lreluSlope
        self.audio_limit = audioLimit

        self.num_kernels = resblockKernelSizes.count
        self.num_upsamples = upsampleRates.count

        var upsampleScale = istftParams["hop_len"] ?? 1
        for rate in upsampleRates {
            upsampleScale *= rate
        }
        self.f0_upsample_scale = upsampleScale

        self._m_source = ModuleInfo(wrappedValue: SourceModuleHnNSF(
            samplingRate: samplingRate,
            upsampleScale: upsampleScale,
            harmonicNum: nbHarmonics,
            sineAmp: nsfAlpha,
            addNoiseStd: nsfSigma,
            voicedThreshold: nsfVoicedThreshold,
            useInterpolation: useInterpolation
        ))
        self._f0_predictor = ModuleInfo(wrappedValue: f0Predictor ?? ConvRNNF0Predictor())

        self._conv_pre = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: inChannels,
            outputChannels: baseChannels,
            kernelSize: 7,
            stride: 1,
            padding: 3
        ))

        let postChannels = baseChannels / Int(pow(2.0, Double(upsampleRates.count)))
        self._conv_post = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: postChannels,
            outputChannels: (istftParams["n_fft"] ?? 16) + 2,
            kernelSize: 7,
            stride: 1,
            padding: 3
        ))

        self._ups = ModuleInfo(wrappedValue: upsampleRates.enumerated().map { idx, u in
            let k = upsampleKernelSizes[idx]
            let inCh = baseChannels / Int(pow(2.0, Double(idx)))
            let outCh = baseChannels / Int(pow(2.0, Double(idx + 1)))
            return ConvTranspose1d(
                inputChannels: inCh,
                outputChannels: outCh,
                kernelSize: k,
                stride: u,
                padding: (k - u) / 2
            )
        })

        let downsampleRates = [1] + Array(upsampleRates.reversed().dropLast())
        var downsampleCumRates: [Int] = []
        var cum = 1
        for rate in downsampleRates {
            cum *= rate
            downsampleCumRates.append(cum)
        }
        let downsampleCumRev = Array(downsampleCumRates.reversed())

        let numSourceBlocks = min(downsampleCumRev.count, sourceResblockKernelSizes.count, sourceResblockDilationSizes.count)
        var sourceDowns: [Conv1d] = []
        var sourceResblocks: [ResBlock] = []
        for i in 0..<numSourceBlocks {
            let u = downsampleCumRev[i]
            let outCh = baseChannels / Int(pow(2.0, Double(i + 1)))
            if u == 1 {
                sourceDowns.append(Conv1d(
                    inputChannels: (istftParams["n_fft"] ?? 16) + 2,
                    outputChannels: outCh,
                    kernelSize: 1,
                    stride: 1
                ))
            } else {
                sourceDowns.append(Conv1d(
                    inputChannels: (istftParams["n_fft"] ?? 16) + 2,
                    outputChannels: outCh,
                    kernelSize: u * 2,
                    stride: u,
                    padding: u / 2
                ))
            }
            sourceResblocks.append(ResBlock(
                channels: outCh,
                kernelSize: sourceResblockKernelSizes[i],
                dilations: sourceResblockDilationSizes[i]
            ))
        }
        self._source_downs = ModuleInfo(wrappedValue: sourceDowns)
        self._source_resblocks = ModuleInfo(wrappedValue: sourceResblocks)

        var resblocks: [ResBlock] = []
        for i in 0..<upsampleRates.count {
            let ch = baseChannels / Int(pow(2.0, Double(i + 1)))
            for j in 0..<resblockKernelSizes.count {
                resblocks.append(ResBlock(
                    channels: ch,
                    kernelSize: resblockKernelSizes[j],
                    dilations: resblockDilationSizes[j]
                ))
            }
        }
        self._resblocks = ModuleInfo(wrappedValue: resblocks)

        self._stft_window.wrappedValue = hannWindowPeriodic(size: istftParams["n_fft"] ?? 16)
        super.init()
    }

    private func f0Upsample(_ f0: MLXArray) -> MLXArray {
        repeated(f0, count: f0_upsample_scale, axis: 2)
    }

    private struct HifiganStftResult {
        let real: MLXArray
        let imag: MLXArray
        let padded: MLXArray
        let frames: MLXArray
        let windowed: MLXArray
    }

    private func hifiganStftDetailed(
        _ x: MLXArray,
        nFFT: Int,
        hopLength: Int,
        window: MLXArray
    ) -> HifiganStftResult {
        var x = x
        if x.ndim == 1 {
            x = x.expandedDimensions(axis: 0)
        }

        let batchSize = x.dim(0)
        let signalLen = x.dim(1)
        let padLength = nFFT / 2

        let leftIdx = MLXArray(Array(stride(from: padLength, through: 1, by: -1)))
        let rightStart = signalLen - 2
        let rightEnd = signalLen - padLength - 1
        let rightIdx = MLXArray(Array(stride(from: rightStart, through: rightEnd, by: -1)))

        let leftPad = MLX.take(x, leftIdx, axis: 1)
        let rightPad = MLX.take(x, rightIdx, axis: 1)
        let xPadded = concatenated([leftPad, x, rightPad], axis: 1)

        let paddedLen = xPadded.dim(1)
        let numFrames = (paddedLen - nFFT) / hopLength + 1
        precondition(numFrames > 0, "STFT input is too short")

        let frameStarts = MLXArray(Array(stride(from: 0, to: numFrames, by: 1))) * hopLength
        let sampleOffsets = MLXArray(Array(stride(from: 0, to: nFFT, by: 1)))
        let allIndices =
            frameStarts.expandedDimensions(axis: 1) + sampleOffsets.expandedDimensions(axis: 0)
        let flatIndices = allIndices.flattened().asType(.int32)

        let frames = MLX.take(xPadded, flatIndices, axis: 1)
            .reshaped([batchSize, numFrames, nFFT])
            .swappedAxes(1, 2)

        let windowExpanded = window.reshaped([1, window.dim(0), 1])
        let windowed = frames * windowExpanded

        let (real, imag) = dftRfft(frames: windowed, nFFT: nFFT)
        return HifiganStftResult(
            real: real,
            imag: imag,
            padded: xPadded,
            frames: frames,
            windowed: windowed
        )
    }

    private func dftRfft(frames: MLXArray, nFFT: Int) -> (MLXArray, MLXArray) {
        let nFreq = nFFT / 2 + 1
        let k = MLXArray(stride(from: 0, to: nFreq, by: 1)).asType(.float32)
        let n = MLXArray(stride(from: 0, to: nFFT, by: 1)).asType(.float32)

        let angles = k.expandedDimensions(axis: 1) * n.expandedDimensions(axis: 0)
        let scale = MLXArray(Float(-2.0 * Float.pi / Float(nFFT)))
        let phase = angles * scale

        let cosTable = MLX.cos(phase)
        let sinTable = MLX.sin(phase)
        let cosT = cosTable.transposed(1, 0)
        let sinT = sinTable.transposed(1, 0)

        let framesT = frames.swappedAxes(1, 2)
        let real = matmul(framesT, cosT).swappedAxes(1, 2)
        let imag = matmul(framesT, sinT).swappedAxes(1, 2)
        return (real, imag)
    }

    private func hifiganStft(
        _ x: MLXArray,
        nFFT: Int,
        hopLength: Int,
        window: MLXArray
    ) -> (MLXArray, MLXArray) {
        let result = hifiganStftDetailed(x, nFFT: nFFT, hopLength: hopLength, window: window)
        return (result.real, result.imag)
    }

    private func hifiganIstft(
        magnitude: MLXArray,
        phase: MLXArray,
        nFFT: Int,
        hopLength: Int,
        window: MLXArray
    ) -> MLXArray {
        let clipped = MLX.clip(magnitude, max: 1e2)
        let real = clipped * cos(phase)
        let imag = clipped * sin(phase)

        let batchSize = real.dim(0)
        let freqBins = real.dim(1)
        let numFrames = real.dim(2)

        let mirrorIdx = MLXArray(Array(stride(from: freqBins - 2, through: 1, by: -1)))
        let realMirror = MLX.take(real, mirrorIdx, axis: 1)
        let imagMirror = MLX.take(imag, mirrorIdx, axis: 1)

        let realFull = concatenated([real, realMirror], axis: 1)
        let imagFull = concatenated([imag, -imagMirror], axis: 1)
        let spectrum = realFull + imagFull.asImaginary()

        var frames = MLXFFT.ifft(spectrum, axis: 1)
        frames = frames.realPart()

        let windowExpanded = window.reshaped([1, window.dim(0), 1])
        frames = frames * windowExpanded

        let outputLength = (numFrames - 1) * hopLength + nFFT
        let frameOffsets = MLXArray(Array(stride(from: 0, to: numFrames, by: 1))) * hopLength
        let sampleIndices = MLXArray(Array(stride(from: 0, to: nFFT, by: 1)))
        let indices =
            frameOffsets.expandedDimensions(axis: 1) + sampleIndices.expandedDimensions(axis: 0)
        let indicesFlat = indices.flattened().asType(.int32)

        let windowSq = window * window
        let windowUpdates = MLX.tiled(windowSq, repetitions: numFrames)
        var windowSum = MLXArray.zeros([outputLength])
        windowSum = windowSum.at[indicesFlat].add(windowUpdates)
        windowSum = MLX.maximum(windowSum, Float(1e-8))

        let frameData = frames.swappedAxes(1, 2)
        let updates = frameData.reshaped([batchSize, numFrames * nFFT])

        var output = MLXArray.zeros([batchSize, outputLength])
        let batchBase = MLXArray(Array(stride(from: 0, to: batchSize, by: 1)))
        let batchIndices = MLX.repeated(batchBase, count: numFrames * nFFT)
        let flatIndices = MLX.tiled(indicesFlat, repetitions: batchSize)
        let linearIndices =
            (batchIndices * MLXArray(outputLength) + flatIndices).asType(.int32)

        var flatOutput = output.reshaped([-1])
        flatOutput = flatOutput.at[linearIndices].add(updates.flattened())
        output = flatOutput.reshaped([batchSize, outputLength])

        output = output / windowSum

        let padLength = nFFT / 2
        return output[0..., padLength..<(outputLength - padLength)]
    }

    private func stftSignal(_ x: MLXArray) -> (MLXArray, MLXArray) {
        hifiganStft(
            x,
            nFFT: istft_params["n_fft"] ?? 16,
            hopLength: istft_params["hop_len"] ?? 4,
            window: stft_window
        )
    }

    private func istftSignal(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        hifiganIstft(
            magnitude: magnitude,
            phase: phase,
            nFFT: istft_params["n_fft"] ?? 16,
            hopLength: istft_params["hop_len"] ?? 4,
            window: stft_window
        )
    }

    public func decode(x: MLXArray, s: MLXArray) -> MLXArray {
        let (sReal, sImag) = stftSignal(s.squeezed(axis: 1))
        let sStft = concatenated([sReal, sImag], axis: 1)

        var out = x.swappedAxes(1, 2)
        out = conv_pre(out)
        out = out.swappedAxes(1, 2)

        for i in 0..<num_upsamples {
            out = leakyRelu(out, negativeSlope: lrelu_slope)
            out = out.swappedAxes(1, 2)
            out = ups[i](out)
            out = out.swappedAxes(1, 2)

            if i == num_upsamples - 1 {
                out = concatenated([out[0..., 0..., 1..<2], out], axis: 2)
            }

            var si = sStft.swappedAxes(1, 2)
            si = source_downs[i](si)
            si = si.swappedAxes(1, 2)
            si = source_resblocks[i](si)
            out = out + si

            let startIdx = i * num_kernels
            let outputs = (0..<num_kernels).map { j in
                resblocks[startIdx + j](out)
            }
            out = mean(stacked(outputs, axis: 0), axis: 0)
        }

        out = leakyRelu(out, negativeSlope: lrelu_slope)
        out = out.swappedAxes(1, 2)
        out = conv_post(out)
        out = out.swappedAxes(1, 2)

        let nFftHalf = (istft_params["n_fft"] ?? 16) / 2 + 1
        let magnitude = exp(out[0..., 0..<nFftHalf, 0...])
        let phase = sin(out[0..., nFftHalf..., 0...])

        var wav = istftSignal(magnitude: magnitude, phase: phase)
        wav = clip(wav, min: -audio_limit, max: audio_limit)
        return wav
    }

    public func decodeDebug(x: MLXArray, s: MLXArray) -> (MLXArray, [String: MLXArray]) {
        let stftResult = hifiganStftDetailed(
            s.squeezed(axis: 1),
            nFFT: istft_params["n_fft"] ?? 16,
            hopLength: istft_params["hop_len"] ?? 4,
            window: stft_window
        )
        let sReal = stftResult.real
        let sImag = stftResult.imag
        let sStft = concatenated([sReal, sImag], axis: 1)

        var out = x.swappedAxes(1, 2)
        out = conv_pre(out)
        out = out.swappedAxes(1, 2)
        let convPreOut = out

        for i in 0..<num_upsamples {
            out = leakyRelu(out, negativeSlope: lrelu_slope)
            out = out.swappedAxes(1, 2)
            out = ups[i](out)
            out = out.swappedAxes(1, 2)

            if i == num_upsamples - 1 {
                out = concatenated([out[0..., 0..., 1..<2], out], axis: 2)
            }

            var si = sStft.swappedAxes(1, 2)
            si = source_downs[i](si)
            si = si.swappedAxes(1, 2)
            si = source_resblocks[i](si)
            out = out + si

            let startIdx = i * num_kernels
            let outputs = (0..<num_kernels).map { j in
                resblocks[startIdx + j](out)
            }
            out = mean(stacked(outputs, axis: 0), axis: 0)
        }

        out = leakyRelu(out, negativeSlope: lrelu_slope)
        out = out.swappedAxes(1, 2)
        out = conv_post(out)
        out = out.swappedAxes(1, 2)

        let nFftHalf = (istft_params["n_fft"] ?? 16) / 2 + 1
        let magnitude = exp(out[0..., 0..<nFftHalf, 0...])
        let phase = sin(out[0..., nFftHalf..., 0...])

        var wav = istftSignal(magnitude: magnitude, phase: phase)
        wav = clip(wav, min: -audio_limit, max: audio_limit)

        let debug: [String: MLXArray] = [
            "hifigan_s_stft": sStft,
            "hifigan_s_stft_real": sReal,
            "hifigan_s_stft_imag": sImag,
            "hifigan_stft_padded": stftResult.padded,
            "hifigan_stft_frames": stftResult.frames,
            "hifigan_stft_windowed": stftResult.windowed,
            "hifigan_conv_pre_out": convPreOut,
            "hifigan_conv_post_out": out,
            "hifigan_mag": magnitude,
            "hifigan_phase": phase,
        ]
        return (wav, debug)
    }

    public func callAsFunction(speechFeat: MLXArray, cacheSource: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let cache = cacheSource ?? MLXArray.zeros([1, 1, 0])

        let f0 = f0_predictor(speechFeat)
        var s = f0Upsample(f0.expandedDimensions(axis: 1))
        s = s.swappedAxes(1, 2)
        let (sine, _, _) = m_source(s)
        s = sine.swappedAxes(1, 2)

        if cache.shape[2] != 0 {
            let cacheLen = cache.shape[2]
            s = concatenated([cache, s[0..., 0..., cacheLen...]], axis: 2)
        }

        let generated = decode(x: speechFeat, s: s)
        return (generated, s)
    }

    public func inference(speechFeat: MLXArray, cacheSource: MLXArray? = nil) -> (MLXArray, MLXArray) {
        callAsFunction(speechFeat: speechFeat, cacheSource: cacheSource)
    }

    public func inferenceDebug(speechFeat: MLXArray, cacheSource: MLXArray? = nil) -> (MLXArray, MLXArray, [String: MLXArray]) {
        let cache = cacheSource ?? MLXArray.zeros([1, 1, 0])

        let f0 = f0_predictor(speechFeat)
        var s = f0Upsample(f0.expandedDimensions(axis: 1))
        s = s.swappedAxes(1, 2)
        let (sine, _, _) = m_source(s)
        s = sine.swappedAxes(1, 2)

        if cache.shape[2] != 0 {
            let cacheLen = cache.shape[2]
            s = concatenated([cache, s[0..., 0..., cacheLen...]], axis: 2)
        }

        let (generated, debug) = decodeDebug(x: speechFeat, s: s)
        return (generated, s, debug)
    }
}
