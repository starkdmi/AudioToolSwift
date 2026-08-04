import Foundation
import MLX

private let melScaleFrequencyBase: Float = 700.0
private let melScaleLogCoefficient: Float = 1127.0
private let millisecondsToSeconds: Float = 0.001

private struct WindowCoefficients {
    static let hannCenter: Float = 0.5
    static let hammingAlpha: Float = 0.54
    static let hammingBeta: Float = 0.46
    static let poveyPower: Float = 0.85
}

struct KaldiCompatArgs {
    let samplingRate: Int
    let frameLengthMs: Float
    let frameShiftMs: Float
    let numMels: Int
    let windowType: String
    let dither: Float
    let preemphasisCoefficient: Float
    let lowFreq: Float
    let highFreq: Float
    let epsilon: Float
    let snipEdges: Bool
    let roundToPowerOfTwo: Bool
    let removeDcOffset: Bool
    let usePower: Bool
    let useLogFbank: Bool

    init(
        samplingRate: Int = 16000,
        frameLengthMs: Float = 25.0,
        frameShiftMs: Float = 10.0,
        numMels: Int = 80,
        windowType: String = "povey",
        dither: Float = 0.0,
        preemphasisCoefficient: Float = 0.97,
        lowFreq: Float = 20.0,
        highFreq: Float = 0.0,
        epsilon: Float = 1.1920928955078125e-07,
        snipEdges: Bool = true,
        roundToPowerOfTwo: Bool = true,
        removeDcOffset: Bool = true,
        usePower: Bool = true,
        useLogFbank: Bool = true
    ) {
        self.samplingRate = samplingRate
        self.frameLengthMs = frameLengthMs
        self.frameShiftMs = frameShiftMs
        self.numMels = numMels
        self.windowType = windowType
        self.dither = dither
        self.preemphasisCoefficient = preemphasisCoefficient
        self.lowFreq = lowFreq
        self.highFreq = highFreq
        self.epsilon = epsilon
        self.snipEdges = snipEdges
        self.roundToPowerOfTwo = roundToPowerOfTwo
        self.removeDcOffset = removeDcOffset
        self.usePower = usePower
        self.useLogFbank = useLogFbank
    }
}

enum KaldiFbankCompat {

    private final class WindowCache: @unchecked Sendable {
        private var cache: [String: MLXArray] = [:]
        private let lock = NSLock()

        func get(_ key: String) -> MLXArray? {
            lock.lock()
            defer { lock.unlock() }
            return cache[key]
        }

        func set(_ key: String, value: MLXArray) {
            lock.lock()
            defer { lock.unlock() }
            cache[key] = value
        }
    }

    private final class MelFilterbankCache: @unchecked Sendable {
        private var cache: [String: (MLXArray, MLXArray)] = [:]
        private let lock = NSLock()

        func get(_ key: String) -> (MLXArray, MLXArray)? {
            lock.lock()
            defer { lock.unlock() }
            return cache[key]
        }

        func set(_ key: String, value: (MLXArray, MLXArray)) {
            lock.lock()
            defer { lock.unlock() }
            cache[key] = value
        }
    }

    private static let windowCache = WindowCache()
    private static let melFilterbankCache = MelFilterbankCache()

    /// High-precision FFT using float64 on CPU to match NumPy's behavior
    /// NumPy upcasts float32 inputs to float64 for FFT, then returns float32.
    /// This gives ~1e-7 precision vs ~1e-4 with pure float32.
    private static func rfft64CPU(_ input: MLXArray, n: Int, axis: Int) -> MLXArray {
        return Stream.withNewDefaultStream(device: .cpu) {
            // Upcast to float64 for precision
            let input64 = input.asType(.float64)
            
            // Perform FFT in float64
            let fftResult = MLXFFT.rfft(input64, n: n, axis: axis)
            eval(fftResult)
            
            // Return complex64 (float32 real/imag components)
            // The abs/power ops will handle the conversion
            return fftResult
        }
    }

    static func compute(_ audioIn: MLXArray, args: KaldiCompatArgs = KaldiCompatArgs()) -> MLXArray {
        let audio: MLXArray
        if audioIn.ndim == 2 {
            audio = audioIn[0]
        } else {
            audio = audioIn
        }

        let sampleFrequency = Float(args.samplingRate)
        let windowShiftSamples = Int(sampleFrequency * args.frameShiftMs * millisecondsToSeconds)
        let windowSize = Int(sampleFrequency * args.frameLengthMs * millisecondsToSeconds)
        let paddedWindowSize = args.roundToPowerOfTwo ? nextPowerOf2(windowSize) : windowSize

        let frames = getOptimizedFrames(
            waveform: audio,
            windowSize: windowSize,
            windowShift: windowShiftSamples,
            snipEdges: args.snipEdges
        )

        if frames.shape[0] == 0 {
            return MLXArray.zeros([0, args.numMels], dtype: audio.dtype)
        }

        var processedFrames = frames
        if args.dither != 0.0 {
            let randGauss = MLXRandom.normal(frames.shape) * args.dither
            processedFrames = processedFrames + randGauss
        }

        if args.removeDcOffset {
            let rowMeans = processedFrames.mean(axis: 1, keepDims: true)
            processedFrames = processedFrames - rowMeans
        }

        if args.preemphasisCoefficient != 0.0 {
            let firstCol = processedFrames[0..., 0..<1]
            let offset = MLX.concatenated([firstCol, processedFrames], axis: 1)
            processedFrames = processedFrames - args.preemphasisCoefficient * offset[0..., 0..<(offset.shape[1] - 1)]
        }

        let window = getCachedWindow(windowType: args.windowType, windowSize: windowSize, dtype: audio.dtype)
        processedFrames = processedFrames * window

        if paddedWindowSize > windowSize {
            let padding = paddedWindowSize - windowSize
            processedFrames = MLX.padded(processedFrames, widths: [IntOrPair(0), IntOrPair([0, padding])])
        }

        // Use float64 CPU FFT for high precision (matches NumPy's float64 upcast behavior)
        let fftResult = rfft64CPU(processedFrames, n: paddedWindowSize, axis: 1)
        var spectrum = MLX.abs(fftResult).asType(.float32)

        if args.usePower {
            spectrum = spectrum ** 2.0
        }

        let (melEnergies, _) = getCachedMelBanks(
            numBins: args.numMels,
            windowLengthPadded: paddedWindowSize,
            sampleFreq: sampleFrequency,
            lowFreq: args.lowFreq,
            highFreq: args.highFreq
        )
        let paddedMelEnergies = MLX.padded(melEnergies, widths: [IntOrPair(0), IntOrPair([0, 1])])
        let melPre = MLX.matmul(spectrum, paddedMelEnergies.T)

        if args.useLogFbank {
            let eps = MLXArray(args.epsilon, dtype: melPre.dtype)
            return MLX.log(MLX.maximum(melPre, eps))
        }
        return melPre
    }

    private static func getOptimizedFrames(
        waveform: MLXArray,
        windowSize: Int,
        windowShift: Int,
        snipEdges: Bool
    ) -> MLXArray {
        let numSamples = waveform.shape[0]

        if snipEdges {
            if numSamples < windowSize {
                return MLXArray.zeros([0, 0])
            }

            let m = 1 + (numSamples - windowSize) / windowShift
            return MLX.asStrided(waveform, [m, windowSize], strides: [windowShift, 1])
        }

        let m = (numSamples + (windowShift / 2)) / windowShift
        let pad = windowSize / 2 - windowShift / 2

        var paddedWaveform = waveform
        if pad > 0 && numSamples > pad {
            let padLeft = waveform[1..<(pad + 1)][.stride(by: -1)]

            let padRight: MLXArray
            if pad > 1 {
                padRight = waveform[(-pad - 1)..<(-1)][.stride(by: -1)]
            } else {
                padRight = waveform[-2..<(-1)]
            }

            paddedWaveform = MLX.concatenated([padLeft, waveform, padRight])
        } else if pad > 0 {
            let zeroPad = MLXArray.zeros([pad], dtype: waveform.dtype)
            paddedWaveform = MLX.concatenated([zeroPad, waveform, zeroPad])
        } else if pad < 0 {
            let trimAmount = min(-pad, numSamples)
            if trimAmount < numSamples {
                paddedWaveform = waveform[trimAmount...]
            }
        }

        return MLX.asStrided(paddedWaveform, [m, windowSize], strides: [windowShift, 1])
    }

    private static func nextPowerOf2(_ x: Int) -> Int {
        if x <= 0 { return 1 }
        if x == 1 { return 1 }

        var power = 1
        while power < x {
            power *= 2
        }
        return power
    }

    private static func getCachedWindow(
        windowType: String,
        windowSize: Int,
        dtype: DType,
        periodic: Bool = false
    ) -> MLXArray {
        let cacheKey = "\(windowType)_\(windowSize)_\(periodic)_\(dtype)"

        if let cachedWindow = windowCache.get(cacheKey) {
            return cachedWindow
        }

        let n = MLXArray(0..<windowSize).asType(dtype)
        let window: MLXArray
        let twoPi = MLXArray(2.0 * Float.pi, dtype: dtype)
        let half = MLXArray(WindowCoefficients.hannCenter, dtype: dtype)

        switch windowType {
        case "hanning":
            let divisor = MLXArray(periodic ? Float(windowSize) : Float(windowSize - 1), dtype: dtype)
            let cosArg = twoPi * n / divisor
            window = half - half * MLX.cos(cosArg)
        case "hamming":
            let divisor = MLXArray(Float(windowSize - 1), dtype: dtype)
            let cosArg = twoPi * n / divisor
            let alpha = MLXArray(WindowCoefficients.hammingAlpha, dtype: dtype)
            let beta = MLXArray(WindowCoefficients.hammingBeta, dtype: dtype)
            window = alpha - beta * MLX.cos(cosArg)
        case "povey":
            let a = (2.0 * Double.pi) / Double(windowSize - 1)
            let aArr = MLXArray(Float(a), dtype: dtype)
            let hann = half - half * MLX.cos(aArr * n)
            window = MLX.pow(hann, WindowCoefficients.poveyPower)
        case "rectangular":
            window = MLXArray.ones([windowSize], dtype: dtype)
        default:
            window = MLXArray.ones([windowSize], dtype: dtype)
        }

        windowCache.set(cacheKey, value: window)
        return window
    }

    private static func melScale(_ freq: MLXArray) -> MLXArray {
        let dtype = freq.dtype
        let one = MLXArray(1.0, dtype: dtype)
        let scale = MLXArray(melScaleLogCoefficient, dtype: dtype)
        let denom = MLXArray(melScaleFrequencyBase, dtype: dtype)
        return scale * MLX.log(one + freq / denom)
    }

    private static func inverseMelScale(_ melFreq: MLXArray) -> MLXArray {
        let dtype = melFreq.dtype
        let one = MLXArray(1.0, dtype: dtype)
        let scale = MLXArray(melScaleFrequencyBase, dtype: dtype)
        let denom = MLXArray(melScaleLogCoefficient, dtype: dtype)
        return scale * (MLX.exp(melFreq / denom) - one)
    }

    private static func melScaleScalar(_ freq: Float) -> Float {
        let value = 1127.0 * log(1.0 + Double(freq) / 700.0)
        return Float(value)
    }

    private static func getCachedMelBanks(
        numBins: Int,
        windowLengthPadded: Int,
        sampleFreq: Float,
        lowFreq: Float,
        highFreq: Float
    ) -> (MLXArray, MLXArray) {
        let cacheKey = "\(numBins)_\(windowLengthPadded)_\(sampleFreq)_\(lowFreq)_\(highFreq)"

        if let cached = melFilterbankCache.get(cacheKey) {
            return cached
        }

        let result = getMelBanks(
            numBins: numBins,
            windowLengthPadded: windowLengthPadded,
            sampleFreq: sampleFreq,
            lowFreq: lowFreq,
            highFreq: highFreq
        )

        melFilterbankCache.set(cacheKey, value: result)
        return result
    }

    private static func getMelBanks(
        numBins: Int,
        windowLengthPadded: Int,
        sampleFreq: Float,
        lowFreq: Float,
        highFreq: Float
    ) -> (MLXArray, MLXArray) {
        assert(numBins > 3, "Must have at least 3 mel bins")
        assert(windowLengthPadded % 2 == 0)

        let numFftBins = windowLengthPadded / 2
        let nyquist = 0.5 * sampleFreq

        var actualHighFreq = highFreq
        if highFreq <= 0.0 {
            actualHighFreq = highFreq + nyquist
        }

        assert(0.0 <= lowFreq && lowFreq < nyquist)
        assert(0.0 < actualHighFreq && actualHighFreq <= nyquist)
        assert(lowFreq < actualHighFreq)

        let fftBinWidth = sampleFreq / Float(windowLengthPadded)
        let melLowFreq = melScaleScalar(lowFreq)
        let melHighFreq = melScaleScalar(actualHighFreq)

        let melFreqDelta = (melHighFreq - melLowFreq) / Float(numBins + 1)

        let binIdx = MLXArray(0..<numBins).asType(.float32).reshaped([-1, 1])
        let melLow = MLXArray(melLowFreq, dtype: .float32)
        let melDelta = MLXArray(melFreqDelta, dtype: .float32)
        let leftMel = melLow + binIdx * melDelta
        let centerMel = melLow + (binIdx + 1.0) * melDelta
        let rightMel = melLow + (binIdx + 2.0) * melDelta

        let centerFreqs = inverseMelScale(centerMel)

        let fftBins = MLXArray(0..<numFftBins).asType(.float32)
        let mel = melScale(fftBins * fftBinWidth).reshaped([1, -1])

        let upSlope = (mel - leftMel) / (centerMel - leftMel)
        let downSlope = (rightMel - mel) / (rightMel - centerMel)

        let zeros = MLXArray.zeros(upSlope.shape, dtype: upSlope.dtype)
        let bins = MLX.maximum(zeros, MLX.minimum(upSlope, downSlope))

        return (bins, centerFreqs.squeezed())
    }
}
