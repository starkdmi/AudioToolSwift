import Foundation
import MLX

private final class DSPArrayCache: @unchecked Sendable {
    private let lock = NSLock()
    private let maximumEntries = 32
    private var values: [String: MLXArray] = [:]
    private var recency: [String] = []

    func value(for key: String) -> MLXArray? {
        lock.lock()
        defer { lock.unlock() }
        guard let value = values[key] else { return nil }
        touch(key)
        return value
    }

    func insertIfAbsent(_ value: MLXArray, for key: String) -> MLXArray {
        eval(value)
        lock.lock()
        defer { lock.unlock() }
        if let existing = values[key] {
            touch(key)
            return existing
        }
        if values.count >= maximumEntries, let evicted = recency.first {
            values.removeValue(forKey: evicted)
            recency.removeFirst()
        }
        values[key] = value
        recency.append(key)
        return value
    }

    /// Called only while `lock` is held.
    private func touch(_ key: String) {
        if let index = recency.firstIndex(of: key) {
            recency.remove(at: index)
        }
        recency.append(key)
    }
}

public enum DSP {
    private static let windowCache = DSPArrayCache()
    private static let melCache = DSPArrayCache()

    /// Periodic Hanning window - matches Python DSP hanning which uses N denominator
    /// (librosa fftbins=True, torch.hann_window(periodic=True))
    public static func hanning(_ size: Int) -> MLXArray {
        cachedWindow("hanning", size: size) {
            let n = MLXArray(0..<size).asType(.float32)
            let denom = MLXArray(Float(size))
            return 0.5 * (1 - MLX.cos(2 * Float.pi * n / denom))
        }
    }

    public static func hamming(_ size: Int) -> MLXArray {
        cachedWindow("hamming", size: size) {
            let n = MLXArray(0..<size).asType(.float32)
            let denom = MLXArray(Float(size - 1))
            return 0.54 - 0.46 * MLX.cos(2 * Float.pi * n / denom)
        }
    }

    public static func blackman(_ size: Int) -> MLXArray {
        cachedWindow("blackman", size: size) {
            let n = MLXArray(0..<size).asType(.float32)
            let denom = MLXArray(Float(size - 1))
            return 0.42
                - 0.5 * MLX.cos(2 * Float.pi * n / denom)
                + 0.08 * MLX.cos(4 * Float.pi * n / denom)
        }
    }

    public static func bartlett(_ size: Int) -> MLXArray {
        cachedWindow("bartlett", size: size) {
            let n = MLXArray(0..<size).asType(.float32)
            let center = MLXArray(Float(size - 1) / 2)
            let denom = MLXArray(Float(size - 1))
            return 1 - 2 * MLX.abs(n - center) / denom
        }
    }

    public static func stft(
        _ x: MLXArray,
        nFFT: Int = 800,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        window: String = "hann",
        center: Bool = true,
        padMode: String = "reflect"
    ) -> MLXArray {
        let hop = hopLength ?? (nFFT / 4)
        let winLen = winLength ?? nFFT
        let w = windowFromString(window, size: winLen)
        return stft(x, nFFT: nFFT, hopLength: hop, winLength: winLen, window: w, center: center, padMode: padMode)
    }

    public static func stft(
        _ x: MLXArray,
        nFFT: Int = 800,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        window: MLXArray,
        center: Bool = true,
        padMode: String = "reflect"
    ) -> MLXArray {
        let hop = hopLength ?? (nFFT / 4)
        var w = window
        if w.shape[0] < nFFT {
            let pad = MLXArray.zeros([nFFT - w.shape[0]], dtype: w.dtype)
            w = MLX.concatenated([w, pad], axis: 0)
        }

        var signal = x
        if center {
            signal = pad1d(signal, padding: nFFT / 2, padMode: padMode)
        }

        let numFrames = 1 + (signal.shape[0] - nFFT) / hop
        guard numFrames > 0 else {
            fatalError("Input is too short for n_fft=\(nFFT) hop_length=\(hop) center=\(center).")
        }

        let frames = MLX.asStrided(signal, [numFrames, nFFT], strides: [hop, 1])
        return MLX.rfft(frames * w, n: nFFT, axis: -1)
    }

    public static func istft(
        _ x: MLXArray,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        window: String = "hann",
        center: Bool = true,
        length: Int? = nil
    ) -> MLXArray {
        let winLen = winLength ?? (x.shape[0] - 1) * 2
        let hop = hopLength ?? (winLen / 4)
        let w = windowFromString(window, size: winLen + 1)[0..<winLen]
        return istft(x, hopLength: hop, winLength: winLen, window: w, center: center, length: length)
    }

    public static func istft(
        _ x: MLXArray,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        window: MLXArray,
        center: Bool = true,
        length: Int? = nil
    ) -> MLXArray {
        let winLen = winLength ?? (x.shape[0] - 1) * 2
        let hop = hopLength ?? (winLen / 4)
        var w = window
        if w.shape[0] < winLen {
            let pad = MLXArray.zeros([winLen - w.shape[0]], dtype: w.dtype)
            w = MLX.concatenated([w, pad], axis: 0)
        }

        let numFrames = x.shape[1]
        let t = (numFrames - 1) * hop + winLen
        var reconstructed = MLXArray.zeros([t], dtype: w.dtype)
        var windowSum = MLXArray.zeros([t], dtype: w.dtype)

        let framesTime = MLX.irfft(x, axis: 0).transposed(1, 0)
        let frameOffsets = MLXArray(0..<numFrames) * hop
        let indices = frameOffsets.expandedDimensions(axis: 1) + MLXArray(0..<winLen)
        let indicesFlat = indices.flattened()

        let updatesReconstructed = (framesTime * w).flattened()
        let updatesWindow = MLX.repeated(w, count: numFrames, axis: 0).flattened()

        reconstructed = reconstructed.at[indicesFlat].add(updatesReconstructed)
        windowSum = windowSum.at[indicesFlat].add(updatesWindow)

        reconstructed = MLX.where(windowSum .!= 0, reconstructed / windowSum, reconstructed)

        if center && length == nil {
            reconstructed = reconstructed[(winLen / 2)..<(reconstructed.shape[0] - winLen / 2)]
        }
        if let length = length {
            reconstructed = reconstructed[0..<min(length, reconstructed.shape[0])]
        }
        return reconstructed
    }

    public static func melFilters(
        sampleRate: Int,
        nFFT: Int,
        nMels: Int,
        fMin: Float = 0,
        fMax: Float? = nil,
        norm: String? = nil,
        melScale: String = "htk"
    ) -> MLXArray {
        let maxFreq = fMax ?? Float(sampleRate) / 2
        let key = "\(sampleRate)|\(nFFT)|\(nMels)|\(fMin)|\(maxFreq)|\(norm ?? "none")|\(melScale)"
        if let cached = melCache.value(for: key) {
            return cached
        }

        func hzToMel(_ freq: Float, scale: String) -> Float {
            if scale == "htk" {
                return 2595.0 * log10(1.0 + freq / 700.0)
            }
            let fMin: Float = 0.0
            let fSp: Float = 200.0 / 3
            var mels = (freq - fMin) / fSp
            let minLogHz: Float = 1000.0
            let minLogMel = (minLogHz - fMin) / fSp
            let logstep = Float(log(6.4) / 27.0)
            if freq >= minLogHz {
                mels = minLogMel + log(freq / minLogHz) / logstep
            }
            return mels
        }

        func melToHz(_ mels: MLXArray, scale: String) -> MLXArray {
            if scale == "htk" {
                return 700.0 * (MLX.pow(MLXArray(10.0), mels / 2595.0) - 1.0)
            }
            let fMin: Float = 0.0
            let fSp: Float = 200.0 / 3
            let freqs = fMin + fSp * mels
            let minLogHz: Float = 1000.0
            let minLogMel = (minLogHz - fMin) / fSp
            let logstep = Float(log(6.4) / 27.0)
            return MLX.where(
                mels .>= MLXArray(minLogMel),
                MLXArray(minLogHz) * MLX.exp(MLXArray(logstep) * (mels - MLXArray(minLogMel))),
                freqs
            )
        }

        let nFreqs = nFFT / 2 + 1
        // Use integer division to match Python: sample_rate // 2
        let allFreqs = MLX.linspace(Float(0), Float(sampleRate / 2), count: nFreqs)

        let mMin = hzToMel(fMin, scale: melScale)
        let mMax = hzToMel(maxFreq, scale: melScale)
        let mPts = MLX.linspace(mMin, mMax, count: nMels + 2)
        let fPts = melToHz(mPts, scale: melScale)

        let fPtsLen = fPts.shape[0]
        let fDiff = fPts[1..<fPtsLen] - fPts[0..<(fPtsLen - 1)]
        let slopes = fPts.expandedDimensions(axis: 0) - allFreqs.expandedDimensions(axis: 1)

        let downSlopes = (-slopes[0..., 0..<(slopes.shape[1] - 2)]) / fDiff[0..<(fDiff.shape[0] - 1)]
        let upSlopes = slopes[0..., 2..<slopes.shape[1]] / fDiff[1..<fDiff.shape[0]]
        var filterbank = MLX.maximum(zerosLike(downSlopes), MLX.minimum(downSlopes, upSlopes))

        if norm == "slaney" {
            let enorm = 2.0 / (fPts[2..<(nMels + 2)] - fPts[0..<nMels])
            filterbank = filterbank * enorm.expandedDimensions(axis: 0)
        }

        filterbank = filterbank.transposed(0, 1)
        return melCache.insertIfAbsent(filterbank, for: key)
    }

    private static func windowFromString(_ window: String, size: Int) -> MLXArray {
        switch window.lowercased() {
        case "hann", "hanning":
            return hanning(size)
        case "hamming":
            return hamming(size)
        case "blackman":
            return blackman(size)
        case "bartlett":
            return bartlett(size)
        default:
            fatalError("Unknown window function: \(window)")
        }
    }

    private static func cachedWindow(_ name: String, size: Int, builder: () -> MLXArray) -> MLXArray {
        let key = "\(name)_\(size)"
        if let cached = windowCache.value(for: key) {
            return cached
        }
        let window = builder()
        return windowCache.insertIfAbsent(window, for: key)
    }

    private static func pad1d(_ x: MLXArray, padding: Int, padMode: String) -> MLXArray {
        guard padding > 0 else { return x }
        switch padMode {
        case "constant":
            return MLX.padded(x, widths: [IntOrPair([padding, padding])])
        case "reflect":
            let leftIdx = MLXArray(Array(stride(from: padding, through: 1, by: -1)))
            let rightStart = x.shape[0] - 2
            let rightEnd = x.shape[0] - padding - 1
            let rightIdx = MLXArray(Array(stride(from: rightStart, through: rightEnd, by: -1)))
            let left = MLX.take(x, leftIdx, axis: 0)
            let right = MLX.take(x, rightIdx, axis: 0)
            return MLX.concatenated([left, x, right], axis: 0)
        default:
            fatalError("Invalid pad_mode \(padMode)")
        }
    }
}
