import Foundation
import Accelerate
import MLX

/// The lowpass a polyphase resampler filters with, as a specification rather than
/// a set of tuned numbers.
///
/// Kaiser's design formulas fix `beta` and the length from a stopband attenuation
/// and a transition width (Kaiser 1974; Oppenheim & Schafer, *Discrete-Time Signal
/// Processing*, 7.5), so a resampler is described by three published constants and
/// not by a search. Both designs below are quoted from their sources; neither was
/// fitted here, and fitting one to match another implementation's output is how you
/// get a filter that tracks a reference and is not otherwise good.
public struct ResamplerDesign: Sendable, Equatable, Hashable {
    /// Sinc zero crossings either side of centre. Sets the filter length.
    public let zeroCrossings: Int
    /// Kaiser window parameter.
    public let beta: Double
    /// Passband edge as a fraction of the target Nyquist.
    public let rolloff: Double

    public init(zeroCrossings: Int, beta: Double, rolloff: Double) {
        self.zeroCrossings = zeroCrossings
        self.beta = beta
        self.rolloff = rolloff
    }

    /// `scipy.signal.resample_poly`'s defaults.
    ///
    /// Kept because matching it is sometimes the requirement: `S3Gen.embed_ref`
    /// resamples with exactly this on the Python side, so reproducing it there is
    /// correctness, not inertia.
    ///
    /// It is not a good filter. The cutoff sits at Nyquist, where a lowpass is only
    /// -6 dB, and 61 taps give it no room to reach stopband: measured alias rejection
    /// is -20.9 dB against -57.7 dB for soxr on the same sweep.
    public static let scipyDefault = ResamplerDesign(
        zeroCrossings: 10, beta: 5.0, rolloff: 1.0
    )

    /// `resampy`'s `kaiser_best`, as published by that project (ISC licensed).
    ///
    /// Designed for -120 dB stopband; measures -55.8 dB alias rejection here, within
    /// 2 dB of soxr's HQ preset. soxr's own header specifies HQ as passband_end 0.913
    /// against this 0.917347 - two independent designs landing on the same passband
    /// edge, which is what makes ~0.915 the answer rather than a preference.
    ///
    /// This is the design, realized through the polyphase path below. It is not a
    /// transcription of resampy's implementation, which interpolates a precomputed
    /// table; for the rational ratios here the two agree, and reusing the existing
    /// `upfirdn` keeps one code path under test instead of two.
    public static let kaiserBest = ResamplerDesign(
        zeroCrossings: 50, beta: 12.9846, rolloff: 0.917347
    )
}

/// Resample with `scipy.signal.resample_poly`'s own filter.
///
/// For call sites whose reference is scipy. Everywhere else prefer
/// ``resampleAudioKaiserBest(_:origSR:targetSR:)`` - see ``ResamplerDesign/scipyDefault``
/// for why this one aliases.
public func resampleAudioPolyphase(_ audio: MLXArray, origSR: Int, targetSR: Int) -> MLXArray {
    resampleAudio(audio, origSR: origSR, targetSR: targetSR, design: .scipyDefault)
}

/// Resample with a band-limited sinc designed for -120 dB stopband.
///
/// The default choice for anything feeding a model. See ``ResamplerDesign/kaiserBest``.
public func resampleAudioKaiserBest(_ audio: MLXArray, origSR: Int, targetSR: Int) -> MLXArray {
    resampleAudio(audio, origSR: origSR, targetSR: targetSR, design: .kaiserBest)
}

public func resampleAudio(
    _ audio: MLXArray, origSR: Int, targetSR: Int, design: ResamplerDesign
) -> MLXArray {
    if origSR == targetSR {
        return audio
    }
    let gcd = greatestCommonDivisor(origSR, targetSR)
    let up = targetSR / gcd
    let down = origSR / gcd
    if up == 1 && down == 1 {
        return audio
    }
    let samples = audio.asArray(Float.self)
    if samples.isEmpty {
        return audio
    }
    return MLXArray(
        resamplePolyphaseScipyEdge(samples, up: up, down: down, design: design, kernel: .vectorized)
    )
}

public func trimSilenceLibrosa(_ audio: MLXArray, topDb: Float) -> MLXArray {
    let frameLength = 2048
    let hopLength = 512
    let padAmount = frameLength / 2
    if audio.dim(0) <= 1 || audio.dim(0) <= padAmount {
        return audio
    }

    let padded = reflectPad1D(audio, padding: padAmount)
    let numFrames = (padded.dim(0) - frameLength) / hopLength + 1
    if numFrames <= 0 {
        return audio
    }

    let frames = MLX.asStrided(
        padded,
        [numFrames, frameLength],
        strides: [hopLength, 1],
        offset: 0
    )
    let power = MLX.mean(frames * frames, axis: 1)
    let rms = MLX.sqrt(power)
    let rmsDb = 20.0 * MLX.log10(MLX.maximum(rms, 1e-10))
    let maxDb = rmsDb.max().item(Float.self)
    let threshold = maxDb - topDb

    let rmsDbArray = rmsDb.asArray(Float.self)
    guard let first = rmsDbArray.firstIndex(where: { $0 > threshold }) else {
        return audio
    }
    let last = rmsDbArray.lastIndex(where: { $0 > threshold }) ?? first
    let startSample = max(0, first * hopLength)
    let endSample = min(audio.dim(0), last * hopLength + frameLength)
    if endSample <= startSample {
        return audio
    }
    return audio[startSample..<endSample]
}

private func reflectPad1D(_ x: MLXArray, padding: Int) -> MLXArray {
    if padding <= 0 {
        return x
    }
    let leftIdx = MLXArray(
        Array(stride(from: padding, through: 1, by: -1))
    )
    let rightStart = x.dim(0) - 2
    let rightEnd = x.dim(0) - padding - 1
    let rightIdx = MLXArray(
        Array(stride(from: rightStart, through: rightEnd, by: -1))
    )
    let left = MLX.take(x, leftIdx, axis: 0)
    let right = MLX.take(x, rightIdx, axis: 0)
    return MLX.concatenated([left, x, right], axis: 0)
}

private func greatestCommonDivisor(_ a: Int, _ b: Int) -> Int {
    var a = a
    var b = b
    while b != 0 {
        let temp = b
        b = a % b
        a = temp
    }
    return a
}

// MARK: - Polyphase resampling

/// Which convolution runs in the polyphase inner loop.
///
/// Both compute the same thing. `scalarReference` performs the additions in scipy's
/// own order, so it is bit-identical to the port this replaced; `vectorized` splits
/// the accumulation four ways to break the FMA dependency chain, which reassociates.
/// `ResamplerNumericsTests` pins both.
enum PolyphaseKernel {
    case vectorized
    case scalarReference
}

/// The designed taps, keyed by everything they depend on.
///
/// `firwinLowpassDouble` costs one `sin` and two `besselI0Double` per tap, and
/// `numtaps` is `2 * zeroCrossings * max(up, down) + 1` - 301 taps for 24k -> 16k
/// but 44101 for 22.05k -> 16k, where the design is ~11% of a call once the
/// convolution is vectorized. None of it depends on the signal, and a pipeline
/// resamples between the same handful of rates for its whole life.
final class DesignedFilterCache: @unchecked Sendable {
    private struct Key: Hashable {
        let up: Int
        let down: Int
        let design: ResamplerDesign
    }

    static let shared = DesignedFilterCache()

    private let lock = NSLock()
    private var entries: [(key: Key, taps: [Double])] = []
    /// Four covers every rate pair a pipeline uses at once. The largest plausible
    /// entry is the 22.05k -> 16k bank at 44101 taps, 345 KB; the common ones are
    /// 301 taps, 2.4 KB.
    private let capacity = 4

    func taps(up: Int, down: Int, design: ResamplerDesign) -> [Double] {
        let key = Key(up: up, down: down, design: design)

        lock.lock()
        if let index = entries.firstIndex(where: { $0.key == key }) {
            let entry = entries.remove(at: index)
            entries.append(entry)
            lock.unlock()
            return entry.taps
        }
        lock.unlock()

        // Built outside the lock. Two threads racing on a cold key both compute the
        // same taps, which is cheaper than serialising every caller behind a design.
        let taps = Self.build(up: up, down: down, design: design)

        lock.lock()
        if !entries.contains(where: { $0.key == key }) {
            entries.append((key, taps))
            if entries.count > capacity {
                entries.removeFirst()
            }
        }
        lock.unlock()
        return taps
    }

    private static func build(up: Int, down: Int, design: ResamplerDesign) -> [Double] {
        let maxRate = max(up, down)
        let halfLen = design.zeroCrossings * maxRate
        var taps = firwinLowpassDouble(
            numtaps: 2 * halfLen + 1,
            cutoff: design.rolloff / Double(maxRate),
            beta: design.beta
        )
        let gain = Double(up)
        for i in 0..<taps.count {
            taps[i] *= gain
        }
        return taps
    }
}

/// `scipy.signal.resample_poly(..., padtype="edge")` over `[Float]`.
///
/// Still the same algorithm as the reference - the filter design, the pre/post pad
/// arithmetic and the `nPreRemove` trim are scipy's, unchanged. What moved is where
/// the work happens: the taps come from a cache, the padded and transposed filter is
/// built in one pass instead of three arrays, and the convolution runs over unsafe
/// buffers with a vector accumulator.
func resamplePolyphaseScipyEdge(
    _ audio: [Float], up: Int, down: Int, design: ResamplerDesign, kernel: PolyphaseKernel
) -> [Float] {
    if up == 1 && down == 1 {
        return audio
    }
    if audio.isEmpty {
        return audio
    }

    let nIn = audio.count
    let nOut = (nIn * up + down - 1) / down

    let maxRate = max(up, down)
    let halfLen = design.zeroCrossings * maxRate
    let taps = DesignedFilterCache.shared.taps(up: up, down: down, design: design)

    let nPrePad = down - (halfLen % down)
    var nPostPad = 0
    let nPreRemove = (halfLen + nPrePad) / down
    while outputLen(taps.count + nPrePad + nPostPad, nIn, up, down) < nOut + nPreRemove {
        nPostPad += 1
    }
    let hPaddedCount = nPrePad + taps.count + nPostPad

    // scipy pads `h` with zeros on both sides, then transposes it into `up` phases
    // and flips each. The pad is all zeros and the bank starts zeroed, so the padded
    // array never has to exist: writing tap `j` to the slot `hPadded[j + nPrePad]`
    // would have landed in produces the same bank, with two fewer allocations.
    let padLen = hPaddedCount + ((up - (hPaddedCount % up)) % up)
    let hPerPhase = padLen / up
    var hTransFlip = [Double](repeating: 0.0, count: padLen)
    hTransFlip.withUnsafeMutableBufferPointer { bank in
        taps.withUnsafeBufferPointer { source in
            for j in 0..<source.count {
                let i = j + nPrePad
                bank[(i % up) * hPerPhase + (hPerPhase - 1 - i / up)] = source[j]
            }
        }
    }

    let lenOut = outputLen(hPaddedCount, nIn, up, down)
    if lenOut <= 0 {
        return []
    }

    // Float -> Double is exact, so this is the same conversion `map(Double.init)` was.
    let xDouble = [Double](unsafeUninitializedCapacity: nIn) { buffer, initialized in
        audio.withUnsafeBufferPointer { source in
            vDSP_vspdp(source.baseAddress!, 1, buffer.baseAddress!, 1, vDSP_Length(nIn))
        }
        initialized = nIn
    }

    var y = [Double](repeating: 0.0, count: lenOut)
    y.withUnsafeMutableBufferPointer { out in
        xDouble.withUnsafeBufferPointer { x in
            hTransFlip.withUnsafeBufferPointer { bank in
                upfirdnEdgeDouble(
                    bank: bank.baseAddress!,
                    hPerPhase: hPerPhase,
                    x: x.baseAddress!,
                    lenX: nIn,
                    up: up,
                    down: down,
                    out: out.baseAddress!,
                    lenOut: lenOut,
                    kernel: kernel
                )
            }
        }
    }

    // scipy keeps y[nPreRemove ..< nPreRemove + nOut]. Go straight to Float from the
    // slice rather than materialising the slice and then mapping it.
    let keep = min(nOut, max(0, lenOut - nPreRemove))
    var result = [Float](repeating: 0.0, count: nOut)
    if keep > 0 {
        y.withUnsafeBufferPointer { source in
            result.withUnsafeMutableBufferPointer { destination in
                vDSP_vdpsp(
                    source.baseAddress! + nPreRemove, 1,
                    destination.baseAddress!, 1,
                    vDSP_Length(keep)
                )
            }
        }
    }
    return result
}

private func outputLen(_ lenH: Int, _ lenX: Int, _ up: Int, _ down: Int) -> Int {
    return ((lenX * up + lenH - up - 1) / down) + 1
}

/// `scipy.signal.upfirdn` with `padtype="edge"`, over a closed-form index.
///
/// scipy walks `(xIdx, t)` forward with a carry. That walk has a closed form: after
/// `k` outputs the accumulated numerator is `k * down`, so `t = (k * down) % up` and
/// `xIdx = (k * down) / up`. Addressing each output independently is what lets the
/// body be a single dot product with no loop-carried state.
private func upfirdnEdgeDouble(
    bank: UnsafePointer<Double>,
    hPerPhase: Int,
    x: UnsafePointer<Double>,
    lenX: Int,
    up: Int,
    down: Int,
    out: UnsafeMutablePointer<Double>,
    lenOut: Int,
    kernel: PolyphaseKernel
) {
    let leftEdge = lenX > 0 ? x[0] : 0.0
    let rightEdge = lenX > 0 ? x[lenX - 1] : 0.0
    let paddedLen = lenX + hPerPhase - 1

    for k in 0..<lenOut {
        let numerator = k * down
        let xIdx = numerator / up
        // scipy stops once `xIdx` runs past the edge-extended signal and leaves the
        // rest of the output zero.
        if xIdx >= paddedLen {
            out[k] = 0.0
            continue
        }

        var hIdx = (numerator % up) * hPerPhase
        let low = xIdx - hPerPhase + 1
        var acc = 0.0

        // The window runs ascending over [low, xIdx], which splits into at most three
        // runs: before the signal, inside it, at or past the end. Taking them in that
        // order is scipy's own addition sequence.
        if low < 0 {
            for _ in low..<0 {
                acc += leftEdge * bank[hIdx]
                hIdx += 1
            }
        }

        let insideLow = max(low, 0)
        let insideHigh = min(xIdx, lenX - 1)
        if insideHigh >= insideLow {
            let count = insideHigh - insideLow + 1
            if kernel == .vectorized && count >= 8 {
                acc += dotProductSIMD(x + insideLow, bank + hIdx, count)
                hIdx += count
            } else {
                for idx in insideLow...insideHigh {
                    acc += x[idx] * bank[hIdx]
                    hIdx += 1
                }
            }
        }

        let pastLow = max(low, lenX)
        if xIdx >= pastLow {
            for _ in pastLow...xIdx {
                acc += rightEdge * bank[hIdx]
                hIdx += 1
            }
        }

        out[k] = acc
    }
}

/// Dot product with four independent accumulators.
///
/// Not `vDSP_dotprD`: one output sample is only `hPerPhase` taps - 151 for
/// 24k -> 16k - and at that length the framework call costs more than the arithmetic
/// it performs. Measured on 10 s at 24k -> 16k, per-sample `vDSP_dotprD` runs the
/// conversion in 5.9 ms against 4.1 ms for this, and both against 13.6 ms for the
/// same loop with one accumulator.
///
/// Four accumulators, because a single one serialises on FMA latency: one add per
/// ~4 cycles is what pinned the scalar loop near 1 GMAC/s however the bounds checks
/// went. Splitting the chain four ways is also the only arithmetic difference from
/// `scalarReference`.
@inline(__always)
private func dotProductSIMD(
    _ a: UnsafePointer<Double>, _ f: UnsafePointer<Double>, _ n: Int
) -> Double {
    var s0 = SIMD2<Double>.zero
    var s1 = SIMD2<Double>.zero
    var s2 = SIMD2<Double>.zero
    var s3 = SIMD2<Double>.zero
    var i = 0
    while i + 8 <= n {
        let a0 = UnsafeRawPointer(a + i).loadUnaligned(as: SIMD2<Double>.self)
        let a1 = UnsafeRawPointer(a + i + 2).loadUnaligned(as: SIMD2<Double>.self)
        let a2 = UnsafeRawPointer(a + i + 4).loadUnaligned(as: SIMD2<Double>.self)
        let a3 = UnsafeRawPointer(a + i + 6).loadUnaligned(as: SIMD2<Double>.self)
        let f0 = UnsafeRawPointer(f + i).loadUnaligned(as: SIMD2<Double>.self)
        let f1 = UnsafeRawPointer(f + i + 2).loadUnaligned(as: SIMD2<Double>.self)
        let f2 = UnsafeRawPointer(f + i + 4).loadUnaligned(as: SIMD2<Double>.self)
        let f3 = UnsafeRawPointer(f + i + 6).loadUnaligned(as: SIMD2<Double>.self)
        s0 = s0.addingProduct(a0, f0)
        s1 = s1.addingProduct(a1, f1)
        s2 = s2.addingProduct(a2, f2)
        s3 = s3.addingProduct(a3, f3)
        i += 8
    }
    let folded = (s0 + s1) + (s2 + s3)
    var total = folded[0] + folded[1]
    while i < n {
        total += a[i] * f[i]
        i += 1
    }
    return total
}

/// Internal rather than private so `ResamplerBenchmarkTests` can time the design on
/// its own. It is the expensive half for large `max(up, down)` and the half the cache
/// removes, so it needs to be measurable apart from the convolution.
func firwinLowpassDouble(numtaps: Int, cutoff: Double, beta: Double) -> [Double] {
    let alpha = 0.5 * Double(numtaps - 1)
    var h = [Double](repeating: 0.0, count: numtaps)
    for n in 0..<numtaps {
        let m = Double(n) - alpha
        h[n] = cutoff * sincDouble(cutoff * m)
    }
    let window = kaiserWindowDouble(numtaps: numtaps, beta: beta)
    for i in 0..<numtaps {
        h[i] *= window[i]
    }
    let scale = h.reduce(0.0, +)
    if scale != 0.0 {
        for i in 0..<numtaps {
            h[i] /= scale
        }
    }
    return h
}

private func sincDouble(_ x: Double) -> Double {
    if abs(x) < 1e-12 {
        return 1.0
    }
    let pix = Double.pi * x
    return sin(pix) / pix
}

private func kaiserWindowDouble(numtaps: Int, beta: Double) -> [Double] {
    if numtaps == 1 {
        return [1.0]
    }
    let alpha = 0.5 * Double(numtaps - 1)
    let denom = besselI0Double(beta)
    var win = [Double](repeating: 0.0, count: numtaps)
    for n in 0..<numtaps {
        let ratio = (Double(n) - alpha) / alpha
        let inside = max(0.0, 1.0 - ratio * ratio)
        let value = beta * sqrt(inside)
        win[n] = besselI0Double(value) / denom
    }
    return win
}

private func besselI0Double(_ x: Double) -> Double {
    let ax = abs(x)
    if ax < 3.75 {
        let y = x / 3.75
        let y2 = y * y
        return 1.0 + y2 * (3.5156229 + y2 * (3.0899424 + y2 * (1.2067492
            + y2 * (0.2659732 + y2 * (0.0360768 + y2 * 0.0045813)))))
    }
    let y = 3.75 / ax
    let expTerm = exp(ax) / sqrt(ax)
    var poly = 0.00392377
    poly = -0.01647633 + y * poly
    poly = 0.02635537 + y * poly
    poly = -0.02057706 + y * poly
    poly = 0.00916281 + y * poly
    poly = -0.00157565 + y * poly
    poly = 0.00225319 + y * poly
    poly = 0.01328592 + y * poly
    poly = 0.39894228 + y * poly
    return expTerm * poly
}
