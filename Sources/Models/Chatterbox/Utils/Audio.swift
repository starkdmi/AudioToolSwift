import Foundation
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
public struct ResamplerDesign: Sendable, Equatable {
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
    return resamplePolyphaseScipyEdge(audio, up: up, down: down, design: design)
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

private func resamplePolyphaseScipyEdge(
    _ audio: MLXArray, up: Int, down: Int, design: ResamplerDesign
) -> MLXArray {
    if up == 1 && down == 1 {
        return audio
    }
    let xFloat = audio.asArray(Float.self)
    if xFloat.isEmpty {
        return audio
    }
    let xDouble = xFloat.map(Double.init)
    let nIn = xDouble.count
    let nOut = (nIn * up + down - 1) / down

    let maxRate = max(up, down)
    let halfLen = design.zeroCrossings * maxRate
    let numtaps = 2 * halfLen + 1
    var h = firwinLowpassDouble(
        numtaps: numtaps,
        cutoff: design.rolloff / Double(maxRate),
        beta: design.beta
    )
    for i in 0..<h.count {
        h[i] *= Double(up)
    }

    let nPrePad = down - (halfLen % down)
    var nPostPad = 0
    let nPreRemove = (halfLen + nPrePad) / down
    while outputLen(h.count + nPrePad + nPostPad, nIn, up, down) < nOut + nPreRemove {
        nPostPad += 1
    }
    var hPadded = [Double](repeating: 0.0, count: nPrePad)
    hPadded.append(contentsOf: h)
    if nPostPad > 0 {
        hPadded.append(contentsOf: [Double](repeating: 0.0, count: nPostPad))
    }

    let y = upfirdnEdgeDouble(h: hPadded, x: xDouble, up: up, down: down)
    let start = nPreRemove
    let end = nPreRemove + nOut
    let yKeep = Array(y[start..<end]).map { Float($0) }
    return MLXArray(yKeep)
}

private func outputLen(_ lenH: Int, _ lenX: Int, _ up: Int, _ down: Int) -> Int {
    return ((lenX * up + lenH - up - 1) / down) + 1
}

private func padHDouble(_ h: [Double], up: Int) -> ([Double], Int) {
    let padLen = h.count + ((up - (h.count % up)) % up)
    var hFull = [Double](repeating: 0.0, count: padLen)
    for i in 0..<h.count {
        hFull[i] = h[i]
    }
    let nPhase = padLen / up
    var hTransFlip = [Double](repeating: 0.0, count: padLen)
    for p in 0..<up {
        for q in 0..<nPhase {
            let src = p + up * q
            let dst = p * nPhase + (nPhase - 1 - q)
            hTransFlip[dst] = hFull[src]
        }
    }
    return (hTransFlip, nPhase)
}

private func extendLeftEdgeDouble(_ x: [Double]) -> Double {
    return x.first ?? 0.0
}

private func extendRightEdgeDouble(_ x: [Double]) -> Double {
    return x.last ?? 0.0
}

private func upfirdnEdgeDouble(h: [Double], x: [Double], up: Int, down: Int) -> [Double] {
    let (hTransFlip, _) = padHDouble(h, up: up)
    let lenH = hTransFlip.count
    let hPerPhase = lenH / up
    let lenOut = outputLen(h.count, x.count, up, down)
    if lenOut <= 0 {
        return []
    }
    var out = [Double](repeating: 0.0, count: lenOut)
    let lenX = x.count
    let paddedLen = lenX + hPerPhase - 1
    var xIdx = 0
    var yIdx = 0
    var t = 0

    while xIdx < lenX {
        var hIdx = t * hPerPhase
        var xConvIdx = xIdx - hPerPhase + 1
        if xConvIdx < 0 {
            for _ in xConvIdx..<0 {
                let xVal = extendLeftEdgeDouble(x)
                out[yIdx] += xVal * hTransFlip[hIdx]
                hIdx += 1
            }
            xConvIdx = 0
        }
        if xConvIdx <= xIdx {
            for idx in xConvIdx...xIdx {
                out[yIdx] += x[idx] * hTransFlip[hIdx]
                hIdx += 1
            }
        }
        yIdx += 1
        if yIdx >= lenOut {
            return out
        }
        t += down
        xIdx += t / up
        t = t % up
    }

    while xIdx < paddedLen {
        var hIdx = t * hPerPhase
        let xConvIdx = xIdx - hPerPhase + 1
        if xConvIdx <= xIdx {
            for idx in xConvIdx...xIdx {
                let xVal: Double
                if idx >= lenX {
                    xVal = extendRightEdgeDouble(x)
                } else if idx < 0 {
                    xVal = extendLeftEdgeDouble(x)
                } else {
                    xVal = x[idx]
                }
                out[yIdx] += xVal * hTransFlip[hIdx]
                hIdx += 1
            }
        }
        yIdx += 1
        if yIdx >= lenOut {
            return out
        }
        t += down
        xIdx += t / up
        t = t % up
    }
    return out
}

private func firwinLowpassDouble(numtaps: Int, cutoff: Double, beta: Double) -> [Double] {
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
