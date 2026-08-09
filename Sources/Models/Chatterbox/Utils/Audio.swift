import Foundation
import AudioUtils
import MLX

// MARK: - Resampling

/// Resample with `scipy.signal.resample_poly`'s own filter.
///
/// For call sites whose reference is scipy. `S3Gen.embed_ref` resamples with exactly
/// this on the Python side, so reproducing it there is correctness, not inertia.
///
/// It is not a good filter: the cutoff sits at Nyquist, where a lowpass is only
/// -6 dB, and 61 taps give it no room to reach stopband. Measured alias rejection is
/// -20.9 dB against -55.8 dB for ``resampleAudioKaiserBest(_:origSR:targetSR:)``, and
/// the S3 tokenizer resolves that 35 dB into different speech tokens. Everywhere the
/// reference is not scipy, prefer the other one.
public func resampleAudioPolyphase(_ audio: MLXArray, origSR: Int, targetSR: Int) -> MLXArray {
    resampleAudio(audio, origSR: origSR, targetSR: targetSR, design: .scipyDefault)
}

/// Resample with a band-limited sinc designed for -120 dB stopband.
///
/// The default choice for anything feeding a model. See `ResamplerDesign.kaiserBest`.
public func resampleAudioKaiserBest(_ audio: MLXArray, origSR: Int, targetSR: Int) -> MLXArray {
    resampleAudio(audio, origSR: origSR, targetSR: targetSR, design: .kaiserBest)
}

/// Chatterbox's entry point into AudioUtils' polyphase resampler.
///
/// The engine, the filter designs and the `resample_poly(padtype: "edge")` semantics
/// all live in `AudioUtils.PolyphaseResampling`; there is nothing model-specific
/// about any of it, and `AudioUtilsTests.PolyphaseResamplingTests` is what pins the
/// arithmetic. What is specific to this model is only *which* design each call site
/// gets, which is the pair of functions above.
public func resampleAudio(
    _ audio: MLXArray, origSR: Int, targetSR: Int, design: ResamplerDesign
) -> MLXArray {
    if origSR == targetSR {
        return audio
    }
    let samples = audio.asArray(Float.self)
    if samples.isEmpty {
        return audio
    }
    return MLXArray(
        AudioUtils.resamplePolyphase(samples, fromRate: origSR, toRate: targetSR, design: design)
    )
}

// MARK: - Silence trimming

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
