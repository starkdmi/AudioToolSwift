//
//  SuperResolutionResampler.swift
//  AudioToolMLX
//
//  The 16 -> 48 kHz upsample MossFormer2 SR runs before inference.
//

import AVFoundation
import AudioToolCore
import Foundation

/// Upsampling for MossFormer2 SR.
///
/// Its own file, and its own resampler, for two reasons.
///
/// The file, because `import AVFoundation` alongside `AudioToolCore` makes
/// `AudioBuffer` ambiguous - CoreAudioTypes has one too. Keeping AVFoundation
/// out of the provider avoids qualifying every mention of ours.
///
/// The resampler, because SR is the only model here where resampling *is*
/// inference. Every other provider takes audio already at its rate; this one
/// reconstructs the band 16 kHz cannot carry, so what the upsample preserves is
/// what the model has to work with. That is why it asks for Mastering at maximum
/// quality rather than a general-purpose default, matching the standalone
/// generator whose numbers this model was validated against
/// (`Models/mossformer2_sr_mlx_swift/Sources/Generate/main.swift:26`).
///
/// What this replaced, and what each part cost:
///
/// - A round trip through the filesystem. The buffer was written to a temp wav
///   and read back purely to reach `AudioLoader`, which has no in-memory entry
///   point. An encode and a decode to perform a resample.
/// - `AudioLoader`'s default resampling method, not Mastering/`.max`. The
///   provider declared `preferredResamplingQuality = .high` and then never
///   consulted it, because the old code built its own configuration.
/// - No drain. `AVAudioConverter` withholds frames for its own latency, so a
///   single `convert` call returns short - 48000 samples at 16 kHz produced
///   141312 at 48 kHz where 144000 is the only correct answer. 2688 samples,
///   56 ms, silently dropped off the end of every 16 kHz call.
///
/// Against the reference upsample the old path agreed at 31.1 dB over the
/// samples it did produce.
enum SuperResolutionResampler {

    /// Upsample mono float samples, preserving the exact ratio.
    ///
    /// - Returns: exactly `samples.count * to / from` samples.
    static func upsample(_ samples: [Float], from sourceRate: Int, to targetRate: Int) throws -> [Float] {
        guard sourceRate != targetRate else { return samples }
        guard !samples.isEmpty else { return [] }

        let source = Double(sourceRate)
        let target = Double(targetRate)

        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: source, channels: 1, interleaved: false
        ), let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: target, channels: 1, interleaved: false
        ) else {
            throw AudioToolError.resourceUnavailable("SR resampler: could not create audio formats")
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioToolError.resourceUnavailable("SR resampler: could not create converter")
        }
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue

        guard let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(samples.count)
        ), let sourceChannel = sourceBuffer.floatChannelData else {
            throw AudioToolError.resourceUnavailable("SR resampler: could not allocate input buffer")
        }
        sourceBuffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer {
            sourceChannel[0].update(from: $0.baseAddress!, count: samples.count)
        }

        let expected = Int((Double(samples.count) * target / source).rounded())
        var resampled = [Float]()
        resampled.reserveCapacity(expected)

        let feed = SingleShotInput(sourceBuffer)

        // Keep pulling until the converter says it has nothing left. One call is
        // not enough: the tail sits inside the converter until end of stream.
        while resampled.count < expected {
            guard let sink = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 16384) else {
                throw AudioToolError.resourceUnavailable(
                    "SR resampler: could not allocate output buffer"
                )
            }

            var failure: NSError?
            let status = converter.convert(to: sink, error: &failure, withInputFrom: feed.next)
            if let failure {
                throw AudioToolError.stageFailed(stage: "MossFormer2SR.resample", underlying: failure)
            }
            if let produced = sink.floatChannelData, sink.frameLength > 0 {
                resampled.append(contentsOf: UnsafeBufferPointer(
                    start: produced[0], count: Int(sink.frameLength)
                ))
            }
            // `.haveData` having produced nothing would spin forever.
            if status == .endOfStream || status == .error || sink.frameLength == 0 { break }
        }

        // Latency compensation can land a sample or two either side of the exact
        // ratio. Pin it: an integer-ratio upsample should be exact, and callers
        // downstream size their buffers from it.
        if resampled.count > expected {
            resampled.removeLast(resampled.count - expected)
        } else if resampled.count < expected {
            resampled.append(contentsOf: [Float](repeating: 0, count: expected - resampled.count))
        }
        return resampled
    }
}

/// Hands the converter the whole input once, then reports end of stream.
///
/// A class rather than a captured `var` because `AVAudioConverterInputBlock` is
/// `@Sendable` under Swift 6 strict concurrency. `@unchecked` is honest here: the
/// block runs synchronously on the thread that called `convert`, never
/// concurrently, so the single mutable flag has no second reader.
private final class SingleShotInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private var supplied = false

    init(_ buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(
        _ requested: AVAudioPacketCount,
        _ status: UnsafeMutablePointer<AVAudioConverterInputStatus>
    ) -> AVAudioBuffer? {
        guard !supplied else {
            status.pointee = .endOfStream
            return nil
        }
        supplied = true
        status.pointee = .haveData
        return buffer
    }
}
