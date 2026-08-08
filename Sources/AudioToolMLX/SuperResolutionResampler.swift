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

    /// Incremental mono sample-rate conversion. The source array uses copy-on-write
    /// storage and only a small AVAudioPCMBuffer is allocated for each converter
    /// request, so long-form callers never materialize a second full-rate input.
    final class Stream {
        let expectedFrameCount: Int

        private let sourceSamples: [Float]
        private let targetFormat: AVAudioFormat?
        private let converter: AVAudioConverter?
        private let feeder: ChunkedConverterInput?
        private var sourcePosition = 0
        private var emittedFrameCount = 0
        private var converterFinished = false

        init(_ samples: [Float], from sourceRate: Int, to targetRate: Int) throws {
            guard sourceRate > 0, targetRate > 0 else {
                throw AudioToolError.invalidAudioFormat(
                    expected: "positive source and target sample rates",
                    found: "\(sourceRate)Hz -> \(targetRate)Hz"
                )
            }

            sourceSamples = samples
            expectedFrameCount = Int(
                (Double(samples.count) * Double(targetRate) / Double(sourceRate)).rounded()
            )

            guard sourceRate != targetRate else {
                targetFormat = nil
                converter = nil
                feeder = nil
                return
            }

            guard let inputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sourceRate),
                channels: 1,
                interleaved: false
            ), let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(targetRate),
                channels: 1,
                interleaved: false
            ) else {
                throw AudioToolError.resourceUnavailable("SR resampler: could not create audio formats")
            }
            guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
                throw AudioToolError.resourceUnavailable("SR resampler: could not create converter")
            }
            converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
            converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue

            targetFormat = outputFormat
            self.converter = converter
            feeder = ChunkedConverterInput(samples: samples, format: inputFormat)
        }

        /// Returns up to `maxFrames`, or nil once the exact mathematical output
        /// length has been produced.
        func next(maxFrames: Int) throws -> [Float]? {
            guard maxFrames > 0, emittedFrameCount < expectedFrameCount else { return nil }
            let wanted = min(maxFrames, expectedFrameCount - emittedFrameCount)

            // Same-rate streams are a bounded view/copy and need no converter.
            guard let converter, let targetFormat, let feeder else {
                let end = min(sourcePosition + wanted, sourceSamples.count)
                var result = Array(sourceSamples[sourcePosition..<end])
                sourcePosition = end
                if result.count < wanted {
                    result.append(contentsOf: repeatElement(0, count: wanted - result.count))
                }
                emittedFrameCount += result.count
                return result
            }

            if converterFinished {
                emittedFrameCount += wanted
                return [Float](repeating: 0, count: wanted)
            }

            var attempts = 0
            while attempts < 8 {
                attempts += 1
                guard let sink = AVAudioPCMBuffer(
                    pcmFormat: targetFormat,
                    frameCapacity: AVAudioFrameCount(wanted)
                ) else {
                    throw AudioToolError.resourceUnavailable("SR resampler: could not allocate output buffer")
                }

                var failure: NSError?
                let status = converter.convert(to: sink, error: &failure, withInputFrom: feeder.next)
                if let failure {
                    throw AudioToolError.stageFailed(stage: "MossFormer2SR.resample", underlying: failure)
                }
                if status == .error {
                    throw AudioToolError.resourceUnavailable("SR resampler: converter failed without an error")
                }
                if status == .endOfStream {
                    converterFinished = true
                }

                if let channels = sink.floatChannelData, sink.frameLength > 0 {
                    let count = min(Int(sink.frameLength), wanted)
                    let result = Array(UnsafeBufferPointer(start: channels[0], count: count))
                    emittedFrameCount += result.count
                    return result
                }

                if converterFinished {
                    emittedFrameCount += wanted
                    return [Float](repeating: 0, count: wanted)
                }
            }

            throw AudioToolError.resourceUnavailable("SR resampler made no forward progress")
        }
    }

    /// Upsample mono float samples, preserving the exact ratio.
    ///
    /// - Returns: exactly `samples.count * to / from` samples.
    static func upsample(_ samples: [Float], from sourceRate: Int, to targetRate: Int) throws -> [Float] {
        let stream = try Stream(samples, from: sourceRate, to: targetRate)
        var resampled = [Float]()
        resampled.reserveCapacity(stream.expectedFrameCount)
        while let chunk = try stream.next(maxFrames: 16_384) {
            resampled.append(contentsOf: chunk)
        }
        return resampled
    }
}

/// Bounded source feeder for AVAudioConverter. The callback is invoked
/// synchronously by `convert`; the unchecked Sendable conformance documents that
/// confinement and avoids claiming AVAudioPCMBuffer itself is generally Sendable.
private final class ChunkedConverterInput: @unchecked Sendable {
    private let samples: [Float]
    private let format: AVAudioFormat
    private let maximumFramesPerBuffer = 16_384
    private var position = 0

    init(samples: [Float], format: AVAudioFormat) {
        self.samples = samples
        self.format = format
    }

    func next(
        _ requested: AVAudioPacketCount,
        _ status: UnsafeMutablePointer<AVAudioConverterInputStatus>
    ) -> AVAudioBuffer? {
        guard position < samples.count else {
            status.pointee = .endOfStream
            return nil
        }

        let requestedFrames = max(1, Int(requested))
        let count = min(maximumFramesPerBuffer, requestedFrames, samples.count - position)
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(count)
        ), let channels = buffer.floatChannelData else {
            status.pointee = .noDataNow
            return nil
        }
        buffer.frameLength = AVAudioFrameCount(count)
        samples.withUnsafeBufferPointer { source in
            channels[0].update(from: source.baseAddress! + position, count: count)
        }
        position += count
        status.pointee = .haveData
        return buffer
    }
}
