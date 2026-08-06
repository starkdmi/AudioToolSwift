//
//  AudioLoaderParityTests.swift
//  AudioToolMLXIntegrationTests
//
//  The package's resampler must agree with the AudioLoader it was copied from.
//

import XCTest
import AVFoundation
import AudioToolCore
import AudioUtils
import MLX
@testable import AudioTool

/// `AudioBuffer.resampled(to:quality:)` against `AudioLoader`, on the same audio.
///
/// There are two resampler implementations in play. `AudioLoader` in SwiftAudio is
/// the one every standalone model generator ran through, so it is what the ports
/// were validated with. `AudioBuffer+Resampling` is this package's own copy, which
/// exists to avoid an MLXArray round trip at the facade's edge. Its own header says
/// "This is a second copy of the implementation in SwiftAudio's AudioUtils; the two
/// should be reconciled rather than left to drift."
///
/// They had already drifted, in a way that was easy to miss. `.auto` was implemented
/// here as `Normal` at `.medium`, copied from the local SwiftAudio working tree -
/// which carries an unreleased change. The pinned dependency, 1.0.0, uses `Normal`
/// at `.high`. So the seam that exists to reproduce the loader was reproducing a
/// version of the loader that has never been published.
///
/// Comparing the two implementations directly is the only check that survives that
/// class of mistake: it does not care what anyone believes the setting is, and it
/// starts failing the moment the dependency moves.
final class AudioLoaderParityTests: MLXTestCase {

    /// Write a mono float WAV, since `AudioLoader` takes a file.
    private func writeWAV(_ samples: [Float], sampleRate: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("parity-\(UUID().uuidString).wav")

        let format = try XCTUnwrap(AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
            channels: 1, interleaved: false))
        let file = try AVAudioFile(forWriting: url, settings: format.settings,
                                   commonFormat: .pcmFormatFloat32, interleaved: false)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)))
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer {
            buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count)
        }
        try file.write(from: buffer)
        return url
    }

    private func signal(count: Int, sampleRate: Int) -> [Float] {
        (0..<count).map { i in
            let t = Float(i) / Float(sampleRate)
            var sum: Float = 0
            for frequency in stride(from: Float(180), to: Float(9000), by: 430) {
                sum += sin(2 * .pi * frequency * t)
            }
            return sum / 21
        }
    }

    private func rms(_ values: [Float]) -> Float {
        sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(max(values.count, 1)))
    }

    /// Run both paths at the same rate change and require them to agree.
    private func assertParity(from: Int, to: Int, line: UInt = #line) throws {
        let source = signal(count: from * 2, sampleRate: from)
        let url = try writeWAV(source, sampleRate: from)
        defer { try? FileManager.default.removeItem(at: url) }

        // The reference: AudioLoader, with the default resampling method.
        let loader = AudioLoader(config: .init(
            targetSampleRate: Double(to),
            normalizationMode: .none
        ))
        let loaded = try loader.loadMono(from: url)
        eval(loaded)
        let viaLoader = loaded.asArray(Float.self)

        // This package's copy, asked for the same thing.
        let viaBuffer = try AudioBuffer(samples: source, sampleRate: from, channels: 1)
            .resampled(to: to, quality: .auto)

        // AudioLoader reads slightly fewer frames from the file than were written -
        // a decoder artifact, not a resampler difference - so compare the common
        // prefix rather than requiring equal lengths. Truncation at the tail does
        // not shift the head, which is what makes a prefix comparison meaningful.
        let count = min(viaBuffer.samples.count, viaLoader.count)
        XCTAssertGreaterThan(count, 0, line: line)
        let difference = rms((0..<count).map { viaBuffer.samples[$0] - viaLoader[$0] })
        let scale = max(rms(Array(viaLoader.prefix(count))), 1e-9)

        XCTAssertLessThan(difference / scale, 0.01,
                          "\(from) -> \(to): `.auto` differs from AudioLoader by \(difference / scale) relative RMS - the two resamplers have drifted, most likely a converter algorithm or quality setting",
                          line: line)
    }

    /// Downsampling is where the setting matters; every model rate pair that a
    /// caller is likely to hit.
    func testDownsamplingMatchesAudioLoader() throws {
        try assertParity(from: 48000, to: 16000)
        try assertParity(from: 44100, to: 16000)
        try assertParity(from: 48000, to: 44100)
        try assertParity(from: 32000, to: 16000)
    }

    /// Upsampling takes the cubic branch in both, so it should agree closely too.
    func testUpsamplingMatchesAudioLoader() throws {
        try assertParity(from: 16000, to: 48000)
        try assertParity(from: 16000, to: 44100)
    }
}
