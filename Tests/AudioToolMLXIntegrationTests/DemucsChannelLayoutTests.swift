//
//  DemucsChannelLayoutTests.swift
//  AudioToolMLXIntegrationTests
//
//  Channel handling for Demucs - no weights, no network.
//

import XCTest
import AudioToolTestSupport
import MLX
import AudioToolCore
@testable import AudioToolMLX

/// How an ``AudioBuffer`` becomes the `(2, frames)` planar pair HTDemucs expects.
///
/// Deliberately not an `IntegrationTestCase`: this is array arithmetic, it needs no
/// weights, and it is the part of the stereo path that was silently wrong for as
/// long as the code existed.
final class DemucsChannelLayoutTests: MLXTestCase {

    /// The regression. `AudioBuffer.samples` is interleaved for multichannel audio,
    /// but the stereo branch read it as planar via `reshaped([2, -1])`: channel 0
    /// came out as the first half of the recording and channel 1 as the second half.
    /// Demucs separates *using* stereo cues, so it was being handed two unrelated
    /// halves of the song and asked to find the vocals across them.
    func testStereoIsDeinterleavedNotSplitInHalf() throws {
        // L = 1, 2, 3, 4  R = -1, -2, -3, -4, interleaved.
        let interleaved: [Float] = [1, -1, 2, -2, 3, -3, 4, -4]
        let buffer = AudioBuffer(samples: interleaved, sampleRate: 44100, channels: 2)

        let pair = DemucsProvider.stereoPair(from: buffer)
        eval(pair)

        XCTAssertEqual(pair.shape, [2, 4])

        let left = pair[0].asArray(Float.self)
        let right = pair[1].asArray(Float.self)

        XCTAssertEqual(left, [1, 2, 3, 4], "left channel should be every even-indexed sample")
        XCTAssertEqual(right, [-1, -2, -3, -4], "right channel should be every odd-indexed sample")

        // What the old code produced, for the record: the first half and second half.
        XCTAssertNotEqual(left, [1, -1, 2, -2],
                          "splitting the flat buffer in half is the defect this guards")
    }

    func testMonoIsDuplicatedToBothChannels() throws {
        let buffer = AudioBuffer(samples: [1, 2, 3, 4], sampleRate: 44100, channels: 1)

        let pair = DemucsProvider.stereoPair(from: buffer)
        eval(pair)

        XCTAssertEqual(pair.shape, [2, 4])
        XCTAssertEqual(pair[0].asArray(Float.self), [1, 2, 3, 4])
        XCTAssertEqual(pair[1].asArray(Float.self), [1, 2, 3, 4])
    }

    /// The model has exactly two input channels, so anything wider is downmixed
    /// rather than truncated - dropping channels 2..N would discard content.
    func testMoreThanTwoChannelsIsDownmixed() throws {
        // Three channels: 0, 3, 6 / 1, 4, 7 / 2, 5, 8 interleaved.
        let interleaved: [Float] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        let buffer = AudioBuffer(samples: interleaved, sampleRate: 44100, channels: 3)

        let pair = DemucsProvider.stereoPair(from: buffer)
        eval(pair)

        XCTAssertEqual(pair.shape, [2, 3])
        let left = pair[0].asArray(Float.self)
        // Frame means: (0+1+2)/3, (3+4+5)/3, (6+7+8)/3
        XCTAssertEqual(left, [1, 4, 7])
        XCTAssertEqual(pair[1].asArray(Float.self), left, "downmix is duplicated to both channels")
    }

    /// Frame count, not sample count, decides whether the chunked path is taken.
    /// With stereo the two differ by a factor of two, and using the sample count
    /// meant a 5-second stereo clip was treated as 10 seconds of audio.
    func testFrameCountDrivesDurationForStereo() throws {
        let frames = 1000
        let buffer = AudioBuffer(samples: [Float](repeating: 0.1, count: frames * 2),
                                 sampleRate: 44100,
                                 channels: 2)

        let pair = DemucsProvider.stereoPair(from: buffer)
        eval(pair)

        XCTAssertEqual(pair.shape[1], frames)
        XCTAssertEqual(pair.shape[1], buffer.frameCount)
    }

    func testRangedConversionMaterializesOnlyRequestedFrames() throws {
        let interleaved: [Float] = [1, -1, 2, -2, 3, -3, 4, -4]
        let buffer = AudioBuffer(samples: interleaved, sampleRate: 44100, channels: 2)

        let pair = DemucsProvider.stereoPair(from: buffer, frames: 1..<3)
        eval(pair)

        XCTAssertEqual(pair.shape, [2, 2])
        XCTAssertEqual(pair[0].asArray(Float.self), [2, 3])
        XCTAssertEqual(pair[1].asArray(Float.self), [-2, -3])
    }
}
