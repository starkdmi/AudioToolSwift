//
//  AudioBuffer+Saving.swift
//  AudioTool
//
//  Writing AudioBuffer to a container, channel count intact.
//

import Foundation
@preconcurrency import AVFoundation
import AudioToolCore

/// Writes an ``AudioBuffer`` to disk in the container the caller asked for.
///
/// This replaces a hand-off to `AudioUtils.AudioSaver`, which is mono-only: it
/// creates its `AVAudioFormat` with `channels: 1` regardless of the source. An
/// interleaved stereo buffer routed through it came back as alternating left and
/// right samples in a single channel - a recording of roughly twice the duration,
/// played at half speed with the channels comb-filtered together.
///
/// The format mapping is also honest here. Both MP3 and FLAC used to fall back to a
/// WAV encoding while keeping the requested file extension, so a `.flac` path held
/// WAV bytes. FLAC is a real CoreAudio encoder and is used as such; MP3 has no
/// encoder on Apple platforms and is rejected rather than silently mislabelled.
enum AudioFileWriter {

    /// Write `buffer` to `url` in `format`.
    ///
    /// - Throws: ``AudioToolError/invalidAudioFormat(expected:found:)`` for `.mp3`,
    ///   which AVFoundation can decode but not encode.
    static func write(
        _ buffer: AudioToolCore.AudioBuffer,
        to url: URL,
        format: AudioFormat
    ) throws {
        guard !buffer.samples.isEmpty else {
            throw AudioToolError.emptyAudioBuffer
        }
        guard buffer.channels > 0 else {
            throw AudioToolError.channelCountMismatch(expected: 1, found: buffer.channels)
        }

        let settings = try outputSettings(
            for: format,
            sampleRate: Double(buffer.sampleRate),
            channels: buffer.channels
        )

        // The source is interleaved Float32 by definition of `AudioBuffer`. Declaring
        // the processing format that way lets one memcpy-shaped copy do the work and
        // leaves any conversion to the encoder behind `AVAudioFile`.
        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(buffer.sampleRate),
            channels: AVAudioChannelCount(buffer.channels),
            interleaved: true
        ) else {
            throw AudioToolError.invalidAudioFormat(
                expected: "float32 PCM",
                found: "\(buffer.sampleRate) Hz, \(buffer.channels) ch"
            )
        }

        let frameCount = AVAudioFrameCount(buffer.frameCount)
        guard frameCount > 0, let pcm = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: frameCount
        ) else {
            throw AudioToolError.emptyAudioBuffer
        }
        pcm.frameLength = frameCount

        guard let destination = pcm.floatChannelData?[0] else {
            throw AudioToolError.resourceUnavailable("Could not access output buffer storage")
        }
        buffer.samples.withUnsafeBufferPointer { source in
            destination.update(from: source.baseAddress!, count: source.count)
        }

        let file: AVAudioFile
        do {
            file = try AVAudioFile(
                forWriting: url,
                settings: settings,
                commonFormat: .pcmFormatFloat32,
                interleaved: true
            )
        } catch {
            throw AudioToolError.resourceUnavailable(
                "Could not create audio file at \(url.path): \(error.localizedDescription)"
            )
        }

        do {
            try file.write(from: pcm)
        } catch {
            throw AudioToolError.resourceUnavailable(
                "Could not write audio to \(url.path): \(error.localizedDescription)"
            )
        }
    }

    /// Encoder settings per container. Sample rate and channel count always come
    /// from the buffer, never from a fixed configuration.
    private static func outputSettings(
        for format: AudioFormat,
        sampleRate: Double,
        channels: Int
    ) throws -> [String: Any] {
        var settings: [String: Any] = [
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channels
        ]

        switch format {
        case .wav:
            settings[AVFormatIDKey] = kAudioFormatLinearPCM
            settings[AVLinearPCMBitDepthKey] = 32
            settings[AVLinearPCMIsFloatKey] = true
            settings[AVLinearPCMIsBigEndianKey] = false
            settings[AVLinearPCMIsNonInterleaved] = false
        case .m4a:
            settings[AVFormatIDKey] = kAudioFormatMPEG4AAC
            settings[AVEncoderBitRateKey] = 128_000
        case .flac:
            settings[AVFormatIDKey] = kAudioFormatFLAC
        case .mp3:
            // kAudioFormatMPEGLayer3 is decode-only on Apple platforms. Writing WAV
            // bytes into a `.mp3` file, which is what the previous mapping did, is
            // worse than refusing: the file is undiagnosably broken downstream.
            throw AudioToolError.invalidAudioFormat(
                expected: "wav/m4a/flac",
                found: "mp3 (no encoder available on this platform)"
            )
        }

        return settings
    }
}
