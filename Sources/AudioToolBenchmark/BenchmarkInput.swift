//
//  BenchmarkInput.swift
//  AudioToolBenchmark
//
//  The signal every case is fed, and why it is synthetic by default.
//

import Foundation
import AudioToolCore
@preconcurrency import AudioUtils
@preconcurrency import MLX

/// Where a run's audio comes from.
///
/// Synthetic by default, and that is the interesting choice. A benchmark that has
/// to travel between machines cannot depend on a fixture that only exists in one
/// checkout, and the two committed fixture directories here are test inputs with
/// licences attached, not benchmark material. The generator below is forty lines
/// of arithmetic with a fixed seed, so every machine measures the same samples
/// without downloading anything.
///
/// This is sound because inference cost in this catalog is a function of length
/// and sample rate, not content: these are convolutional and attention models run
/// over fixed-size chunks, with no early exit and no data-dependent branching. The
/// one thing synthetic input cannot measure is output *quality*. Completed
/// cross-implementation comparisons are recorded in `Docs/conversion.md`, and
/// nothing here duplicates them.
///
/// Pass `--input` for a real file when the question is about a specific recording.
public enum BenchmarkInput {

    /// Deterministic pseudo-speech: a harmonic stack under a syllable-rate envelope,
    /// plus a little broadband hiss.
    ///
    /// Not white noise. A denoiser fed white noise does roughly the same work as one
    /// fed speech, but the STFT magnitudes it produces are unrepresentative, and any
    /// future case with a magnitude-dependent path - a VAD gate, a silence skip -
    /// would be measured on input it will never see. Harmonics plus an envelope
    /// costs nothing extra to generate and removes that whole class of surprise.
    ///
    /// - Parameters:
    ///   - seconds: length of audio to produce.
    ///   - sampleRate: rate to produce it at.
    public static func synthetic(
        seconds: Double,
        sampleRate: Int,
        channels: Int = 1
    ) -> AudioBuffer {
        precondition(seconds > 0, "input length must be positive")
        precondition(sampleRate > 0, "sample rate must be positive")
        precondition(channels >= 1, "channel count must be positive")

        let frameCount = Int(seconds * Double(sampleRate))
        var samples = [Float](repeating: 0, count: frameCount)

        // xorshift64*, seeded identically on every machine. Foundation's RNG is
        // not reproducible and `arc4random` is not seedable, so neither can be
        // used for input that must be byte-identical across hosts.
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        func noise() -> Double {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Double(state % 2_000_003) / 1_000_001.5 - 1.0
        }

        // A 110 Hz fundamental with harmonics to just under Nyquist, so the same
        // routine is band-appropriate at 8 kHz and at 48 kHz without aliasing.
        let fundamental = 110.0
        let ceiling = Double(sampleRate) * 0.45
        let harmonics = max(1, Int(ceiling / fundamental))

        for index in 0..<frameCount {
            let t = Double(index) / Double(sampleRate)
            // ~3.5 syllables per second, never fully silent so no case can take a
            // silence shortcut on part of the input and not on another.
            let envelope = 0.35 + 0.65 * abs(sin(2 * .pi * 3.5 * t))
            var value = 0.0
            for harmonic in 1...harmonics {
                let frequency = fundamental * Double(harmonic)
                // 1/n rolloff: a sawtooth-like spectrum, which is what voiced
                // speech looks like before the vocal tract shapes it.
                value += sin(2 * .pi * frequency * t + Double(harmonic)) / Double(harmonic)
            }
            value = value / 3.0 * envelope + 0.01 * noise()
            samples[index] = Float(max(-1.0, min(1.0, value * 0.5)))
        }

        guard channels > 1 else {
            return AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
        }
        return interleave(samples, into: channels, sampleRate: sampleRate)
    }

    /// Spread a mono signal across `channels`, interleaved.
    ///
    /// Channels beyond the first are delayed by a few samples rather than copied.
    /// Dual-mono is a degenerate input for a separator - Demucs works on the
    /// difference between channels, and handing it two identical ones measures a
    /// case its callers do not have. A small delay is the cheapest way to make the
    /// channels genuinely distinct without pretending to model a stereo field.
    private static func interleave(
        _ mono: [Float],
        into channels: Int,
        sampleRate: Int
    ) -> AudioBuffer {
        let frames = mono.count
        var interleaved = [Float](repeating: 0, count: frames * channels)
        for frame in 0..<frames {
            for channel in 0..<channels {
                let delay = channel * 13  // ~0.3 ms at 44.1 kHz
                let source = frame >= delay ? mono[frame - delay] : 0
                interleaved[frame * channels + channel] = source
            }
        }
        return AudioBuffer(samples: interleaved, sampleRate: sampleRate, channels: channels)
    }

    /// Load a file, resampled to `sampleRate` and trimmed or looped to `seconds`.
    ///
    /// Looped rather than zero-padded when the file is short: padding would hand
    /// the model silence and quietly lower the measured cost of any case whose
    /// input happened to be shorter than the requested length.
    public static func file(
        at url: URL,
        seconds: Double,
        sampleRate: Int,
        channels: Int = 1
    ) throws -> AudioBuffer {
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: Double(sampleRate),
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: url)
        eval(audio)
        let source = audio.asArray(Float.self)
        guard !source.isEmpty else {
            throw BenchmarkInputError.emptyFile(url.path)
        }

        let wanted = Int(seconds * Double(sampleRate))
        var samples = [Float]()
        samples.reserveCapacity(wanted)
        while samples.count < wanted {
            samples.append(contentsOf: source.prefix(wanted - samples.count))
        }
        guard channels > 1 else {
            return AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
        }
        // `loadMono` is the only loader the providers use, so a multi-channel case
        // gets the same delayed spread the synthetic path produces rather than a
        // second decode at a different channel count.
        return interleave(samples, into: channels, sampleRate: sampleRate)
    }

    /// Input for one case, honouring the run's source setting.
    ///
    /// `channels` comes from the case, not the run. It was originally a workaround:
    /// `DemucsProvider` rejected mono outright, so a benchmark feeding every model a
    /// mono buffer measured ten models and reported `channelCountMismatch` for two.
    /// That guard is gone - Demucs converts any layout itself now - but the per-case
    /// count stays, for the reason that should have been the reason all along: a
    /// music separator works on stereo cues, so measuring it on a duplicated mono
    /// channel would measure something the model is not for.
    public static func make(
        source: Source,
        seconds: Double,
        sampleRate: Int,
        channels: Int = 1
    ) throws -> AudioBuffer {
        switch source {
        case .synthetic:
            return synthetic(seconds: seconds, sampleRate: sampleRate, channels: channels)
        case .file(let url):
            return try file(at: url, seconds: seconds, sampleRate: sampleRate, channels: channels)
        }
    }

    public enum Source: Sendable, Equatable {
        case synthetic
        case file(URL)

        /// How this appears in ``RunConfiguration/inputSource``.
        public var description: String {
            switch self {
            case .synthetic: "synthetic"
            case .file(let url): url.path
            }
        }
    }
}

public enum BenchmarkInputError: Error, CustomStringConvertible {
    case emptyFile(String)

    public var description: String {
        switch self {
        case .emptyFile(let path): "input file decoded to zero samples: \(path)"
        }
    }
}
