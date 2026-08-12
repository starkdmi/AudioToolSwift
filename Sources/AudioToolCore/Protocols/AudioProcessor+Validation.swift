//
//  AudioProcessor+Validation.swift
//  AudioToolCore
//
//  Input-format precondition shared by every provider
//

import Foundation

public extension AudioProcessor {

    /// Verify audio arrives in the exact format this processor consumes.
    ///
    /// Providers validate rather than resample. Resampling inside a provider looks
    /// convenient but hides its cost and, in a chain, compounds: with every stage
    /// silently converting to its own rate and back again, `enhance` at 16 kHz
    /// followed by `upscale` on 48 kHz input ran 48 → 16 → 48 → 16 → 48, handing the
    /// super-resolution model - whose entire job is reconstructing detail lost below
    /// 16 kHz - audio that had just been through a round trip.
    ///
    /// So a provider is exact about its rate and the caller adapts, which lets the
    /// facade and the pipeline convert once at the edge instead of once per stage.
    ///
    /// - Throws: ``AudioToolError/sampleRateMismatch(expected:found:)`` or
    ///   ``AudioToolError/channelCountMismatch(expected:found:)``.
    func validateSampleRate(_ audio: AudioBuffer) throws {
        guard audio.sampleRate == sampleRate else {
            throw AudioToolError.sampleRateMismatch(
                expected: sampleRate,
                found: audio.sampleRate
            )
        }
        try validateInputChannels(audio)
    }

    /// Validate only the channel layout. This is for the small number of legacy
    /// provider APIs that intentionally perform their own sample-rate conversion;
    /// they still must never interpret interleaved multichannel storage as mono.
    func validateInputChannels(_ audio: AudioBuffer) throws {
        guard audio.channels == inputChannels else {
            throw AudioToolError.channelCountMismatch(
                expected: inputChannels,
                found: audio.channels
            )
        }
    }

    /// Validate only the sample rate, for a provider that adapts any channel layout
    /// itself.
    ///
    /// The rate is never negotiable - resampling inside a provider is the thing the
    /// note above exists to prevent - but the channel count sometimes is. A provider
    /// may declare ``AudioProcessor/inputChannels`` to describe the tensor its model
    /// takes while accepting anything it can correctly convert to that tensor, and
    /// then this is the check it wants: rejecting a layout it already knows how to
    /// handle would make its own conversion unreachable.
    ///
    /// Use it only where that conversion demonstrably exists and understands
    /// interleaving. The default remains ``validateSampleRate(_:)``.
    func validateInputRate(_ audio: AudioBuffer) throws {
        guard audio.sampleRate == sampleRate else {
            throw AudioToolError.sampleRateMismatch(
                expected: sampleRate,
                found: audio.sampleRate
            )
        }
    }


    /// Preferred spelling for new code. `validateSampleRate` remains as a source-
    /// compatible alias because providers outside this package already call it.
    func validateInputFormat(_ audio: AudioBuffer) throws {
        try validateSampleRate(audio)
    }
}
