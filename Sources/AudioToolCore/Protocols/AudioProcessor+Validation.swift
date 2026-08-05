//
//  AudioProcessor+Validation.swift
//  AudioToolCore
//
//  Sample-rate precondition shared by every provider
//

import Foundation

public extension AudioProcessor {

    /// Verify audio arrives at the rate this processor consumes.
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
    /// - Throws: ``AudioToolError/sampleRateMismatch(expected:found:)``
    func validateSampleRate(_ audio: AudioBuffer) throws {
        guard audio.sampleRate == sampleRate else {
            throw AudioToolError.sampleRateMismatch(
                expected: sampleRate,
                found: audio.sampleRate
            )
        }
    }
}
