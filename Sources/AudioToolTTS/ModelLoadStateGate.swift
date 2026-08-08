//
//  ModelLoadStateGate.swift
//  AudioToolTTS
//
//  Keeps asynchronous download progress generation-safe and monotonic
//

import AudioToolCore

enum ModelLoadStateGate {
    static func acceptsProgress(
        _ proposedProgress: Double,
        generation: UInt64,
        currentGeneration: UInt64,
        state: ModelState
    ) -> Bool {
        guard generation == currentGeneration,
              proposedProgress.isFinite,
              case .downloading(let currentProgress) = state
        else { return false }
        return proposedProgress >= currentProgress
    }
}
