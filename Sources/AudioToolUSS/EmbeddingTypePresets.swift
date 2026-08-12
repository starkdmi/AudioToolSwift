//
//  EmbeddingTypePresets.swift
//  AudioToolUSS
//
//  Bridges USS's seven named targets to their SoundEmbedding presets
//

import AudioToolCore
import USSMLXSwift

extension EmbeddingLoader.EmbeddingType {

    /// The preset this case names.
    ///
    /// The two types live in different modules on purpose: `EmbeddingType` is part of
    /// the vendored USS port, which knows nothing about `AudioToolCore`, while
    /// `SoundEmbedding` is the general form that any 527-d target takes. This is the
    /// only place they meet, and it exists so the provider's `type:` convenience API
    /// and its `target:` API resolve to the same vectors rather than to a file and a
    /// class list that could drift apart.
    var embedding: SoundEmbedding {
        switch self {
        case .speech: .speech
        case .music: .music
        case .noise: .noise
        case .nature: .nature
        case .things: .things
        case .animal: .animal
        case .human: .human
        }
    }
}
