import Foundation

/// The seven conditioning targets USS ships presets for.
///
/// This used to also load them: seven ~2 KB `.safetensors` resources, one per case,
/// read from the module bundle during `load()`. Each turned out to be a normalised
/// multi-hot vector over the AudioSet class list - n classes at 1/n, zero elsewhere -
/// so `AudioToolCore.SoundEmbedding`'s presets reconstruct them exactly from the
/// indices alone, and the files and the loader are both gone.
///
/// The enum stays. It is what the provider's convenience API is keyed on, and "the
/// seven bundled targets" remains a useful thing to name. Anything outside these
/// seven is a `SoundEmbedding` built by the caller: the model takes any 527-d vector
/// and never saw this enum.
public enum EmbeddingLoader {

    /// Available embedding types
    public enum EmbeddingType: String, CaseIterable, Sendable {
        case speech = "speech"
        case music = "music"
        case noise = "noise"
        case nature = "nature"
        case things = "things"
        case animal = "animal"
        case human = "human"
    }
}
