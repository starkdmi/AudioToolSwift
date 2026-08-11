import Foundation

/// How a MossFormer2 SE checkpoint was quantized, in the terms `MLXNN.quantize` needs.
///
/// Two published layouts describe the same thing differently and both are in use:
///
/// | Repository | Weights | Metadata |
/// | --- | --- | --- |
/// | `starkdmi/MossFormer2_SE_48K_MLX` | `model_int{4,6,8}.safetensors` | none |
/// | `starkdmi/MossFormer2-SE-{4,6,8}bit` | `model.safetensors` | `config.json` |
///
/// The flat repository predates the convention and records nothing: its group size
/// exists only as a literal `64` in the generator that produced it,
/// `mossformer2_se_mlx/public/python/generate.py`. The per-precision repositories
/// carry `quantization_config`, so they describe themselves and a re-quantization at
/// some other group size cannot silently mismatch the loader.
///
/// ``resolve(forWeightsAt:)`` prefers the sidecar and falls back to the filename, so
/// a cache holding either layout loads without the caller having to know which it
/// got. A checkpoint that is neither resolves to `nil`, which means "not quantized"
/// - the fp32 and fp16 files take that path unchanged.
public struct QuantizationParameters: Sendable, Equatable {

    public let groupSize: Int
    public let bits: Int

    public init(groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
    }

    /// The group size the Python generator quantized with.
    ///
    /// Correct only for the flat repository, which records nothing. Anything
    /// carrying a `config.json` is read rather than assumed - see
    /// ``resolve(forWeightsAt:)``.
    public static let generatorGroupSize = 64

    /// Quantization for the checkpoint at `path`, or `nil` if it is not quantized.
    ///
    /// Sidecar first, filename second. Both are checked because the two published
    /// layouts disagree about where the information lives, and a machine can hold
    /// either - or, after a partial migration, one of each.
    public static func resolve(forWeightsAt path: String) -> QuantizationParameters? {
        let url = URL(fileURLWithPath: path)
        return fromSidecar(beside: url) ?? fromFilename(url.lastPathComponent)
    }

    /// `quantization_config` from a `config.json` sitting next to the weights.
    ///
    /// Absent config, absent key, and unreadable JSON are all "no metadata here",
    /// not failures: the flat repository legitimately has no config at all, and its
    /// quantized checkpoints still have to load.
    static func fromSidecar(beside weights: URL) -> QuantizationParameters? {
        let config = weights.deletingLastPathComponent().appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: config),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        // `quantization_config` is what the MossFormer2-SE-*bit repos write;
        // `quantization` is the spelling Chatterbox's repos use. Accepting both
        // costs one line and means a checkpoint converted by either script loads.
        let block = (root["quantization_config"] ?? root["quantization"]) as? [String: Any]
        guard let block, let bits = block["bits"] as? Int else { return nil }
        let groupSize = block["group_size"] as? Int ?? generatorGroupSize
        return QuantizationParameters(groupSize: groupSize, bits: bits)
    }

    /// Bit width from a `model_int<n>.safetensors` filename.
    ///
    /// The fallback for the flat repository. Group size is the generator's, because
    /// there is nowhere else for it to come from; if that repository is ever
    /// re-quantized at a different group size it must gain a `config.json`, and the
    /// sidecar path above will then take precedence.
    static func fromFilename(_ filename: String) -> QuantizationParameters? {
        guard let range = filename.range(of: #"(?<=_int)\d+"#, options: .regularExpression),
              let bits = Int(filename[range])
        else { return nil }
        return QuantizationParameters(groupSize: generatorGroupSize, bits: bits)
    }
}
