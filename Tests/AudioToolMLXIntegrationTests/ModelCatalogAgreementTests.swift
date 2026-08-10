//
//  ModelCatalogAgreementTests.swift
//  AudioToolMLXIntegrationTests
//
//  The catalog and the providers must describe the same downloads. No network.
//

import XCTest
import AudioToolCore
import AudioToolUSS
import AudioToolTTS
@testable import AudioToolMLX

/// What the catalog advertises has to be what the providers fetch.
///
/// These are two different consumers of the same facts. The catalog is what a host
/// app lists and what a pre-download step pulls; the provider constant is what runs
/// at `load()`. When they disagree the failure is nasty and late: a user pre-downloads
/// one repo, then inference fails looking in another, and nothing before that point
/// reports a problem.
///
/// They had drifted in five places - SR's repo was missing its sample rate, SS was one
/// entry for three separately trained models, USS advertised `model.safetensors` when
/// the weights are `resunet30_*`, Demucs advertised one file when there are four, and
/// FRCRN advertised a repo no code could fetch.
///
/// `ModelRepository` and `ModelFiles` in `AudioToolCore` are now the single source, so
/// the strings cannot disagree. This test covers what a shared constant cannot: that
/// each provider is actually wired to the entry describing it, and that entries are
/// internally coherent.
///
/// This test target can see both sides; `AudioToolCore` cannot see the providers,
/// which is why the check lives here rather than next to the catalog.
final class ModelCatalogAgreementTests: XCTestCase {

    private var catalog: ModelCatalog { .shared }

    private func variant(_ id: String, file: StaticString = #filePath, line: UInt = #line) throws -> ModelVariant {
        try XCTUnwrap(catalog.variant(id: id), "no catalog variant '\(id)'", file: file, line: line)
    }

    // MARK: - Repositories

    func testProviderRepositoriesMatchTheCatalog() throws {
        let pairs: [(provider: String, variantID: String, label: String)] = [
            (MossFormer2SE48KProvider.repo, "mossformer2_se_fp32", "MossFormer2 SE"),
            (MossFormer2SR48KProvider.repo, "mossformer2_sr_fp32", "MossFormer2 SR"),
            (FRCRNSE16KProvider.repo, "frcrn_se_fp32", "FRCRN SE"),
            (DemucsProvider.repo, "demucs_fp32", "Demucs"),
            (USSMLXProvider.repo, "uss_fp32", "USS"),
        ]

        for (providerRepo, variantID, label) in pairs {
            let entry = try variant(variantID)
            XCTAssertEqual(providerRepo, entry.repo,
                           "\(label): provider downloads from '\(providerRepo)' but the catalog advertises '\(entry.repo)'")
        }
    }

    /// Speaker separation is three models, not three quantizations of one.
    func testEverySpeakerSeparationModelHasItsOwnEntry() throws {
        let pairs: [(model: MossFormer2SSProvider.Model, variantID: String)] = [
            (.twoSpeaker, "mossformer2_ss_2spk_16k_fp32"),
            (.threeSpeaker, "mossformer2_ss_3spk_8k_fp32"),
            (.twoSpeakerWHAMR, "mossformer2_ss_2spk_whamr_8k_fp32"),
        ]

        for (model, variantID) in pairs {
            let entry = try variant(variantID)
            XCTAssertEqual(model.huggingFaceRepo, entry.repo,
                           "SS \(model.rawValue): provider uses '\(model.huggingFaceRepo)', catalog advertises '\(entry.repo)'")
        }

        // And no entry points at a repo no provider knows about.
        let providerRepos = Set(MossFormer2SSProvider.Model.allCases.map(\.huggingFaceRepo))
        let catalogRepos = Set(catalog.variants(for: "mossformer2_ss").map(\.repo))
        XCTAssertEqual(catalogRepos, providerRepos,
                       "catalog and provider disagree on which SS repos exist")
    }

    // MARK: - File layouts

    /// A repo entry that names the wrong file downloads successfully and then fails
    /// at load, which is the most expensive place to find out.
    func testCatalogFilesMatchWhatProvidersAskFor() throws {
        // Precision-suffixed convention.
        for (variantID, precision) in [("mossformer2_se_fp32", ModelPrecision.fp32),
                                       ("mossformer2_se_fp16", .fp16),
                                       ("mossformer2_sr_fp32", .fp32),
                                       ("frcrn_se_fp32", .fp32)] {
            let entry = try variant(variantID)
            XCTAssertTrue(entry.files.contains(precision.weightsFilename),
                          "\(variantID) should list '\(precision.weightsFilename)', lists \(entry.files)")
        }

        // What must be present to *load* is narrower than what gets downloaded,
        // for every one of these but super-resolution. Their providers build their
        // configuration in code and read nothing but the safetensors, so a snapshot
        // without the config.json is loadable and must not be reported as missing.
        //
        // This is pinned rather than left to the default because the default is
        // `files`, and taking it made the catalog and the loaders disagree about
        // the word "installed": a machine holding the weights was told it had
        // nothing, and every load paid a repository round-trip to fetch a file it
        // would never open. Offline, it failed outright.
        for (variantID, precision) in [("mossformer2_se_fp32", ModelPrecision.fp32),
                                       ("mossformer2_se_fp16", .fp16),
                                       ("frcrn_se_fp32", .fp32),
                                       ("mossformer2_ss_2spk_16k_fp32", .fp32),
                                       ("mossformer2_ss_3spk_8k_fp32", .fp32),
                                       ("mossformer2_ss_2spk_whamr_8k_fp32", .fp32)] {
            let entry = try variant(variantID)
            XCTAssertEqual(
                entry.requiredFiles, ModelFiles.standardRequired(precision),
                "\(variantID) loads the safetensors alone; requiredFiles is \(entry.requiredFiles)"
            )
            XCTAssertFalse(
                entry.requiredFiles.contains("config.json"),
                "\(variantID)'s provider never opens a config.json"
            )
            XCTAssertTrue(
                entry.files.contains("config.json"),
                "\(variantID) should still fetch the config when it downloads, so a "
                    + "snapshot arrives complete"
            )
        }

        // Super-resolution is the exception, and the reason the two lists exist:
        // MossFormer2SR48KProvider reads its architecture out of config.json, so a
        // snapshot without one genuinely cannot load.
        let sr = try variant("mossformer2_sr_fp32")
        XCTAssertEqual(sr.requiredFiles, ModelFiles.standard(.fp32))
        XCTAssertTrue(sr.requiredFiles.contains("config.json"))

        // USS names its checkpoints after the architecture.
        let uss = try variant("uss_fp32")
        XCTAssertEqual(uss.files, ["resunet30_fp32.safetensors"],
                       "USS loads resunet30_<precision>.safetensors, not a generic model file")

        // Demucs is four files, one per stem.
        let demucs = try variant("demucs_fp32")
        XCTAssertEqual(Set(demucs.files), Set(DemucsProvider.Stem.allCases.map { "\($0.rawValue).safetensors" }),
                       "Demucs needs one weight file per stem, catalog lists \(demucs.files)")
        XCTAssertEqual(demucs.files.count, 4)

        let kokoro = try variant("kokoro_tts_bf16")
        XCTAssertEqual(kokoro.repo, ModelRepository.kokoroBF16)
        XCTAssertEqual(kokoro.repo, ModelPrecision.bf16.repo(base: KokoroTTSProvider.baseRepo))
        XCTAssertEqual(kokoro.files, ModelFiles.kokoro)

        let chatterbox = try variant("chatterbox_tts_fp32")
        XCTAssertEqual(chatterbox.repo, ChatterboxTTSProvider.baseRepo)
        XCTAssertEqual(chatterbox.files, ModelFiles.chatterboxDownload(for: .fp32))
        XCTAssertEqual(chatterbox.requiredFiles, ModelFiles.chatterboxRequired(for: .fp32))
        XCTAssertTrue(chatterbox.files.contains("conds.safetensors"))
        XCTAssertTrue(chatterbox.files.contains("s3tokenizer.safetensors"))
        XCTAssertFalse(chatterbox.requiredFiles.contains("conds.safetensors"))
    }

    func testMossFormerEnhancementAdvertisesOnlySupportedPrecisions() {
        let precisions = Set(catalog.variants(for: "mossformer2_se").map(\.quantization))
        XCTAssertEqual(precisions, Set([.fp32, .fp16]))
    }

    /// A size that is wrong by a factor of four misinforms a storage precheck, which
    /// is the one thing the catalog is consulted for before anything is downloaded.
    func testDemucsAdvertisesAllFourStems() throws {
        let demucs = try variant("demucs_fp32")
        // Each HTDemucs stem checkpoint is ~84 MB.
        XCTAssertGreaterThan(demucs.sizeBytes, 300_000_000,
                             "Demucs size should cover four stem files, not one")
    }

    // MARK: - Internal coherence

    func testVariantIdentifiersAreUnique() {
        let ids = catalog.allVariants.map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count,
                       "duplicate variant ids: \(Dictionary(grouping: ids, by: { $0 }).filter { $0.value.count > 1 }.keys)")
    }

    func testEveryVariantNamesAtLeastOneFile() {
        for entry in catalog.allVariants {
            XCTAssertFalse(entry.files.isEmpty, "\(entry.id) lists no files to download")
            XCTAssertFalse(
                entry.requiredFiles.isEmpty,
                "\(entry.id) lists no files required for verification"
            )
            XCTAssertTrue(
                Set(entry.requiredFiles).isSubset(of: Set(entry.files)),
                "\(entry.id) requires files its download manifest does not request"
            )
            XCTAssertFalse(entry.repo.isEmpty, "\(entry.id) has no repo")
        }
    }
}
