//
//  ModelPinTests.swift
//  AudioTool
//
//  Tests for pinned revisions and downloaded-file verification
//

import Testing
import CryptoKit
import Foundation
@testable import AudioToolCore
@testable import AudioToolTTS

@Suite("Model Pins")
struct ModelPinTests {

    @Test("Every catalog repository is either pinned or listed as unpublished")
    func testCatalogRepositoriesArePinnedOrDeclared() {
        for variant in ModelCatalog.shared.allVariants {
            let pinned = ModelPins.pin(for: variant.repo) != nil
            let declared = ModelPins.unpublished.contains(variant.repo)
            #expect(
                pinned || declared,
                """
                \(variant.repo) (\(variant.id)) has no pin and is not declared \
                unpublished. Add its revision to ModelPins, or add it to \
                ModelPins.unpublished with the reason.
                """
            )
        }
    }

    @Test("Every Kokoro precision the factory advertises is pinned")
    func testKokoroPrecisionRepositoriesArePinned() {
        // The catalog lists one Kokoro variant, so the catalog sweep above cannot see
        // this: `TTSProviders.kokoro(precision:)` derives a repository name from the
        // precision, and those repositories were following mutable `main`.
        for precision in KokoroTTSProvider.supportedPrecisions {
            let repo = precision.repo(base: KokoroTTSProvider.baseRepo)
            #expect(
                ModelPins.pin(for: repo) != nil,
                "\(repo) is reachable from TTSProviders.kokoro(precision: .\(precision.rawValue)) but has no pin"
            )
        }
    }

    @Test("Unpinned repositories resolve to the default branch")
    func testUnpinnedRepositoriesFallBackToMain() {
        // `unpublished` is empty as of 2026-08-12 - USS, FRCRN and Demucs shipped and
        // are pinned - so this loop guards the next conversion to be listed there
        // rather than anything currently in the catalog.
        for repo in ModelPins.unpublished {
            #expect(ModelPins.pin(for: repo) == nil)
            #expect(ModelPins.revision(for: repo) == "main")
        }
        #expect(ModelPins.revision(for: "nobody/nothing") == "main")
    }

    @Test("Pinned revisions are full commit hashes, not branches or tags")
    func testPinnedRevisionsAreCommitHashes() {
        // A branch name here would silently reintroduce exactly the mutability the
        // pin exists to remove, and would still look pinned.
        let hex = CharacterSet(charactersIn: "0123456789abcdef")
        for (repo, pin) in ModelPins.all {
            #expect(pin.revision.count == 40, "\(repo) revision is not a full SHA-1")
            #expect(
                CharacterSet(charactersIn: pin.revision).isSubset(of: hex),
                "\(repo) revision is not hexadecimal"
            )
            for (file, hash) in pin.fileHashes {
                #expect(hash.count == 64, "\(repo)/\(file) hash is not a SHA-256")
                #expect(
                    CharacterSet(charactersIn: hash).isSubset(of: hex),
                    "\(repo)/\(file) hash is not hexadecimal"
                )
            }
        }
    }
}

@Suite("Model Integrity")
struct ModelIntegrityTests {

    private func makeSnapshot(_ files: [String: Data]) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for (name, data) in files {
            let url = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: url)
        }
        return root
    }

    @Test("Chunked hashing matches CryptoKit over the whole file")
    func testChunkedHashMatchesWholeFile() throws {
        // The reader is chunked because the largest pinned file is 2.7 GB, and a
        // chunk-boundary bug would show up as a wrong digest on exactly one size.
        let root = try makeSnapshot([:])
        defer { try? FileManager.default.removeItem(at: root) }

        for size in [0, 1, 4 * 1024 * 1024 - 1, 4 * 1024 * 1024, 4 * 1024 * 1024 + 1] {
            var generator = SystemRandomNumberGenerator()
            let bytes = (0..<size).map { _ in UInt8.random(in: 0...255, using: &generator) }
            let data = Data(bytes)
            let url = root.appendingPathComponent("blob-\(size).bin")
            try data.write(to: url)

            let expected = SHA256.hash(data: data)
                .map { String(format: "%02x", $0) }
                .joined()
            #expect(try ModelDownloader.sha256(ofFileAt: url) == expected, "size \(size)")
        }
    }

    @Test("An unpinned repository reports as unpinned rather than as verified")
    func testUnpinnedRepositoryIsNotReportedAsVerified() throws {
        let root = try makeSnapshot(["model_fp32.safetensors": Data([1, 2, 3])])
        defer { try? FileManager.default.removeItem(at: root) }

        // A coordinate deliberately absent from the registry. This used to be
        // `ModelRepository.uss`, which was unpinned only because it was unpublished;
        // now that every catalog repository is pinned, the case needs a repository
        // that is not one of them rather than one that happens not to be pinned yet.
        let report = try ModelDownloader.shared.verify(repo: "nobody/nothing", at: root)
        #expect(!report.isPinned)
        #expect(report.passed)
        #expect(report.verified.isEmpty)
    }

    @Test("Matching bytes verify and altered bytes are reported")
    func testVerificationDetectsAlteredBytes() throws {
        let repo = ModelRepository.mossFormer2SE48K
        let pin = try #require(ModelPins.pin(for: repo))
        let file = "model_fp16.safetensors"
        let expected = try #require(pin.fileHashes[file])

        // Stand-in bytes: the point is that verify() compares what is on disk to
        // the pin, so a snapshot whose file hashes to something else must fail
        // rather than pass by being present.
        let root = try makeSnapshot([file: Data("not the weights".utf8)])
        defer { try? FileManager.default.removeItem(at: root) }

        let report = try ModelDownloader.shared.verify(repo: repo, at: root)
        #expect(report.isPinned)
        #expect(!report.passed)
        #expect(report.mismatches.count == 1)
        #expect(report.mismatches.first?.file == file)
        #expect(report.mismatches.first?.expected == expected)
    }

    @Test("A snapshot holding only some pinned files still passes")
    func testAbsentFilesAreNotFailures() throws {
        // Downloading fp16 alone is a complete, valid install. Treating the absent
        // fp32 as a failure would make every precision-specific download look
        // corrupt.
        let root = try makeSnapshot([:])
        defer { try? FileManager.default.removeItem(at: root) }

        let report = try ModelDownloader.shared.verify(
            repo: ModelRepository.mossFormer2SE48K,
            at: root
        )
        #expect(report.isPinned)
        #expect(report.passed)
        #expect(report.verified.isEmpty)
        #expect(report.mismatches.isEmpty)
    }
}
