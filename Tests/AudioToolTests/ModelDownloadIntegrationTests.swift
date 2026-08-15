//
//  ModelDownloadIntegrationTests.swift
//  AudioTool
//
//  Integration tests for model download system with real HuggingFace models
//  Uses one small fixture for both variants: two ~10 KB text files from
//  FluidInference/silero-vad-coreml, which download fast and change rarely
//

import Testing
import Foundation
@testable import AudioTool
@testable import AudioToolCore

// MARK: - Real Download Tests

/// Integration tests that download real models from HuggingFace
/// These tests require network access and may take time
///
/// Opt-in, like every other suite that leaves the machine. This one is the reason
/// `swift test` was not offline-clean even after the slow Apple-framework suites
/// were gated: it reached HuggingFace on every run. The hermetic half of the
/// downloader's behaviour is covered by `ModelDownloadTests`.
@Suite("Model Download Integration", .tags(.integration), .serialized,
       .enabled(if: TestConfiguration.runIntegrationTests,
                "integration test - set RUN_INTEGRATION_TESTS=1 (downloads from HuggingFace)"))
struct ModelDownloadIntegrationTests {
    
    // Small test models - use actual public repos
    // Use simple files (config.json, README.md) to avoid Hub library issues with nested directories
    static let testVariantGAN = ModelVariant(
        id: "test_silero_vad_1",
        name: "Silero VAD Config (Test 1)",
        quantization: .fp16,
        sizeBytes: 10_000,  // ~10 KB
        repo: "FluidInference/silero-vad-coreml",
        files: ["config.json", "README.md"]  // Simple files for reliable testing
    )
    
    static let testVariantVAD = ModelVariant(
        id: "test_silero_vad_2",
        name: "Silero VAD Config (Test 2)",
        quantization: .fp16,
        sizeBytes: 10_000,  // ~10 KB
        repo: "FluidInference/silero-vad-coreml",
        files: ["config.json", "README.md"]  // Simple files for reliable testing
    )
    
    // MARK: - Download Tests
    
    @Test("Download GAN SE model with progress")
    func testDownloadGANWithProgress() async throws {
        let variant = Self.testVariantGAN
        var progressUpdates: [DownloadProgress] = []
        
        print("Starting download of \(variant.name) from \(variant.repo)")
        
        for try await progress in await DownloadCoordinator.shared.download(variant: variant) {
            progressUpdates.append(progress)
            print("  Progress: \(progress.percentComplete)%")
        }
        
        // Should have received progress updates
        #expect(!progressUpdates.isEmpty)
        
        // Last progress should be 100%
        if let last = progressUpdates.last {
            #expect(last.fractionCompleted == 1.0)
        }
        
        // Model should now be marked as downloaded
        let isDownloaded = ModelDownloader.shared.isDownloaded(variant: variant)
        #expect(isDownloaded == true)
        
        print("✓ Download completed successfully")
    }
    
    @Test("Download and get path")
    func testDownloadAndGetPath() async throws {
        let variant = Self.testVariantGAN
        
        let path = try await DownloadCoordinator.shared.downloadAndGetPath(
            variant: variant
        ) { progress in
            print("  \(progress.percentComplete)%")
        }
        
        #expect(FileManager.default.fileExists(atPath: path.path))
        print("✓ Model downloaded to: \(path.path)")
    }
    
    @Test("Check local path after download")
    func testLocalPathAfterDownload() async throws {
        let variant = Self.testVariantGAN
        
        // Ensure it's downloaded first
        _ = try await DownloadCoordinator.shared.downloadAndGetPath(variant: variant)
        
        // Check local path
        let localPath = ModelDownloader.shared.localPath(for: variant)
        #expect(localPath != nil)
        
        if let path = localPath {
            print("✓ Local path: \(path.path)")
            #expect(path.path.contains("huggingface"))
        }
    }
    
    @Test("Cache size after download")
    func testCacheSizeAfterDownload() async throws {
        let variant = Self.testVariantGAN
        
        // Ensure downloaded
        _ = try await DownloadCoordinator.shared.downloadAndGetPath(variant: variant)
        
        // Check cache size
        let size = ModelDownloader.shared.cacheSize(for: variant.repo)
        #expect(size > 0)
        
        print("✓ Cache size: \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))")
    }
    
    // MARK: - Download and Delete Cycle
    
    @Test("Download and delete cycle")
    func testDownloadDeleteCycle() async throws {
        let variant = Self.testVariantVAD
        
        print("1. Downloading \(variant.name)...")
        _ = try await DownloadCoordinator.shared.downloadAndGetPath(variant: variant)
        
        #expect(ModelDownloader.shared.isDownloaded(variant: variant) == true)
        print("   ✓ Downloaded")
        
        print("2. Deleting...")
        try await DownloadCoordinator.shared.delete(variant: variant)
        
        // Verify deleted
        let stillExists = ModelDownloader.shared.isDownloaded(variant: variant)
        #expect(stillExists == false)
        print("   ✓ Deleted")
    }
    
    // MARK: - Cancellation Tests
    
    @Test("Cancel download")
    func testCancelDownload() async throws {
        let variant = Self.testVariantGAN

        // Not `try?`: if the cache cannot be cleared, the download below may not
        // happen at all and the test would be measuring nothing.
        try await ModelDownloader.shared.delete(repo: variant.repo)

        // Cancel to completion *before* consuming a single element. That ordering
        // is what makes this deterministic rather than merely likely.
        //
        // `download` registers the operation and returns the stream from inside
        // the actor, so there is always something to cancel by the time it hands
        // back. `cancel` then cancels the task and awaits it, and it can do that
        // with nobody consuming: `AsyncThrowingStream.makeStream()` buffers
        // `.unbounded` by default, so the producer never suspends on a yield
        // waiting for a reader. When this returns, the download is over one way or
        // the other and the loop below only drains what it left.
        //
        // Both earlier versions were races. One cancelled after progress passed
        // 10%, which on a 10 KB fixture may never happen. The other spawned
        // `Task { cancel() }`, which merely schedules cancellation - the download
        // could still win on a fast response and fail the test although the code
        // was behaving correctly.
        let stream = await DownloadCoordinator.shared.download(variant: variant)
        await DownloadCoordinator.shared.cancel(variantId: variant.id)

        var progressCount = 0
        var terminatingError: Error?
        do {
            for try await _ in stream { progressCount += 1 }
        } catch {
            terminatingError = error
        }

        guard let terminatingError else {
            Issue.record(
                Comment(rawValue: """
                    the stream completed after \(progressCount) progress events \
                    although cancel() returned before it was consumed - cancel() \
                    awaits the operation, so nothing can have been in flight after it
                    """)
            )
            return
        }
        guard let downloadError = terminatingError as? DownloadError,
              case .cancelled = downloadError else {
            throw terminatingError
        }

        // The documented contract: cancel() cancels the operation, waits for it,
        // and removes it. Not that the partial download is rolled back - nothing
        // promises that, and asserting it would fail whenever Hub happened to
        // finish writing before the cancellation was observed.
        let active = await DownloadCoordinator.shared.activeDownloadIds
        #expect(active.contains(variant.id) == false, "still active after cancellation: \(active)")
        #expect(await DownloadCoordinator.shared.isDownloading(variantId: variant.id) == false)
    }

    // MARK: - Storage Check Tests
    
    @Test("Storage check passes for small model")
    func testStorageCheckPasses() async throws {
        let variant = Self.testVariantGAN
        
        // Should not throw - the fixture is ~10 KB
        try await DownloadCoordinator.shared.checkStorageAvailable(for: variant)
        print("✓ Storage check passed")
    }
    
    @Test("Storage check fails for huge model")
    func testStorageCheckFails() async throws {
        let hugeVariant = ModelVariant(
            id: "huge_model",
            name: "Huge Model",
            quantization: .fp32,
            sizeBytes: 1_000_000_000_000_000,  // 1 PB - impossible
            repo: "test/huge",
            files: []
        )
        
        do {
            try await DownloadCoordinator.shared.checkStorageAvailable(for: hugeVariant)
            Issue.record("Should have thrown insufficientStorage error")
        } catch let error as DownloadError {
            if case .insufficientStorage = error {
                print("✓ Correctly rejected: \(error.localizedDescription)")
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        }
    }
    
    // MARK: - Multiple Downloads
    
    @Test("Download multiple variants sequentially")
    func testMultipleDownloads() async throws {
        let variants = [Self.testVariantGAN, Self.testVariantVAD]
        
        for (index, variant) in variants.enumerated() {
            print("Downloading \(index + 1)/\(variants.count): \(variant.name)")
            
            _ = try await DownloadCoordinator.shared.downloadAndGetPath(variant: variant)
            
            #expect(ModelDownloader.shared.isDownloaded(variant: variant))
        }
        
        print("✓ All \(variants.count) models downloaded")
    }
    
    // MARK: - Coordinator State Tests
    
    @Test("Active downloads tracking")
    func testActiveDownloadsTracking() async throws {
        let ids = await DownloadCoordinator.shared.activeDownloadIds

        // `ids.count >= 0` was the assertion here, which every array satisfies.
        // The suite is `.serialized` and does contain download tests, so the
        // invariant worth stating is that they clean up after themselves: by the
        // time this runs, nothing is still in flight.
        #expect(ids.isEmpty, "a previous test left a download in flight: \(ids)")
    }
    
    @Test("Check is downloading status")
    func testIsDownloadingStatus() async throws {
        // Check a fake ID
        let isDownloading = await DownloadCoordinator.shared.isDownloading(variantId: "fake_id_xyz")
        #expect(isDownloading == false)
    }
    
    @Test("Cancel all when none active")
    func testCancelAllEmpty() async throws {
        await DownloadCoordinator.shared.cancelAll()
        
        let ids = await DownloadCoordinator.shared.activeDownloadIds
        #expect(ids.isEmpty)
    }
}

// MARK: - Test Tags

extension Tag {
    /// Tests that require network access and real downloads
    @Tag static var integration: Self
}
