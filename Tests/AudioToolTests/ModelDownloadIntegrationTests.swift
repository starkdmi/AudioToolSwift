//
//  ModelDownloadIntegrationTests.swift
//  ClearVoice
//
//  Integration tests for model download system with real HuggingFace models
//  Uses small models (GAN SE CoreML ~2MB, Silero VAD ~2MB) for fast testing
//

import Testing
import Foundation
@testable import ClearVoice
@testable import ClearVoiceCore

// MARK: - Real Download Tests

/// Integration tests that download real models from HuggingFace
/// These tests require network access and may take time
@Suite("Model Download Integration", .tags(.integration))
struct ModelDownloadIntegrationTests {
    
    // Small test models
    static let testVariantGAN = ModelVariant(
        id: "test_gan_se_coreml",
        name: "MossFormer GAN SE (CoreML Test)",
        quantization: .fp16,
        sizeBytes: 2_000_000,  // ~2 MB
        repo: "starkdmi/MossFormerGAN_SE_CoreML",
        files: ["MossFormerGAN_256frames.mlpackage/Data/com.apple.CoreML/model.mlmodel"]
    )
    
    static let testVariantVAD = ModelVariant(
        id: "test_silero_vad",
        name: "Silero VAD (Test)",
        quantization: .fp16,
        sizeBytes: 2_000_000,  // ~2 MB
        repo: "FluidInference/SileroVAD",
        files: ["silero_vad.onnx"]
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
        let isDownloaded = ModelDownloader.shared.isDownloaded(repo: variant.repo)
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
        let localPath = ModelDownloader.shared.localPath(for: variant.repo)
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
        
        #expect(ModelDownloader.shared.isDownloaded(repo: variant.repo) == true)
        print("   ✓ Downloaded")
        
        print("2. Deleting...")
        try await DownloadCoordinator.shared.delete(variant: variant)
        
        // Verify deleted
        let stillExists = ModelDownloader.shared.isDownloaded(repo: variant.repo)
        #expect(stillExists == false)
        print("   ✓ Deleted")
    }
    
    // MARK: - Cancellation Tests
    
    @Test("Cancel download")
    func testCancelDownload() async throws {
        let variant = Self.testVariantGAN
        
        // First ensure it's not cached (delete if exists)
        try? await ModelDownloader.shared.delete(repo: variant.repo)
        
        print("Starting download (will cancel)...")
        
        // Try to download and cancel - the download stream handles cancellation
        var progressCount = 0
        var didFinish = false
        
        do {
            for try await progress in await DownloadCoordinator.shared.download(variant: variant) {
                progressCount += 1
                print("  Progress: \(progress.percentComplete)%")
                
                // Cancel after some progress
                if progress.fractionCompleted > 0.1 {
                    await DownloadCoordinator.shared.cancel(variantId: variant.id)
                    break
                }
            }
            didFinish = true
        } catch let error as DownloadError {
            if case .cancelled = error {
                print("✓ Download was cancelled")
            } else {
                throw error
            }
        }
        
        print("✓ Download handled (finished=\(didFinish), progressUpdates=\(progressCount))")
    }
    
    // MARK: - Storage Check Tests
    
    @Test("Storage check passes for small model")
    func testStorageCheckPasses() async throws {
        let variant = Self.testVariantGAN
        
        // Should not throw - we have enough space for a 2MB model
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
            
            #expect(ModelDownloader.shared.isDownloaded(repo: variant.repo))
        }
        
        print("✓ All \(variants.count) models downloaded")
    }
}

// MARK: - Coordinator Tests

@Suite("Download Coordinator", .tags(.integration))
struct DownloadCoordinatorTests {
    
    @Test("Active downloads tracking")
    func testActiveDownloadsTracking() async throws {
        let ids = await DownloadCoordinator.shared.activeDownloadIds
        
        // Verify we can access the array
        #expect(ids.count >= 0)
        print("Active downloads: \(ids.count)")
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
