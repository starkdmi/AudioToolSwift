//
//  ModelDownloadTests.swift
//  AudioTool
//
//  Tests for model download system: types, coordinator, and manager
//

import Testing
import Foundation
@testable import AudioTool
@testable import AudioToolCore

// MARK: - Type Tests

@Suite("Model Download Types")
struct ModelDownloadTypeTests {
    
    // MARK: - Quantization
    
    @Test("Quantization display names")
    func testQuantizationDisplayNames() {
        #expect(Quantization.fp32.displayName == "Full Precision (FP32)")
        #expect(Quantization.fp16.displayName == "Half Precision (FP16)")
        #expect(Quantization.int8.displayName == "8-bit Quantized")
        #expect(Quantization.int4.displayName == "4-bit Quantized")
    }
    
    @Test("Quantization short names")
    func testQuantizationShortNames() {
        #expect(Quantization.fp32.shortName == "FP32")
        #expect(Quantization.int8.shortName == "INT8")
    }
    
    @Test("Quantization memory multipliers")
    func testQuantizationMemoryMultipliers() {
        #expect(Quantization.fp32.memoryMultiplier == 1.0)
        #expect(Quantization.fp16.memoryMultiplier == 0.5)
        #expect(Quantization.int8.memoryMultiplier == 0.25)
        #expect(Quantization.int4.memoryMultiplier == 0.125)
    }
    
    // MARK: - ModelCategory
    
    @Test("Model category display names")
    func testCategoryDisplayNames() {
        #expect(ModelCategory.speechEnhancement.displayName == "Speech Enhancement")
        #expect(ModelCategory.textToSpeech.displayName == "Text to Speech")
        #expect(ModelCategory.vad.displayName == "Voice Activity Detection")
    }
    
    @Test("Model category icons")
    func testCategoryIcons() {
        #expect(!ModelCategory.speechEnhancement.iconName.isEmpty)
        #expect(!ModelCategory.vad.iconName.isEmpty)
    }
    
    // MARK: - ModelVariant
    
    @Test("ModelVariant size string formatting")
    func testVariantSizeString() {
        let variant = ModelVariant(
            id: "test_variant",
            name: "Test Variant",
            quantization: .fp16,
            sizeBytes: 100_000_000,
            repo: "test/repo",
            files: ["model.safetensors"]
        )
        
        #expect(variant.sizeString.contains("100") || variant.sizeString.contains("MB"))
        #expect(variant.id == "test_variant")
        #expect(variant.quantization == .fp16)
    }
    
    @Test("ModelVariant is hashable")
    func testVariantHashable() {
        let v1 = ModelVariant(id: "a", name: "A", quantization: .fp32, sizeBytes: 100, repo: "r", files: [])
        let v2 = ModelVariant(id: "a", name: "A", quantization: .fp32, sizeBytes: 100, repo: "r", files: [])
        
        #expect(v1 == v2)
        #expect(v1.hashValue == v2.hashValue)
    }
    
    // MARK: - ModelDefinition
    
    @Test("ModelDefinition default variant")
    func testDefinitionDefaultVariant() {
        let variant1 = ModelVariant(id: "v1", name: "V1", quantization: .fp32, sizeBytes: 200, repo: "r", files: [])
        let variant2 = ModelVariant(id: "v2", name: "V2", quantization: .int8, sizeBytes: 50, repo: "r", files: [])
        
        let definition = ModelDefinition(
            id: "model",
            name: "Model",
            category: .speechEnhancement,
            description: "Desc",
            variants: [variant1, variant2]
        )
        
        #expect(definition.defaultVariant?.id == "v1")
        #expect(definition.smallestVariant?.id == "v2")
    }
    
    // MARK: - ModelPackage
    
    @Test("ModelPackage properties")
    func testPackageProperties() {
        let package = ModelPackage(
            id: "pkg",
            name: "Package",
            description: "A package",
            variantIds: ["v1", "v2", "v3"],
            totalSizeBytes: 500_000_000
        )
        
        #expect(package.modelCount == 3)
        #expect(package.sizeString.contains("500") || package.sizeString.contains("MB"))
    }
    
    // MARK: - DownloadStatus
    
    @Test("DownloadStatus states")
    func testDownloadStatus() {
        #expect(DownloadStatus.queued.isActive == true)
        #expect(DownloadStatus.downloading.isActive == true)
        #expect(DownloadStatus.completed.isActive == false)
        #expect(DownloadStatus.cancelled.isActive == false)
        
        #expect(DownloadStatus.completed.isFinished == true)
        #expect(DownloadStatus.failed("error").isFinished == true)
        #expect(DownloadStatus.downloading.isFinished == false)
    }
    
    // MARK: - DownloadError
    
    @Test("DownloadError descriptions")
    func testDownloadErrorDescriptions() {
        let storageError = DownloadError.insufficientStorage(required: 200_000_000, available: 50_000_000)
        #expect(storageError.localizedDescription.contains("available"))
        
        let networkError = DownloadError.networkUnavailable
        #expect(networkError.localizedDescription.contains("internet"))
        
        let cancelledError = DownloadError.cancelled
        #expect(cancelledError.localizedDescription.contains("cancelled"))
    }
    
    // MARK: - DownloadProgress
    
    @Test("DownloadProgress formatting")
    func testDownloadProgressFormatting() {
        let progress = DownloadProgress(
            fractionCompleted: 0.75,
            completedBytes: 75_000_000,
            totalBytes: 100_000_000,
            bytesPerSecond: 5_000_000
        )
        
        #expect(progress.percentComplete == 75)
        #expect(progress.speedString?.contains("MB/s") == true)
    }
    
    @Test("DownloadProgress slow speed formatting")
    func testDownloadProgressSlowSpeed() {
        let progress = DownloadProgress(
            fractionCompleted: 0.1,
            completedBytes: 10_000,
            totalBytes: 100_000,
            bytesPerSecond: 500
        )
        
        #expect(progress.speedString?.contains("B/s") == true)
    }
}

// MARK: - Registry Tests

@Suite("Model Registry")
struct ModelRegistryTests {
    
    @Test("Registry has models")
    func testRegistryHasModels() {
        let registry = ModelRegistry.shared
        
        #expect(!registry.models.isEmpty)
        #expect(!registry.allVariants.isEmpty)
    }
    
    @Test("Registry has packages")
    func testRegistryHasPackages() {
        let registry = ModelRegistry.shared
        
        #expect(!registry.packages.isEmpty)
    }
    
    @Test("Registry lookup by ID")
    func testRegistryLookup() {
        let registry = ModelRegistry.shared
        
        // Should find a model
        if let model = registry.models.first {
            let found = registry.model(id: model.id)
            #expect(found?.id == model.id)
        }
        
        // Should find a variant
        if let variant = registry.allVariants.first {
            let found = registry.variant(id: variant.id)
            #expect(found?.id == variant.id)
        }
    }
    
    @Test("Registry filter by category")
    func testRegistryFilterByCategory() {
        let registry = ModelRegistry.shared
        
        let enhancementModels = registry.models(in: .speechEnhancement)
        #expect(enhancementModels.allSatisfy { $0.category == .speechEnhancement })
    }
    
    @Test("Registry statistics")
    func testRegistryStatistics() {
        let registry = ModelRegistry.shared
        
        #expect(registry.modelCount == registry.models.count)
        #expect(registry.variantCount == registry.allVariants.count)
        #expect(registry.packageCount == registry.packages.count)
        #expect(registry.totalCatalogSize > 0)
    }
}

// MARK: - ModelDownloader Tests

@Suite("Model Downloader Cache")
struct ModelDownloaderCacheTests {
    
    @Test("Check cache path format")
    func testCachePathFormat() {
        let path = ModelDownloader.shared.localPath(for: "testuser/test-model")
        
        // Path should be nil if not downloaded (unless this was previously cached)
        // If it exists, it should be in the HF cache structure
        if let path {
            #expect(path.path.contains(".cache/huggingface/hub"))
        }
    }
    
    @Test("List cached repositories")
    func testListCachedRepos() {
        let repos = ModelDownloader.shared.cachedRepositories()
        
        // Verify we can access the array
        #expect(repos.count >= 0)
    }
    
    @Test("Total cache size")
    func testTotalCacheSize() {
        let size = ModelDownloader.shared.totalCacheSize()
        
        // Size should be non-negative
        #expect(size >= 0)
    }
    
    @Test("Check downloaded status")
    func testIsDownloadedStatus() {
        // Check a repo that definitely doesn't exist
        let isDownloaded = ModelDownloader.shared.isDownloaded(repo: "nonexistent/fake-model-xyz123")
        #expect(isDownloaded == false)
    }
}

// MARK: - FileManager Extensions Tests

@Suite("FileManager Extensions")
struct FileManagerExtensionsTests {
    
    @Test("Available storage space")
    func testAvailableStorageSpace() throws {
        let space = try FileManager.default.availableStorageSpace()
        
        // Should have at least 1 GB available on most systems
        #expect(space > 1_000_000_000)
    }
    
    @Test("Directory size calculation")
    func testDirectorySize() throws {
        // Create a temporary directory with a file
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_dir_size_\(UUID().uuidString)")
        
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        
        // Write some data
        let testData = Data(repeating: 0x42, count: 1000)
        let testFile = tempDir.appendingPathComponent("test.bin")
        try testData.write(to: testFile)
        
        // Check size
        let size = FileManager.default.directorySize(at: tempDir)
        #expect(size >= 1000)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
}

// MARK: - Download Task Tests

@Suite("Download Task")
struct DownloadTaskTests {
    
    @Test("Download task initialization")
    func testTaskInit() {
        let variant = ModelVariant(
            id: "test",
            name: "Test",
            quantization: .fp16,
            sizeBytes: 1000,
            repo: "test/repo",
            files: ["model.bin"]
        )
        
        let task = DownloadTask(
            id: variant.id,
            variant: variant,
            status: .downloading,
            startedAt: Date()
        )
        
        #expect(task.id == "test")
        #expect(task.status == .downloading)
        #expect(task.elapsedTime != nil)
    }
    
    @Test("Download task estimated time")
    func testTaskEstimatedTime() {
        let variant = ModelVariant(
            id: "test",
            name: "Test",
            quantization: .fp16,
            sizeBytes: 100_000_000,
            repo: "test/repo",
            files: ["model.bin"]
        )
        
        let progress = DownloadProgress(
            fractionCompleted: 0.5,
            completedBytes: 50_000_000,
            totalBytes: 100_000_000,
            bytesPerSecond: 10_000_000  // 10 MB/s
        )
        
        let task = DownloadTask(
            id: variant.id,
            variant: variant,
            status: .downloading,
            progress: progress,
            startedAt: Date()
        )
        
        // 50 MB remaining at 10 MB/s = 5 seconds
        #expect(task.estimatedTimeRemaining != nil)
        if let eta = task.estimatedTimeRemaining {
            #expect(eta > 4 && eta < 6)
        }
    }
}

// MARK: - Package Download Progress Tests

@Suite("Package Download Progress")
struct PackageDownloadProgressTests {
    
    @Test("Overall progress calculation")
    func testOverallProgress() {
        let variant = ModelVariant(
            id: "test",
            name: "Test",
            quantization: .fp16,
            sizeBytes: 1000,
            repo: "test/repo",
            files: []
        )
        
        let variantProgress = DownloadProgress(
            fractionCompleted: 0.5,
            completedBytes: 500,
            totalBytes: 1000,
            bytesPerSecond: nil
        )
        
        // 1 completed, currently on 2nd (50%), 4 total
        // Overall: 1/4 + (0.5/4) = 0.25 + 0.125 = 0.375
        let progress = PackageDownloadProgress(
            currentVariant: variant,
            variantProgress: variantProgress,
            completedCount: 1,
            totalCount: 4
        )
        
        #expect(progress.overallFraction > 0.3 && progress.overallFraction < 0.4)
        #expect(progress.overallPercent == Int(progress.overallFraction * 100))
    }
}
