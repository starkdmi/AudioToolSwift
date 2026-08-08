//
//  ModelDownloader.swift
//  AudioToolCore
//
//  HuggingFace Hub model downloading with progress and cancellation
//

import Foundation
import Hub

/// Progress information for model downloads
public struct DownloadProgress: Sendable {
    /// Fraction completed (0.0 - 1.0)
    public let fractionCompleted: Double
    
    /// Downloaded bytes
    public let completedBytes: Int64
    
    /// Total bytes to download
    public let totalBytes: Int64
    
    /// Download speed in bytes per second (nil if unknown)
    public let bytesPerSecond: Double?
    
    /// Progress percentage (0-100)
    public var percentComplete: Int {
        Int(fractionCompleted * 100)
    }
    
    /// Human-readable speed string
    public var speedString: String? {
        guard let speed = bytesPerSecond else { return nil }
        if speed > 1_000_000 {
            return String(format: "%.1f MB/s", speed / 1_000_000)
        } else if speed > 1_000 {
            return String(format: "%.1f KB/s", speed / 1_000)
        } else {
            return String(format: "%.0f B/s", speed)
        }
    }
}

/// Model download manager using HuggingFace Hub
///
/// Downloads model files from HuggingFace with progress reporting and cancellation.
/// Uses standard HF cache location (`~/.cache/huggingface/hub/`).
///
/// Usage:
/// ```swift
/// // Download with progress
/// for try await progress in ModelDownloader.shared.download(repo: "starkdmi/MossFormer2_SE_48K_MLX") {
///     print("\(progress.percentComplete)% - \(progress.speedString ?? "")")
/// }
///
/// // Check if cached
/// if let path = ModelDownloader.shared.localPath(
///     for: "starkdmi/MossFormer2_SE_48K_MLX",
///     matching: ["model_fp16.safetensors", "config.json"]
/// ) {
///     print("Model at: \(path)")
/// }
/// ```
public actor ModelDownloader {
    /// Shared instance
    public static let shared = ModelDownloader()
    
    private init() {}
    
    // MARK: - Download
    
    /// Download model files from HuggingFace Hub
    /// - Parameters:
    ///   - repo: Repository ID (e.g., "starkdmi/MossFormer2_SE_48K_MLX")
    ///   - matching: Glob patterns for files to download
    /// - Returns: AsyncThrowingStream of progress updates, final URL on completion
    public func download(
        repo: String,
        matching globs: [String] = ["*.safetensors", "config.json"]
    ) -> AsyncThrowingStream<DownloadProgress, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    try Task.checkCancellation()
                    _ = try await self.downloadAndGetPath(
                        repo: repo,
                        matching: globs
                    ) { progress in
                        _ = continuation.yield(progress)
                    }
                    try Task.checkCancellation()
                    // Yield final 100% progress
                    _ = continuation.yield(DownloadProgress(
                        fractionCompleted: 1.0,
                        completedBytes: 0,
                        totalBytes: 0,
                        bytesPerSecond: nil
                    ))
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }
    
    /// Download model and return final path
    /// - Parameters:
    ///   - repo: Repository ID
    ///   - matching: Glob patterns
    ///   - progress: Progress callback
    /// - Returns: Local directory URL
    public func downloadAndGetPath(
        repo: String,
        matching globs: [String] = ["*.safetensors", "config.json"],
        progress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> URL {
        guard Self.repositoryComponents(for: repo) != nil else {
            throw AudioToolError.modelNotFound(
                "Invalid HuggingFace repository identifier '\(repo)'. Expected 'owner/name'."
            )
        }
        return try await RepositoryCacheAccess.shared.withExclusiveAccess(to: repo) {
            try Task.checkCancellation()
            return try await HubApi.shared.snapshot(
                from: Hub.Repo(id: repo),
                matching: globs
            ) { foundationProgress, speed in
                progress(DownloadProgress(
                    fractionCompleted: foundationProgress.fractionCompleted,
                    completedBytes: foundationProgress.completedUnitCount,
                    totalBytes: foundationProgress.totalUnitCount,
                    bytesPerSecond: speed
                ))
            }
        }
    }
    
    // MARK: - Cache Management
    
    /// Check whether any repository cache entry exists, without validating files.
    public nonisolated func isDownloaded(repo: String) -> Bool {
        localPath(for: repo) != nil
    }

    /// A repository directory is not evidence that a particular precision or
    /// model variant is installed. Every required glob must match an entry.
    public nonisolated func isDownloaded(variant: ModelVariant) -> Bool {
        localPath(for: variant) != nil
    }

    /// Check whether any snapshot contains every requested file pattern.
    public nonisolated func isDownloaded(
        repo: String,
        matching globs: [String]
    ) -> Bool {
        localPath(for: repo, matching: globs) != nil
    }

    /// Return a repository snapshot only when it contains every file required to
    /// load this variant. Optional owned/download files do not determine whether
    /// an otherwise usable snapshot is installed.
    public nonisolated func localPath(for variant: ModelVariant) -> URL? {
        localPath(for: variant.repo, matching: variant.requiredFiles)
    }

    /// Return the highest-priority cache location containing every requested pattern.
    ///
    /// Filtering all candidates is important because HuggingFace can leave a
    /// newer, partial snapshot next to an older complete one after interruption.
    public nonisolated func localPath(
        for repo: String,
        matching globs: [String]
    ) -> URL? {
        Self.firstCompletePath(
            in: candidatePaths(for: repo),
            matching: globs
        )
    }
    
    /// Get any local cache path for a repo (nil if no cache entry exists).
    ///
    /// This does not establish that a model is loadable. Model loaders should use
    /// `localPath(for:matching:)` or `localPath(for:)` with a `ModelVariant`.
    /// Checks both Swift Hub library location (~/Documents/huggingface/models/)
    /// and Python hf CLI location (~/.cache/huggingface/hub/)
    public nonisolated func localPath(for repo: String) -> URL? {
        candidatePaths(for: repo).first
    }

    /// Select the highest-priority complete candidate. Python snapshot candidates
    /// are ordered newest-first; keeping selection separate also makes
    /// interrupted-snapshot behavior deterministic to test.
    nonisolated static func firstCompletePath(
        in candidates: [URL],
        matching globs: [String]
    ) -> URL? {
        candidates.first { hasRequiredFiles(at: $0, patterns: globs) }
    }

    private nonisolated func candidatePaths(for repo: String) -> [URL] {
        guard let roots = Self.repositoryCacheRoots(
            for: repo,
            homeDirectory: FileManager.default.homeDirectoryForCurrentUser
        ) else { return [] }

        var candidates: [URL] = []
        let containers = Self.cacheContainerRoots(homeDirectory: FileManager.default.homeDirectoryForCurrentUser)
        // Swift Hub library uses ~/Documents/huggingface/models/{owner}/{repo}
        if FileManager.default.fileExists(atPath: roots.swiftHub.path),
           Self.isStrictDescendant(roots.swiftHub, of: containers[0]) {
            candidates.append(roots.swiftHub)
        }
        
        // Python hf CLI uses ~/.cache/huggingface/hub/models--{owner}--{repo}/snapshots/{hash}
        if FileManager.default.fileExists(atPath: roots.pythonHub.path) {
            let snapshotsDir = roots.pythonHub.appendingPathComponent("snapshots")
            if let snapshots = try? FileManager.default.contentsOfDirectory(
                at: snapshotsDir,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            ) {
                candidates.append(contentsOf: snapshots.filter {
                    Self.isStrictDescendant($0, of: containers[1])
                }.sorted { lhs, rhs in
                    let left = (try? lhs.resourceValues(
                        forKeys: [.contentModificationDateKey]
                    ).contentModificationDate) ?? .distantPast
                    let right = (try? rhs.resourceValues(
                        forKeys: [.contentModificationDateKey]
                    ).contentModificationDate) ?? .distantPast
                    return left > right
                })
            }
        }
        return candidates
    }

    /// Parse a HuggingFace repository identifier without allowing path
    /// components to escape either cache root. Repository IDs accepted by this
    /// package are deliberately limited to the canonical `owner/name` form.
    nonisolated static func repositoryComponents(
        for repository: String
    ) -> (owner: String, name: String)? {
        let components = repository.split(
            separator: "/",
            omittingEmptySubsequences: false
        )
        guard components.count == 2 else { return nil }

        let owner = String(components[0])
        let name = String(components[1])
        let invalidComponents = CharacterSet(charactersIn: "\\/")
            .union(.controlCharacters)
        guard !owner.isEmpty,
              !name.isEmpty,
              owner != ".",
              owner != "..",
              name != ".",
              name != "..",
              owner.rangeOfCharacter(from: invalidComponents) == nil,
              name.rangeOfCharacter(from: invalidComponents) == nil
        else { return nil }

        return (owner, name)
    }

    /// Resolve the two cache locations from validated components. Keeping this
    /// helper injectable makes the containment guarantee independently testable.
    nonisolated static func repositoryCacheRoots(
        for repository: String,
        homeDirectory: URL
    ) -> (swiftHub: URL, pythonHub: URL)? {
        guard let components = repositoryComponents(for: repository) else {
            return nil
        }
        let standardizedHome = homeDirectory.standardizedFileURL
        let swiftModels = standardizedHome
            .appendingPathComponent("Documents/huggingface/models", isDirectory: true)
        let pythonModels = standardizedHome
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
        let swiftHub = swiftModels
            .appendingPathComponent(components.owner, isDirectory: true)
            .appendingPathComponent(components.name, isDirectory: true)
            .standardizedFileURL
        let pythonHub = pythonModels
            .appendingPathComponent(
                "models--\(components.owner)--\(components.name)",
                isDirectory: true
            )
            .standardizedFileURL

        guard swiftHub.path.hasPrefix(swiftModels.standardizedFileURL.path + "/"),
              pythonHub.path.hasPrefix(pythonModels.standardizedFileURL.path + "/")
        else { return nil }
        return (swiftHub, pythonHub)
    }

    /// Test whether every requested file pattern exists below a snapshot root.
    public nonisolated static func hasRequiredFiles(
        at root: URL,
        patterns: [String]
    ) -> Bool {
        guard !patterns.isEmpty else { return false }
        let entries = relativeFileEntries(at: root, includeUnusableEntries: false)
        return patterns.allSatisfy { pattern in
            entries.contains { path($0, matches: pattern) }
        }
    }

    /// Test whether at least one file owned by a variant exists below a cache
    /// root. Unlike installation verification, deletion must also find failed or
    /// cancelled downloads whose required file set is incomplete.
    nonisolated static func hasAnyMatchingFile(
        at root: URL,
        patterns: [String]
    ) -> Bool {
        guard !patterns.isEmpty else { return false }
        let entries = relativeFileEntries(at: root, includeUnusableEntries: true)
        return entries.contains { entry in
            patterns.contains { path(entry, matches: $0) }
        }
    }

    /// Minimal HuggingFace-compatible glob matcher. `*` stays within one path
    /// component, `**` may cross directories, and `?` matches one non-separator.
    public nonisolated static func path(_ path: String, matches glob: String) -> Bool {
        let normalizedPath = path.replacingOccurrences(of: "\\", with: "/")
        let normalizedGlob = glob.replacingOccurrences(of: "\\", with: "/")
        let characters = Array(normalizedGlob)
        var expression = "^"
        var index = 0
        while index < characters.count {
            let character = characters[index]
            if character == "*" {
                if index + 1 < characters.count, characters[index + 1] == "*" {
                    // A recursive component followed by `/` also matches zero
                    // directories, so `**/*.safetensors` finds both `model…` at
                    // the root and `weights/model…` below it.
                    if index + 2 < characters.count, characters[index + 2] == "/" {
                        expression += "(?:.*/)?"
                        index += 3
                    } else {
                        expression += ".*"
                        index += 2
                    }
                    continue
                }
                expression += "[^/]*"
            } else if character == "?" {
                expression += "[^/]"
            } else {
                if "\\.^$|+()[]{}".contains(character) { expression += "\\" }
                expression.append(character)
            }
            index += 1
        }
        expression += "$"
        return normalizedPath.range(of: expression, options: .regularExpression) != nil
    }

    private nonisolated static func relativeFileEntries(
        at root: URL,
        includeUnusableEntries: Bool
    ) -> [String] {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }
        let rootPath = root.standardizedFileURL.path
        return enumerator.compactMap { element in
            guard let url = element as? URL else { return nil }
            // A directory named like a model file is not a valid installation.
            let isDirectory = try? url.resourceValues(
                forKeys: [.isDirectoryKey]
            ).isDirectory
            if isDirectory == true {
                return nil
            }
            // Broken snapshot symlinks must remain discoverable for deletion but
            // cannot establish that a model is usable.
            if !includeUnusableEntries,
               (isDirectory == nil || !FileManager.default.fileExists(atPath: url.path)) {
                return nil
            }
            let path = url.standardizedFileURL.path
            guard path.hasPrefix(rootPath) else { return nil }
            return String(path.dropFirst(rootPath.count)).trimmingCharacters(
                in: CharacterSet(charactersIn: "/")
            )
        }
    }
    
    // MARK: - Deletion
    
    /// Delete cached model files for a repository
    /// - Parameter repo: Repository ID to delete
    /// - Throws: FileManager errors if deletion fails
    public func delete(repo: String) async throws {
        guard Self.repositoryComponents(for: repo) != nil else {
            throw AudioToolError.modelNotFound(
                "Invalid HuggingFace repository identifier '\(repo)'. Expected 'owner/name'."
            )
        }
        try await RepositoryCacheAccess.shared.withExclusiveAccess(to: repo) {
            let home = FileManager.default.homeDirectoryForCurrentUser
            try Self.removeRepositoryCacheEntries(for: repo, homeDirectory: home)
            guard let roots = Self.repositoryCacheRoots(for: repo, homeDirectory: home)
            else { return }
            Self.cleanupEmptyHuggingFaceCache(for: repo)
            Self.cleanupEmptySwiftHubCache(for: roots.swiftHub)
        }
    }

    /// Delete only files owned by a variant, retaining files shared with any
    /// installed sibling variant from the same repository.
    public func delete(
        variant: ModelVariant,
        preserving siblings: [ModelVariant]
    ) async throws {
        guard Self.repositoryComponents(for: variant.repo) != nil else {
            throw AudioToolError.modelNotFound(
                "Invalid HuggingFace repository identifier '\(variant.repo)'. Expected 'owner/name'."
            )
        }
        try await RepositoryCacheAccess.shared.withExclusiveAccess(
            to: variant.repo
        ) { [self] in
            let roots = candidatePaths(for: variant.repo).filter {
                Self.hasAnyMatchingFile(at: $0, patterns: variant.files)
            }
            guard !roots.isEmpty else { return }
            let installedSiblings = siblings.filter {
                $0.repo == variant.repo &&
                    $0.id != variant.id &&
                    isDownloaded(variant: $0)
            }
            let protectedPatterns = installedSiblings.flatMap(\.files)

            for root in roots {
                try Self.deleteVariantFiles(
                    at: root,
                    targetPatterns: variant.files,
                    protectedPatterns: protectedPatterns
                )
            }
            Self.cleanupEmptyHuggingFaceCache(for: variant.repo)
        }
    }

    private nonisolated static func cleanupEmptyHuggingFaceCache(for repo: String) {
        guard let cache = repositoryCacheRoots(
            for: repo,
            homeDirectory: FileManager.default.homeDirectoryForCurrentUser
        )?.pythonHub else { return }
        let snapshots = cache.appendingPathComponent("snapshots")
        guard let contents = try? FileManager.default.contentsOfDirectory(
            atPath: snapshots.path
        ) else { return }
        if contents.allSatisfy({ $0.hasPrefix(".") }) {
            try? FileManager.default.removeItem(at: cache)
        }
    }

    private nonisolated static func cleanupEmptySwiftHubCache(for repository: URL) {
        let owner = repository.deletingLastPathComponent()
        guard (try? FileManager.default.contentsOfDirectory(atPath: owner.path).isEmpty) == true
        else { return }
        try? FileManager.default.removeItem(at: owner)
    }

    nonisolated static func deleteVariantFiles(
        at root: URL,
        targetPatterns: [String],
        protectedPatterns: [String]
    ) throws {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return }
        let rootPath = root.standardizedFileURL.path
        let entries = enumerator.compactMap { $0 as? URL }.sorted {
            $0.pathComponents.count > $1.pathComponents.count
        }
        let blobContext = huggingFaceBlobContext(forSnapshotRoot: root)
        var candidateBlobs: Set<URL> = []

        for url in entries {
            let entryPath = url.standardizedFileURL.path
            guard entryPath.hasPrefix(rootPath) else { continue }
            let relative = String(entryPath.dropFirst(rootPath.count)).trimmingCharacters(
                in: CharacterSet(charactersIn: "/")
            )
            let belongsToTarget = targetPatterns.contains { path(relative, matches: $0) }
            let isShared = protectedPatterns.contains { path(relative, matches: $0) }
            if belongsToTarget && !isShared {
                if let blobContext,
                   let blob = blobTarget(forSymlink: url, in: blobContext) {
                    candidateBlobs.insert(blob)
                }
                try FileManager.default.removeItem(at: url)
            }
        }

        // Remove now-empty subdirectories, deepest first, without broadening the
        // deletion beyond this snapshot root.
        for url in entries {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
                  isDirectory.boolValue,
                  (try? FileManager.default.contentsOfDirectory(atPath: url.path).isEmpty) == true
            else { continue }
            try FileManager.default.removeItem(at: url)
        }
        if (try? FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty) == true {
            try FileManager.default.removeItem(at: root)
        }

        if let blobContext, !candidateBlobs.isEmpty {
            try removeUnreferencedBlobs(candidateBlobs, in: blobContext)
        }
    }

    private struct HuggingFaceBlobContext {
        let snapshotsDirectory: URL
        let blobsDirectory: URL
    }

    /// Recognize only the standard Python HuggingFace cache layout. This keeps
    /// symlink cleanup scoped to `models--owner--repo/blobs` and prevents an
    /// arbitrary snapshot symlink from broadening deletion outside its cache.
    private nonisolated static func huggingFaceBlobContext(
        forSnapshotRoot root: URL
    ) -> HuggingFaceBlobContext? {
        let snapshotsDirectory = root.standardizedFileURL.deletingLastPathComponent()
        guard snapshotsDirectory.lastPathComponent == "snapshots" else { return nil }

        let blobsDirectory = snapshotsDirectory
            .deletingLastPathComponent()
            .appendingPathComponent("blobs", isDirectory: true)
            .standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: blobsDirectory.path,
            isDirectory: &isDirectory
        ), isDirectory.boolValue,
              (try? FileManager.default.destinationOfSymbolicLink(
                atPath: blobsDirectory.path
              )) == nil else { return nil }

        return HuggingFaceBlobContext(
            snapshotsDirectory: snapshotsDirectory,
            blobsDirectory: blobsDirectory
        )
    }

    private nonisolated static func blobTarget(
        forSymlink link: URL,
        in context: HuggingFaceBlobContext
    ) -> URL? {
        guard let destination = try? FileManager.default.destinationOfSymbolicLink(
            atPath: link.path
        ) else { return nil }

        let unresolvedTarget: URL
        if destination.hasPrefix("/") {
            unresolvedTarget = URL(fileURLWithPath: destination)
        } else {
            unresolvedTarget = link.deletingLastPathComponent()
                .appendingPathComponent(destination)
        }
        let target = unresolvedTarget.standardizedFileURL.resolvingSymlinksInPath()
        let blobRootPath = context.blobsDirectory.path
        guard target.path.hasPrefix(blobRootPath + "/") else { return nil }

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: target.path, isDirectory: &isDirectory),
              !isDirectory.boolValue else { return nil }
        return target
    }

    private nonisolated static func removeUnreferencedBlobs(
        _ candidates: Set<URL>,
        in context: HuggingFaceBlobContext
    ) throws {
        var referencedBlobPaths: Set<String> = []
        if let enumerator = FileManager.default.enumerator(
            at: context.snapshotsDirectory,
            includingPropertiesForKeys: nil,
            options: []
        ) {
            for case let entry as URL in enumerator {
                if let target = blobTarget(forSymlink: entry, in: context) {
                    referencedBlobPaths.insert(target.path)
                }
            }
        }

        for blob in candidates where !referencedBlobPaths.contains(blob.path) {
            // Revalidate immediately before deletion in case the candidate was
            // concurrently removed by another variant cleanup.
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: blob.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else { continue }
            try FileManager.default.removeItem(at: blob)
        }
    }
    
    /// Get size of cached model on disk
    /// - Parameter repo: Repository ID
    /// - Returns: Size in bytes, or 0 if not cached
    public nonisolated func cacheSize(for repo: String) -> Int64 {
        guard let roots = Self.repositoryCacheRoots(
            for: repo,
            homeDirectory: FileManager.default.homeDirectoryForCurrentUser
        ) else { return 0 }
        return [roots.swiftHub, roots.pythonHub].reduce(0) {
            $0 + FileManager.default.directorySize(at: $1)
        }
    }
    
    /// Get total size of all cached models
    /// - Returns: Total size in bytes
    public nonisolated func totalCacheSize() -> Int64 {
        Self.totalCacheSize(
            homeDirectory: FileManager.default.homeDirectoryForCurrentUser
        )
    }
    
    /// Clear all cached models
    /// - Throws: FileManager errors if deletion fails
    public func clearCache() throws {
        try Self.clearCache(
            homeDirectory: FileManager.default.homeDirectoryForCurrentUser
        )
    }
    
    /// List all cached repository IDs
    /// - Returns: Array of repository IDs
    public nonisolated func cachedRepositories() -> [String] {
        Self.cachedRepositories(
            homeDirectory: FileManager.default.homeDirectoryForCurrentUser
        )
    }


    // MARK: - Cache Root Operations

    nonisolated static func totalCacheSize(homeDirectory: URL) -> Int64 {
        cacheContainerRoots(homeDirectory: homeDirectory).reduce(0) {
            $0 + FileManager.default.directorySize(at: $1)
        }
    }

    nonisolated static func clearCache(homeDirectory: URL) throws {
        let roots = cacheContainerRoots(homeDirectory: homeDirectory)
        let swiftModels = roots[0]
        if let owners = try? FileManager.default.contentsOfDirectory(
            at: swiftModels,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            for owner in owners {
                try FileManager.default.removeItem(at: owner)
            }
        }

        let pythonHub = roots[1]
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: pythonHub,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            for entry in entries where entry.lastPathComponent.hasPrefix("models--") {
                try FileManager.default.removeItem(at: entry)
            }
        }
    }

    /// Delete a repository from both cache layouts after resolving symlinks and
    /// proving each existing target remains within its approved cache container.
    nonisolated static func removeRepositoryCacheEntries(
        for repository: String,
        homeDirectory: URL
    ) throws {
        guard let roots = repositoryCacheRoots(
            for: repository,
            homeDirectory: homeDirectory
        ) else {
            throw AudioToolError.modelNotFound(
                "Invalid HuggingFace repository identifier '\(repository)'. Expected 'owner/name'."
            )
        }
        let containers = cacheContainerRoots(homeDirectory: homeDirectory)
        for (target, container) in [
            (roots.swiftHub, containers[0]),
            (roots.pythonHub, containers[1]),
        ] where FileManager.default.fileExists(atPath: target.path) {
            guard isStrictDescendant(target, of: container) else {
                throw AudioToolError.resourceUnavailable(
                    "Refusing to delete cache entry outside its approved root: \(target.path)"
                )
            }
            try FileManager.default.removeItem(at: target)
        }
    }

    private nonisolated static func isStrictDescendant(
        _ candidate: URL,
        of container: URL
    ) -> Bool {
        let resolvedContainer = container.standardizedFileURL
            .resolvingSymlinksInPath().path
        let resolvedCandidate = candidate.standardizedFileURL
            .resolvingSymlinksInPath().path
        return resolvedCandidate.hasPrefix(resolvedContainer + "/")
    }

    nonisolated static func cachedRepositories(homeDirectory: URL) -> [String] {
        let roots = cacheContainerRoots(homeDirectory: homeDirectory)
        var repositories: Set<String> = []

        if let owners = try? FileManager.default.contentsOfDirectory(
            at: roots[0],
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for owner in owners {
                guard (try? owner.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true,
                      let models = try? FileManager.default.contentsOfDirectory(
                        at: owner,
                        includingPropertiesForKeys: [.isDirectoryKey],
                        options: [.skipsHiddenFiles]
                      )
                else { continue }
                for model in models
                where (try? model.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                    repositories.insert("\(owner.lastPathComponent)/\(model.lastPathComponent)")
                }
            }
        }

        if let entries = try? FileManager.default.contentsOfDirectory(
            at: roots[1],
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for entry in entries {
                let name = entry.lastPathComponent
                guard name.hasPrefix("models--") else { continue }
                let components = name.dropFirst("models--".count)
                    .components(separatedBy: "--")
                guard components.count == 2 else { continue }
                repositories.insert("\(components[0])/\(components[1])")
            }
        }

        return repositories.sorted()
    }

    private nonisolated static func cacheContainerRoots(
        homeDirectory: URL
    ) -> [URL] {
        let home = homeDirectory.standardizedFileURL
        return [
            home.appendingPathComponent(
                "Documents/huggingface/models",
                isDirectory: true
            ),
            home.appendingPathComponent(
                ".cache/huggingface/hub",
                isDirectory: true
            ),
        ]
    }
}
