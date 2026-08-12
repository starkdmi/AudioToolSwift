//
//  ModelIntegrity.swift
//  AudioToolCore
//
//  SHA-256 verification of downloaded model files against their pins
//

import CryptoKit
import Foundation

// MARK: - Report

/// One file whose bytes on disk do not match the pinned hash.
public struct ModelFileMismatch: Sendable, Hashable {

    /// Repository-relative path of the file.
    public let file: String

    /// SHA-256 recorded in ``ModelPin/fileHashes``.
    public let expected: String

    /// SHA-256 actually computed from disk.
    public let actual: String

    public init(file: String, expected: String, actual: String) {
        self.file = file
        self.expected = expected
        self.actual = actual
    }
}

/// The result of checking a snapshot on disk against its pin.
public struct ModelIntegrityReport: Sendable, Hashable {

    /// Repository the snapshot belongs to.
    public let repo: String

    /// Revision it is pinned at, or `nil` if the repository has no pin.
    public let revision: String?

    /// Files that were present and hashed successfully.
    public let verified: [String]

    /// Files that were present and hashed to something else.
    public let mismatches: [ModelFileMismatch]

    /// Whether a pin existed at all. An unpinned repository always "passes",
    /// which is exactly why this is reported separately rather than folded into
    /// ``passed``.
    public var isPinned: Bool { revision != nil }

    /// No file contradicted its pin. True for an unpinned repository, and true
    /// for a pinned one whose files are simply not downloaded yet.
    public var passed: Bool { mismatches.isEmpty }

    /// One line per mismatch, for an error message.
    public var failureDescription: String {
        mismatches
            .map { "\($0.file): expected \($0.expected), found \($0.actual)" }
            .joined(separator: "; ")
    }

    public init(
        repo: String,
        revision: String?,
        verified: [String],
        mismatches: [ModelFileMismatch]
    ) {
        self.repo = repo
        self.revision = revision
        self.verified = verified
        self.mismatches = mismatches
    }
}

// MARK: - Provenance by content

/// Remembers which cache directories have already been shown to hold their pinned
/// bytes.
///
/// The cache lookup that consults this is synchronous and runs on every load, so
/// without a memo a repository whose provenance is not recorded on disk would be
/// re-hashed on each one. Deliberately per-process and in memory: the alternative,
/// writing a marker into the cache directory, puts a file of ours inside a layout
/// HuggingFace owns and that this package's own deletion and empty-directory cleanup
/// walks.
///
/// Keyed by the identity of the pinned files themselves - path, size and modification
/// date of each - so a snapshot that changes after being accepted is hashed again.
///
/// Deliberately not the containing directory's modification date, which is what an
/// earlier version used: a directory's mtime moves when an entry is added or removed,
/// not when an existing file is overwritten in place. A weight truncated or replaced
/// after this cache had accepted it would have kept its entry and never been rehashed,
/// which is the one case the hashing exists to catch.
final class PinnedContentCache: @unchecked Sendable {

    static let shared = PinnedContentCache()

    private struct Key: Hashable {
        let path: String
        let revision: String
        /// Per pinned file: relative path, size, modification date.
        let files: [String]
    }

    private let lock = NSLock()
    private var answers: [Key: Bool] = [:]

    /// Whether every pinned file present in `candidate` hashes to its pinned value,
    /// and at least one did - a directory that happens to contain none of the pinned
    /// files proves nothing about its revision.
    ///
    /// Hashes against the `pin` it is given rather than looking one up by repository:
    /// the caller has already resolved which pin applies, and re-deriving it here
    /// would make this answer for a repository nobody asked about.
    /// - Parameter covering: The files that must all be accounted for. The caller
    ///   has already established that each has a pinned hash; this checks the bytes.
    func matchesPinnedHashes(_ candidate: URL, pin: ModelPin, covering files: [String]) -> Bool {
        guard !pin.fileHashes.isEmpty, !files.isEmpty else { return false }

        let key = Key(
            path: candidate.standardizedFileURL.path,
            revision: pin.revision,
            files: Self.fingerprints(of: files, in: candidate)
        )

        if let answer = lock.withLock({ answers[key] }) { return answer }

        let matches = Self.hashesMatch(candidate, pin: pin, covering: files)
        lock.withLock { answers[key] = matches }
        return matches
    }

    /// Size and modification date per file, as a cheap stand-in for its contents.
    ///
    /// Not proof against a same-size, same-timestamp rewrite - nothing short of
    /// rehashing is - but it catches every ordinary way a cached weight changes:
    /// truncation, replacement, a resumed download completing.
    private static func fingerprints(of files: [String], in candidate: URL) -> [String] {
        files.map { file in
            let url = candidate.appendingPathComponent(file)
            let values = try? url.resourceValues(
                forKeys: [.fileSizeKey, .contentModificationDateKey]
            )
            let size = values?.fileSize.map(String.init) ?? "-"
            let modified = values?.contentModificationDate
                .map { String($0.timeIntervalSince1970) } ?? "-"
            return "\(file):\(size):\(modified)"
        }
    }

    private static func hashesMatch(
        _ candidate: URL,
        pin: ModelPin,
        covering files: [String]
    ) -> Bool {
        var checkedAny = false
        for file in files {
            guard let expected = pin.fileHashes[file] else { return false }
            let path = candidate.appendingPathComponent(file)
            // Absent is not a contradiction: a cache holding fp16 alone has no fp32.
            guard FileManager.default.fileExists(atPath: path.path) else { continue }
            guard let actual = try? ModelDownloader.sha256(ofFileAt: path),
                  actual == expected
            else { return false }
            checkedAny = true
        }
        return checkedAny
    }

    /// Forget everything. For tests, and for a cache that has just been cleared.
    func invalidate() {
        lock.withLock { answers.removeAll() }
    }
}

// MARK: - Verification

public extension ModelDownloader {

    /// Check a downloaded snapshot against ``ModelPins``.
    ///
    /// Only files that are *both* pinned and present are hashed. A missing file is
    /// not a failure: a variant that downloaded fp16 alone has no fp32 on disk and
    /// is perfectly valid. What this catches is a file whose bytes disagree with the
    /// pin - a truncated or resumed-wrong download, a corrupted blob, or a cache
    /// directory filled from a different revision, which the Swift Hub cache layout
    /// makes easy because it stores one directory per repository with no revision in
    /// the path.
    ///
    /// - Parameters:
    ///   - repo: Repository ID.
    ///   - directory: Snapshot to check. Defaults to the highest-priority cache
    ///     location holding any of the pinned files.
    /// - Returns: What was checked and what failed. Never throws on mismatch - the
    ///   caller decides whether a mismatch is fatal.
    nonisolated func verify(
        repo: String,
        at directory: URL? = nil
    ) throws -> ModelIntegrityReport {
        guard let pin = ModelPins.pin(for: repo) else {
            return ModelIntegrityReport(
                repo: repo,
                revision: nil,
                verified: [],
                mismatches: []
            )
        }

        let root: URL?
        if let directory {
            root = directory
        } else {
            root = localPath(for: repo, matching: Array(pin.fileHashes.keys))
                ?? localPath(for: repo)
        }
        guard let root else {
            return ModelIntegrityReport(
                repo: repo,
                revision: pin.revision,
                verified: [],
                mismatches: []
            )
        }

        var verified: [String] = []
        var mismatches: [ModelFileMismatch] = []
        for (file, expected) in pin.fileHashes.sorted(by: { $0.key < $1.key }) {
            let path = root.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: path.path) else { continue }
            let actual = try Self.sha256(ofFileAt: path)
            if actual == expected {
                verified.append(file)
            } else {
                mismatches.append(
                    ModelFileMismatch(file: file, expected: expected, actual: actual)
                )
            }
        }

        return ModelIntegrityReport(
            repo: repo,
            revision: pin.revision,
            verified: verified,
            mismatches: mismatches
        )
    }

    /// SHA-256 of a file, read in chunks.
    ///
    /// Chunked because the largest pinned file here is a 2.7 GB Chatterbox
    /// checkpoint, and `Data(contentsOf:)` would hold all of it at once on a machine
    /// that is about to load the model anyway.
    nonisolated static func sha256(ofFileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        let chunkSize = 4 * 1024 * 1024
        while true {
            let chunk = try handle.read(upToCount: chunkSize) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
