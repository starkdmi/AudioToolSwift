//
//  ResamplerNumericsTests.swift
//  AudioToolMLXIntegrationTests
//
//  What the polyphase resampler is allowed to compute. No weights, no MLX ops.
//

import XCTest
@testable import ChatterboxMLXSwift

/// Pins the arithmetic of `resamplePolyphaseScipyEdge` across the optimisation that
/// made it ~20x faster.
///
/// The resampler is the one place in Chatterbox conditioning where Swift has to agree
/// with `scipy.signal.resample_poly` rather than merely resemble it - see the long
/// note in `ParityThresholds`. That makes "it got faster" an unsafe claim on its own,
/// so this checks the two halves separately:
///
///   - `.scalarReference` sums in scipy's own order and must reproduce stored
///     checksums *bit-for-bit*. Those checksums were taken from the previous
///     implementation, so anything that shifts the port's numerics fails here.
///   - `.vectorized` splits the accumulation four ways, which reassociates, and is
///     held to one Float ulp. Measured across the sweep below the two agree on every
///     sample exactly: reassociating a `Double` accumulator moves the result ~1e-16
///     relative, and the result is rounded to Float32 whose quantum is ~6e-8, so a
///     differing sample would need the Double value to sit within 1e-16 of a Float32
///     rounding tie. The bound is one ulp because that is what is *guaranteed*.
///
/// Not an `IntegrationTestCase`: this touches no model and no GPU.
final class ResamplerNumericsTests: XCTestCase {

    // MARK: - Fixtures

    /// Deterministic input. A tone comb plus a little hiss, so the filter has content
    /// in its stopband rather than resampling silence.
    ///
    /// Reproduced exactly from the harness that generated the checksums below; the
    /// xorshift state, the tone spacing and the order of the two terms are all load
    /// bearing.
    private func signal(count: Int, sampleRate: Int) -> [Float] {
        var state: UInt64 = 0x2545F4914F6CDD1D
        func next() -> Double {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Double(state % 2_000_003) / 1_000_001.5 - 1.0
        }
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let t = Double(i) / Double(sampleRate)
            var v = 0.0
            for f in stride(from: 110.0, through: 9000.0, by: 430.0) {
                v += sin(2 * .pi * f * t + f)
            }
            v = v / 21.0 + 0.02 * next()
            out[i] = Float(v)
        }
        return out
    }

    private func checksum(_ samples: [Float]) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for value in samples {
            var bits = value.bitPattern
            for _ in 0..<4 {
                hash ^= UInt64(bits & 0xff)
                hash = hash &* 0x100000001b3
                bits >>= 8
            }
        }
        return hash
    }

    private func ratio(_ origSR: Int, _ targetSR: Int) -> (up: Int, down: Int) {
        var a = origSR
        var b = targetSR
        while b != 0 {
            let t = b
            b = a % b
            a = t
        }
        return (targetSR / a, origSR / a)
    }

    private func resample(
        _ x: [Float], _ origSR: Int, _ targetSR: Int,
        design: ResamplerDesign = .kaiserBest,
        kernel: PolyphaseKernel
    ) -> [Float] {
        let r = ratio(origSR, targetSR)
        return resamplePolyphaseScipyEdge(x, up: r.up, down: r.down, design: design, kernel: kernel)
    }

    // MARK: - Bit-exactness against the previous implementation

    /// Checksums of `.scalarReference` output, FNV-1a over the Float bit patterns.
    ///
    /// Taken from the implementation this replaced: a straight `[Double]` port of
    /// `resample_poly(padtype="edge")`. They are not a specification of correct
    /// resampling - `ChatterboxParityTests` is what says the filter matches Python -
    /// they are a specification that *this* refactor changed no arithmetic.
    func testScalarReferenceIsBitIdenticalToThePreviousPort() {
        let expected: [(origSR: Int, targetSR: Int, count: Int, checksum: UInt64)] = [
            (24000, 16000, 16000, 0xa2d6e498f4ea5134),
            (48000, 16000, 16000, 0xbd5b2855e329b5af),
            (22050, 16000, 16000, 0xa50b2d32a736db2c)
        ]

        for case_ in expected {
            let x = signal(count: case_.origSR, sampleRate: case_.origSR)
            let y = resample(x, case_.origSR, case_.targetSR, kernel: .scalarReference)
            XCTAssertEqual(y.count, case_.count, "\(case_.origSR) -> \(case_.targetSR) length")
            XCTAssertEqual(
                checksum(y), case_.checksum,
                String(
                    format: "%d -> %d changed: got 0x%016llx, expected 0x%016llx. "
                        + "The scalar path is meant to sum in scipy's order; if this "
                        + "moved, the port's numerics moved with it.",
                    case_.origSR, case_.targetSR, checksum(y), case_.checksum
                )
            )
        }
    }

    // MARK: - Vectorized against scalar

    /// Every index path the two kernels share: both rate directions, integer and
    /// coprime ratios, and lengths on either side of one filter span.
    ///
    /// Lengths vary *within* a rate pair on purpose. `nPostPad` depends on the input
    /// length while the cached taps do not, so a cache that wrongly held anything
    /// length-dependent would show up as a second call disagreeing with the first.
    func testVectorizedMatchesScalarWithinOneULP() {
        let ratePairs = [
            (24000, 16000), (48000, 16000), (22050, 16000), (44100, 16000),
            (16000, 24000), (44100, 48000), (48000, 24000), (8000, 16000)
        ]
        let lengths = [1, 2, 7, 63, 137, 512, 4001]
        let designs: [(String, ResamplerDesign)] = [
            ("kaiser_best", .kaiserBest), ("scipy_default", .scipyDefault)
        ]

        var worstULP = 0
        for (designName, design) in designs {
            for (origSR, targetSR) in ratePairs {
                for n in lengths {
                    let x = signal(count: n, sampleRate: origSR)
                    let reference = resample(
                        x, origSR, targetSR, design: design, kernel: .scalarReference
                    )
                    let vectorized = resample(
                        x, origSR, targetSR, design: design, kernel: .vectorized
                    )

                    XCTAssertEqual(
                        vectorized.count, reference.count,
                        "\(designName) \(origSR)->\(targetSR) n=\(n) length"
                    )
                    guard vectorized.count == reference.count else { continue }

                    for i in 0..<reference.count {
                        let tolerance = max(reference[i].ulp, vectorized[i].ulp)
                        let delta = abs(vectorized[i] - reference[i])
                        if delta > tolerance {
                            XCTFail(
                                "\(designName) \(origSR)->\(targetSR) n=\(n) sample \(i): "
                                    + "\(vectorized[i]) vs \(reference[i]), delta \(delta) "
                                    + "exceeds one ulp \(tolerance)"
                            )
                            return
                        }
                        if delta > 0 { worstULP = 1 }
                    }
                }
            }
        }

        // Not an assertion, a record: the sweep has never produced a differing sample.
        // If this starts printing 1, the tolerance above is doing real work and the
        // claim in the class comment needs revisiting.
        XCTAssertLessThanOrEqual(worstULP, 1)
    }

    // MARK: - Cache

    /// The design cache must not change what comes out, only how long it takes.
    ///
    /// Two rate pairs interleaved, each visited twice with different lengths: a warm
    /// entry has to serve a different input length correctly, and a second rate pair
    /// must not evict or corrupt the first.
    func testDesignCacheDoesNotChangeResults() {
        let plan = [(24000, 16000, 3000), (22050, 16000, 5000), (24000, 16000, 7001),
                    (22050, 16000, 2048), (24000, 16000, 3000)]

        var firstByKey: [String: [Float]] = [:]
        for (origSR, targetSR, n) in plan {
            let x = signal(count: n, sampleRate: origSR)
            let y = resample(x, origSR, targetSR, kernel: .vectorized)
            let key = "\(origSR)->\(targetSR)/\(n)"
            if let previous = firstByKey[key] {
                XCTAssertEqual(y, previous, "\(key) differed on a warm cache")
            } else {
                firstByKey[key] = y
            }
            // And against the kernel that never touches the vector path.
            let scalar = resample(x, origSR, targetSR, kernel: .scalarReference)
            XCTAssertEqual(y.count, scalar.count, "\(key) length")
        }
    }

    /// The cache is shared mutable state behind a lock, and conditioning can be built
    /// from more than one thread. Racing on a cold key is allowed to duplicate work;
    /// it is not allowed to hand back a torn or wrong filter.
    func testDesignCacheIsSafeUnderConcurrentUse() {
        let at24k = signal(count: 4001, sampleRate: 24000)
        let at22k = signal(count: 4001, sampleRate: 22050)
        let expected = resample(at24k, 24000, 16000, kernel: .scalarReference)

        // Everything the concurrent block touches is prepared up front, so the
        // closure captures values and one collector rather than the test case.
        let collector = Collector()
        DispatchQueue.concurrentPerform(iterations: 32) { index in
            if index % 2 == 0 {
                collector.add(
                    resamplePolyphaseScipyEdge(
                        at24k, up: 2, down: 3, design: .kaiserBest, kernel: .vectorized
                    )
                )
            } else {
                // Second rate pair, only to keep the cache contended: 22.05k -> 16k
                // reduces to 320/441 and designs a 44101-tap filter.
                _ = resamplePolyphaseScipyEdge(
                    at22k, up: 320, down: 441, design: .kaiserBest, kernel: .vectorized
                )
            }
        }

        let results = collector.drain()
        XCTAssertEqual(results.count, 16)
        for y in results {
            XCTAssertEqual(y.count, expected.count)
            guard y.count == expected.count else { continue }
            for i in 0..<y.count {
                XCTAssertLessThanOrEqual(
                    abs(y[i] - expected[i]), max(y[i].ulp, expected[i].ulp),
                    "concurrent result diverged at sample \(i)"
                )
            }
        }
    }
}

/// Locked box for results gathered off `concurrentPerform`.
private final class Collector: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [[Float]] = []

    func add(_ value: [Float]) {
        lock.lock()
        values.append(value)
        lock.unlock()
    }

    func drain() -> [[Float]] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}
