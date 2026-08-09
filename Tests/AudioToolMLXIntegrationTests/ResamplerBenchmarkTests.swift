//
//  ResamplerBenchmarkTests.swift
//  AudioToolMLXIntegrationTests
//
//  What the polyphase resampler costs, in numbers someone else can reproduce.
//

import XCTest
import AudioToolTestSupport
import AudioUtils

/// Timings for AudioUtils' polyphase resampler, reported rather than asserted.
///
/// The engine lives in SwiftAudio; `AudioUtilsTests.PolyphaseResamplingTests` is what
/// pins its arithmetic. This measures it at the rates *this* package resamples at,
/// and on the call pattern Chatterbox conditioning actually performs, neither of
/// which the library knows about.
///
/// This exists because the alternative is a claim. `kaiser_best` uses 50 zero
/// crossings where `scipy.signal.resample_poly`'s default uses 10, so the
/// conditioning resample does ~100 multiply-accumulates per output sample instead of
/// ~20 - and that ratio is inherent to the design, since the taps are what buy the
/// -120 dB stopband. Whether it *matters* is a different question from whether it is
/// true, and only measurement answers it.
///
/// ```bash
/// ./Scripts/build_mlx_metallib.sh release
/// RUN_BENCHMARKS=1 swift test -c release -Xswiftc -enable-testing \
///   --filter ResamplerBenchmarkTests
/// ```
///
/// `-c release` is not optional and the suite skips without it; `-enable-testing`
/// is what keeps `@testable import` working in a release configuration. A debug build of this
/// code is ~70x slower and moves the two kernels apart by 17x rather than 3x, because
/// bounds checks and unspecialised generics dominate everything the optimiser was
/// supposed to remove. Those numbers describe the compiler, not the resampler.
///
/// Every number is printed as a ratio as well as an absolute, because the ratio is
/// the part that survives moving to a different machine.
///
/// Recorded on an M1 Pro (8 cores, 16 GB), Swift 6.3, release:
///
/// | conversion      | scalar   | vectorized | ratio |
/// | --------------- | -------- | ---------- | ----- |
/// | 24k    -> 16k   | 13.5 ms  | 4.1 ms     | 3.3x  |
/// | 48k    -> 16k   | 30.0 ms  | 7.7 ms     | 3.9x  |
/// | 22.05k -> 16k   | 13.0 ms  | 4.7 ms     | 2.8x  |
///
/// against 86.5 / 173.5 / 80.4 ms for the `[Double]` port these replaced, and
/// 24.0 / 54.9 / 22.1 ms for scipy's own C `upfirdn` on the same conversions. The
/// Swift is now faster than scipy at `kaiser_best`, and faster than scipy is at its
/// own 10-zero-crossing default.
///
/// Not an `IntegrationTestCase`: no model, no weights, no GPU.
final class ResamplerBenchmarkTests: XCTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipUnless(TestGate.runBenchmarks, TestGate.benchmarksDisabled)
        #if DEBUG
        throw XCTSkip(
            "benchmark needs an optimised build - run: "
                + "RUN_BENCHMARKS=1 swift test -c release -Xswiftc -enable-testing "
                + "--filter ResamplerBenchmarkTests"
        )
        #endif
    }

    // MARK: - Fixtures

    /// A tone comb plus a little hiss, so the filter is rejecting something.
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

    /// Best of `repeats`, after one warm-up.
    ///
    /// Best rather than mean: the thing being measured is deterministic, so every
    /// millisecond above the minimum is another process on the machine, not variance
    /// in the code.
    private func best(repeats: Int = 5, _ body: () -> Void) -> Double {
        body()
        var fastest = Double.infinity
        for _ in 0..<repeats {
            let start = DispatchTime.now().uptimeNanoseconds
            body()
            fastest = min(fastest, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        return fastest
    }

    // MARK: - Conversions

    /// The three conversions Chatterbox conditioning actually performs.
    func testResamplerThroughput() {
        let conversions = [(24000, 16000), (48000, 16000), (22050, 16000)]
        let seconds = 10

        print("")
        print("resampler, \(seconds) s of audio, kaiser_best, best of 5")
        print("conversion         taps     scalar  vectorized   ratio   x realtime")

        for (origSR, targetSR) in conversions {
            let x = signal(count: origSR * seconds, sampleRate: origSR)
            let r = ratio(origSR, targetSR)
            let numtaps = 2 * ResamplerDesign.kaiserBest.zeroCrossings * max(r.up, r.down) + 1

            let scalar = best {
                _ = resamplePolyphase(
                    x, up: r.up, down: r.down, design: .kaiserBest, kernel: .scalarReference
                )
            }
            let vectorized = best {
                _ = resamplePolyphase(
                    x, up: r.up, down: r.down, design: .kaiserBest, kernel: .vectorized
                )
            }

            print(String(
                format: "%6d -> %5d  %7d  %8.2f ms %8.2f ms  %5.1fx  %9.0f",
                origSR, targetSR, numtaps, scalar, vectorized,
                scalar / vectorized, Double(seconds) * 1000 / vectorized
            ))

            // A floor, not a target. Measured 2.8-3.9x; 1.5x is low enough to survive
            // a loaded machine or a slower core and high enough that deleting the
            // vector path fails here instead of silently costing 3x.
            XCTAssertGreaterThan(
                scalar / vectorized, 1.5,
                "\(origSR) -> \(targetSR): the vectorized kernel is no longer pulling "
                    + "its weight (scalar \(scalar) ms, vectorized \(vectorized) ms)"
            )
        }
    }

    // MARK: - Where the cost actually lands

    /// One `setReferenceAudio`, not one generated token.
    ///
    /// `ChatterboxTTSProvider.rebuildReferenceConditioning` resamples three times: the
    /// prompt to 16 kHz for the voice encoder, the prompt to 24 kHz for S3Gen, and
    /// the first 10 s of that back to 16 kHz for the log-mel front end. This is the
    /// number that decides whether the filter's cost is worth an argument - it is
    /// paid once per reference voice and never per token, so against a multi-second
    /// generation it is close to invisible.
    ///
    /// Recorded on an M1 Pro: 16.0 ms, against 290.8 ms for the port this replaced.
    func testConditioningPathCost() {
        let sourceSR = 22050
        let prompt = signal(count: sourceSR * 12, sampleRate: sourceSR)

        let elapsed = best {
            _ = resamplePolyphase(prompt, fromRate: sourceSR, toRate: 16000)
            let at24k = resamplePolyphase(prompt, fromRate: sourceSR, toRate: 24000)
            let window = Array(at24k[0..<min(10 * 24000, at24k.count)])
            _ = resamplePolyphase(window, fromRate: 24000, toRate: 16000)
        }

        print("")
        print(String(
            format: "setReferenceAudio, 12 s prompt at %d Hz: %.1f ms of resampling",
            sourceSR, elapsed
        ))
    }

    // MARK: - What the design cache is worth

    /// What `DesignedFilterCache` saves, measured end to end rather than in isolation.
    ///
    /// `numtaps` is `2 * zeroCrossings * max(up, down) + 1`, which is 301 for
    /// 24k -> 16k but 44101 for 22.05k -> 16k, and every tap costs a `sin` and two
    /// `besselI0`. Timing a cold call against a warm one is the honest form of that
    /// number: it is the difference a caller sees, and it grew as a share of the
    /// total when the convolution got faster.
    ///
    /// Recorded on an M1 Pro: 0.57 ms for the 44101-tap design, ~11% of a cold call.
    func testFilterDesignCost() {
        print("")
        print("filter design, cold call minus warm call")
        print("conversion         taps      cold      warm     design")

        for (origSR, targetSR) in [(24000, 16000), (48000, 16000), (22050, 16000)] {
            let x = signal(count: origSR * 10, sampleRate: origSR)
            let r = ratio(origSR, targetSR)
            let numtaps = 2 * ResamplerDesign.kaiserBest.zeroCrossings * max(r.up, r.down) + 1

            // Cold: cache emptied before every run, so each pays the design once.
            var cold = Double.infinity
            for _ in 0..<5 {
                DesignedFilterCache.shared.removeAll()
                let start = DispatchTime.now().uptimeNanoseconds
                _ = resamplePolyphase(x, fromRate: origSR, toRate: targetSR)
                cold = min(cold, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
            let warm = best {
                _ = resamplePolyphase(x, fromRate: origSR, toRate: targetSR)
            }

            print(String(
                format: "%6d -> %5d  %7d  %6.2f ms  %6.2f ms  %6.2f ms",
                origSR, targetSR, numtaps, cold, warm, cold - warm
            ))
        }
    }
}
