//
//  SharedMLXCacheConcurrencyTests.swift
//  AudioToolMLXIntegrationTests
//
//  Concurrent access regressions for process-wide model utility caches
//

import XCTest
@preconcurrency import MLX
@preconcurrency @testable import USSMLXSwift
@preconcurrency import DemucsMLXSwift
@preconcurrency import ChatterboxMLXSwift

final class SharedMLXCacheConcurrencyTests: MLXTestCase {

    func testUSSPrewarmPopulatesTheOwnedWindowEntry() {
        let istft = ISTFT(n_fft: 16, hop_length: 4, win_length: 16)

        XCTAssertFalse(istft.isNormalizationPrewarmed(numFrames: 4))
        istft.prewarmNormalization(numFrames: 4)
        XCTAssertTrue(istft.isNormalizationPrewarmed(numFrames: 4))

        // An equivalent but separately-owned window intentionally has a
        // different identity and must not claim the first instance's entry.
        let otherISTFT = ISTFT(n_fft: 16, hop_length: 4, win_length: 16)
        XCTAssertFalse(otherISTFT.isNormalizationPrewarmed(numFrames: 4))
    }

    func testDemucsPeriodicHannWindowReusesNormalizationEntry() {
        clearMLXNormBufferCache()
        defer { clearMLXNormBufferCache() }

        let real = MLXArray.zeros([1, 9, 4])
        let imaginary = MLXArray.zeros([1, 9, 4])
        let firstWindow = DemucsMLXSwift.createPeriodicHannWindow(winLength: 16)
        let first = DemucsMLXSwift.istft(
            realPart: real,
            imagPart: imaginary,
            nFFT: 16,
            hopLength: 4,
            winLength: 16,
            window: firstWindow,
            center: false,
            audioLength: 40
        )
        let secondWindow = DemucsMLXSwift.createPeriodicHannWindow(winLength: 16)
        let second = DemucsMLXSwift.istft(
            realPart: real,
            imagPart: imaginary,
            nFFT: 16,
            hopLength: 4,
            winLength: 16,
            window: secondWindow,
            center: false,
            audioLength: 40
        )
        eval(first, second)

        XCTAssertTrue(firstWindow === secondWindow)
        let matchingKeys = getMLXCacheInfo().keys.filter {
            $0.hasPrefix("4_4_16_")
        }
        XCTAssertEqual(matchingKeys.count, 1)
    }

    func testSharedCachesSupportConcurrentInferenceUtilities() async {
        let resultShapes = await withTaskGroup(
            of: ([Int], [Int], [Int], [Int]).self,
            returning: [([Int], [Int], [Int], [Int])].self
        ) { group in
            // More jobs than each cache's bound exercises concurrent insertion,
            // lookup and eviction rather than merely racing the first write.
            for _ in 0..<64 {
                group.addTask {
                    let real = MLXArray.zeros([1, 9, 4])
                    let imaginary = MLXArray.zeros([1, 9, 4])

                    let ussWindow = USSMLXSwift.createHannWindow(16, periodic: true)
                    let ussOutput = USSMLXSwift.istft(
                        realPart: real,
                        imagPart: imaginary,
                        nFFT: 16,
                        hopLength: 4,
                        winLength: 16,
                        window: ussWindow,
                        center: false,
                        audioLength: 40
                    )

                    let demucsWindow = DemucsMLXSwift.createPeriodicHannWindow(winLength: 16)
                    let demucsOutput = DemucsMLXSwift.istft(
                        realPart: real,
                        imagPart: imaginary,
                        nFFT: 16,
                        hopLength: 4,
                        winLength: 16,
                        window: demucsWindow,
                        center: false,
                        audioLength: 40
                    )

                    let chatterboxWindow = DSP.hanning(16)
                    let melFilters = DSP.melFilters(
                        sampleRate: 16_000,
                        nFFT: 16,
                        nMels: 4
                    )

                    eval(ussOutput, demucsOutput, chatterboxWindow, melFilters)
                    return (
                        ussOutput.shape,
                        demucsOutput.shape,
                        chatterboxWindow.shape,
                        melFilters.shape
                    )
                }
            }

            var values: [([Int], [Int], [Int], [Int])] = []
            for await value in group { values.append(value) }
            return values
        }

        XCTAssertEqual(resultShapes.count, 64)
        for shapes in resultShapes {
            XCTAssertEqual(shapes.0, [1, 40])
            XCTAssertEqual(shapes.1, [1, 40])
            XCTAssertEqual(shapes.2, [16])
            XCTAssertEqual(shapes.3, [9, 4])
        }
    }
}
