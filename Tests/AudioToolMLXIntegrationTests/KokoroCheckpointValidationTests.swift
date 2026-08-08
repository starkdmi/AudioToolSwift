//
//  KokoroCheckpointValidationTests.swift
//  AudioToolMLXIntegrationTests
//
//  Malformed external checkpoints must fail without terminating the process
//

import Foundation
import Testing
import MLX
@testable import KokoroSwift

@Suite("Kokoro checkpoint validation")
struct KokoroCheckpointValidationTests {

    @Test("Corrupt safetensors data is reported as an error")
    func corruptCheckpointThrows() throws {
        let checkpoint = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: checkpoint) }
        try Data("not a safetensors checkpoint".utf8).write(to: checkpoint)

        do {
            _ = try WeightLoader.loadWeights(modelPath: checkpoint)
            Issue.record("corrupt checkpoint unexpectedly loaded")
        } catch {
            #expect(!error.localizedDescription.isEmpty)
        }
    }

    @Test("Missing required tensors throw a descriptive error")
    func missingTensorThrows() {
        let weights: [String: MLXArray] = [:]

        #expect(throws: KokoroModelLoadingError.self) {
            _ = try weights.required("bert.pooler.weight")
        }
    }

    @Test("Present tensors are returned unchanged")
    func presentTensorIsReturned() throws {
        let expected = MLXArray([Float(1), 2, 3])
        let weights = ["tensor": expected]
        let actual = try weights.required("tensor")

        #expect(actual.shape == expected.shape)
        #expect(actual.asArray(Float.self) == [1, 2, 3])
    }

    @Test("Malformed convolution ranks throw before transposition")
    func malformedConvolutionRankThrows() {
        #expect(throws: KokoroModelLoadingError.self) {
            _ = try WeightLoader.transposedConvWeight(
                MLXArray(Float(1)),
                name: "predictor.F0_proj.weight"
            )
        }
    }
}
