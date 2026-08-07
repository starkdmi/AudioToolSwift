//
//  GANDiagnosticTests.swift
//  AudioToolParityTests
//
//  Evidence for the MossFormerGAN stride finding, kept as a regression guard.
//
//  Fixed in MossFormerGANCoreMLProvider.parseModelOutputToMLX; the parity case now
//  reads 129.0/129.6 dB where it read -1.1/-1.2 dB. This suite stays because the
//  parity test only sees the symptom, and this prints the cause - if CoreML ever
//  changes the padding, the `read=linear` line here says so directly.
//
//  What it establishes:
//
//    CoreML returns `var_9026` as [1, 2, 256, 201] with strides
//    [106496, 53248, 208, 1]. The innermost dimension is 201 wide but rows are
//    208 apart - padded up to a 16-float boundary. `parseModelOutputToMLX`
//    copies 2*T*F floats linearly off `dataPointer` and reshapes, which assumes
//    a row stride of 201, so every row after the first is read 7 floats further
//    out of place than the last.
//
//    The padding is present under cpuOnly, cpuAndGPU, cpuAndNeuralEngine and
//    all, so it is a property of the model rather than of where it runs. This
//    path has never produced correct output; nothing regressed into it.
//
//    Python is unaffected because coremltools hands back a numpy array that
//    carries its own strides, which is why run.py's enhanced_output_no_chunk.wav
//    is clean while the Swift-produced ganse_*.wav files are not.
//

import AudioToolCore
import AudioToolCoreML
import AudioToolTestSupport
import CoreML
import MLX
import XCTest

final class GANDiagnosticTests: ParityTestCase {

    /// Dump the CoreML tensor layout and the stage-by-stage signal levels.
    ///
    /// The parity run put this case at -1.1 dB, which is far too low to be
    /// explained by anything numeric - it means the two sides are not computing
    /// the same thing at all. This narrows *where*.
    func testGANLayoutDiagnostic() async throws {
        let artifact = try artifact("mossformer_gan_se_16k_coreml")
        let modelPath = try reference(
            "Models/python/mossformer_gan_se_coreml/MossFormerGAN_256frames.mlpackage"
        )

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        let compiled = try await MLModel.compileModel(at: modelPath)
        let model = try await MLModel.load(contentsOf: compiled, configuration: config)

        print("DIAG inputs:")
        for (name, description) in model.modelDescription.inputDescriptionsByName {
            print("DIAG   in  \(name): \(description.multiArrayConstraint?.shape ?? []) ")
        }
        for (name, description) in model.modelDescription.outputDescriptionsByName {
            print("DIAG   out \(name): \(description.multiArrayConstraint?.shape ?? [])")
        }

        // Drive one exact segment through the same steps the provider uses, so the
        // intermediate levels can be read off against the Python run.
        let segment = try XCTUnwrap(artifact.tensor("input_segment"))
        let reference = try XCTUnwrap(artifact.tensor("enhanced_segment"))
        let input = MLXArray(segment).expandedDimensions(axis: 0).asType(.float32)

        let inputLen = MLXArray(Float(segment.count))
        let sumSquares = sum(input * input, axis: 1, keepDims: true)
        let normFactor = sqrt(inputLen / (sumSquares + 1e-9))
        let normed = input * normFactor
        eval(normed, normFactor)
        print(String(format: "DIAG normFactor=%.6f", normFactor.item(Float.self)))

        let window = createPeriodicHannWindow(length: 400)
        let (real, imag) = mlxSTFT(normed, nFFT: 400, hopLength: 100, winLength: 400,
                                   window: window, center: true)
        eval(real, imag)
        print("DIAG stft shape=\(real.shape) rms=\(rms(real)) / \(rms(imag))")

        // Feed CoreML directly, bypassing the provider, so the output layout can be
        // inspected before anything reinterprets it.
        let realT = real.transposed(0, 2, 1)
        let imagT = imag.transposed(0, 2, 1)
        let magnitude = MLX.sqrt(real * real + imag * imag + 1e-9)
        let phase = MLX.atan2(imag, real)
        let compressed = MLX.pow(magnitude, MLXArray(Float(0.3)))
        let realC = (compressed * MLX.cos(phase)).transposed(0, 2, 1)
        let imagC = (compressed * MLX.sin(phase)).transposed(0, 2, 1)
        eval(realT, imagT, realC, imagC)

        let frames = realC.shape[1]
        let bins = realC.shape[2]
        print("DIAG coreml input T=\(frames) F=\(bins)")

        let spec = MLX.stacked([realC, imagC], axis: 1)
        eval(spec)
        let multiArray = try MLMultiArray(
            shape: [1, 2, NSNumber(value: frames), NSNumber(value: bins)], dataType: .float32
        )
        let flat = spec.flattened().asArray(Float.self)
        multiArray.dataPointer.assumingMemoryBound(to: Float.self)
            .update(from: flat, count: flat.count)
        print("DIAG input strides=\(multiArray.strides.map(\.intValue)) count=\(multiArray.count)")

        let prediction = try await model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["spectrogram": multiArray])
        )
        let name = try XCTUnwrap(prediction.featureNames.first)
        let out = try XCTUnwrap(prediction.featureValue(for: name)?.multiArrayValue)

        // The reinterpretation in parseModelOutputToMLX assumes C-contiguous data.
        // If CoreML hands back anything else, a flat copy plus reshape is silently
        // wrong, which is exactly what a -1 dB result looks like.
        print("DIAG output name=\(name) shape=\(out.shape.map(\.intValue)) "
              + "strides=\(out.strides.map(\.intValue)) dtype=\(out.dataType.rawValue)")
        let expectedStrides = [2 * frames * bins, frames * bins, bins, 1]
        print("DIAG expected C-order strides=\(expectedStrides)")

        print(String(format: "DIAG reference enhanced_segment rms=%.6f", ParityMetrics.rms(reference)))
        print(String(format: "DIAG input_segment rms=%.6f", ParityMetrics.rms(segment)))

        // Decisive check: finish the pipeline twice from the *same* prediction, once
        // reading the buffer linearly (what the provider does) and once honouring the
        // strides, and see which one reconstructs the recorded reference.
        let outStrides = out.strides.map(\.intValue)
        let count = 2 * frames * bins
        let source = out.dataPointer.assumingMemoryBound(to: Float.self)

        var linear = [Float](repeating: 0, count: count)
        linear.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: source, count: count) }

        var strided = [Float](repeating: 0, count: count)
        for part in 0..<2 {
            for frame in 0..<frames {
                let rowStart = part * outStrides[1] + frame * outStrides[2]
                let destination = (part * frames + frame) * bins
                for bin in 0..<bins {
                    strided[destination + bin] = source[rowStart + bin * outStrides[3]]
                }
            }
        }

        for (label, data) in [("linear", linear), ("strided", strided)] {
            let parsed = MLXArray(data).reshaped([1, 2, frames, bins])
            let outReal = parsed[0..., 0, 0..., 0...].transposed(0, 2, 1)
            let outImag = parsed[0..., 1, 0..., 0...].transposed(0, 2, 1)
            let magnitudeOut = MLX.sqrt(outReal * outReal + outImag * outImag + 1e-9)
            let phaseOut = MLX.atan2(outImag, outReal)
            let expanded = MLX.pow(magnitudeOut, MLXArray(Float(1.0 / 0.3)))
            let reconstructed = mlxISTFT(
                realPart: expanded * MLX.cos(phaseOut),
                imagPart: expanded * MLX.sin(phaseOut),
                nFFT: 400, hopLength: 100, winLength: 400,
                window: window, center: true, audioLength: segment.count
            )
            let denormalized = (reconstructed / normFactor).squeezed(axis: 0)
            eval(denormalized)
            let snr = ParityMetrics.snrDB(
                reference: reference, candidate: denormalized.asArray(Float.self)
            )
            print(String(format: "DIAG read=%@ SNR vs reference = %.1f dB", label, snr))
        }

        // Is the padding a property of the model, or of where it runs? That decides
        // whether this ever worked and stopped, or never worked on this path.
        for units in [MLComputeUnits.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine, .all] {
            let probeConfig = MLModelConfiguration()
            probeConfig.computeUnits = units
            let probe = try await MLModel.load(contentsOf: compiled, configuration: probeConfig)
            let probeOut = try await probe.prediction(
                from: MLDictionaryFeatureProvider(dictionary: ["spectrogram": multiArray])
            )
            guard let probeName = probeOut.featureNames.first,
                  let array = probeOut.featureValue(for: probeName)?.multiArrayValue else { continue }
            let strides = array.strides.map(\.intValue)
            print("DIAG computeUnits=\(units.rawValue) strides=\(strides) "
                  + "contiguous=\(strides == expectedStrides)")
        }
    }

    private func rms(_ array: MLXArray) -> Float {
        let value = MLX.sqrt(MLX.mean(array * array))
        eval(value)
        return value.item(Float.self)
    }
}
