//
//  AudioBufferTests.swift
//  AudioTool
//
//  Tests for AudioBuffer operations
//

import Testing
@testable import AudioTool
@testable import AudioToolCore

@Suite("AudioBuffer Tests")
struct AudioBufferTests {
    
    // MARK: - Initialization
    
    @Test("Create buffer from samples")
    func testCreateFromSamples() {
        let samples: [Float] = [0.1, 0.2, 0.3, 0.4]
        let buffer = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        
        #expect(buffer.samples.count == 4)
        #expect(buffer.sampleRate == 16000)
        #expect(buffer.channels == 1)
        #expect(buffer.frameCount == 4)
    }
    
    @Test("Create silence buffer")
    func testCreateSilence() {
        let buffer = AudioBuffer.silence(duration: 1.0, sampleRate: 16000)
        
        #expect(buffer.frameCount == 16000)
        #expect(buffer.duration == 1.0)
        #expect(buffer.samples.allSatisfy { $0 == 0 })
    }
    
    @Test("Create sine wave buffer")
    func testCreateSine() {
        let buffer = AudioBuffer.sine(frequency: 440, duration: 0.1, sampleRate: 16000, amplitude: 0.5)
        
        #expect(buffer.frameCount == 1600)
        #expect(buffer.samples.max()! <= 0.5)
        #expect(buffer.samples.min()! >= -0.5)
    }
    
    @Test("Create noise buffer")
    func testCreateNoise() {
        let buffer = AudioBuffer.noise(duration: 0.1, sampleRate: 16000, amplitude: 0.3)
        
        #expect(buffer.frameCount == 1600)
        #expect(buffer.samples.max()! <= 0.3)
        #expect(buffer.samples.min()! >= -0.3)
    }
    
    // MARK: - Properties
    
    @Test("Duration calculation")
    func testDuration() {
        let buffer = AudioBuffer(samples: [Float](repeating: 0, count: 32000), sampleRate: 16000, channels: 1)
        #expect(buffer.duration == 2.0)
        
        let stereo = AudioBuffer(samples: [Float](repeating: 0, count: 32000), sampleRate: 16000, channels: 2)
        #expect(stereo.duration == 1.0)  // Half the frames due to 2 channels
    }
    
    @Test("Empty buffer")
    func testEmptyBuffer() {
        let buffer = AudioBuffer(samples: [], sampleRate: 16000, channels: 1)
        #expect(buffer.isEmpty)
        #expect(buffer.duration == 0)
        #expect(buffer.frameCount == 0)
    }
    
    // MARK: - Slicing
    
    @Test("Slice by time range")
    func testSliceByTime() {
        let buffer = AudioBuffer.silence(duration: 4.0, sampleRate: 16000)
        let slice = buffer.slice(1.0..<3.0)
        
        #expect(slice.duration == 2.0)
        #expect(slice.frameCount == 32000)
        #expect(slice.sampleRate == 16000)
    }
    
    @Test("Slice by sample range")
    func testSliceBySamples() {
        let buffer = AudioBuffer.silence(duration: 2.0, sampleRate: 16000)
        let slice = buffer.slice(samples: 0..<8000)
        
        #expect(slice.frameCount == 8000)
        #expect(slice.duration == 0.5)
    }
    
    @Test("Slice out of bounds clamps to valid range")
    func testSliceOutOfBounds() {
        let buffer = AudioBuffer.silence(duration: 1.0, sampleRate: 16000)
        let slice = buffer.slice(0.5..<5.0)  // End is beyond duration
        
        #expect(slice.duration == 0.5)  // Should clamp to 1.0s end
    }
    
    // MARK: - Operations
    
    @Test("Subtract buffers")
    func testSubtract() {
        let a = AudioBuffer(samples: [1.0, 2.0, 3.0], sampleRate: 16000)
        let b = AudioBuffer(samples: [0.5, 0.5, 0.5], sampleRate: 16000)
        let result = a.subtracting(b)
        
        #expect(result.samples[0] == 0.5)
        #expect(result.samples[1] == 1.5)
        #expect(result.samples[2] == 2.5)
    }
    
    @Test("Mix buffers")
    func testMix() {
        let a = AudioBuffer(samples: [1.0, 1.0, 1.0], sampleRate: 16000)
        let b = AudioBuffer(samples: [0.5, 0.5], sampleRate: 16000)
        let result = a.mixing(with: b)
        
        #expect(result.samples[0] == 1.5)
        #expect(result.samples[1] == 1.5)
        #expect(result.samples[2] == 1.0)
    }

    @Test("Mix with a negative offset preserves both timelines")
    func testMixAtNegativeOffset() {
        let a = AudioBuffer(samples: [1, 1, 1], sampleRate: 1)
        let b = AudioBuffer(samples: [2, 2], sampleRate: 1)

        let result = a.mixing(with: b, at: -1)

        #expect(result.samples == [2, 3, 1, 1])
    }
    
    @Test("Append buffers")
    func testAppend() {
        let a = AudioBuffer(samples: [1.0, 2.0], sampleRate: 16000)
        let b = AudioBuffer(samples: [3.0, 4.0], sampleRate: 16000)
        let result = a.appending(b)
        
        #expect(result.samples == [1.0, 2.0, 3.0, 4.0])
        #expect(result.frameCount == 4)
    }
    
    @Test("Scale amplitude")
    func testScale() {
        let buffer = AudioBuffer(samples: [1.0, 2.0, -1.0], sampleRate: 16000)
        let scaled = buffer.scaled(by: 0.5)
        
        #expect(scaled.samples[0] == 0.5)
        #expect(scaled.samples[1] == 1.0)
        #expect(scaled.samples[2] == -0.5)
    }
    
    @Test("Replace segment")
    func testReplace() {
        // 2 seconds of zeros
        var buffer = AudioBuffer.silence(duration: 2.0, sampleRate: 16000)
        
        // 0.5 seconds of ones to insert at 0.5s
        let replacement = AudioBuffer(
            samples: [Float](repeating: 1.0, count: 8000),
            sampleRate: 16000
        )
        
        buffer = buffer.replacing(0.5..<1.0, with: replacement)
        
        // First 0.5s should be zeros
        #expect(buffer.samples[0] == 0)
        #expect(buffer.samples[7999] == 0)
        
        // 0.5-1.0s should be ones (but replacement is 0.5s worth)
        // Actually the replacement has 8000 samples but we're replacing a 8000 sample range
        #expect(buffer.samples[8000] == 1.0)
    }

    @Test("Replacing a range outside the buffer is a no-op")
    func testReplaceOutOfBounds() {
        let buffer = AudioBuffer(samples: [1, 2, 3], sampleRate: 1)
        let replacement = AudioBuffer(samples: [9], sampleRate: 1)

        #expect(buffer.replacing(5..<6, with: replacement) == buffer)
        #expect(buffer.replacing(-2 ..< -1, with: replacement) == buffer)
    }

    @Test("Partially overlapping replacement ranges clamp to the buffer")
    func testReplacePartiallyOutOfBounds() {
        let buffer = AudioBuffer(samples: [1, 2, 3], sampleRate: 1)
        let replacement = AudioBuffer(samples: [9], sampleRate: 1)

        #expect(buffer.replacing(-2..<1, with: replacement).samples == [9, 2, 3])
        #expect(buffer.replacing(2..<6, with: replacement).samples == [1, 2, 9])
    }

    @Test("Unbounded time slices clamp without integer conversion traps")
    func testUnboundedSlice() {
        let buffer = AudioBuffer(samples: [1, 2, 3], sampleRate: 1)

        #expect(buffer.slice(-Double.infinity..<Double.infinity) == buffer)
    }
    
    // MARK: - Hashable
    
    @Test("Buffers with same content are equal")
    func testHashable() {
        let a = AudioBuffer(samples: [1.0, 2.0, 3.0], sampleRate: 16000)
        let b = AudioBuffer(samples: [1.0, 2.0, 3.0], sampleRate: 16000)
        
        #expect(a == b)
        #expect(a.hashValue == b.hashValue)
    }
    
    @Test("Buffers with different sample rates are not equal")
    func testDifferentSampleRates() {
        let a = AudioBuffer(samples: [1.0, 2.0], sampleRate: 16000)
        let b = AudioBuffer(samples: [1.0, 2.0], sampleRate: 48000)
        
        #expect(a != b)
    }
}
