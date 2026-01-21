# Streaming Pipeline Tests Plan

## Overview

Implement comprehensive pipeline tests for **VAD -> Diarization -> Speech Enhancement -> USS** workflows with streaming diarization support, embedding switching, and performance benchmarks.

**Primary Test Target:** `harry_potter.wav` (2:16 movie trailer with speech, music, animal sounds, effects)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         harry_potter.wav (2:16)                              │
│                   Speech + Music + Animal sounds + Effects                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VAD (Silero @ 16kHz)                               │
│              256ms chunks, yields speech/non-speech segments                 │
└─────────────────────────────────────────────────────────────────────────────┘
              │                                         │
              ▼                                         ▼
       [Speech Segments]                        [Non-Speech Segments]
              │                                         │
              ▼                                         │
┌──────────────────────────┐                            │
│      Diarization         │                            │
│  Sortformer (streaming)  │                            │
│  OR Pyannote (batch)     │                            │
│  (16kHz input)           │                            │
└──────────────────────────┘                            │
              │                                         │
              ▼                                         │
       [Speaker Timeline]                               │
              │                                         │
              ▼                                         ▼
┌──────────────────────────┐              ┌──────────────────────────────────┐
│   *** RESAMPLE ***       │              │         *** RESAMPLE ***         │
│   16kHz -> 48kHz         │              │         16kHz -> 32kHz           │
└──────────────────────────┘              └──────────────────────────────────┘
              │                                         │
              ▼                                         ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────┐
│  Speech Enhancement (SE 48K)         │  │      USS Multi-Type Separation   │
│  4s chunks, 25% overlap              │  │      on VAD-excluded segments    │
│  -> enhanced speech + background     │  │   .music, .animal, .nature, etc. │
│  (48kHz in/out)                      │  │   (32kHz in/out)                 │
└──────────────────────────────────────┘  └──────────────────────────────────┘
              │                    │
              ▼                    ▼
       [Enhanced Speech]    [Background Track @ 48kHz]
                                   │
                                   ▼
                      ┌──────────────────────────────────┐
                      │         *** RESAMPLE ***         │
                      │         48kHz -> 32kHz           │
                      └──────────────────────────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────────────┐
                      │    USS Multi-Type Separation     │
                      │    on SE background residual     │
                      │  .music, .animal, .noise, etc.   │
                      │  (32kHz in/out)                  │
                      └──────────────────────────────────┘
```

---

## Sample Rate Transitions (Critical)

The pipeline crosses multiple sample rates. **Explicit resampling is required** to avoid silent mismatches.

| Stage | Input Rate | Output Rate | Resampling Needed |
|-------|------------|-------------|-------------------|
| Load audio | Native | 16kHz | Yes, for VAD/Diarization |
| VAD (Silero) | 16kHz | 16kHz (segments) | No |
| Diarization | 16kHz | 16kHz (timeline) | No |
| SE 48K | **48kHz** | 48kHz | **Resample 16k->48k before** |
| USS | **32kHz** | 32kHz | **Resample input to 32kHz** |
| SE Background -> USS | 48kHz | 32kHz | **Resample 48k->32k before USS** |

### Resampling Implementation in Tests

```swift
// Load audio at multiple sample rates using AudioLoader
func loadAudioAtRates(url: URL) throws -> (audio16k: AudioBuffer, audio48k: AudioBuffer, audio32k: AudioBuffer) {
    let loader16k = AudioLoader(config: .init(targetSampleRate: 16000))
    let loader48k = AudioLoader(config: .init(targetSampleRate: 48000))
    let loader32k = AudioLoader(config: .init(targetSampleRate: 32000))
    
    let mlx16k = try loader16k.loadMono(from: url)
    let mlx48k = try loader48k.loadMono(from: url)
    let mlx32k = try loader32k.loadMono(from: url)
    
    eval(mlx16k); eval(mlx48k); eval(mlx32k)
    
    return (
        AudioBuffer(samples: mlx16k.asArray(Float.self), sampleRate: 16000),
        AudioBuffer(samples: mlx48k.asArray(Float.self), sampleRate: 48000),
        AudioBuffer(samples: mlx32k.asArray(Float.self), sampleRate: 32000)
    )
}

// Resample SE background (48kHz) to USS input (32kHz)
func resample48kTo32k(_ buffer: AudioBuffer) -> AudioBuffer {
    // Simple linear interpolation (or use vDSP for quality)
    let ratio = 32000.0 / 48000.0
    let newLength = Int(Double(buffer.samples.count) * ratio)
    var resampled = [Float](repeating: 0, count: newLength)
    
    for i in 0..<newLength {
        let srcIdx = Double(i) / ratio
        let idx0 = Int(srcIdx)
        let idx1 = min(idx0 + 1, buffer.samples.count - 1)
        let frac = Float(srcIdx - Double(idx0))
        resampled[i] = buffer.samples[idx0] * (1 - frac) + buffer.samples[idx1] * frac
    }
    
    return AudioBuffer(samples: resampled, sampleRate: 32000)
}
```

### Validation in Tests

Each test should verify sample rates match model expectations:

```swift
// Before VAD/Diarization
XCTAssertEqual(audio.sampleRate, 16000, "VAD/Diarization requires 16kHz")

// Before SE 48K
XCTAssertEqual(audio.sampleRate, 48000, "SE 48K requires 48kHz input")

// Before USS
XCTAssertEqual(audio.sampleRate, 32000, "USS requires 32kHz input")
```

---

## Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | **USSProviders extensions** | Add factory methods for all 7 types + embedding switching API |
| 2 | **StreamingPipelineTests.swift** | New XCTest file with comprehensive pipeline tests |
| 3 | **Sortformer config exposure** | Expose low latency preset (high latency pending model availability) |
| 4 | **Progress reporting** | Wire `PipelineEvent.progress` for all pipeline stages (VAD, Diarization, Enhancement, USS, Transcription) |
| 5 | **Sample rate utilities** | Explicit resampling helpers to avoid silent mismatches between stages |

---

## Phase 1: USSProviders Extensions + Embedding Switching API [COMPLETED]

**Commit:** `a26c7cb - Add USS embedding swap for efficient multi-type separation`

### 1.1 Factory Methods (Implemented)

**File:** `Sources/ClearVoiceUSS/USSProviders.swift`

All 7 embedding types now have factory methods:

```swift
USSProviders.speechSeparation()   // .speech
USSProviders.musicSeparation()    // .music
USSProviders.noiseSeparation()    // .noise
USSProviders.animalSeparation()   // .animal
USSProviders.natureSeparation()   // .nature
USSProviders.humanSeparation()    // .human
USSProviders.thingsSeparation()   // .things

// Generic factory
USSProviders.separation(type: .animal)
```

### 1.2 Embedding Switching API (Implemented)

**File:** `Sources/ClearVoiceUSS/USSMLXProvider.swift`

Key features:
- **Embedding cache**: All 7 embeddings loaded on init (~14KB total)
- **Instant switching**: `setConditioning()` switches embeddings in ~0ms
- **Stateless processing**: `process(_:type:)` for one-off separation (recommended)
- **Batch processing**: `processMultiple()` for multiple types at once
- **Pre-warming**: `prewarmEmbeddings()` for target embeddings

```swift
// Load model once
let uss = USSProviders.speechSeparation()
try await uss.load()

// Process with different embedding types - no reload needed
let music = try await uss.process(audio, type: .music)      // Recommended
let animal = try await uss.process(audio, type: .animal)

// Or switch embedding and use default process()
try await uss.setConditioning(.music)
let musicAlt = try await uss.process(audio)

// Or batch multiple types
let results = try await uss.processMultiple(audio, types: [.music, .animal, .noise])
```

### 1.3 Benchmark Results

| Method | Time (30s audio, 2 types) | RTF | Notes |
|--------|---------------------------|-----|-------|
| Sequential `process(_:type:)` | 1.1s | 27x | **Fastest, recommended** |
| `processMultiple()` | 2.4s | 12x | Slower due to logging overhead |

JIT compilation happens once, embedding swap is instant since all cached.

---

## Phase 2: Sortformer Configuration Exposure [COMPLETED]

**Commit:** TBD

### 2.1 Sortformer Config Details (from FluidAudio v0.10.0)

| Config | Latency | Right Context | Use Case |
|--------|---------|---------------|----------|
| `.default` / `.nvidiaLowLatency` | ~1.04s | 7 frames | Real-time streaming |
| `.nvidiaHighLatency` | ~30.4s | 40 frames | Batch, best quality |

**Latency formula:**
```
latency = (chunkLen + rightContext) × subsamplingFactor × melStride / sampleRate
        = (6 + rightContext) × 8 × 160 / 16000
```

### 2.2 CoreML Model Binary Mapping (Critical)

CoreML models have **static input shapes baked in at conversion time**. The config MUST match the model's compiled shape or runtime errors will occur.

| Config | Required Model Binary | HuggingFace Repo | Notes |
|--------|----------------------|------------------|-------|
| `.default` / `.nvidiaLowLatency` | `sortformer_low_latency.mlpackage` | `FluidInference/sortformer-diar-msdd-base` | Default in FluidAudio v0.10.0 |
| `.nvidiaHighLatency` | `sortformer_high_latency.mlpackage` | TBD (may not be converted yet) | Requires separate model conversion |

**Current Status:**
- FluidAudio v0.10.0 ships with **low latency model only** (`.default` config)
- High latency config requires a separately converted CoreML model
- For initial tests, use `.default` config which matches the shipped model

### 2.3 Factory Presets (Implemented)

**File:** `Sources/ClearVoiceFluidAudio/FluidAudioProviders.swift`

```swift
// Low latency (~1.04s) - uses default CoreML model (RECOMMENDED)
FluidAudioProviders.sortformerLowLatency()

// High latency (~30.4s) - best quality, requires matching CoreML model
FluidAudioProviders.sortformerHighLatency()

// NVIDIA low latency benchmark config - may require matching model
FluidAudioProviders.sortformerNVIDIALowLatency()

// Generic with custom config
FluidAudioProviders.sortformer(config: .default)
```

### 2.4 Provider Enhancements (Implemented)

**File:** `Sources/ClearVoiceFluidAudio/FluidAudioSortformerProvider.swift`

New features added:
- **`configuration` property**: Exposes the Sortformer config used by the provider
- **`estimatedLatency` property**: Computed latency in seconds based on config
- **`validateConfigCompatibility()`**: Logs warnings for non-default configs

```swift
let provider = FluidAudioProviders.sortformerLowLatency()
print(provider.estimatedLatency)  // 1.04 seconds
print(provider.configuration.chunkLen)  // 6 frames
```

### 2.5 Documentation Updates (Implemented)

- Added comprehensive doc comments explaining configuration presets
- Added table of config options with latency and use cases
- Added warning about CoreML model compatibility
- Added recommended factory method examples

---

## Phase 3: StreamingPipelineTests.swift [COMPLETED]

**File:** `Tests/ClearVoiceFluidAudioTests/StreamingPipelineTests.swift`

### Test Suite Structure

```swift
import XCTest
import ClearVoiceFluidAudio
import ClearVoiceMLX
import ClearVoiceUSS
import ClearVoiceCore
import AudioUtils
import MLX

final class StreamingPipelineTests: XCTestCase {
    
    // MARK: - Test Fixtures
    
    /// Load harry_potter.wav for all tests
    private func loadHarryPotterAudio(sampleRate: Int) throws -> AudioBuffer
    
    // MARK: - Full Pipeline Tests (Batch Baseline)
    
    /// VAD -> Sortformer -> SE -> USS pipeline (batch mode)
    /// Tests complete workflow with Sortformer diarization (max 4 speakers)
    func testFullBatchPipeline_Sortformer_HarryPotter() async throws {
        // 1. Load harry_potter.wav
        // 2. VAD: detect speech vs non-speech segments
        // 3. Diarization (Sortformer): identify speakers (capped at 4)
        // 4. Enhancement (SE 48K): enhance speech, extract background
        // 5. USS on non-speech: separate .music, .animal
        // 6. USS on SE background: separate .music, .animal
        // 7. Measure RTF per stage, validate outputs
    }
    
    /// VAD -> Pyannote -> SE -> USS pipeline (batch mode)
    /// Tests complete workflow with Pyannote diarization (unlimited speakers)
    func testFullBatchPipeline_Pyannote_HarryPotter() async throws {
        // Same as above but with Pyannote (better for 10-12 speakers)
        // Compare speaker detection with Sortformer
    }
    
    // MARK: - Streaming Tests
    
    /// Streaming VAD -> Sortformer coordination
    /// VAD produces 256ms chunks, accumulated to ~10s, fed to Sortformer streaming
    func testStreamingVADToSortformer_HarryPotter() async throws {
        // 1. Stream audio through VAD in 256ms chunks
        // 2. Accumulate speech samples until ~10s (WeSpeaker window size)
        // 3. Feed to Sortformer.processChunk() for streaming diarization
        // 4. Report intermediate speaker timelines
        // 5. Compare final timeline with batch mode result
    }
    
    // MARK: - Multi-Type USS Tests
    
    /// USS on non-speech segments: .music + .animal separation
    /// Tests USS on VAD-excluded (non-speech) segments
    func testMultiTypeUSS_NonSpeech_HarryPotter() async throws {
        // 1. VAD -> get non-speech segments
        // 2. Run USS with .music embedding
        // 3. Run USS with .animal embedding
        // 4. Validate separated outputs are non-silent
        // 5. Measure RTF for each USS type
    }
    
    /// USS on SE background residual: .music + .animal separation
    /// Tests USS on speech enhancement background track
    func testMultiTypeUSS_SEBackground_HarryPotter() async throws {
        // 1. VAD -> get speech segments
        // 2. Enhancement (SE 48K) with background extraction
        // 3. Run USS .music on background
        // 4. Run USS .animal on background
        // 5. Validate music/animal detected in background
    }
    
    /// USS embedding switching efficiency
    /// Load model once, run multiple separation types using process(_:type:)
    func testUSSEmbeddingSwitching_HarryPotter() async throws {
        // 1. Load USS model once (any initial type works)
        // 2. Process with .music using process(audio, type: .music)
        // 3. Process with .animal using process(audio, type: .animal)
        // 4. Verify both outputs valid and non-silent
        // 5. Measure time - should be ~27x RTF per type
    }
    
    /// USS processMultiple - batch multiple embeddings
    func testUSSProcessMultiple_HarryPotter() async throws {
        // 1. Load USS model once
        // 2. Call processMultiple(audio, types: [.music, .animal, .noise])
        // 3. Verify all three outputs valid
        // 4. Note: ~12x RTF (slower than sequential due to logging)
    }
    
    // MARK: - Diarization Comparison
    
    /// Sortformer vs Pyannote diarization comparison
    /// Compare speaker detection, overlap detection, timing accuracy
    func testDiarizationComparison_HarryPotter() async throws {
        // 1. Run Sortformer on harry_potter.wav
        // 2. Run Pyannote on harry_potter.wav
        // 3. Compare:
        //    - Speaker count (Sortformer capped at 4, Pyannote unlimited)
        //    - Segment count
        //    - Max overlapping speakers
        //    - RTF (Sortformer ~120x, Pyannote ~5-10x)
        // 4. Note: Pyannote likely more accurate for 10-12 speaker scenario
    }
    
    // MARK: - Performance Benchmark
    
    /// Full performance benchmark on harry_potter.wav
    func testPipelinePerformanceBenchmark_HarryPotter() async throws {
        // Measure and report:
        // - Model load times (per model)
        // - Per-stage RTF (VAD, Diarization, Enhancement, USS)
        // - Total pipeline RTF
        // - Peak memory usage (if measurable)
        // - Audio duration (2:16 = 136s) vs processing time
    }
    
    // MARK: - Progress Reporting
    
    /// Verify progress events emitted during long-running stages
    func testProgressReportingDuringPipeline() async throws {
        // 1. Run pipeline with onEvent handler
        // 2. Collect all .progress events
        // 3. Validate progress increases monotonically per stage
        // 4. Validate .stageComplete events at end of each stage
    }
}
```

---

## Phase 4: Progress Reporting in Pipeline Execution [COMPLETED]

### 4.1 Current State

`PipelineEvent.progress(stage:, percent:)` is now emitted during pipeline execution for all stages.

### 4.2 Scope of Progress Reporting

Progress events will be wired for **all stages that support chunked/streaming processing**:

| Stage | Progress Support | Implementation |
|-------|------------------|----------------|
| VAD (detect) | Yes | Emit per-chunk as streaming VAD processes |
| Diarization | Limited | Sortformer streaming: emit per chunk. Pyannote batch: 0% -> 100% only |
| Enhancement | Yes | StreamableOutput: emit per chunk yielded |
| USS | Yes | Can emit per 2s segment processed |
| Transcription | Yes | Emit per segment recognized |
| Separation | Limited | Batch processing: 0% -> 100% only |
| Upscale | Yes | StreamableOutput: emit per chunk |

### 4.3 Implementation Details

**File:** `Sources/ClearVoice/ClearVoice.swift` (executePipeline method)

#### Enhancement Stage (StreamableOutput)

```swift
case .enhance(let model):
    let enhanced: ClearVoiceCore.AudioBuffer
    
    // Check if provider supports streaming output
    if let streamable = enhancerProviders[model.modelName] as? StreamableOutput {
        // Stream processing with progress reporting
        var chunks: [AudioBuffer] = []
        var processedSamples = 0
        let totalSamples = context.currentAudio.samples.count
        
        for try await chunk in streamable.processStream(context.currentAudio) {
            chunks.append(chunk)
            processedSamples += chunk.samples.count
            let percent = min(100.0, Double(processedSamples) / Double(totalSamples) * 100)
            await eventHandler?(.progress(stage: "enhance", percent: percent))
        }
        
        // Concatenate chunks
        enhanced = AudioBuffer.concatenate(chunks)
    } else {
        // Fallback to batch processing - emit 0% and 100% only
        await eventHandler?(.progress(stage: "enhance", percent: 0))
        if let segments = context.analysis?.speechSegments, !segments.isEmpty {
            enhanced = try await enhance(context.currentAudio, segments: segments, model: model)
        } else {
            enhanced = try await enhance(context.currentAudio, model: model)
        }
        await eventHandler?(.progress(stage: "enhance", percent: 100))
    }
    // ... rest of case
```

#### VAD Stage (Streaming)

```swift
case .detect(let model):
    await eventHandler?(.progress(stage: "vad", percent: 0))
    let segments = try await detect(context.currentAudio, model: model)
    // VAD is typically fast, emit 100% after completion
    await eventHandler?(.progress(stage: "vad", percent: 100))
    // ... rest of case
```

#### Diarization Stage

```swift
case .diarize:
    await eventHandler?(.progress(stage: "diarize", percent: 0))
    let timeline = try await diarize(context.currentAudio)
    await eventHandler?(.progress(stage: "diarize", percent: 100))
    // ... rest of case
```

#### Transcription Stage (Streaming)

```swift
case .transcribe(let model):
    await eventHandler?(.progress(stage: "transcribe", percent: 0))
    let transcription = try await transcribe(context.currentAudio, model: model)
    for (idx, segment) in transcription.segments.enumerated() {
        await eventHandler?(.transcriptionSegment(segment))
        let percent = Double(idx + 1) / Double(transcription.segments.count) * 100
        await eventHandler?(.progress(stage: "transcribe", percent: percent))
    }
    // ... rest of case
```

### 4.4 Test Expectations

The progress reporting test should expect:

```swift
func testProgressReportingDuringPipeline() async throws {
    var progressEvents: [(stage: String, percent: Double)] = []
    var stageCompleteEvents: [String] = []
    
    let result = try await voice.pipeline()
        .detect(.silero)
        .diarize()
        .enhance(.mossformerSE48k)
        .onEvent { event in
            switch event {
            case .progress(let stage, let percent):
                progressEvents.append((stage, percent))
            case .stageComplete(let stage, _):
                stageCompleteEvents.append(stage)
            default:
                break
            }
        }
        .process(audio: audio)
    
    // Verify progress events exist for each stage
    let vadProgress = progressEvents.filter { $0.stage == "vad" }
    let diarizeProgress = progressEvents.filter { $0.stage == "diarize" }
    let enhanceProgress = progressEvents.filter { $0.stage == "enhance" }
    
    // Each stage should have at least start (0%) and end (100%) progress
    XCTAssertTrue(vadProgress.contains { $0.percent == 0 })
    XCTAssertTrue(vadProgress.contains { $0.percent == 100 })
    
    XCTAssertTrue(diarizeProgress.contains { $0.percent == 0 })
    XCTAssertTrue(diarizeProgress.contains { $0.percent == 100 })
    
    // Enhancement with StreamableOutput should have intermediate progress
    XCTAssertTrue(enhanceProgress.count > 2, "Enhancement should emit intermediate progress")
    
    // Progress should be monotonically increasing within each stage
    for stage in ["vad", "diarize", "enhance"] {
        let stageProgress = progressEvents.filter { $0.stage == stage }.map { $0.percent }
        for i in 1..<stageProgress.count {
            XCTAssertGreaterThanOrEqual(stageProgress[i], stageProgress[i-1],
                "Progress should increase monotonically for \(stage)")
        }
    }
    
    // Verify stageComplete events
    XCTAssertTrue(stageCompleteEvents.contains("vad"))
    XCTAssertTrue(stageCompleteEvents.contains("diarize"))
    XCTAssertTrue(stageCompleteEvents.contains("enhance"))
}

---

## Diarization Provider Comparison

| Feature | Sortformer | Pyannote + WeSpeaker |
|---------|------------|----------------------|
| **Max Speakers** | 4 | Unlimited |
| **Streaming** | Yes (`processChunk`) | No (batch only) |
| **Chunk Size** | ~1s (default config) | 10s windows (WeSpeaker) |
| **RTF** | ~120x | ~5-10x |
| **Overlap Detection** | Native (per-frame scores) | VBx clustering |
| **Best For** | Real-time, <=4 speakers | Batch, many speakers |

### Overlap Detection for Future Speaker Separation

Both providers expose `maxOverlappingSpeakers` in `SpeakerTimeline`. Future pipeline pattern:

```
Diarization -> detect overlap regions
     │
     ▼
[If overlapping speakers >= 2]
     │
     ▼
Speaker Separation (MossFormer2 SS 2spk/3spk) on original audio
     │
     ▼
[Separated speaker tracks]
```

This conditional separation is already partially supported via `.conditionally()` in PipelineBuilder.

---

## USS Chunking Strategy

### Decision: Separate Streams (Option 1)

For USS/Demucs, process VAD non-speech and SE background as **separate inputs** (not merged by timestamp).

**Rationale:**
- USS (ResUNet30 + FiLM) and Demucs are **stateless per chunk**
- FiLM conditioning doesn't use cross-chunk recurrent state
- Each 2s chunk is processed independently
- Simpler implementation, natural parallelism

**Alternative (not implemented initially):**
Merge non-speech + SE background by timestamp into continuous track, then chunk and process. This could be added later if audio quality requires temporal continuity.

---

## Test Audio

| File | Duration | Content | Speakers | Use Case |
|------|----------|---------|----------|----------|
| `harry_potter.wav` | 2:16 (136s) | Movie trailer | 10-12 | Primary test - speech, music, animals (Hedwig), effects |

---

## Files to Modify/Create

| File | Action | Status | Description |
|------|--------|--------|-------------|
| `USSProviders.swift` | Modify | **DONE** | Added all 7 type factories |
| `USSMLXProvider.swift` | Modify | **DONE** | Added `setConditioning()`, `process(_:type:)`, `processMultiple()`, `prewarmEmbeddings()` |
| `USSEmbeddingSwapTests.swift` | Create | **DONE** | Tests for embedding switching |
| `USSPrewarmBenchmarkTests.swift` | Create | **DONE** | Benchmark tests |
| `FluidAudioProviders.swift` | Modify | **DONE** | Added `sortformerLowLatency()`, `sortformerHighLatency()`, `sortformerNVIDIALowLatency()` |
| `FluidAudioSortformerProvider.swift` | Modify | **DONE** | Added `configuration`, `estimatedLatency`, `validateConfigCompatibility()` |
| `StreamingPipelineTests.swift` | **Create** | **DONE** | Comprehensive pipeline test file |
| `ClearVoice.swift` | Modify | **DONE** | Wire progress events for all pipeline stages |
| `AudioBuffer+Resampling.swift` | Consider | N/A | Resampling implemented inline in tests |

### Test File Internal Structure

The test file should include:

```swift
// MARK: - Sample Rate Utilities

/// Load audio at multiple sample rates for pipeline stages
private func loadAudioAtRates(url: URL) throws -> (audio16k: AudioBuffer, audio48k: AudioBuffer, audio32k: AudioBuffer)

/// Resample 48kHz SE background to 32kHz USS input
private func resample48kTo32k(_ buffer: AudioBuffer) -> AudioBuffer

/// Extract time range from higher sample rate audio using VAD segments from 16kHz
private func extractSegment(from audio: AudioBuffer, timeRange: TimeRange) -> AudioBuffer

// MARK: - Validation Helpers

/// Assert audio sample rate matches expected
private func assertSampleRate(_ audio: AudioBuffer, expected: Int, context: String)
```

---

## Expected Outcomes

1. **Full pipeline validation** - VAD -> Diarization -> Enhancement -> USS produces valid outputs
2. **Streaming proof** - Sortformer streaming works with VAD chunks (~10s accumulation)
3. **Multi-type USS** - Successfully separate music AND animal sounds from harry_potter.wav
4. **Embedding switching** - Load USS model once, switch embeddings efficiently
5. **Provider comparison** - Document Sortformer vs Pyannote differences
6. **Performance baseline** - Documented RTF for each stage on 2:16 audio
7. **Progress events** - `PipelineEvent.progress` emitted for all stages (0%/100% minimum, intermediate for streaming stages)
8. **Sample rate safety** - Explicit resampling with validation to prevent silent mismatches

---

## Technical Notes

### Sortformer Latency Calculation

```
latency = (chunkLen + rightContext) × subsamplingFactor × melStride / sampleRate

Default config (~1.04s):
  = (6 + 7) × 8 × 160 / 16000
  = 13 × 8 × 0.01 = 1.04 seconds

NVIDIA High Latency (~30.4s):
  = (340 + 40) × 8 × 160 / 16000
  = 380 × 8 × 0.01 = 30.4 seconds
```

### USS Embedding Sizes

| Component | Size | Notes |
|-----------|------|-------|
| ResUNet30 weights (FP16) | ~53MB | Load once |
| ResUNet30 weights (FP32) | ~106MB | Load once |
| Each embedding | ~2KB | 527 floats, switch freely |

### Chunk Sizes Summary

| Model | Chunk Size | Overlap | Sample Rate |
|-------|------------|---------|-------------|
| VAD (Silero) | 256ms (4096 samples) | None | 16kHz |
| Sortformer | ~1s (configurable) | N/A | 16kHz |
| Pyannote + WeSpeaker | 10s windows | N/A | 16kHz |
| MossFormer2 SE 48K | 4s | 25% | 48kHz |
| USS | 2s | None (hopLength=segmentDuration) | 32kHz |
| Demucs | 7.8s | 25% | 44.1kHz |
