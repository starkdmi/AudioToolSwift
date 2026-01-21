# Speaker Separation Pipeline Plan

## Overview

This document outlines the design for handling overlapping speech in ClearVoice, including:
- Diarization model selection (Sortformer vs Pyannote)
- Overlap detection and speaker separation
- Speaker re-identification after separation

---

## Diarization Models Comparison

### Sortformer (Recommended for most cases)

**Architecture:** End-to-end neural model with 4 fixed speaker slots

```
Audio (16kHz) → Mel Spectrogram → CoreML Model → Speaker Probabilities [T', 4]
```

**Strengths:**
- Handles noisy environments well
- Native overlapping speech handling (scores all 4 speakers per frame)
- Real-time streaming with configurable latency
- Consistent speaker slots within session (maintained by `spkcache`)

**Limitations:**
- Maximum 4 speakers (hardcoded in model)
- No cross-session speaker memory
- May miss quiet/background speech (trained to ignore)

**Configurations:**

| Config | Latency | Quality | Use Case |
|--------|---------|---------|----------|
| `default` | ~1.04s | Good | Real-time |
| `nvidiaLowLatency` | ~1.04s | Better | Real-time (larger FIFO) |
| `nvidiaHighLatency` | ~30.4s | Best | Batch processing |

**Speaker Slot Assignment:**
- Slots 0-3 are assigned based on speaker characteristics
- First speaker detected → Slot 0, etc.
- `spkcache` maintains speaker identity across chunks within session

### Pyannote (DiarizerManager)

**Architecture:** 3-stage pipeline with WeSpeaker embeddings

```
Audio → VAD/Segmentation → WeSpeaker Embeddings → VBx/AHC Clustering → Timeline
```

**Strengths:**
- Unlimited speakers
- Cross-session speaker recognition (via embeddings)
- Known speaker enrollment
- Full embedding API via SpeakerManager

**Limitations:**
- Struggles with background noise
- Similar-sounding speakers may be confused
- Higher computational cost

**When to Use:**
- 5+ speakers in conversation
- Need cross-session speaker recognition
- Known speaker enrollment required

### Selection Strategy

```swift
// Recommended selection logic
func selectDiarizer(expectedSpeakers: Int?) -> DiarizationProvider {
    if let count = expectedSpeakers, count >= 5 {
        return pyannoteProvider  // Unlimited speakers
    }
    return sortformerProvider    // Better noise handling, faster
}
```

---

## Speaker Separation Models

### MossFormer2 Speaker Separation

| Model | Speakers | Sample Rate | Best For |
|-------|----------|-------------|----------|
| `twoSpeakerWHAMR` | 2 | 8kHz | **Noisy environments** (recommended) |
| `twoSpeaker` | 2 | 16kHz | Clean 2-speaker |
| `threeSpeaker` | 3 | 8kHz | 3-speaker scenarios |

**Auto-Selection Logic (implemented):**

```swift
// SeparationModel.forOverlappingSpeakers(_:)
switch overlappingSpeakers {
case 0, 1: return nil           // No separation needed
case 2:    return .mossformerWhamr  // Best for noisy
case 3:    return .mossformer3spk   // 3-speaker model
default:   return nil           // 4+ not supported
}
```

**Processing Characteristics:**
- Chunking: 4s chunks, 25% overlap, triangular blending
- Progress reporting: Per-chunk (0-90% processing, 90-100% reassembly)
- Peak normalization to 1.0

### Resampling Considerations

| From | To | Method |
|------|----|----- |
| Original (any) | 8kHz (WHAMR/3spk) | `audio.resampled(to: 8000)` |
| 8kHz (separated) | 16kHz (Sortformer) | `audio.resampled(to: 16000)` |

Note: MossFormer2 SR (super-resolution) is 16kHz → 48kHz, not suitable for this use case. Standard resampling is sufficient.

---

## Speaker Re-identification After Separation

### The Problem

After separating overlapping speech:
- Track 1: Clean audio of one speaker
- Track 2: Clean audio of another speaker
- **Unknown:** Which track belongs to which diarized speaker

### Solution: Sortformer Re-identification

**Key Insight:** Sortformer maintains speaker slot assignments via `spkcache`. Running separated tracks through the **same Sortformer instance** will identify which slot each track belongs to.

```
1. Run Sortformer on full audio
   → Timeline: Speaker 0 (0-5s), Overlap 0+1 (5-10s), Speaker 1 (10-15s)
   → spkcache now contains Speaker 0 and Speaker 1 characteristics
   
2. Separate overlap region (5-10s) → Track A, Track B

3. Run Sortformer on Track A (SAME instance, preserved state)
   → Probabilities: [0.95, 0.02, 0.01, 0.01]
   → Track A = Speaker 0 (highest probability)

4. Run Sortformer on Track B (SAME instance)
   → Probabilities: [0.03, 0.92, 0.02, 0.01]
   → Track B = Speaker 1
```

### Implementation Plan

#### Phase 1: Core Re-identification (Priority: High)

**1.1 Add `identifySpeaker` method to Sortformer provider**

```swift
public struct SpeakerIdentification: Sendable {
    public let speakerSlot: Int           // 0-3
    public let confidence: Float          // Probability for that slot
    public let allProbabilities: [Float]  // [4] probabilities
}

extension FluidAudioSortformerProvider {
    /// Identify which speaker slot an audio segment belongs to
    /// Uses preserved spkcache from previous diarization
    /// - Parameter audio: Audio to identify (will be resampled to 16kHz)
    /// - Returns: Speaker identification with slot and confidence
    public func identifySpeaker(_ audio: AudioBuffer) async throws -> SpeakerIdentification
}
```

**1.2 Add `SeparatedSpeakerTrack` result type**

```swift
public struct SeparatedSpeakerTrack: Sendable {
    public let audio: AudioBuffer
    public let speakerSlot: Int?          // From re-identification
    public let speakerID: SpeakerID?      // Mapped from diarization
    public let confidence: Float
    public let sourceTimeRange: TimeRange // Original overlap region
}
```

**1.3 Add `separateAndIdentify` to pipeline**

```swift
extension ClearVoice {
    /// Separate overlapping speech and identify speakers
    /// - Parameters:
    ///   - audio: Mixed audio with overlapping speakers
    ///   - timeline: Diarization timeline (for speaker mapping)
    ///   - diarizer: Sortformer instance with preserved state
    /// - Returns: Separated tracks with speaker identification
    public func separateAndIdentify(
        _ audio: AudioBuffer,
        timeline: SpeakerTimeline,
        diarizer: FluidAudioSortformerProvider
    ) async throws -> [SeparatedSpeakerTrack]
}
```

#### Phase 2: Pipeline Integration (Priority: High)

**2.1 Add `.separateOverlap` stage to PipelineBuilder**

```swift
public enum OverlapHandling: Sendable {
    case skip                    // Don't separate
    case separate                // Separate but don't identify
    case separateAndIdentify     // Full pipeline
}

extension PipelineBuilder {
    /// Separate overlapping speech regions
    /// Automatically selects model based on overlap speaker count
    public func separateOverlap(
        _ handling: OverlapHandling = .separateAndIdentify
    ) -> PipelineBuilder
}
```

**2.2 Add `separatedTracks` to PipelineResult**

```swift
public struct PipelineResult {
    // Existing...
    public let separatedTracks: [AudioBuffer]?
    
    // New
    public let identifiedTracks: [SeparatedSpeakerTrack]?
}
```

**2.3 Add pipeline events**

```swift
public enum PipelineEvent {
    // Existing...
    case speakerSeparated(speakerIndex: Int, audio: AudioBuffer)
    
    // New
    case overlapDetected(timeRange: TimeRange, speakerCount: Int)
    case trackIdentified(track: SeparatedSpeakerTrack)
}
```

#### Phase 3: Full Pipeline Example (Priority: Medium)

```swift
let voice = ClearVoice(...)

// Register providers
let sortformer = FluidAudioProviders.sortformerLowLatency()
let separator = MLXProviders.mossformer2SS(model: .twoSpeakerWHAMR)

await voice.register(diarization: sortformer)
await voice.register(separator: separator, for: .mossformerWhamr)

// Run pipeline
let result = try await voice.pipeline()
    .detect(.silero)
    .diarize()
    .separateOverlap(.separateAndIdentify)  // Auto-handles 2-3 speakers
    .enhance(.mossformerSE48k)              // Optional: enhance separated tracks
    .transcribe(.parakeet)
    .onEvent { event in
        switch event {
        case .overlapDetected(let range, let count):
            print("Overlap at \(range): \(count) speakers")
        case .trackIdentified(let track):
            print("Track identified: Speaker \(track.speakerSlot!)")
        default: break
        }
    }
    .process(audio: audio)

// Access results
for track in result.identifiedTracks ?? [] {
    print("Speaker \(track.speakerID!): \(track.audio.duration)s")
}
```

#### Phase 4: Pyannote Fallback (Priority: Low)

For 5+ speakers (where Sortformer can't be used):

**4.1 WeSpeaker embedding extraction**

```swift
extension ClearVoice {
    /// Extract speaker embedding from audio
    /// Uses WeSpeaker via Pyannote's EmbeddingExtractor
    public func extractSpeakerEmbedding(_ audio: AudioBuffer) async throws -> [Float]
}
```

**4.2 Embedding-based re-identification**

```swift
/// Match separated tracks to speakers using embeddings
/// For use when Sortformer is not available (5+ speakers)
public func identifyByEmbedding(
    tracks: [AudioBuffer],
    referenceEmbeddings: [SpeakerID: [Float]]
) async throws -> [SeparatedSpeakerTrack]
```

---

## API Summary

### New Types

```swift
// Speaker identification result
public struct SpeakerIdentification: Sendable {
    public let speakerSlot: Int
    public let confidence: Float
    public let allProbabilities: [Float]
}

// Separated track with identification
public struct SeparatedSpeakerTrack: Sendable {
    public let audio: AudioBuffer
    public let speakerSlot: Int?
    public let speakerID: SpeakerID?
    public let confidence: Float
    public let sourceTimeRange: TimeRange
}

// Overlap handling options
public enum OverlapHandling: Sendable {
    case skip
    case separate
    case separateAndIdentify
}
```

### New Methods

```swift
// Sortformer provider
func identifySpeaker(_ audio: AudioBuffer) async throws -> SpeakerIdentification

// ClearVoice
func separateAndIdentify(_:timeline:diarizer:) async throws -> [SeparatedSpeakerTrack]
func extractSpeakerEmbedding(_:) async throws -> [Float]  // Pyannote fallback

// PipelineBuilder
func separateOverlap(_: OverlapHandling) -> PipelineBuilder

// SpeakerTimeline
func overlappingRanges() -> [TimeRange]  // Already exists
func nonOverlappingSegments(for speaker: SpeakerID) -> [TimeRange]  // New
```

### New Pipeline Events

```swift
case overlapDetected(timeRange: TimeRange, speakerCount: Int)
case trackIdentified(track: SeparatedSpeakerTrack)
```

---

## Testing Plan

### Unit Tests

1. **Sortformer re-identification**
   - Process audio → separate overlap → identify → verify correct slots

2. **Auto model selection**
   - Verify WHAMR selected for 2 speakers
   - Verify 3spk selected for 3 speakers
   - Verify nil for 4+ speakers

3. **Resampling pipeline**
   - Original → 8kHz (separation) → 16kHz (identification) → verify quality

### Integration Tests

1. **Full pipeline with Harry Potter audio**
   - Diarize → detect overlaps → separate → identify → verify speaker continuity

2. **Progress reporting**
   - Verify events emitted for each stage

3. **Edge cases**
   - Very short overlaps (<0.5s)
   - No overlaps detected
   - Single speaker (no separation needed)

---

## Implementation Priority

| Phase | Description | Priority | Effort |
|-------|-------------|----------|--------|
| 1.1 | `identifySpeaker` method | High | Medium |
| 1.2 | `SeparatedSpeakerTrack` type | High | Low |
| 1.3 | `separateAndIdentify` method | High | Medium |
| 2.1 | `.separateOverlap` pipeline stage | High | Medium |
| 2.2 | Pipeline result updates | High | Low |
| 2.3 | Pipeline events | Medium | Low |
| 3 | Full pipeline example/tests | Medium | Medium |
| 4.1 | WeSpeaker embedding extraction | Low | Medium |
| 4.2 | Embedding-based identification | Low | Medium |

---

## Open Questions

1. **State persistence**: Should Sortformer state be saveable/restorable for multi-file processing?

2. **Confidence thresholds**: What confidence level indicates reliable identification?

3. **Partial overlaps**: How to handle regions where only part of the audio overlaps?

4. **Multi-overlap**: Audio with multiple overlapping regions - process sequentially or batch?

5. **Quality metrics**: How to measure separation + identification quality?

---

## References

- FluidAudio Sortformer docs: `Docs/temp/FluidAudio/Documentation/Sortformer.md`
- FluidAudio SpeakerManager docs: `Docs/temp/FluidAudio/Documentation/SpeakerManager.md`
- MossFormer2 SS implementation: `Sources/ClearVoiceMLX/MLXSeparatorProvider.swift`
- Current separation tests: `Tests/ClearVoiceFluidAudioTests/StreamingPipelineTests.swift`
