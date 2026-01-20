# Testing Harness Changes (Chunking, Preprocessing, etc.)

Since model correctness is already validated (PyTorch -> Python MLX -> Swift MLX, and PyTorch -> CoreML), we need testing for quality/performance degradation on Swift harness updates.

This tests: **"Does my Swift infrastructure preserve model quality?"**

---

## What Can Break in Harness Updates

| Component | Failure Mode |
|-----------|--------------|
| Chunking | Boundary artifacts, discontinuities |
| Overlap-add | Wrong window function, incorrect blending |
| Resampling | Aliasing, wrong rate |
| Normalization | Clipping, wrong scale |
| Streaming | Latency drift, buffer misalignment |
| Edge padding | Wrong padding mode, length mismatch |

---

## Recommended Test Strategy

### 1. Swift Goldens at Strategic Lengths

```
short.wav     ->  2s   (no chunking)
exact.wav     ->  4s   (exactly 1 chunk)  
boundary.wav  ->  4.1s (just over boundary)
long.wav      ->  12s  (3 chunks)
```

If chunking is 4s with 25% overlap, these cases catch:
- No-op path (short)
- Edge case (exact boundary)
- Minimal overlap (boundary + epsilon)
- Full pipeline (multiple chunks)

### 2. Boundary-Specific Assertions

Don't just compare full output - check chunk join regions specifically:

```
Samples around t=4.0s, t=8.0s should:
- Have no discontinuity (derivative check)
- Match expected overlap blend
- Not have audible clicks (zero-crossing check)
```

### 3. RTF Benchmarks

Track performance over time:

| Model          | Baseline RTF | Threshold |
|----------------|--------------|-----------|
| FRCRN 16K      | 8.2x         | > 6.0x    |
| MossFormer2 SE | 3.5x         | > 2.5x    |

Flag if RTF drops significantly. Could indicate accidental O(n^2) or memory pressure.

---

## Why Goldens Work Well Here

1. **Determinism**: Same model + same input = same output (on same hardware)
2. **Harness changes should be invisible**: If chunking works correctly, output is identical regardless of chunk size
3. **Cheap to run**: No Python, no model loading beyond what tests already do
4. **Catches real bugs**: Off-by-one in overlap, wrong window function, etc.

---

## What Goldens Won't Catch

- Performance regressions (need RTF benchmarks)
- Memory leaks (need memory profiling)
- Streaming latency issues (need timing assertions)
- "It's different but equally good" (but this shouldn't happen for harness changes)

---

## Implementation Examples

### Golden generation (one-time or on model update)

```swift
// Generate with known-good harness, save output
let output = try await provider.process(testInput)
try saveGolden(output.samples, to: "frcrn_12s_golden.safetensors")
```

### Test assertion

```swift
let output = try await provider.process(testInput)
let golden = try loadGolden("frcrn_12s_golden.safetensors")

// Tight tolerance - same model, same hardware
#expect(maxDiff(output.samples, golden) < 1e-5, 
        "Harness change affected output")
```

### Boundary check

```swift
// Check discontinuity at chunk boundary (4.0s = sample 64000 at 16kHz)
let boundary = 64000
let derivative = abs(output.samples[boundary] - output.samples[boundary-1])
#expect(derivative < 0.1, "Click detected at chunk boundary")
```

---

## Summary

For harness validation: **Swift goldens + boundary checks + RTF tracking**

- Goldens catch correctness regressions
- Boundary checks catch chunking bugs specifically
- RTF tracking catches performance regressions

No need for PyTorch or Python MLX comparison - model correctness is already validated. Now we're just ensuring Swift code doesn't break what's already working.
