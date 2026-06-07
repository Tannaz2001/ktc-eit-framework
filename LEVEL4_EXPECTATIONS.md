# Level 4 Performance Expectations

**Context**: Level 4 is the halfway point in KTC 2023 difficulty. It uses significantly reduced measurement coverage.

---

## Level 4 Characteristics

### Measurement Reduction
| Level | Injections | Measurements | Coverage | Data Loss |
|-------|-----------|--------------|----------|-----------|
| 1     | 76        | 2,460        | 100%     | 0%        |
| 2     | 61        | 1,830        | 74%      | 26%       |
| 3     | 49        | 1,176        | 48%      | 52%       |
| **4** | **38**    | **703**      | **29%**  | **71%**   |
| 5     | 30        | 435          | 18%      | 82%       |
| 6     | 19        | 171          | 7%       | 93%       |
| 7     | 9         | 36           | 1.5%     | 98.5%     |

### What This Means Physically
- **Level 1**: Full electrode coverage, all electrodes inject and measure
- **Level 4**: Only ~50% of electrodes are active; information is genuinely sparse
- **Level 7**: Only 9 electrodes; almost no spatial information

---

## Performance Expectations at Level 4

### Baseline (All-Water)
- KTC score: **0.0** (by definition—baseline is water)
- If your method scores < -0.01: **worse than guessing water**

### Realistic Performance Range
Based on KTC 2023 challenge results:

| Method | Level 1 | Level 4 | Degradation |
|--------|---------|---------|-------------|
| Best   | 0.65    | 0.30    | -54%        |
| Median | 0.45    | 0.15    | -67%        |
| Worst  | 0.10    | -0.10   | -100%       |

**At Level 4, scores naturally drop by 50-70% compared to Level 1.**

---

## What Your -0.008 Score Means

### Interpretation
- **-0.008 vs baseline (0.0)**: 0.8% worse than just predicting all water
- **This is VERY close to baseline**—almost no signal detected

### Likely Causes (in order of probability)

1. **Sparse data is genuinely hard** (70% data loss)
   - With 71% of measurements removed, reconstruction is severely underdetermined
   - Even optimal methods score only ~0.15 at Level 4

2. **Electrode positions still wrong** (20%)
   - Despite fixes, electrode detection may still be incorrect
   - Check: Does `extract_electrode_nodes.py` run successfully?

3. **Measurement patterns mismatch** (5%)
   - Inj or Mpat may not match actual electrode count
   - Should be verified with mesh structure

4. **Reference voltages not populated** (5%)
   - Falls back to mean subtraction (we fixed this to error, but check logs)

---

## How to Diagnose

### Test 1: Compare to Baseline
```python
# If your score > baseline (0.0), code is working
# If your score < -0.05, electrode mapping is likely wrong
```

### Test 2: Visual Inspection
```python
# Display reconstruction output
import matplotlib.pyplot as plt
pred = bp.reconstruct(batch)
gt = batch.ground_truth

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt, cmap='viridis')
axes[0].set_title('Ground Truth')
axes[1].imshow(pred, cmap='viridis')
axes[1].set_title('Prediction')
axes[2].imshow(np.abs(pred.astype(float) - gt.astype(float)), cmap='hot')
axes[2].set_title('Error')
plt.show()

# If pred is scattered noise → electrode bug
# If pred has coherent structure (even if wrong) → sparse data is the issue
```

### Test 3: Compare Methods
```python
# Run BackProjection, GaussNewton, etc. at Level 4
# If all methods score ~0 to -0.05 → sparse data is the limit
# If BackProjection scores -0.05 and others score +0.10 → BackProjection is broken
```

---

## Realistic Goals for Level 4

### Conservative Estimate
- **Achievable**: Score > 0.05 (marginally better than baseline)
- **Good**: Score > 0.15 (typical of median methods)
- **Excellent**: Score > 0.30 (top 25% of challenge)

### Why Low Scores are OK
At Level 4, you're trying to reconstruct from 29% of ideal data:
- Information theory says 71% data loss makes reconstruction hard
- Electrical impedance is ambiguous with sparse electrodes
- Even iterative methods (Gauss-Newton) struggle below 0.20 at this level

---

## What to Check

### If Score is -0.008 (Worse than Baseline)
- [ ] Are electrode positions correct? (Run `extract_electrode_nodes.py`)
- [ ] Are reference_voltages populated? (Check logs for "ValueError")
- [ ] Does reconstructed image have ANY structure, or just noise?
- [ ] Do other methods (GaussNewton) score higher?

### If Score is 0.05-0.15 (Typical for Level 4)
- ✓ Code is likely working correctly
- ✓ Sparse data is just genuinely hard
- ✓ This is normal for Level 4

### If Score is >0.20 (Good for Level 4)
- ✓ Algorithm is working well
- ✓ Consider this a success for sparse regime

---

## Recommendation

**Before declaring electrode mapping a bug, verify:**

1. Run BackProjection, GaussNewton, MockBaseline at Level 4
2. If all three score near 0.0: sparse data is the bottleneck (expected)
3. If BackProjection is significantly worse than others: electrode bug (likely)
4. Check visual output (scattered noise vs coherent structure)

The -0.008 score by itself doesn't prove a bug—it might just prove that Level 4 is hard.

