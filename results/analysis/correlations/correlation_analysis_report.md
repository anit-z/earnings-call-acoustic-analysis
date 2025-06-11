# CORRELATION ANALYSIS REPORT
================================================================================

## REPRODUCIBILITY NOTE
--------------------------------------------------
Random seed used for bootstrap procedures: 42
Bootstrap iterations for confidence intervals: 1000

## 1. SIGNIFICANT CORRELATIONS SUMMARY
--------------------------------------------------

### 1.1 Full Correlations
Total significant correlations: 18

**Strong correlations (|r| ≥ 0.6):**
- Sentiment Negative & Sentiment Neutral: r = -0.992 (negative, p = 0.0000)
- Sentiment Neutral & Sentiment Negative: r = -0.992 (negative, p = 0.0000)
- Is Downgrade & Is Affirm: r = -0.798 (negative, p = 0.0001)
- Is Affirm & Is Downgrade: r = -0.798 (negative, p = 0.0001)
- Sentiment Negative & Sentiment Variability: r = -0.746 (negative, p = 0.0006)

**Moderate correlations (0.3 ≤ |r| < 0.6):**
- F0 Cv & Jitter Local: r = 0.594 (positive, p = 0.0209)
- Jitter Local & F0 Cv: r = 0.594 (positive, p = 0.0209)
- Is Upgrade & Is Affirm: r = -0.552 (negative, p = 0.0429)
- Is Affirm & Is Upgrade: r = -0.552 (negative, p = 0.0429)
- F0 Cv & F0 Std: r = 0.542 (positive, p = 0.0453)

### 1.1 Acoustic-Acoustic Correlations
Total significant correlations: 6

**Strong correlations (|r| ≥ 0.6):**
- F0 Cv & Pause Frequency: r = 0.647 (positive, p = 0.0038)
- Pause Frequency & F0 Cv: r = 0.647 (positive, p = 0.0038)

**Moderate correlations (0.3 ≤ |r| < 0.6):**
- F0 Cv & Jitter Local: r = 0.594 (positive, p = 0.0067)
- Jitter Local & F0 Cv: r = 0.594 (positive, p = 0.0067)
- F0 Cv & F0 Std: r = 0.542 (positive, p = 0.0124)
- F0 Std & F0 Cv: r = 0.542 (positive, p = 0.0124)

### 1.1 Semantic-Semantic Correlations
Total significant correlations: 8

**Strong correlations (|r| ≥ 0.6):**
- Sentiment Negative & Sentiment Neutral: r = -0.992 (negative, p = 0.0000)
- Sentiment Neutral & Sentiment Negative: r = -0.992 (negative, p = 0.0000)
- Sentiment Negative & Sentiment Variability: r = -0.746 (negative, p = 0.0001)
- Sentiment Variability & Sentiment Negative: r = -0.746 (negative, p = 0.0001)
- Sentiment Neutral & Sentiment Variability: r = 0.691 (positive, p = 0.0004)

## 2. ACOUSTIC-SEMANTIC CORRELATIONS
--------------------------------------------------

### 2.1 Convergent Stress Indicators
Looking for high correlations between stress-related acoustic features and negative sentiment:
- F0 Cv & Sentiment Negative: r = -0.165 (weak negative)
- F0 Cv & Sentiment Variability: r = 0.110 (weak positive)
- F0 Std & Sentiment Negative: r = 0.187 (weak positive)
- F0 Std & Sentiment Variability: r = -0.126 (weak negative)
- Pause Frequency & Sentiment Negative: r = -0.031 (weak negative)
- Pause Frequency & Sentiment Variability: r = -0.093 (weak negative)
- Jitter Local & Sentiment Negative: r = -0.179 (weak negative)
- Jitter Local & Sentiment Variability: r = 0.233 (weak positive)

## 3. METHODOLOGICAL NOTES
--------------------------------------------------

This correlation analysis implements the methodology specified in the thesis:

1. **Acoustic-Semantic Convergence Thresholds** (as per Section 3.4):
   - Strong correlation: |r| ≥ 0.6
   - Moderate correlation: 0.3 ≤ |r| < 0.6
   - Weak correlation: |r| < 0.3

2. **Multiple Comparison Correction**: P-values are adjusted using the fdr_bh method to control for false discovery rate.

3. **Significance Threshold**: Correlations are considered significant at p < 0.05 after correction.

4. **Bootstrap Confidence Intervals**: Each correlation includes bootstrap-based confidence intervals (1000 iterations, seed=42).

5. **Sample Size Considerations**: Given n=24 with extreme class imbalance (21:2:1), correlations should be interpreted as exploratory baselines rather than definitive findings.

6. **Validation Approach**: FinBERT sentiment serves as a directional validator for acoustic stress markers, not as a fused feature.

7. **Constant Features**: Features with identical values across all samples cannot have meaningful correlations and are reported as NaN.
