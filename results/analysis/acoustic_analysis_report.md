# EARNINGS CALL ACOUSTIC ANALYSIS REPORT
================================================================================

## REPRODUCIBILITY NOTE
--------------------------------------------------
Random seed used for all bootstrap procedures: 42
Number of bootstrap iterations: 5000
Confidence level: 95.0%

## 1. BASELINE DISTRIBUTION SUMMARY
--------------------------------------------------
Total Baseline Calls: 21

### 1.1 Acoustic Features

**F0 Cv**:
  - Mean: 0.4847
  - Std Dev: 0.4036
  - Median: 0.5000
  - MAD: 0.5000
  - IQR: 0.9692

**F0 Std**:
  - Mean: 0.4704
  - Std Dev: 0.3802
  - Median: 0.5000
  - MAD: 0.4677
  - IQR: 0.8059

**Pause Frequency**:
  - Mean: 0.5127
  - Std Dev: 0.3879
  - Median: 0.5298
  - MAD: 0.4546
  - IQR: 0.8987

**Jitter Local**:
  - Mean: 0.4483
  - Std Dev: 0.4080
  - Median: 0.4139
  - MAD: 0.4139
  - IQR: 0.9932

### 1.2 Semantic Features

**Sentiment Negative**:
  - Mean: 0.4493
  - Std Dev: 0.2687
  - Median: 0.3803
  - MAD: 0.1608

**Sentiment Positive**:
  - Mean: 0.3215
  - Std Dev: 0.2938
  - Median: 0.2924
  - MAD: 0.2530

**Sentiment Variability**:
  - Mean: 0.6596
  - Std Dev: 0.2389
  - Median: 0.7221
  - MAD: 0.0857

## 2. NON-AFFIRMATION GROUP ANALYSIS
--------------------------------------------------

### 2.1 UPGRADE GROUP (N=1)

**Pattern**: Convergent Stress
**Strength**: Moderate
**Confidence**: Low
**F0 CV Percentile**: 88.1%
**Semantic Percentile**: 90.5%
**F0 CV Cohen's d**: 1.28
**F0 CV MAD Effect**: 1.00
**Semantic Cohen's d**: 1.76
**Semantic MAD Effect**: 3.37

**Key Acoustic Features**:
  - F0 Cv: 1.0000 (88.1%ile, MAD effect: 1.00)
  - F0 Std: 1.0000 (90.5%ile, MAD effect: 1.07)
  - Pause Frequency: 1.0000 (90.5%ile, MAD effect: 1.03)
  - Jitter Local: 1.0000 (88.1%ile, MAD effect: 1.42)

**Key Semantic Features**:
  - Sentiment Negative: 0.9222 (90.5%ile, MAD effect: 3.37)
  - Sentiment Positive: 0.0968 (28.6%ile, MAD effect: -0.77)
  - Sentiment Variability: 0.2448 (9.5%ile, MAD effect: -5.57)

### 2.1 DOWNGRADE GROUP (N=2)

**Pattern**: Mixed Pattern
**Strength**: Moderate
**Confidence**: Low
**F0 CV Percentile**: 38.1%
**Semantic Percentile**: 69.0%
**F0 CV Cohen's d**: -0.65
**F0 CV MAD Effect**: -0.55
**Semantic Cohen's d**: 0.45
**Semantic MAD Effect**: 1.18

**Key Acoustic Features**:
  - F0 Cv: 0.2231 (38.1%ile, MAD effect: -0.55)
  - F0 Std: 0.5000 (51.2%ile, MAD effect: 0.00)
  - Pause Frequency: 0.7536 (69.0%ile, MAD effect: 0.49)
  - Jitter Local: 0.3944 (47.6%ile, MAD effect: -0.05)

**Key Semantic Features**:
  - Sentiment Negative: 0.5701 (69.0%ile, MAD effect: 1.18)
  - Sentiment Positive: 0.2072 (47.6%ile, MAD effect: -0.34)
  - Sentiment Variability: 0.6704 (54.8%ile, MAD effect: -0.60)

## 3. INDIVIDUAL CASE STUDIES
--------------------------------------------------

### 3.1 Case: 4368670 (UPGRADE)

**Communication Pattern**: Mixed Pattern

**Key Percentiles**:
  - F0 Cv: 1.0000 (88.1%ile, MAD effect: 1.00)
  - F0 Std: 1.0000 (90.5%ile, MAD effect: 1.07)
  - Pause Frequency: 1.0000 (90.5%ile, MAD effect: 1.03)
  - Jitter Local: 1.0000 (88.1%ile, MAD effect: 1.42)
  - Sentiment Negative: 0.9222 (90.5%ile, MAD effect: 3.37)
  - Sentiment Positive: 0.0968 (28.6%ile, MAD effect: -0.77)
  - Sentiment Variability: 0.2448 (9.5%ile, MAD effect: -5.57)

### 3.2 Case: 4346923 (DOWNGRADE)

**Communication Pattern**: Baseline Stability

**Key Percentiles**:
  - F0 Cv: 0.2089 (38.1%ile, MAD effect: -0.58)
  - F0 Std: 0.0000 (11.9%ile, MAD effect: -1.07)
  - Pause Frequency: 1.0000 (90.5%ile, MAD effect: 1.03)
  - Jitter Local: 0.4072 (47.6%ile, MAD effect: -0.02)
  - Sentiment Negative: 0.4415 (57.1%ile, MAD effect: 0.38)
  - Sentiment Positive: 0.3980 (71.4%ile, MAD effect: 0.42)
  - Sentiment Variability: 0.9004 (95.2%ile, MAD effect: 2.08)

### 3.3 Case: 4384683 (DOWNGRADE)

**Communication Pattern**: Moderate Stress

**Key Percentiles**:
  - F0 Cv: 0.2373 (38.1%ile, MAD effect: -0.53)
  - F0 Std: 1.0000 (90.5%ile, MAD effect: 1.07)
  - Pause Frequency: 0.5072 (47.6%ile, MAD effect: -0.05)
  - Jitter Local: 0.3815 (47.6%ile, MAD effect: -0.08)
  - Sentiment Negative: 0.6987 (81.0%ile, MAD effect: 1.98)
  - Sentiment Positive: 0.0163 (23.8%ile, MAD effect: -1.09)
  - Sentiment Variability: 0.4405 (14.3%ile, MAD effect: -3.28)

## 4. METHODOLOGY NOTE
--------------------------------------------------

This analysis follows a descriptive exploration methodology appropriate for small-sample analysis:

1. Baseline Establishment: Affirmation calls (N≈21) provide the reference distribution for acoustic and semantic features.

2. Percentile Ranking: Non-affirmation cases are ranked against this baseline, with bootstrap confidence intervals for uncertainty quantification (seed=42 for reproducibility).

3. MAD-Based Standardization: In addition to traditional Cohen's d, we report MAD-based effect sizes which are:
   - Robust to outliers and non-normal distributions
   - Calculable for any sample size (including n=1)
   - More appropriate for small samples than mean/SD-based measures

4. Pattern Classification: Cases are classified based on acoustic-semantic alignment following Russell's Circumplex Model of Affect.

5. Case Studies: Individual cases receive detailed analysis rather than attempting group-level inference inappropriate for small samples.

This approach aligns with contemporary scientific frameworks for financial speech analysis while acknowledging sample size constraints.
