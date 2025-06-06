# EARNINGS CALL ACOUSTIC ANALYSIS REPORT
================================================================================

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

**Jitter Local**:
  - Mean: 0.4483
  - Std Dev: 0.4080
  - Median: 0.4139
  - MAD: 0.4139
  - IQR: 0.9932

**Pause Frequency**:
  - Mean: 0.5127
  - Std Dev: 0.3879
  - Median: 0.5298
  - MAD: 0.4546
  - IQR: 0.8987

**Acoustic Volatility Index**:
  - Mean: 0.4429
  - Std Dev: 0.3984
  - Median: 0.3670
  - MAD: 0.3670
  - IQR: 0.9922

### 1.2 Semantic Features

**Sentiment Negative**:
  - Mean: 0.4493
  - Std Dev: 0.2687
  - Median: 0.3803

**Sentiment Positive**:
  - Mean: 0.3215
  - Std Dev: 0.2938
  - Median: 0.2924

**Sentiment Variability**:
  - Mean: 0.6596
  - Std Dev: 0.2389
  - Median: 0.7221

## 2. NON-AFFIRMATION GROUP ANALYSIS
--------------------------------------------------

### 2.1 UPGRADE GROUP (N=1)

**Pattern**: Divergent Semantic Stress
**Strength**: Strong
**Confidence**: Low
**Acoustic Percentile**: 38.1%
**Semantic Percentile**: 90.5%
**Acoustic Effect Size**: -0.59
**Semantic Effect Size**: 1.76

**Key Acoustic Features**:
  - F0 Cv: 1.0000 (88.1%ile)
  - Jitter Local: 1.0000 (88.1%ile)
  - Pause Frequency: 1.0000 (90.5%ile)
  - Acoustic Volatility Index: 0.2087 (38.1%ile)

**Key Semantic Features**:
  - Sentiment Negative: 0.9222 (90.5%ile)
  - Sentiment Positive: 0.0968 (28.6%ile)
  - Sentiment Variability: 0.2448 (9.5%ile)

### 2.1 DOWNGRADE GROUP (N=2)

**Pattern**: Mixed Pattern
**Strength**: Moderate
**Confidence**: Low
**Acoustic Percentile**: 41.7%
**Semantic Percentile**: 69.0%
**Acoustic Effect Size**: 0.06
**Semantic Effect Size**: 0.45

**Key Acoustic Features**:
  - F0 Cv: 0.2231 (38.1%ile)
  - Jitter Local: 0.3944 (47.6%ile)
  - Pause Frequency: 0.7536 (69.0%ile)
  - Acoustic Volatility Index: 0.4664 (41.7%ile)

**Key Semantic Features**:
  - Sentiment Negative: 0.5701 (69.0%ile)
  - Sentiment Positive: 0.2072 (47.6%ile)
  - Sentiment Variability: 0.6704 (54.8%ile)

## 3. INDIVIDUAL CASE STUDIES
--------------------------------------------------

### 3.1 Case: 4368670 (UPGRADE)

**Communication Pattern**: Mixed Pattern

**Key Percentiles**:
  - F0 Cv: 1.0000 (88.1%ile)
  - Jitter Local: 1.0000 (88.1%ile)
  - Pause Frequency: 1.0000 (90.5%ile)
  - Acoustic Volatility Index: 0.2087 (38.1%ile)
  - Sentiment Negative: 0.9222 (90.5%ile)
  - Sentiment Positive: 0.0968 (28.6%ile)
  - Sentiment Variability: 0.2448 (9.5%ile)

### 3.2 Case: 4346923 (DOWNGRADE)

**Communication Pattern**: Baseline Stability

**Key Percentiles**:
  - F0 Cv: 0.2089 (38.1%ile)
  - Jitter Local: 0.4072 (47.6%ile)
  - Pause Frequency: 1.0000 (90.5%ile)
  - Acoustic Volatility Index: 0.0000 (11.9%ile)
  - Sentiment Negative: 0.4415 (57.1%ile)
  - Sentiment Positive: 0.3980 (71.4%ile)
  - Sentiment Variability: 0.9004 (95.2%ile)

### 3.3 Case: 4384683 (DOWNGRADE)

**Communication Pattern**: Moderate Stress

**Key Percentiles**:
  - F0 Cv: 0.2373 (38.1%ile)
  - Jitter Local: 0.3815 (47.6%ile)
  - Pause Frequency: 0.5072 (47.6%ile)
  - Acoustic Volatility Index: 0.9329 (71.4%ile)
  - Sentiment Negative: 0.6987 (81.0%ile)
  - Sentiment Positive: 0.0163 (23.8%ile)
  - Sentiment Variability: 0.4405 (14.3%ile)

## 4. METHODOLOGY NOTE
--------------------------------------------------

This analysis follows a descriptive exploration methodology appropriate for small-sample analysis:

1. Baseline Establishment: Affirmation calls (N≈21) provide the reference distribution for acoustic and semantic features.

2. Percentile Ranking: Non-affirmation cases are ranked against this baseline, with bootstrap confidence intervals for uncertainty quantification.

3. Pattern Classification: Cases are classified based on acoustic-semantic alignment following Russell's Circumplex Model of Affect.

4. Case Studies: Individual cases receive detailed analysis rather than attempting group-level inference inappropriate for small samples.

This approach aligns with contemporary scientific frameworks for financial speech analysis while acknowledging sample size constraints.
