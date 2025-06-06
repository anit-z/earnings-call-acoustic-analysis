# CORRELATION ANALYSIS REPORT
================================================================================

## 1. SIGNIFICANT CORRELATIONS SUMMARY
--------------------------------------------------

### 1.1 Full Correlations
Total significant correlations: 14

Top 10 significant correlations:
1. **Sentiment Negative** and **Sentiment Neutral**: -0.992 (strong negative, p=0.0000)
2. **Sentiment Neutral** and **Sentiment Negative**: -0.992 (strong negative, p=0.0000)
3. **Is Downgrade** and **Is Affirm**: -0.798 (strong negative, p=0.0001)
4. **Is Affirm** and **Is Downgrade**: -0.798 (strong negative, p=0.0001)
5. **Sentiment Negative** and **Sentiment Variability**: -0.746 (strong negative, p=0.0009)
6. **Sentiment Variability** and **Sentiment Negative**: -0.746 (strong negative, p=0.0009)
7. **Sentiment Neutral** and **Sentiment Variability**: 0.691 (strong positive, p=0.0042)
8. **Sentiment Variability** and **Sentiment Neutral**: 0.691 (strong positive, p=0.0042)
9. **F0 Cv** and **Pause Frequency**: 0.647 (strong positive, p=0.0103)
10. **Pause Frequency** and **F0 Cv**: 0.647 (strong positive, p=0.0103)

### 1.1 Acoustic-Acoustic Correlations
Total significant correlations: 8

Top 8 significant correlations:
1. **F0 Cv** and **Pause Frequency**: 0.647 (strong positive, p=0.0094)
2. **Pause Frequency** and **F0 Cv**: 0.647 (strong positive, p=0.0094)
3. **F0 Cv** and **Jitter Local**: 0.594 (strong positive, p=0.0167)
4. **Jitter Local** and **F0 Cv**: 0.594 (strong positive, p=0.0167)
5. **F0 Cv** and **F0 Std**: 0.542 (strong positive, p=0.0309)
6. **F0 Std** and **F0 Cv**: 0.542 (strong positive, p=0.0309)
7. **F0 Std** and **Acoustic Volatility Index**: 0.510 (strong positive, p=0.0408)
8. **Acoustic Volatility Index** and **F0 Std**: 0.510 (strong positive, p=0.0408)

### 1.1 Semantic-Semantic Correlations
Total significant correlations: 8

Top 8 significant correlations:
1. **Sentiment Negative** and **Sentiment Neutral**: -0.992 (strong negative, p=0.0000)
2. **Sentiment Neutral** and **Sentiment Negative**: -0.992 (strong negative, p=0.0000)
3. **Sentiment Negative** and **Sentiment Variability**: -0.746 (strong negative, p=0.0001)
4. **Sentiment Variability** and **Sentiment Negative**: -0.746 (strong negative, p=0.0001)
5. **Sentiment Neutral** and **Sentiment Variability**: 0.691 (strong positive, p=0.0004)
6. **Sentiment Variability** and **Sentiment Neutral**: 0.691 (strong positive, p=0.0004)
7. **Sentiment Positive** and **Sentiment Variability**: 0.644 (strong positive, p=0.0010)
8. **Sentiment Variability** and **Sentiment Positive**: 0.644 (strong positive, p=0.0010)

## 2. ACOUSTIC-SEMANTIC CORRELATIONS
--------------------------------------------------

Strongest acoustic-semantic correlations:

## 3. FEATURE-RATING CORRELATIONS
--------------------------------------------------

Features most correlated with downgrades:
- **Pause Frequency**: 0.156 (weak positive, p=0.4661)
- **Sentiment Negative**: 0.101 (weak positive, p=0.6393)
- **Speech Rate**: 0.053 (weak positive, p=0.8069)
- **Acoustic Semantic Alignment**: 0.042 (weak positive, p=0.8452)
- **Sentiment Variability**: 0.033 (weak positive, p=0.8779)

Features most correlated with upgrades:
- **Sentiment Negative**: 0.339 (moderate positive, p=0.1049)
- **Jitter Local**: 0.280 (weak positive, p=0.1859)
- **F0 Cv**: 0.269 (weak positive, p=0.2031)
- **F0 Std**: 0.265 (weak positive, p=0.2116)
- **Pause Frequency**: 0.241 (weak positive, p=0.2571)

## 4. PATTERN-BASED CORRELATION INTERPRETATION
--------------------------------------------------

Communication pattern distribution:
- Mixed Pattern: 11
- Baseline Stability: 6
- Moderate Stress: 4
- High Excitement: 1
- Moderate Excitement: 1
- High Stress: 1

Top acoustic-semantic correlations from pattern analysis:
- **Sentiment Positive** and **Jitter Local**: 0.448 (moderate positive)
- **Sentiment Positive** and **F0 Cv**: 0.269 (weak positive)
- **Sentiment Positive** and **Speech Rate**: 0.247 (weak positive)
- **Sentiment Variability** and **Jitter Local**: 0.233 (weak positive)
- **Sentiment Neutral** and **F0 Std**: -0.195 (weak negative)

## 5. METHODOLOGICAL NOTES
--------------------------------------------------

This correlation analysis implements a comprehensive framework for examining relationships in acoustic-semantic-rating data:

1. Multiple Comparison Correction: P-values are adjusted using the fdr_bh method to control for false discovery rate.

2. Significance Threshold: Correlations are considered significant at p < 0.05 after correction.

3. Effect Size Interpretation: Correlation strength is categorized as:
   - Strong: |r| > 0.5
   - Moderate: 0.3 < |r| < 0.5
   - Weak: |r| < 0.3

4. Sample Size Considerations: Given the small sample size, correlations should be interpreted cautiously, particularly for sector-specific analyses.

5. Convergent Evidence: Findings are most reliable when supported by multiple correlation indicators and aligned with theoretical expectations.
