# demonstrator/backend/services/stress_analyzer.py
"""Core stress analysis service with baseline comparison"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy import stats
from pathlib import Path
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.config import FEATURES_DIR, BOOTSTRAP_ITERATIONS

class StressAnalyzer:
    def __init__(self):
        self.baseline_features = self._load_baseline_features()
        self.feature_names = [
            'f0_cv', 'jitter', 'speech_rate', 'pause_ratio',
            'spectral_centroid_mean', 'mfcc_0_mean'
        ]
        
    def _load_baseline_features(self) -> pd.DataFrame:
        """Load baseline features from affirmation calls"""
        try:
            return pd.read_csv(FEATURES_DIR / "baseline_affirmations.csv")
        except:
            # Return mock data for demonstration
            return pd.DataFrame({
                'f0_cv': np.random.normal(0.15, 0.05, 100),
                'jitter': np.random.normal(0.02, 0.01, 100),
                'speech_rate': np.random.normal(3.5, 0.5, 100),
                'pause_ratio': np.random.normal(0.2, 0.1, 100)
            })
    
    def analyze_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze features against baseline with confidence intervals"""
        results = {
            'composite_stress_score': 0,
            'feature_percentiles': {},
            'confidence_intervals': {},
            'pattern_classification': None
        }
        
        stress_scores = []
        
        for feature_name in self.feature_names:
            if feature_name in features and feature_name in self.baseline_features.columns:
                value = features[feature_name]
                baseline = self.baseline_features[feature_name].values
                
                # Calculate percentile rank with bootstrap CI
                percentile_result = self._calculate_percentile_with_ci(
                    value, baseline
                )
                
                results['feature_percentiles'][feature_name] = percentile_result
                
                # Contribute to composite score
                if feature_name in ['f0_cv', 'jitter', 'pause_ratio']:
                    stress_scores.append(percentile_result['percentile'])
        
        # Composite stress score
        if stress_scores:
            results['composite_stress_score'] = np.mean(stress_scores)
        
        # Pattern classification
        results['pattern_classification'] = self._classify_stress_pattern(results)
        
        return results
    
    def _calculate_percentile_with_ci(
        self, 
        value: float, 
        baseline: np.ndarray
    ) -> Dict[str, float]:
        """Calculate percentile rank with bootstrap confidence interval"""
        percentile = (np.sum(baseline <= value) / len(baseline)) * 100
        
        # Bootstrap for CI
        bootstrap_percentiles = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            bootstrap_sample = np.random.choice(
                baseline, 
                size=len(baseline), 
                replace=True
            )
            boot_percentile = (np.sum(bootstrap_sample <= value) / 
                              len(bootstrap_sample)) * 100
            bootstrap_percentiles.append(boot_percentile)
        
        ci_lower = np.percentile(bootstrap_percentiles, 2.5)
        ci_upper = np.percentile(bootstrap_percentiles, 97.5)
        
        return {
            'percentile': percentile,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'uncertainty_range': ci_upper - ci_lower
        }
    
    def _classify_stress_pattern(self, results: Dict[str, Any]) -> str:
        """Classify stress pattern based on percentile results"""
        composite_score = results['composite_stress_score']
        
        if composite_score >= 90:
            return "High Stress - Significant deviation from baseline"
        elif composite_score >= 75:
            return "Moderate Stress - Notable elevation in stress indicators"
        elif composite_score >= 50:
            return "Mild Stress - Some stress indicators present"
        else:
            return "Low Stress - Within normal baseline range"
    
    def calculate_correlation(
        self, 
        acoustic_features: Dict[str, float],
        sentiment_results: Dict[str, float]
    ) -> float:
        """Calculate acoustic-semantic correlation"""
        if not sentiment_results:
            return 0.0
        
        # Extract stress indicators and sentiment
        stress_score = np.mean([
            acoustic_features.get('f0_cv', 0),
            acoustic_features.get('jitter', 0),
            acoustic_features.get('pause_ratio', 0)
        ])
        
        negative_sentiment = sentiment_results.get('negative_score', 0)
        
        # Simple correlation for demonstration
        return min(stress_score * negative_sentiment * 2, 1.0)
    
    def quick_analysis(self, features: Dict[str, float]) -> float:
        """Quick stress level calculation for real-time processing"""
        stress_indicators = [
            features.get('f0_cv', 0),
            features.get('jitter', 0),
            features.get('pause_ratio', 0)
        ]
        
        # Normalize and average
        normalized_stress = [min(s / 0.3, 1.0) for s in stress_indicators]
        return np.mean(normalized_stress)
    
    def get_baseline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive baseline statistics"""
        stats = {
            'n_samples': len(self.baseline_features),
            'distributions': {},
            'percentiles': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        for feature in self.feature_names:
            if feature in self.baseline_features.columns:
                data = self.baseline_features[feature]
                stats['distributions'][feature] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'median': float(data.median()),
                    'mad': float(stats.median_abs_deviation(data))
                }
                stats['percentiles'][feature] = {
                    'p5': float(data.quantile(0.05)),
                    'p25': float(data.quantile(0.25)),
                    'p50': float(data.quantile(0.50)),
                    'p75': float(data.quantile(0.75)),
                    'p95': float(data.quantile(0.95))
                }
        
        return stats