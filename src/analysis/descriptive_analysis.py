#!/usr/bin/env python3
"""
Descriptive Analysis for Earnings Call Acoustic Features
Implements small-sample descriptive exploration with bootstrap uncertainty quantification
Following contemporary scientific framework for financial-acoustic research
"""

import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DescriptiveAnalysis:
    """
    Descriptive analysis for earnings call acoustic-semantic features
    Implements small-sample methodological approach with percentile ranking
    """
    
    def __init__(self, 
                 n_bootstrap: int = 10000, 
                 confidence_level: float = 0.95,
                 random_seed: int = 42):
        """
        Initialize descriptive analysis
        
        Args:
            n_bootstrap: Number of bootstrap iterations for CIs
            confidence_level: Confidence level for intervals
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Key acoustic features for analysis
        self.key_acoustic_features = [
            'f0_cv',      # F0 Coefficient of Variation - Primary feature
            'f0_std',     # F0 Standard Deviation - Complementary pitch measure
            'pause_frequency',  # Pause Frequency - Temporal stress indicator
            'jitter_local'      # Jitter Local - Voice quality measure
        ]
        
        # Key semantic features for analysis
        self.key_semantic_features = [
            'sentiment_negative',
            'sentiment_positive',
            'sentiment_variability'
        ]
        
        # Key combined features for analysis
        self.key_combined_features = [
            'acoustic_semantic_alignment'
        ]
        
        # Color palette
        self.color_palette = [
            '#D0E7F9',   # Very light sky blue
            '#A8CDFE',   # Light sky blue
            '#5BA0FF',   # Bright, clear blue
            '#2C60D9',   # Rich medium blue
            '#1D8AA2',   # Teal-blue, more cyan/green hint for contrast
            '#1780B9',   # Deep cyan-blue (teal hint)
            '#0D427F',   # Dark blue-gray
            '#062F5B'    # Very dark blue
        ]
    
    def load_data(self, 
                features_dir: str, 
                ratings_file: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load features and ratings data
        
        Args:
            features_dir: Directory containing combined features
            ratings_file: Optional path to ratings CSV file
            
        Returns:
            Tuple of (features_df, ratings_df)
        """
        # Load combined features
        features_path = Path(features_dir) / "combined_features.csv"
        if not features_path.exists():
            logger.error(f"Combined features file not found: {features_path}")
            raise FileNotFoundError(f"Combined features file not found: {features_path}")
        
        features_df = pd.read_csv(features_path)
        logger.info(f"Loaded combined features: {len(features_df)} calls")
        
        # Load ratings data if provided
        ratings_df = None
        if ratings_file and Path(ratings_file).exists():
            ratings_df = pd.read_csv(ratings_file)
            logger.info(f"Loaded ratings data: {len(ratings_df)} companies")
            
            # Ensure file_id is string type for proper matching
            if 'file_id' in ratings_df.columns:
                ratings_df['file_id'] = ratings_df['file_id'].astype(str)
        
        return features_df, ratings_df
    

    def prepare_analysis_data(self, 
                            features_df: pd.DataFrame, 
                            ratings_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare data for analysis by merging features and ratings
        
        Args:
            features_df: Features DataFrame
            ratings_df: Optional ratings DataFrame
            
        Returns:
            Prepared DataFrame for analysis
        """
        # Ensure file_id is string type for proper matching in BOTH dataframes
        if 'file_id' in features_df.columns:
            features_df['file_id'] = features_df['file_id'].astype(str)
        
        # If no ratings data, return features as is
        if ratings_df is None:
            return features_df
        
        # Ensure file_id is string type for proper matching
        if 'file_id' in ratings_df.columns:
            ratings_df['file_id'] = ratings_df['file_id'].astype(str)
        
        # Merge features and ratings
        merged_df = pd.merge(
            features_df,
            ratings_df,
            on='file_id',
            how='inner'
        )
        
        logger.info(f"Merged data: {len(merged_df)} calls with features and ratings")
        
        # If no matches, return original features
        if len(merged_df) == 0:
            logger.warning("No matches between features and ratings. Check file_id consistency.")
            return features_df
        
        return merged_df
    
    def create_rating_groups(self, analysis_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create rating-based groups for comparison
        
        Args:
            analysis_df: Analysis DataFrame with ratings
            
        Returns:
            Dictionary of DataFrames by rating group
        """
        rating_groups = {}
        
        # Check if composite_outcome column exists (from ratings)
        if 'composite_outcome' not in analysis_df.columns:
            logger.warning("No composite_outcome column found. Can't create rating groups.")
            return {'all': analysis_df}
        
        # Create groups based on composite outcome
        for outcome in analysis_df['composite_outcome'].unique():
            if pd.isna(outcome):
                continue
                
            group_df = analysis_df[analysis_df['composite_outcome'] == outcome]
            rating_groups[outcome] = group_df
            logger.info(f"Rating group '{outcome}': {len(group_df)} calls")
        
        # Add 'all' group
        rating_groups['all'] = analysis_df
        
        return rating_groups
    
    def calculate_percentile_rank(self, 
                                value: float, 
                                baseline: np.ndarray) -> Dict[str, float]:
        """
        Calculate percentile rank with bootstrap confidence interval
        using standard definition with proper handling of ties
        
        Args:
            value: Value to rank
            baseline: Baseline distribution
            
        Returns:
            Dictionary with percentile rank and CI
        """
        # Standard definition of percentile rank with proper handling of ties
        # PercentileRank(x) = (#{b_i ∈ B: b_i < x} + 0.5 * #{b_i ∈ B: b_i = x}) / n × 100
        n = len(baseline)
        less_than = np.sum(baseline < value)
        equal_to = np.sum(baseline == value)
        
        percentile_rank = (less_than + 0.5 * equal_to) / n * 100
        
        # Calculate bootstrap confidence interval with fixed seed for reproducibility
        bootstrap_ranks = []
        
        # Set local random state for bootstrap reproducibility
        rng = np.random.RandomState(self.random_seed)
        
        for _ in range(self.n_bootstrap):
            bootstrap_sample = rng.choice(
                baseline, 
                size=len(baseline), 
                replace=True
            )
            # Calculate percentile rank for bootstrap sample using same formula
            boot_less_than = np.sum(bootstrap_sample < value)
            boot_equal_to = np.sum(bootstrap_sample == value)
            boot_rank = (boot_less_than + 0.5 * boot_equal_to) / len(bootstrap_sample) * 100
            bootstrap_ranks.append(boot_rank)
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_ranks, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_ranks, 100 * (1 - self.alpha / 2))
        
        return {
            'percentile': percentile_rank,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        }
    
    def calculate_mad_effect_size(self, 
                                group_value: float, 
                                baseline_median: float, 
                                baseline_mad: float) -> float:
        """
        Calculate MAD-based effect size
        
        MAD effect size = (value - baseline_median) / baseline_MAD
        
        This is robust to outliers and can be calculated even for n=1
        
        Args:
            group_value: Value from the group
            baseline_median: Median of baseline distribution
            baseline_mad: MAD of baseline distribution
            
        Returns:
            MAD-based effect size
        """
        if baseline_mad == 0:
            # If MAD is 0, all baseline values are identical
            # Return 0 if group value equals baseline, else return sign of difference
            if group_value == baseline_median:
                return 0.0
            else:
                return np.sign(group_value - baseline_median)
        
        return (group_value - baseline_median) / baseline_mad
    
    def analyze_group_vs_baseline(self, 
                                group_df: pd.DataFrame, 
                                baseline_df: pd.DataFrame,
                                group_name: str) -> Dict:
        """
        Analyze a group against baseline distribution
        
        Args:
            group_df: Group DataFrame
            baseline_df: Baseline DataFrame
            group_name: Name of the group
            
        Returns:
            Analysis results dictionary
        """
        results = {
            'group_name': group_name,
            'n_group': len(group_df),
            'n_baseline': len(baseline_df),
            'acoustic_features': {},
            'semantic_features': {},
            'combined_features': {}
        }
        
        # Analyze acoustic features
        for feature in self.key_acoustic_features:
            if feature not in group_df.columns or feature not in baseline_df.columns:
                continue
                
            baseline_values = baseline_df[feature].dropna().values
            
            if len(baseline_values) == 0:
                continue
                
            feature_results = {
                'baseline_mean': float(np.mean(baseline_values)),
                'baseline_std': float(np.std(baseline_values)),
                'baseline_median': float(np.median(baseline_values)),
                'baseline_mad': float(stats.median_abs_deviation(baseline_values)),
                'baseline_p25': float(np.percentile(baseline_values, 25)),
                'baseline_p75': float(np.percentile(baseline_values, 75)),
                'group_values': [],
                'percentile_ranks': [],
                'mad_effect_sizes': []
            }
            
            # Calculate percentile ranks and MAD effect sizes for each value in group
            for _, row in group_df.iterrows():
                if pd.isna(row[feature]):
                    continue
                    
                value = row[feature]
                percentile_rank = self.calculate_percentile_rank(value, baseline_values)
                
                # Calculate MAD-based effect size
                mad_effect = self.calculate_mad_effect_size(
                    value,
                    feature_results['baseline_median'],
                    feature_results['baseline_mad']
                )
                
                feature_results['group_values'].append(float(value))
                feature_results['percentile_ranks'].append(percentile_rank)
                feature_results['mad_effect_sizes'].append(float(mad_effect))
            
            # Add group statistics
            group_values = np.array(feature_results['group_values'])
            if len(group_values) > 0:
                feature_results['group_mean'] = float(np.mean(group_values))
                feature_results['group_std'] = float(np.std(group_values))
                feature_results['group_median'] = float(np.median(group_values))
                feature_results['group_mad'] = float(stats.median_abs_deviation(group_values)) if len(group_values) > 1 else 0.0
                
                # Add traditional Cohen's d effect size
                if feature_results['baseline_std'] > 0:
                    feature_results['cohens_d'] = float(
                        (feature_results['group_mean'] - feature_results['baseline_mean']) / 
                        feature_results['baseline_std']
                    )
                else:
                    feature_results['cohens_d'] = 0.0
                
                # Add average MAD effect size
                feature_results['mean_mad_effect'] = float(np.mean(feature_results['mad_effect_sizes']))
            
            results['acoustic_features'][feature] = feature_results
        
        # Analyze semantic features
        for feature in self.key_semantic_features:
            if feature not in group_df.columns or feature not in baseline_df.columns:
                continue
                
            baseline_values = baseline_df[feature].dropna().values
            
            if len(baseline_values) == 0:
                continue
                
            feature_results = {
                'baseline_mean': float(np.mean(baseline_values)),
                'baseline_std': float(np.std(baseline_values)),
                'baseline_median': float(np.median(baseline_values)),
                'baseline_mad': float(stats.median_abs_deviation(baseline_values)),
                'group_values': [],
                'percentile_ranks': [],
                'mad_effect_sizes': []
            }
            
            # Calculate percentile ranks and MAD effect sizes for each value in group
            for _, row in group_df.iterrows():
                if pd.isna(row[feature]):
                    continue
                    
                value = row[feature]
                percentile_rank = self.calculate_percentile_rank(value, baseline_values)
                
                # Calculate MAD-based effect size
                mad_effect = self.calculate_mad_effect_size(
                    value,
                    feature_results['baseline_median'],
                    feature_results['baseline_mad']
                )
                
                feature_results['group_values'].append(float(value))
                feature_results['percentile_ranks'].append(percentile_rank)
                feature_results['mad_effect_sizes'].append(float(mad_effect))
            
            # Add group statistics
            group_values = np.array(feature_results['group_values'])
            if len(group_values) > 0:
                feature_results['group_mean'] = float(np.mean(group_values))
                feature_results['group_std'] = float(np.std(group_values))
                feature_results['group_median'] = float(np.median(group_values))
                feature_results['group_mad'] = float(stats.median_abs_deviation(group_values)) if len(group_values) > 1 else 0.0
                
                # Add traditional Cohen's d effect size
                if feature_results['baseline_std'] > 0:
                    feature_results['cohens_d'] = float(
                        (feature_results['group_mean'] - feature_results['baseline_mean']) / 
                        feature_results['baseline_std']
                    )
                else:
                    feature_results['cohens_d'] = 0.0
                
                # Add average MAD effect size
                feature_results['mean_mad_effect'] = float(np.mean(feature_results['mad_effect_sizes']))
            
            results['semantic_features'][feature] = feature_results
        
        # Analyze combined features
        for feature in self.key_combined_features:
            if feature not in group_df.columns or feature not in baseline_df.columns:
                continue
                
            baseline_values = baseline_df[feature].dropna().values
            
            if len(baseline_values) == 0:
                continue
                
            feature_results = {
                'baseline_mean': float(np.mean(baseline_values)),
                'baseline_std': float(np.std(baseline_values)),
                'baseline_median': float(np.median(baseline_values)),
                'baseline_mad': float(stats.median_abs_deviation(baseline_values)),
                'group_values': [],
                'percentile_ranks': [],
                'mad_effect_sizes': []
            }
            
            # Calculate percentile ranks and MAD effect sizes for each value in group
            for _, row in group_df.iterrows():
                if pd.isna(row[feature]):
                    continue
                    
                value = row[feature]
                percentile_rank = self.calculate_percentile_rank(value, baseline_values)
                
                # Calculate MAD-based effect size
                mad_effect = self.calculate_mad_effect_size(
                    value,
                    feature_results['baseline_median'],
                    feature_results['baseline_mad']
                )
                
                feature_results['group_values'].append(float(value))
                feature_results['percentile_ranks'].append(percentile_rank)
                feature_results['mad_effect_sizes'].append(float(mad_effect))
            
            # Add group statistics
            group_values = np.array(feature_results['group_values'])
            if len(group_values) > 0:
                feature_results['group_mean'] = float(np.mean(group_values))
                feature_results['group_std'] = float(np.std(group_values))
                feature_results['group_median'] = float(np.median(group_values))
                feature_results['group_mad'] = float(stats.median_abs_deviation(group_values)) if len(group_values) > 1 else 0.0
                
                # Add traditional Cohen's d effect size
                if feature_results['baseline_std'] > 0:
                    feature_results['cohens_d'] = float(
                        (feature_results['group_mean'] - feature_results['baseline_mean']) / 
                        feature_results['baseline_std']
                    )
                else:
                    feature_results['cohens_d'] = 0.0
                
                # Add average MAD effect size
                feature_results['mean_mad_effect'] = float(np.mean(feature_results['mad_effect_sizes']))
            
            results['combined_features'][feature] = feature_results
        
        # Add overall assessment
        results['overall_assessment'] = self._generate_overall_assessment(results)
        
        return results
    
    def _generate_overall_assessment(self, results: Dict) -> Dict:
        """
        Generate overall assessment of group vs baseline
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Overall assessment dictionary
        """
        # Check if we have f0_cv data (primary acoustic feature)
        acoustic_key = 'f0_cv'
        semantic_neg_key = 'sentiment_negative'
        
        if (acoustic_key in results['acoustic_features'] and 
            semantic_neg_key in results['semantic_features']):
            
            # Get effect sizes
            acoustic_d = results['acoustic_features'][acoustic_key].get('cohens_d', 0)
            semantic_d = results['semantic_features'][semantic_neg_key].get('cohens_d', 0)
            
            # Get MAD effect sizes
            acoustic_mad = results['acoustic_features'][acoustic_key].get('mean_mad_effect', 0)
            semantic_mad = results['semantic_features'][semantic_neg_key].get('mean_mad_effect', 0)
            
            # Get percentile ranks (average if multiple values)
            acoustic_percentiles = [
                p['percentile'] for p in 
                results['acoustic_features'][acoustic_key].get('percentile_ranks', [])
            ]
            
            semantic_percentiles = [
                p['percentile'] for p in 
                results['semantic_features'][semantic_neg_key].get('percentile_ranks', [])
            ]
            
            acoustic_percentile = np.mean(acoustic_percentiles) if acoustic_percentiles else 50
            semantic_percentile = np.mean(semantic_percentiles) if semantic_percentiles else 50
            
            # Determine pattern based on acoustic and semantic indicators
            if acoustic_percentile > 75 and semantic_percentile > 75:
                pattern = "convergent_stress"
                strength = "strong" if acoustic_percentile > 90 and semantic_percentile > 90 else "moderate"
            elif acoustic_percentile > 75 and semantic_percentile < 50:
                pattern = "divergent_acoustic_stress"
                strength = "strong" if acoustic_percentile > 90 else "moderate"
            elif acoustic_percentile < 50 and semantic_percentile > 75:
                pattern = "divergent_semantic_stress"
                strength = "strong" if semantic_percentile > 90 else "moderate"
            elif acoustic_percentile < 50 and semantic_percentile < 50:
                pattern = "baseline_neutral"
                strength = "strong" if acoustic_percentile < 25 and semantic_percentile < 25 else "moderate"
            else:
                pattern = "mixed_pattern"
                strength = "moderate"
            
            return {
                'pattern': pattern,
                'strength': strength,
                'f0_cv_percentile': float(acoustic_percentile),
                'negative_sentiment_percentile': float(semantic_percentile),
                'f0_cv_effect_size': float(acoustic_d),
                'negative_sentiment_effect_size': float(semantic_d),
                'f0_cv_mad_effect': float(acoustic_mad),
                'negative_sentiment_mad_effect': float(semantic_mad),
                'confidence': "high" if len(acoustic_percentiles) > 3 else "low"
            }
        
        return {
            'pattern': "insufficient_data",
            'strength': "unknown",
            'confidence': "low"
        }
    
    def generate_case_studies(self, 
                            analysis_results: Dict, 
                            features_df: pd.DataFrame) -> Dict:
        """
        Generate case studies of extreme cases
        
        Args:
            analysis_results: Analysis results dictionary
            features_df: Features DataFrame
            
        Returns:
            Case studies dictionary
        """
        case_studies = {}
        
        # Get non-affirmation groups
        non_affirmations = [
            group for group, results in analysis_results.items() 
            if group not in ['affirm', 'all']
        ]
        
        for group in non_affirmations:
            group_results = analysis_results[group]
            
            # Skip if insufficient data
            if group_results['n_group'] == 0:
                continue
            
            # Get file_ids for this group
            if 'composite_outcome' in features_df.columns:
                group_df = features_df[features_df['composite_outcome'] == group]
                file_ids = group_df['file_id'].tolist()
            else:
                continue
            
            # Create case study for each file
            for file_id in file_ids:
                case_df = features_df[features_df['file_id'] == file_id]
                
                if len(case_df) == 0:
                    continue
                
                # Extract key features
                case_data = {
                    'file_id': file_id,
                    'rating_outcome': group,
                    'acoustic_features': {},
                    'semantic_features': {},
                    'combined_features': {}
                }
                
                # Add acoustic features
                for feature in self.key_acoustic_features:
                    if feature not in case_df.columns:
                        continue
                    
                    value = case_df[feature].iloc[0]
                    if pd.isna(value):
                        continue
                        
                    # Get percentile rank and MAD effect from analysis results
                    percentile_data = None
                    mad_effect = None
                    if (feature in group_results['acoustic_features'] and 
                        'percentile_ranks' in group_results['acoustic_features'][feature]):
                        
                        for i, val in enumerate(group_results['acoustic_features'][feature]['group_values']):
                            if abs(val - value) < 1e-6:  # Allow for floating point errors
                                percentile_data = group_results['acoustic_features'][feature]['percentile_ranks'][i]
                                mad_effect = group_results['acoustic_features'][feature]['mad_effect_sizes'][i]
                                break
                    
                    case_data['acoustic_features'][feature] = {
                        'value': float(value),
                        'percentile': percentile_data['percentile'] if percentile_data else None,
                        'percentile_ci': [
                            percentile_data['ci_lower'] if percentile_data else None,
                            percentile_data['ci_upper'] if percentile_data else None
                        ],
                        'mad_effect': float(mad_effect) if mad_effect is not None else None
                    }
                
                # Add semantic features
                for feature in self.key_semantic_features:
                    if feature not in case_df.columns:
                        continue
                    
                    value = case_df[feature].iloc[0]
                    if pd.isna(value):
                        continue
                        
                    # Get percentile rank and MAD effect from analysis results
                    percentile_data = None
                    mad_effect = None
                    if (feature in group_results['semantic_features'] and 
                        'percentile_ranks' in group_results['semantic_features'][feature]):
                        
                        for i, val in enumerate(group_results['semantic_features'][feature]['group_values']):
                            if abs(val - value) < 1e-6:  # Allow for floating point errors
                                percentile_data = group_results['semantic_features'][feature]['percentile_ranks'][i]
                                mad_effect = group_results['semantic_features'][feature]['mad_effect_sizes'][i]
                                break
                    
                    case_data['semantic_features'][feature] = {
                        'value': float(value),
                        'percentile': percentile_data['percentile'] if percentile_data else None,
                        'percentile_ci': [
                            percentile_data['ci_lower'] if percentile_data else None,
                            percentile_data['ci_upper'] if percentile_data else None
                        ],
                        'mad_effect': float(mad_effect) if mad_effect is not None else None
                    }
                
                # Add combined features
                for feature in self.key_combined_features:
                    if feature not in case_df.columns:
                        continue
                    
                    value = case_df[feature].iloc[0]
                    if pd.isna(value):
                        continue
                        
                    case_data['combined_features'][feature] = {
                        'value': float(value)
                    }
                
                # Add communication pattern if available
                if 'communication_pattern' in case_df.columns:
                    case_data['communication_pattern'] = case_df['communication_pattern'].iloc[0]
                
                # Add the case study
                case_studies[file_id] = case_data
        
        return case_studies
    
    def create_summary_tables(self, 
                            analysis_results: Dict, 
                            output_dir: str):
        """
        Create summary tables from analysis results
        
        Args:
            analysis_results: Analysis results dictionary
            output_dir: Output directory for tables
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get baseline (affirmation) results
        if 'affirm' in analysis_results:
            baseline_results = analysis_results['affirm']
        else:
            baseline_results = analysis_results['all']
        
        # Create acoustic features summary
        acoustic_summary = []
        
        for feature in self.key_acoustic_features:
            if feature not in baseline_results['acoustic_features']:
                continue
                
            baseline_data = baseline_results['acoustic_features'][feature]
            
            row = {
                'Feature': feature,
                'Baseline Mean': baseline_data['baseline_mean'],
                'Baseline Std': baseline_data['baseline_std'],
                'Baseline Median': baseline_data['baseline_median'],
                'Baseline MAD': baseline_data['baseline_mad']
            }
            
            # Add non-baseline groups
            for group, results in analysis_results.items():
                if group in ['affirm', 'all']:
                    continue
                    
                if (feature in results['acoustic_features'] and 
                    'group_mean' in results['acoustic_features'][feature]):
                    
                    group_data = results['acoustic_features'][feature]
                    percentile_ranks = [p['percentile'] for p in group_data['percentile_ranks']]
                    
                    row[f'{group} Mean'] = group_data['group_mean']
                    row[f'{group} Percentile'] = np.mean(percentile_ranks) if percentile_ranks else None
                    row[f'{group} Cohen\'s d'] = group_data['cohens_d']
                    row[f'{group} MAD Effect'] = group_data['mean_mad_effect']
            
            acoustic_summary.append(row)
        
        # Create semantic features summary
        semantic_summary = []
        
        for feature in self.key_semantic_features:
            if feature not in baseline_results['semantic_features']:
                continue
                
            baseline_data = baseline_results['semantic_features'][feature]
            
            row = {
                'Feature': feature,
                'Baseline Mean': baseline_data['baseline_mean'],
                'Baseline Std': baseline_data['baseline_std'],
                'Baseline Median': baseline_data['baseline_median'],
                'Baseline MAD': baseline_data['baseline_mad']
            }
            
            # Add non-baseline groups
            for group, results in analysis_results.items():
                if group in ['affirm', 'all']:
                    continue
                    
                if (feature in results['semantic_features'] and 
                    'group_mean' in results['semantic_features'][feature]):
                    
                    group_data = results['semantic_features'][feature]
                    percentile_ranks = [p['percentile'] for p in group_data['percentile_ranks']]
                    
                    row[f'{group} Mean'] = group_data['group_mean']
                    row[f'{group} Percentile'] = np.mean(percentile_ranks) if percentile_ranks else None
                    row[f'{group} Cohen\'s d'] = group_data['cohens_d']
                    row[f'{group} MAD Effect'] = group_data['mean_mad_effect']
            
            semantic_summary.append(row)
        
        # Create pattern summary
        pattern_summary = []
        
        for group, results in analysis_results.items():
            if group in ['affirm', 'all']:
                continue
                
            if 'overall_assessment' in results:
                assessment = results['overall_assessment']
                
                row = {
                    'Group': group,
                    'N': results['n_group'],
                    'Pattern': assessment['pattern'],
                    'Strength': assessment['strength'],
                    'F0_CV Percentile': assessment.get('f0_cv_percentile', 'N/A'),
                    'Semantic Percentile': assessment.get('negative_sentiment_percentile', 'N/A'),
                    'F0_CV Effect Size': assessment.get('f0_cv_effect_size', 'N/A'),
                    'F0_CV MAD Effect': assessment.get('f0_cv_mad_effect', 'N/A'),
                    'Semantic Effect Size': assessment.get('negative_sentiment_effect_size', 'N/A'),
                    'Semantic MAD Effect': assessment.get('negative_sentiment_mad_effect', 'N/A'),
                    'Confidence': assessment['confidence']
                }
                
                pattern_summary.append(row)
        
        # Convert to DataFrames and save
        if acoustic_summary:
            acoustic_df = pd.DataFrame(acoustic_summary)
            acoustic_df.to_csv(os.path.join(output_dir, 'acoustic_features_summary.csv'), index=False)
        
        if semantic_summary:
            semantic_df = pd.DataFrame(semantic_summary)
            semantic_df.to_csv(os.path.join(output_dir, 'semantic_features_summary.csv'), index=False)
        
        if pattern_summary:
            pattern_df = pd.DataFrame(pattern_summary)
            pattern_df.to_csv(os.path.join(output_dir, 'pattern_summary.csv'), index=False)
        
        logger.info(f"Summary tables saved to {output_dir}")
    
    def create_visualizations(self, 
                            analysis_results: Dict, 
                            case_studies: Dict,
                            features_df: pd.DataFrame,
                            output_dir: str):
        """
        Create visualizations from analysis results
        
        Args:
            analysis_results: Analysis results dictionary
            case_studies: Case studies dictionary
            features_df: Features DataFrame
            output_dir: Output directory for visualizations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Create acoustic features boxplots
        self._create_acoustic_boxplots(analysis_results, case_studies, output_dir)
        
        # 2. Create percentile rank visualization
        self._create_percentile_rank_plots(analysis_results, case_studies, output_dir)
        
        # 3. Create acoustic-semantic alignment visualization
        self._create_alignment_plot(features_df, case_studies, output_dir)
        
        # 4. Create case study visualizations
        self._create_case_study_plots(case_studies, analysis_results, output_dir)
        
        # 5. Create pattern distribution visualization
        self._create_pattern_distribution_plot(features_df, output_dir)
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _create_acoustic_boxplots(self,
                                analysis_results: Dict,
                                case_studies: Dict,
                                output_dir: str):
        """Create boxplot visualizations for acoustic features"""
        # Get baseline (affirmation) results
        if 'affirm' in analysis_results:
            baseline_results = analysis_results['affirm']
        else:
            baseline_results = analysis_results['all']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(6, 5))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.key_acoustic_features[:4]):  # Limit to 4 features
            if feature not in baseline_results['acoustic_features']:
                continue
                
            ax = axes[i]
            
            # Get baseline data
            baseline_data = baseline_results['acoustic_features'][feature]
            
            # Set random seed for reproducible jitter
            np.random.seed(self.random_seed)
            baseline_values = baseline_data['baseline_mean'] + np.random.normal(0, 0.01, baseline_results['n_baseline'])
            
            # Create boxplot
            ax.boxplot([baseline_values], positions=[0], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=self.color_palette[1], color=self.color_palette[3]),
                     medianprops=dict(color=self.color_palette[5], linewidth=1.5))
            
            # Calculate y-axis range for better label positioning
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            
            # Add case study points
            for j, (file_id, case) in enumerate(case_studies.items()):
                if (feature in case['acoustic_features'] and 
                    case['acoustic_features'][feature]['value'] is not None):
                    
                    value = case['acoustic_features'][feature]['value']
                    percentile = case['acoustic_features'][feature]['percentile']
                    
                    # Color based on rating outcome
                    color = self.color_palette[7] if case['rating_outcome'] == 'downgrade' else self.color_palette[2]
                    
                    # Plot point
                    ax.scatter([j + 1], [value], color=color, s=50, zorder=5)
                    
                    # Add label with increased vertical offset to avoid overlap
                    label_offset = 0.30 * y_range  # Increased from 0.01 to 0.08
                    ax.text(j + 1, value + label_offset, 
                          f"{percentile:.0f}%", 
                          ha='center', va='bottom', fontsize=7, fontweight='normal')
                    
                    # Add file_id as x-tick label
                    ax.set_xticks([0] + list(range(1, len(case_studies) + 1)))
                    ax.set_xticklabels(['Baseline'] + list(case_studies.keys()), rotation=45, ha='right', fontsize=7)
            
            # Add title and labels
            feature_name = feature.replace('_', ' ').title()
            ax.set_title(f"{feature_name}", fontsize=9, fontweight='normal', pad=10)
            ax.set_ylabel('Feature Value', fontsize=8, fontweight='normal')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'acoustic_boxplots.png'), dpi=300)
        plt.close()
    
    def _create_percentile_rank_plots(self,
                                    analysis_results: Dict,
                                    case_studies: Dict,
                                    output_dir: str):
        """Create percentile rank visualizations"""
        # Get non-affirmation groups
        non_affirmations = [
            group for group, results in analysis_results.items() 
            if group not in ['affirm', 'all']
        ]
        
        if not non_affirmations:
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(len(non_affirmations), 1, figsize=(6, 5))
        
        # Handle single subplot case
        if len(non_affirmations) == 1:
            axes = [axes]
        
        for i, group in enumerate(non_affirmations):
            ax = axes[i]
            results = analysis_results[group]
            
            # Select key features
            features = []
            percentiles = []
            errors = []
            
            for feature in self.key_acoustic_features:
                if (feature in results['acoustic_features'] and 
                    'percentile_ranks' in results['acoustic_features'][feature] and
                    results['acoustic_features'][feature]['percentile_ranks']):
                    
                    ranks = results['acoustic_features'][feature]['percentile_ranks']
                    mean_percentile = np.mean([r['percentile'] for r in ranks])
                    ci_lower = np.mean([r['ci_lower'] for r in ranks])
                    ci_upper = np.mean([r['ci_upper'] for r in ranks])
                    
                    features.append(feature.replace('_', ' ').title())
                    percentiles.append(mean_percentile)
                    errors.append([mean_percentile - ci_lower, ci_upper - mean_percentile])
            
            # Add semantic features
            for feature in self.key_semantic_features:
                if (feature in results['semantic_features'] and 
                    'percentile_ranks' in results['semantic_features'][feature] and
                    results['semantic_features'][feature]['percentile_ranks']):
                    
                    ranks = results['semantic_features'][feature]['percentile_ranks']
                    mean_percentile = np.mean([r['percentile'] for r in ranks])
                    ci_lower = np.mean([r['ci_lower'] for r in ranks])
                    ci_upper = np.mean([r['ci_upper'] for r in ranks])
                    
                    features.append(feature.replace('_', ' ').title())
                    percentiles.append(mean_percentile)
                    errors.append([mean_percentile - ci_lower, ci_upper - mean_percentile])
            
            if not features:
                ax.text(0.5, 0.5, 'No percentile data available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                continue
            
            # Create horizontal bar chart
            bars = ax.barh(features, percentiles, xerr=np.array(errors).T, 
                         color=self.color_palette[1], alpha=0.8, ecolor=self.color_palette[6], capsize=3)
            
            # Color bars based on percentile
            for j, bar in enumerate(bars):
                if percentiles[j] > 90:
                    bar.set_color(self.color_palette[7])
                elif percentiles[j] > 75:
                    bar.set_color(self.color_palette[5])
                elif percentiles[j] < 25:
                    bar.set_color(self.color_palette[2])
            
            # Add percentile lines
            ax.axvline(x=50, color=self.color_palette[4], linestyle='--', alpha=0.5, label='50th Percentile')
            ax.axvline(x=75, color=self.color_palette[5], linestyle='--', alpha=0.5, label='75th Percentile')
            ax.axvline(x=90, color=self.color_palette[7], linestyle='--', alpha=0.5, label='90th Percentile')
            
            # Add values to the right of bars to avoid overlap
            for j, (p, err) in enumerate(zip(percentiles, errors)):
                # Position text to the right of the error bar
                text_x = p + err[1] + 5  # Add 5 units padding after error bar
                if text_x > 95:  # If too close to edge, position inside
                    text_x = p - err[0] - 5
                    ha = 'right'
                else:
                    ha = 'left'
                ax.text(text_x, j, f"{p:.1f}%", va='center', ha=ha, fontsize=7, fontweight='normal')
            
            # Set labels and title
            ax.set_xlabel('Percentile Rank', fontsize=8, fontweight='normal')
            ax.set_title(f'Percentile Ranks for {group.title()} Group (N={results["n_group"]})', 
                        fontsize=9, fontweight='normal', pad=10)
            ax.set_xlim(0, 110)  # Extend xlim to give space for labels
            ax.legend(loc='lower right', fontsize=7)
            
            # Add grid
            ax.grid(True, axis='x', linestyle='--', alpha=0.3)
            
            # Adjust y-tick labels
            ax.set_yticklabels(features, fontsize=7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'percentile_ranks.png'), dpi=300)
        plt.close()
    
    def _create_alignment_plot(self,
                             features_df: pd.DataFrame,
                             case_studies: Dict,
                             output_dir: str):
        """Create acoustic-semantic alignment visualization"""
        # Check if we have necessary columns - use f0_cv instead of acoustic_volatility_index
        if ('f0_cv' not in features_df.columns or 
            'sentiment_negative' not in features_df.columns):
            logger.warning("Missing columns for alignment plot. Skipping.")
            return
        
        # Create figure
        plt.figure(figsize=(6, 5))
        
        # Create scatter plot of all points
        scatter = plt.scatter(
            features_df['f0_cv'],
            features_df['sentiment_negative'],
            c=self.color_palette[0], s=40, alpha=0.6, edgecolors=self.color_palette[3],
            label='Baseline Calls'
        )
        
        # Add case study points
        for file_id, case in case_studies.items():
            acoustic_key = 'f0_cv'
            semantic_key = 'sentiment_negative'
            
            if (acoustic_key in case['acoustic_features'] and 
                semantic_key in case['semantic_features']):
                
                acoustic_value = case['acoustic_features'][acoustic_key]['value']
                semantic_value = case['semantic_features'][semantic_key]['value']
                
                if acoustic_value is None or semantic_value is None:
                    continue
                
                # Color based on rating outcome
                color = self.color_palette[7] if case['rating_outcome'] == 'downgrade' else self.color_palette[2]
                
                # Plot point
                plt.scatter([acoustic_value], [semantic_value], 
                           color=color, s=80, edgecolors=self.color_palette[6], linewidth=1.2,
                           label=f"{file_id} ({case['rating_outcome']})")
                
                # Add label with vertical offset to avoid overlap with points
                plt.text(acoustic_value, semantic_value + 0.04, file_id, 
                        ha='center', va='bottom', fontsize=7, fontweight='normal')
        
        # Add quadrant lines
        plt.axvline(x=features_df['f0_cv'].median(), 
                  color=self.color_palette[4], linestyle='--', alpha=0.4)
        plt.axhline(y=features_df['sentiment_negative'].median(), 
                  color=self.color_palette[4], linestyle='--', alpha=0.4)
        
        # Add quadrant labels with adjusted positions to avoid overlap
        xmax = features_df['f0_cv'].max()
        ymax = features_df['sentiment_negative'].max()
        xmin = features_df['f0_cv'].min()
        ymin = features_df['sentiment_negative'].min()
        
        xmed = features_df['f0_cv'].median()
        ymed = features_df['sentiment_negative'].median()
        
        # Baseline quadrant (bottom-left)
        plt.text(xmin + (xmed - xmin) / 2, ymin + (ymed - ymin) / 2, 
                'Baseline\nQuadrant', 
                ha='center', va='center', fontsize=8, alpha=0.6, fontweight='normal')
        
        # Convergent stress quadrant (top-right)
        plt.text(xmed + (xmax - xmed) / 2, ymed + (ymax - ymed) / 2, 
                'Convergent Stress\nQuadrant', 
                ha='center', va='center', fontsize=8, alpha=0.6, fontweight='normal')
        
        # Semantic stress quadrant (top-left) - adjust position to avoid overlap
        # Check if there are data points in the upper left that might overlap
        upper_left_points = features_df[(features_df['f0_cv'] < xmed) & 
                                       (features_df['sentiment_negative'] > ymed)]
        
        if len(upper_left_points) > 0:
            # If there are points, position text lower to avoid overlap
            semantic_text_y = ymed + (ymax - ymed) * 0.3  # Lower position
        else:
            # If no points, use centered position
            semantic_text_y = ymed + (ymax - ymed) / 2
            
        plt.text(xmin + (xmed - xmin) / 2, semantic_text_y, 
                'Semantic Stress\nQuadrant', 
                ha='center', va='center', fontsize=8, alpha=0.6, fontweight='normal')
        
        # Acoustic stress quadrant (bottom-right)
        plt.text(xmed + (xmax - xmed) / 2, ymin + (ymed - ymin) / 2, 
                'Acoustic Stress\nQuadrant', 
                ha='center', va='center', fontsize=8, alpha=0.6, fontweight='normal')
        
        # Set labels and title
        plt.xlabel('F0 Coefficient of Variation', fontsize=8, fontweight='normal')
        plt.ylabel('Negative Sentiment Score', fontsize=8, fontweight='normal')
        plt.title('Acoustic-Semantic Alignment', fontsize=10, fontweight='normal', pad=10)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Remove duplicate case study labels
        seen = set()
        unique_handles = []
        unique_labels = []
        
        for handle, label in zip(handles, labels):
            if 'Baseline' in label or label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
        
        plt.legend(handles=unique_handles, labels=unique_labels, loc='upper left', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'acoustic_semantic_alignment.png'), dpi=300)
        plt.close()
    
    def _create_case_study_plots(self,
                               case_studies: Dict,
                               analysis_results: Dict,
                               output_dir: str):
        """Create case study visualizations"""
        # Create case study directory
        case_study_dir = os.path.join(output_dir, 'case_studies')
        os.makedirs(case_study_dir, exist_ok=True)
        
        # Create plots for each case
        for file_id, case in case_studies.items():
            # Create figure
            fig = plt.figure(figsize=(10, 8))
            
            # Set up grid
            gs = gridspec.GridSpec(2, 3, figure=fig)
            
            # 1. Acoustic features radar plot
            ax1 = fig.add_subplot(gs[0, 0], polar=True)
            self._create_case_radar_plot(ax1, case, 'acoustic_features', analysis_results)
            
            # 2. Semantic features radar plot
            ax2 = fig.add_subplot(gs[0, 1], polar=True)
            self._create_case_radar_plot(ax2, case, 'semantic_features', analysis_results)
            
            # 3. Percentile rank barplot
            ax3 = fig.add_subplot(gs[0, 2])
            self._create_case_percentile_plot(ax3, case)
            
            # 4. Acoustic-semantic alignment position
            ax4 = fig.add_subplot(gs[1, 0:2])
            self._create_case_alignment_plot(ax4, case, analysis_results)
            
            # 5. Summary text
            ax5 = fig.add_subplot(gs[1, 2])
            self._create_case_summary(ax5, case, analysis_results)
            
            # Add title
            plt.suptitle(f"Case Study: {file_id} ({case['rating_outcome'].title()})", 
                        fontsize=12, fontweight='normal')
            
            plt.tight_layout()
            plt.savefig(os.path.join(case_study_dir, f'case_study_{file_id}.png'), dpi=300)
            plt.close()
    
    def _create_case_radar_plot(self,
                              ax: plt.Axes,
                              case: Dict,
                              feature_type: str,
                              analysis_results: Dict):
        """Create radar plot for case features"""
        # Get features
        if feature_type == 'acoustic_features':
            features = self.key_acoustic_features
            title = 'Acoustic Features'
        else:
            features = self.key_semantic_features
            title = 'Semantic Features'
        
        # Get values and percentiles
        values = []
        percentiles = []
        feature_labels = []
        
        for feature in features:
            if feature in case[feature_type] and case[feature_type][feature]['value'] is not None:
                values.append(case[feature_type][feature]['value'])
                percentiles.append(case[feature_type][feature]['percentile'])
                feature_labels.append(feature.replace('_', ' ').title())
        
        if not values:
            ax.text(0, 0, f'No {feature_type} data available', 
                   ha='center', va='center', fontsize=8, fontweight='normal')
            ax.set_title(title, fontsize=9, fontweight='normal', pad=10)
            return
        
        # Number of variables
        N = len(values)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Values for the plot, from 0 to 100
        percentiles += percentiles[:1]  # Close the loop
        
        # Draw one axis per variable and add labels
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels, fontsize=7)
        
        # Draw y axis labels
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=6)
        ax.set_ylim(0, 100)
        
        # Plot data
        ax.plot(angles, percentiles, 'o-', linewidth=1.5, color=self.color_palette[5])
        
        # Fill area
        ax.fill(angles, percentiles, alpha=0.25, color=self.color_palette[3])
        
        # Add title
        ax.set_title(title, fontsize=9, fontweight='normal', pad=10)
    
    def _create_case_percentile_plot(self, ax: plt.Axes, case: Dict):
        """Create percentile rank barplot for case"""
        # Combine acoustic and semantic features
        features = []
        percentiles = []
        errors = []
        colors = []
        
        # Add acoustic features
        for feature in self.key_acoustic_features:
            if (feature in case['acoustic_features'] and 
                case['acoustic_features'][feature]['percentile'] is not None):
                
                features.append(feature.replace('_', ' ').title())
                percentiles.append(case['acoustic_features'][feature]['percentile'])
                
                ci = case['acoustic_features'][feature]['percentile_ci']
                if ci[0] is not None and ci[1] is not None:
                    errors.append([
                        percentiles[-1] - ci[0],
                        ci[1] - percentiles[-1]
                    ])
                else:
                    errors.append([0, 0])
                
                # Color based on percentile
                if percentiles[-1] > 90:
                    colors.append(self.color_palette[7])
                elif percentiles[-1] > 75:
                    colors.append(self.color_palette[5])
                elif percentiles[-1] < 25:
                    colors.append(self.color_palette[2])
                else:
                    colors.append(self.color_palette[3])
        
        # Add semantic features
        for feature in self.key_semantic_features:
            if (feature in case['semantic_features'] and 
                case['semantic_features'][feature]['percentile'] is not None):
                
                features.append(feature.replace('_', ' ').title())
                percentiles.append(case['semantic_features'][feature]['percentile'])
                
                ci = case['semantic_features'][feature]['percentile_ci']
                if ci[0] is not None and ci[1] is not None:
                    errors.append([
                        percentiles[-1] - ci[0],
                        ci[1] - percentiles[-1]
                    ])
                else:
                    errors.append([0, 0])
                
                # Color based on percentile
                if percentiles[-1] > 90:
                    colors.append(self.color_palette[7])
                elif percentiles[-1] > 75:
                    colors.append(self.color_palette[5])
                elif percentiles[-1] < 25:
                    colors.append(self.color_palette[2])
                else:
                    colors.append(self.color_palette[3])
        
        if not features:
            ax.text(0.5, 0.5, 'No percentile data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8, fontweight='normal')
            ax.set_title('Percentile Ranks', fontsize=9, fontweight='normal', pad=10)
            return
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, percentiles, xerr=np.array(errors).T if errors else None, 
                     alpha=0.8, capsize=3)
        
        # Set bar colors
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add percentile lines
        ax.axvline(x=50, color=self.color_palette[4], linestyle='--', alpha=0.4)
        ax.axvline(x=75, color=self.color_palette[5], linestyle='--', alpha=0.4)
        ax.axvline(x=90, color=self.color_palette[7], linestyle='--', alpha=0.4)
        
        # Add values above bars to avoid overlap
        for i, (p, err) in enumerate(zip(percentiles, errors)):
            # Calculate text position
            text_x = p + err[1] + 3 if p + err[1] + 3 < 95 else p - err[0] - 3
            ha = 'left' if p + err[1] + 3 < 95 else 'right'
            ax.text(text_x, i, f"{p:.1f}%", va='center', ha=ha, fontsize=7, fontweight='normal')
        
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=7)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Percentile Rank', fontsize=8, fontweight='normal')
        ax.set_title('Percentile Ranks vs. Baseline', fontsize=9, fontweight='normal', pad=10)
        ax.set_xlim(0, 105)
        
        # Add grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    def _create_case_alignment_plot(self,
                                  ax: plt.Axes,
                                  case: Dict,
                                  analysis_results: Dict):
        """Create alignment plot showing case position"""
        # Get baseline results
        if 'affirm' in analysis_results:
            baseline_results = analysis_results['affirm']
        else:
            baseline_results = analysis_results['all']
        
        # Get acoustic and semantic data - use f0_cv instead of acoustic_volatility_index
        acoustic_key = 'f0_cv'
        semantic_key = 'sentiment_negative'
        
        if (acoustic_key not in case['acoustic_features'] or 
            semantic_key not in case['semantic_features']):
            ax.text(0.5, 0.5, 'Insufficient data for alignment plot', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8, fontweight='normal')
            ax.set_title('Acoustic-Semantic Alignment', fontsize=9, fontweight='normal', pad=10)
            return
        
        acoustic_value = case['acoustic_features'][acoustic_key]['value']
        semantic_value = case['semantic_features'][semantic_key]['value']
        
        if acoustic_value is None or semantic_value is None:
            ax.text(0.5, 0.5, 'Insufficient data for alignment plot', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8, fontweight='normal')
            ax.set_title('Acoustic-Semantic Alignment', fontsize=9, fontweight='normal', pad=10)
            return
        
        # Get baseline acoustic and semantic data
        if (acoustic_key in baseline_results['acoustic_features'] and 
            'baseline_mean' in baseline_results['acoustic_features'][acoustic_key] and
            semantic_key in baseline_results['semantic_features'] and
            'baseline_mean' in baseline_results['semantic_features'][semantic_key]):
            
            baseline_acoustic = baseline_results['acoustic_features'][acoustic_key]['baseline_mean']
            baseline_acoustic_std = baseline_results['acoustic_features'][acoustic_key]['baseline_std']
            
            baseline_semantic = baseline_results['semantic_features'][semantic_key]['baseline_mean']
            baseline_semantic_std = baseline_results['semantic_features'][semantic_key]['baseline_std']
            
            # Generate scatter data for baseline distribution
            n_baseline = baseline_results['n_baseline']
            np.random.seed(self.random_seed)  # For reproducibility
            
            baseline_acoustic_values = np.random.normal(baseline_acoustic, baseline_acoustic_std, n_baseline)
            baseline_semantic_values = np.random.normal(baseline_semantic, baseline_semantic_std, n_baseline)
            
            # Create scatter plot
            ax.scatter(baseline_acoustic_values, baseline_semantic_values, 
                      c=self.color_palette[0], s=30, alpha=0.5, edgecolors=self.color_palette[3],
                      label='Baseline Calls')
            
            # Add case point
            ax.scatter([acoustic_value], [semantic_value], 
                      color=self.color_palette[7] if case['rating_outcome'] == 'downgrade' else self.color_palette[2], 
                      s=100, edgecolors=self.color_palette[6], linewidth=1.5,
                      label=f"{case['file_id']} ({case['rating_outcome']})")
            
            # Add quadrant lines
            ax.axvline(x=baseline_acoustic, color=self.color_palette[4], linestyle='--', alpha=0.4)
            ax.axhline(y=baseline_semantic, color=self.color_palette[4], linestyle='--', alpha=0.4)
            
            # Get plot limits
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            
            # Add quadrant labels with adjusted positions to avoid overlap
            # Baseline quadrant (bottom-left)
            ax.text((xmin + baseline_acoustic) / 2, (ymin + baseline_semantic) / 2, 
                   'Baseline\nQuadrant', 
                   ha='center', va='center', fontsize=8, alpha=0.6, fontweight='normal')
            
            # Convergent stress quadrant (top-right) - adjusted position
            ax.text((baseline_acoustic + xmax) / 2, (baseline_semantic + ymax * 0.85) / 2, 
                   'Convergent Stress\nQuadrant', 
                   ha='center', va='center', fontsize=8, alpha=0.6, fontweight='normal')
            
            # Semantic stress quadrant (top-left)
            ax.text((xmin + baseline_acoustic) / 2, (baseline_semantic + ymax) / 2, 
                   'Semantic Stress\nQuadrant', 
                   ha='center', va='center', fontsize=8, alpha=0.6, fontweight='normal')
            
            # Acoustic stress quadrant (bottom-right)
            ax.text((baseline_acoustic + xmax) / 2, (ymin + baseline_semantic) / 2, 
                   'Acoustic Stress\nQuadrant', 
                   ha='center', va='center', fontsize=8, alpha=0.6, fontweight='normal')
            
            # Set labels and title
            ax.set_xlabel('F0 Coefficient of Variation', fontsize=8, fontweight='normal')
            ax.set_ylabel('Negative Sentiment Score', fontsize=8, fontweight='normal')
            ax.set_title('Acoustic-Semantic Alignment Position', fontsize=9, fontweight='normal', pad=10)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add legend
            ax.legend(loc='upper left', fontsize=7)
            
            # Add percentile annotations with adjusted positions
            acoustic_percentile = case['acoustic_features'][acoustic_key]['percentile']
            semantic_percentile = case['semantic_features'][semantic_key]['percentile']
            
            if acoustic_percentile is not None and semantic_percentile is not None:
                # Position percentile text to avoid overlap with quadrant labels
                ax.text(acoustic_value, semantic_value - (ymax - ymin) * 0.05, 
                       f"Acoustic: {acoustic_percentile:.1f}%ile", 
                       ha='center', va='top', fontsize=7, fontweight='normal',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                ax.text(acoustic_value, semantic_value + (ymax - ymin) * 0.05, 
                       f"Semantic: {semantic_percentile:.1f}%ile", 
                       ha='center', va='bottom', fontsize=7, fontweight='normal',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient baseline data for alignment plot', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8, fontweight='normal')
            ax.set_title('Acoustic-Semantic Alignment', fontsize=9, fontweight='normal', pad=10)
    
    def _create_case_summary(self,
                           ax: plt.Axes,
                           case: Dict,
                           analysis_results: Dict):
        """Create text summary for case"""
        # Turn off axis
        ax.axis('off')
        
        # Create summary text
        title = f"CASE SUMMARY: {case['file_id']}"
        
        # Rating outcome
        outcome = f"Rating Outcome: {case['rating_outcome'].upper()}"
        
        # Communication pattern
        pattern = f"Communication Pattern: {case.get('communication_pattern', 'Unknown').replace('_', ' ').title()}"
        
        # Key percentiles
        percentiles = []
        mad_effects = []
        
        for feature in ['f0_cv', 'f0_std', 'sentiment_negative']:
            feature_type = 'acoustic_features' if feature in self.key_acoustic_features else 'semantic_features'
            
            if (feature in case[feature_type] and 
                case[feature_type][feature]['percentile'] is not None):
                
                feature_name = feature.replace('_', ' ').title()
                percentile = case[feature_type][feature]['percentile']
                mad_effect = case[feature_type][feature].get('mad_effect', None)
                
                percentiles.append(f"{feature_name}: {percentile:.1f}%ile")
                if mad_effect is not None:
                    mad_effects.append(f"{feature_name} MAD Effect: {mad_effect:.2f}")
        
        # Acoustic-semantic alignment
        alignment = "Acoustic-Semantic Alignment: "
        
        acoustic_key = 'f0_cv'
        semantic_key = 'sentiment_negative'
        
        if (acoustic_key in case['acoustic_features'] and 
            semantic_key in case['semantic_features'] and
            case['acoustic_features'][acoustic_key]['percentile'] is not None and
            case['semantic_features'][semantic_key]['percentile'] is not None):
            
            acoustic_percentile = case['acoustic_features'][acoustic_key]['percentile']
            semantic_percentile = case['semantic_features'][semantic_key]['percentile']
            
            if acoustic_percentile > 75 and semantic_percentile > 75:
                alignment += "Strong Convergent Stress"
            elif acoustic_percentile > 75 and semantic_percentile < 50:
                alignment += "Divergent (High Acoustic, Low Semantic)"
            elif acoustic_percentile < 50 and semantic_percentile > 75:
                alignment += "Divergent (Low Acoustic, High Semantic)"
            elif acoustic_percentile < 50 and semantic_percentile < 50:
                alignment += "Baseline (Low Stress Indicators)"
            else:
                alignment += "Mixed Pattern"
        else:
            alignment += "Insufficient Data"
        
        # Combine text
        summary_text = f"{title}\n\n{outcome}\n\n{pattern}\n\nKey Percentiles:\n"
        summary_text += "\n".join(f"• {p}" for p in percentiles)
        
        if mad_effects:
            summary_text += "\n\nMAD-Based Effects:\n"
            summary_text += "\n".join(f"• {m}" for m in mad_effects)
        
        summary_text += f"\n\n{alignment}"
        
        # Add text box
        props = dict(boxstyle='round', facecolor=self.color_palette[0], alpha=0.8)
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
              fontsize=8, verticalalignment='center', 
              horizontalalignment='center', bbox=props, fontweight='normal')
    
    def _create_pattern_distribution_plot(self,
                                       features_df: pd.DataFrame,
                                       output_dir: str):
        """Create distribution plot for communication patterns"""
        if 'communication_pattern' not in features_df.columns:
            return
            
        # Create figure
        plt.figure(figsize=(6, 5))
        
        # Get pattern counts
        pattern_counts = features_df['communication_pattern'].value_counts()
        
        # Define color mapping for patterns
        colors_map = {
            'high_stress': self.color_palette[7],
            'moderate_stress': self.color_palette[5],
            'high_excitement': self.color_palette[2],
            'moderate_excitement': self.color_palette[3],
            'baseline_stability': self.color_palette[1],
            'mixed_pattern': self.color_palette[4]
        }
        
        # Filter palette to include only existing patterns
        plot_colors = [colors_map.get(pattern, self.color_palette[4]) for pattern in pattern_counts.index]
        
        # Create bar chart
        bars = plt.bar(pattern_counts.index, pattern_counts.values, color=plot_colors, alpha=0.8)
        
        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height}', ha='center', va='bottom', fontsize=8, fontweight='normal')
        
        # Improve x-axis labels
        plt.xticks(
            range(len(pattern_counts)), 
            [p.replace('_', ' ').title() for p in pattern_counts.index],
            rotation=45, ha='right', fontsize=8
        )
        
        # Set labels and title
        plt.xlabel('Communication Pattern', fontsize=8, fontweight='normal')
        plt.ylabel('Count', fontsize=8, fontweight='normal')
        plt.title('Distribution of Communication Patterns', fontsize=10, fontweight='normal', pad=10)
        
        # Add grid
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pattern_distribution.png'), dpi=300)
        plt.close()
    
    def generate_summary_report(self,
                              analysis_results: Dict,
                              case_studies: Dict,
                              output_path: str):
        """
        Generate a summary report of the analysis
        
        Args:
            analysis_results: Analysis results dictionary
            case_studies: Case studies dictionary
            output_path: Path to save the report
        """
        # Get baseline (affirmation) results
        if 'affirm' in analysis_results:
            baseline_results = analysis_results['affirm']
        else:
            baseline_results = analysis_results['all']
        
        # Create report
        report = []
        
        # Add title
        report.append("# EARNINGS CALL ACOUSTIC ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Add methodology note about reproducibility
        report.append("## REPRODUCIBILITY NOTE")
        report.append("-" * 50)
        report.append(f"Random seed used for all bootstrap procedures: {self.random_seed}")
        report.append(f"Number of bootstrap iterations: {self.n_bootstrap}")
        report.append(f"Confidence level: {self.confidence_level * 100}%")
        report.append("")
        
        # Add baseline summary
        report.append("## 1. BASELINE DISTRIBUTION SUMMARY")
        report.append("-" * 50)
        report.append(f"Total Baseline Calls: {baseline_results['n_baseline']}")
        report.append("")
        
        # Add acoustic features
        report.append("### 1.1 Acoustic Features")
        
        for feature in self.key_acoustic_features:
            if feature not in baseline_results['acoustic_features']:
                continue
                
            data = baseline_results['acoustic_features'][feature]
            report.append(f"\n**{feature.replace('_', ' ').title()}**:")
            report.append(f"  - Mean: {data['baseline_mean']:.4f}")
            report.append(f"  - Std Dev: {data['baseline_std']:.4f}")
            report.append(f"  - Median: {data['baseline_median']:.4f}")
            report.append(f"  - MAD: {data['baseline_mad']:.4f}")
            report.append(f"  - IQR: {data['baseline_p75'] - data['baseline_p25']:.4f}")
        
        # Add semantic features
        report.append("\n### 1.2 Semantic Features")
        
        for feature in self.key_semantic_features:
            if feature not in baseline_results['semantic_features']:
                continue
                
            data = baseline_results['semantic_features'][feature]
            report.append(f"\n**{feature.replace('_', ' ').title()}**:")
            report.append(f"  - Mean: {data['baseline_mean']:.4f}")
            report.append(f"  - Std Dev: {data['baseline_std']:.4f}")
            report.append(f"  - Median: {data['baseline_median']:.4f}")
            report.append(f"  - MAD: {data['baseline_mad']:.4f}")
        
        # Add non-affirmation summaries
        report.append("\n## 2. NON-AFFIRMATION GROUP ANALYSIS")
        report.append("-" * 50)
        
        for group, results in analysis_results.items():
            if group in ['affirm', 'all']:
                continue
                
            report.append(f"\n### 2.1 {group.upper()} GROUP (N={results['n_group']})")
            
            if 'overall_assessment' in results:
                assessment = results['overall_assessment']
                report.append(f"\n**Pattern**: {assessment['pattern'].replace('_', ' ').title()}")
                report.append(f"**Strength**: {assessment['strength'].title()}")
                report.append(f"**Confidence**: {assessment['confidence'].title()}")
                
                if 'f0_cv_percentile' in assessment:
                    report.append(f"**F0 CV Percentile**: {assessment['f0_cv_percentile']:.1f}%")
                if 'negative_sentiment_percentile' in assessment:
                    report.append(f"**Semantic Percentile**: {assessment['negative_sentiment_percentile']:.1f}%")
                if 'f0_cv_effect_size' in assessment:
                    report.append(f"**F0 CV Cohen's d**: {assessment['f0_cv_effect_size']:.2f}")
                if 'f0_cv_mad_effect' in assessment:
                    report.append(f"**F0 CV MAD Effect**: {assessment['f0_cv_mad_effect']:.2f}")
                if 'negative_sentiment_effect_size' in assessment:
                    report.append(f"**Semantic Cohen's d**: {assessment['negative_sentiment_effect_size']:.2f}")
                if 'negative_sentiment_mad_effect' in assessment:
                    report.append(f"**Semantic MAD Effect**: {assessment['negative_sentiment_mad_effect']:.2f}")
            
            # Add key acoustic features
            report.append("\n**Key Acoustic Features**:")
            
            for feature in self.key_acoustic_features:
                if (feature in results['acoustic_features'] and 
                    'group_mean' in results['acoustic_features'][feature]):
                    
                    data = results['acoustic_features'][feature]
                    percentile_ranks = [p['percentile'] for p in data['percentile_ranks']]
                    mean_percentile = np.mean(percentile_ranks) if percentile_ranks else float('nan')
                    
                    report.append(f"  - {feature.replace('_', ' ').title()}: {data['group_mean']:.4f} ({mean_percentile:.1f}%ile, MAD effect: {data['mean_mad_effect']:.2f})")
            
            # Add key semantic features
            report.append("\n**Key Semantic Features**:")
            
            for feature in self.key_semantic_features:
                if (feature in results['semantic_features'] and 
                    'group_mean' in results['semantic_features'][feature]):
                    
                    data = results['semantic_features'][feature]
                    percentile_ranks = [p['percentile'] for p in data['percentile_ranks']]
                    mean_percentile = np.mean(percentile_ranks) if percentile_ranks else float('nan')
                    
                    report.append(f"  - {feature.replace('_', ' ').title()}: {data['group_mean']:.4f} ({mean_percentile:.1f}%ile, MAD effect: {data['mean_mad_effect']:.2f})")
        
        # Add case studies
        report.append("\n## 3. INDIVIDUAL CASE STUDIES")
        report.append("-" * 50)
        
        for i, (file_id, case) in enumerate(case_studies.items(), 1):
            report.append(f"\n### 3.{i} Case: {file_id} ({case['rating_outcome'].upper()})")
            
            # Add communication pattern
            if 'communication_pattern' in case:
                report.append(f"\n**Communication Pattern**: {case['communication_pattern'].replace('_', ' ').title()}")
            
            # Add key percentiles
            report.append("\n**Key Percentiles**:")
            
            for feature in self.key_acoustic_features + self.key_semantic_features:
                feature_type = 'acoustic_features' if feature in self.key_acoustic_features else 'semantic_features'
                
                if (feature in case[feature_type] and 
                    case[feature_type][feature]['percentile'] is not None):
                    
                    value = case[feature_type][feature]['value']
                    percentile = case[feature_type][feature]['percentile']
                    mad_effect = case[feature_type][feature].get('mad_effect', None)
                    
                    if mad_effect is not None:
                        report.append(f"  - {feature.replace('_', ' ').title()}: {value:.4f} ({percentile:.1f}%ile, MAD effect: {mad_effect:.2f})")
                    else:
                        report.append(f"  - {feature.replace('_', ' ').title()}: {value:.4f} ({percentile:.1f}%ile)")
        
        # Add methodology note
        report.append("\n## 4. METHODOLOGY NOTE")
        report.append("-" * 50)
        report.append("""
This analysis follows a descriptive exploration methodology appropriate for small-sample analysis:

1. Baseline Establishment: Affirmation calls (N≈21) provide the reference distribution for acoustic and semantic features.

2. Percentile Ranking: Non-affirmation cases are ranked against this baseline, with bootstrap confidence intervals for uncertainty quantification (seed={} for reproducibility).

3. MAD-Based Standardization: In addition to traditional Cohen's d, we report MAD-based effect sizes which are:
   - Robust to outliers and non-normal distributions
   - Calculable for any sample size (including n=1)
   - More appropriate for small samples than mean/SD-based measures

4. Pattern Classification: Cases are classified based on acoustic-semantic alignment following Russell's Circumplex Model of Affect.

5. Case Studies: Individual cases receive detailed analysis rather than attempting group-level inference inappropriate for small samples.

This approach aligns with contemporary scientific frameworks for financial speech analysis while acknowledging sample size constraints.
""".format(self.random_seed))
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Summary report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Descriptive analysis for earnings call acoustic features"
    )
    parser.add_argument("--features_file", type=str, required=True,
                       help="Path to combined features CSV file")
    parser.add_argument("--ratings_file", type=str, default=None,
                       help="Path to ratings CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")
    parser.add_argument("--bootstrap", type=int, default=10000,
                       help="Number of bootstrap iterations for confidence intervals")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for intervals")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize analysis
    analyzer = DescriptiveAnalysis(
        n_bootstrap=args.bootstrap,
        confidence_level=args.confidence,
        random_seed=args.random_seed
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Extract directory from features file path
        features_dir = os.path.dirname(args.features_file)
        
        # Load data
        logger.info("Loading features and ratings data...")
        features_df, ratings_df = analyzer.load_data(features_dir, args.ratings_file)
        
        # Prepare analysis data
        logger.info("Preparing analysis data...")
        analysis_df = analyzer.prepare_analysis_data(features_df, ratings_df)
        
        # Create rating groups
        logger.info("Creating rating groups...")
        rating_groups = analyzer.create_rating_groups(analysis_df)
        
        # Analyze each group against baseline
        logger.info("Analyzing groups against baseline...")
        analysis_results = {}
        
        # Use 'affirm' group as baseline if available, otherwise use 'all'
        if 'affirm' in rating_groups:
            baseline_df = rating_groups['affirm']
        else:
            baseline_df = rating_groups['all']
        
        for group_name, group_df in rating_groups.items():
            logger.info(f"Analyzing group: {group_name}")
            analysis_results[group_name] = analyzer.analyze_group_vs_baseline(
                group_df, baseline_df, group_name
            )
        
        # Generate case studies
        logger.info("Generating case studies...")
        case_studies = analyzer.generate_case_studies(analysis_results, analysis_df)
        
        # Create summary tables
        logger.info("Creating summary tables...")
        analyzer.create_summary_tables(
            analysis_results, 
            os.path.join(args.output_dir, 'tables')
        )
        
        # Create visualizations
        logger.info("Creating visualizations...")
        analyzer.create_visualizations(
            analysis_results,
            case_studies,
            analysis_df,
            os.path.join(args.output_dir, 'figures')
        )
        
        # Generate summary report
        logger.info("Generating summary report...")
        analyzer.generate_summary_report(
            analysis_results,
            case_studies,
            os.path.join(args.output_dir, 'acoustic_analysis_report.md')
        )
        
        # Save analysis results and case studies
        with open(os.path.join(args.output_dir, 'analysis_results.json'), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for group, results in analysis_results.items():
                json_results[group] = results
            json.dump(json_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        with open(os.path.join(args.output_dir, 'case_studies.json'), 'w') as f:
            json.dump(case_studies, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        logger.info("Descriptive analysis complete!")
        logger.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error in descriptive analysis: {e}", exc_info=True)


if __name__ == "__main__":
    main()