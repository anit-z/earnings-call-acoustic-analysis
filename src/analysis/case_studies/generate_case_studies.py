#!/usr/bin/env python3
"""
Case Study Generation for Earnings Call Acoustic Analysis
Creates detailed case studies of specific earnings calls with notable patterns
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
import csv
import librosa
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CaseStudyGenerator:
    """
    Generates detailed case studies of notable earnings calls
    Implements methodological triangulation through small-sample case analysis
    """
    
    def __init__(self, 
                 num_case_studies: int = 5, 
                 selection_criteria: str = 'combined',
                 n_bootstrap: int = 10000,
                 confidence_level: float = 0.95,
                 random_seed: int = 42):
        """
        Initialize case study generator
        
        Args:
            num_case_studies: Number of case studies to generate
            selection_criteria: Criteria for selecting case studies 
                               ('acoustic', 'semantic', 'combined', 'rating_outcome')
            n_bootstrap: Number of bootstrap iterations for confidence intervals
            confidence_level: Confidence level for intervals
            random_seed: Random seed for reproducibility
        """
        self.num_case_studies = num_case_studies
        self.selection_criteria = selection_criteria
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Key acoustic features for case studies (from thesis)
        self.key_acoustic_features = [
            'f0_cv',      # F0 Coefficient of Variation - Primary feature
            'f0_std',     # F0 Standard Deviation - Complementary pitch measure
            'pause_frequency',  # Pause Frequency - Temporal stress indicator
            'jitter_local'      # Jitter Local - Voice quality measure
        ]
        
        # Key semantic features for case studies
        self.key_semantic_features = [
            'sentiment_positive',
            'sentiment_negative',
            'sentiment_neutral',
            'sentiment_variability'
        ]
        
        # Key combined features for case studies
        self.key_combined_features = [
            'acoustic_semantic_alignment'
        ]
        
        # Color palette (same as descriptive_analysis.py)
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
                audio_dir: Optional[str] = None,
                ratings_file: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load combined features, audio metadata, and ratings data
        
        Args:
            features_dir: Directory containing combined features
            audio_dir: Optional directory containing processed audio
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
        logger.info(f"Loaded combined features: {len(features_df)} calls, {features_df.shape[1]} features")
        
        # Ensure file_id is string type
        if 'file_id' in features_df.columns:
            features_df['file_id'] = features_df['file_id'].astype(str)
        
        # Load fusion summary if available
        fusion_summary_path = Path(features_dir) / "fusion_summary.json"
        if fusion_summary_path.exists():
            with open(fusion_summary_path, 'r') as f:
                self.fusion_summary = json.load(f)
            logger.info(f"Loaded fusion summary from {fusion_summary_path}")
        else:
            self.fusion_summary = None
        
        # Load PCA results if available
        pca_path = Path(features_dir) / "pca_results.csv"
        if pca_path.exists():
            self.pca_results = pd.read_csv(pca_path)
            logger.info(f"Loaded PCA results from {pca_path}")
        else:
            self.pca_results = None
        
        # Load feature importance if available
        importance_path = Path(features_dir) / "feature_importance.csv"
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path, index_col=0)
            logger.info(f"Loaded feature importance from {importance_path}")
        else:
            self.feature_importance = None
        
        # Load audio metadata if available
        self.audio_metadata = None
        if audio_dir:
            audio_stats_path = Path(audio_dir) / "preprocessing_stats.csv"
            if audio_stats_path.exists():
                self.audio_metadata = pd.read_csv(audio_stats_path)
                logger.info(f"Loaded audio metadata: {len(self.audio_metadata)} files")
                
                # Ensure file_id is string type
                if 'file_id' in self.audio_metadata.columns:
                    self.audio_metadata['file_id'] = self.audio_metadata['file_id'].astype(str)
            else:
                logger.warning(f"Audio metadata file not found: {audio_stats_path}")
        
        # Load ratings data if provided
        ratings_df = None
        if ratings_file and Path(ratings_file).exists():
            ratings_df = pd.read_csv(ratings_file)
            logger.info(f"Loaded ratings data: {len(ratings_df)} companies")
            
            # Ensure file_id is string type
            if 'file_id' in ratings_df.columns:
                ratings_df['file_id'] = ratings_df['file_id'].astype(str)
        
        return features_df, ratings_df
    
    def prepare_analysis_data(self, 
                            features_df: pd.DataFrame, 
                            ratings_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare data for case study analysis
        
        Args:
            features_df: Features DataFrame
            ratings_df: Optional ratings DataFrame
            
        Returns:
            Prepared DataFrame for analysis
        """
        # If no ratings data, return features as is
        if ratings_df is None:
            return features_df
        
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
        
        # Create numeric rating change column if needed
        if 'rating_change_magnitude' not in merged_df.columns and 'composite_outcome' in merged_df.columns:
            # Map rating outcomes to numeric values
            outcome_map = {
                'upgrade': 1,
                'affirm': 0,
                'downgrade': -1
            }
            merged_df['rating_change_magnitude'] = merged_df['composite_outcome'].map(outcome_map).fillna(0)
        
        # If audio metadata is available, add it
        if self.audio_metadata is not None:
            merged_df = pd.merge(
                merged_df,
                self.audio_metadata,
                on='file_id',
                how='left'
            )
        
        return merged_df
    
    def select_case_studies(self, analysis_df: pd.DataFrame) -> List[str]:
        """
        Select case studies based on specified criteria
        
        Args:
            analysis_df: Analysis DataFrame
            
        Returns:
            List of file_ids selected for case studies
        """
        # Set random state for reproducible selection
        rng = np.random.RandomState(self.random_seed)
        
        # Selection criteria
        if self.selection_criteria == 'acoustic':
            # Select based on acoustic features
            # Focus on extreme values of f0_cv (primary acoustic feature)
            if 'f0_cv' in analysis_df.columns:
                # Get high and low extremes
                high_f0cv = analysis_df.nlargest(self.num_case_studies // 2, 'f0_cv')
                low_f0cv = analysis_df.nsmallest(self.num_case_studies // 2, 'f0_cv')
                
                # Combine and deduplicate
                selected = pd.concat([high_f0cv, low_f0cv])
                selected = selected.drop_duplicates(subset=['file_id'])
                
                # If we need more, add based on pause_frequency
                if len(selected) < self.num_case_studies and 'pause_frequency' in analysis_df.columns:
                    remaining = analysis_df[~analysis_df['file_id'].isin(selected['file_id'])]
                    more_selected = remaining.nlargest(self.num_case_studies - len(selected), 'pause_frequency')
                    selected = pd.concat([selected, more_selected])
            else:
                # Fallback to random selection
                selected = analysis_df.sample(min(self.num_case_studies, len(analysis_df)), random_state=rng)
        
        elif self.selection_criteria == 'semantic':
            # Select based on semantic features
            # Focus on extreme values of sentiment
            if all(col in analysis_df.columns for col in ['sentiment_negative', 'sentiment_positive']):
                # Get high negative and high positive sentiment
                high_negative = analysis_df.nlargest(self.num_case_studies // 2, 'sentiment_negative')
                high_positive = analysis_df.nlargest(self.num_case_studies // 2, 'sentiment_positive')
                
                # Combine and deduplicate
                selected = pd.concat([high_negative, high_positive])
                selected = selected.drop_duplicates(subset=['file_id'])
                
                # If we need more, add based on sentiment_variability
                if len(selected) < self.num_case_studies and 'sentiment_variability' in analysis_df.columns:
                    remaining = analysis_df[~analysis_df['file_id'].isin(selected['file_id'])]
                    more_selected = remaining.nlargest(self.num_case_studies - len(selected), 'sentiment_variability')
                    selected = pd.concat([selected, more_selected])
            else:
                # Fallback to random selection
                selected = analysis_df.sample(min(self.num_case_studies, len(analysis_df)), random_state=rng)
        
        elif self.selection_criteria == 'rating_outcome':
            # Select based on rating outcomes
            if 'composite_outcome' in analysis_df.columns:
                # Prioritize downgrades and upgrades
                downgrades = analysis_df[analysis_df['composite_outcome'] == 'downgrade']
                upgrades = analysis_df[analysis_df['composite_outcome'] == 'upgrade']
                affirms = analysis_df[analysis_df['composite_outcome'] == 'affirm']
                
                # Balance selection across outcomes
                num_downgrades = min(len(downgrades), self.num_case_studies // 2)
                num_upgrades = min(len(upgrades), self.num_case_studies // 4)
                num_affirms = min(len(affirms), self.num_case_studies - num_downgrades - num_upgrades)
                
                selected_downgrades = downgrades.sample(num_downgrades, random_state=rng) if num_downgrades > 0 else pd.DataFrame()
                selected_upgrades = upgrades.sample(num_upgrades, random_state=rng) if num_upgrades > 0 else pd.DataFrame()
                selected_affirms = affirms.sample(num_affirms, random_state=rng) if num_affirms > 0 else pd.DataFrame()
                
                selected = pd.concat([selected_downgrades, selected_upgrades, selected_affirms])
                
                # If we need more, add random selections
                if len(selected) < self.num_case_studies:
                    remaining = analysis_df[~analysis_df['file_id'].isin(selected['file_id'])]
                    more_selected = remaining.sample(min(self.num_case_studies - len(selected), len(remaining)), random_state=rng)
                    selected = pd.concat([selected, more_selected])
            else:
                # Fallback to random selection
                selected = analysis_df.sample(min(self.num_case_studies, len(analysis_df)), random_state=rng)
        
        else:  # 'combined' or any other value
            # Select based on a combination of criteria
            
            # First, look for cases with clear patterns
            if 'communication_pattern' in analysis_df.columns:
                # Prioritize high stress and high excitement patterns
                high_stress = analysis_df[analysis_df['communication_pattern'] == 'high_stress']
                high_excitement = analysis_df[analysis_df['communication_pattern'] == 'high_excitement']
                moderate_stress = analysis_df[analysis_df['communication_pattern'] == 'moderate_stress']
                moderate_excitement = analysis_df[analysis_df['communication_pattern'] == 'moderate_excitement']
                baseline = analysis_df[analysis_df['communication_pattern'] == 'baseline_stability']
                
                # Balance selection across patterns
                num_high_stress = min(len(high_stress), 1)
                num_high_excitement = min(len(high_excitement), 1)
                num_moderate_stress = min(len(moderate_stress), 1)
                num_moderate_excitement = min(len(moderate_excitement), 1)
                num_baseline = min(len(baseline), 1)
                
                # Create initial selection
                selected = pd.concat([
                    high_stress.sample(num_high_stress, random_state=rng) if num_high_stress > 0 else pd.DataFrame(),
                    high_excitement.sample(num_high_excitement, random_state=rng) if num_high_excitement > 0 else pd.DataFrame(),
                    moderate_stress.sample(num_moderate_stress, random_state=rng) if num_moderate_stress > 0 else pd.DataFrame(),
                    moderate_excitement.sample(num_moderate_excitement, random_state=rng) if num_moderate_excitement > 0 else pd.DataFrame(),
                    baseline.sample(num_baseline, random_state=rng) if num_baseline > 0 else pd.DataFrame()
                ])
                
                # If we need more and have rating outcomes, add based on that
                if len(selected) < self.num_case_studies and 'composite_outcome' in analysis_df.columns:
                    remaining = analysis_df[~analysis_df['file_id'].isin(selected['file_id'])]
                    downgrades = remaining[remaining['composite_outcome'] == 'downgrade']
                    upgrades = remaining[remaining['composite_outcome'] == 'upgrade']
                    
                    num_remaining = self.num_case_studies - len(selected)
                    num_downgrades = min(len(downgrades), num_remaining // 2)
                    num_upgrades = min(len(upgrades), num_remaining - num_downgrades)
                    
                    more_selected = pd.concat([
                        downgrades.sample(num_downgrades, random_state=rng) if num_downgrades > 0 else pd.DataFrame(),
                        upgrades.sample(num_upgrades, random_state=rng) if num_upgrades > 0 else pd.DataFrame()
                    ])
                    
                    selected = pd.concat([selected, more_selected])
                
                # If still need more, add based on acoustic-semantic alignment
                if len(selected) < self.num_case_studies and 'acoustic_semantic_alignment' in analysis_df.columns:
                    remaining = analysis_df[~analysis_df['file_id'].isin(selected['file_id'])]
                    
                    # Get both high and low alignment cases
                    high_alignment = remaining.nlargest((self.num_case_studies - len(selected)) // 2, 'acoustic_semantic_alignment')
                    low_alignment = remaining.nsmallest((self.num_case_studies - len(selected)) - len(high_alignment), 'acoustic_semantic_alignment')
                    
                    more_selected = pd.concat([high_alignment, low_alignment])
                    selected = pd.concat([selected, more_selected])
            
            else:
                # Fallback to combined acoustic and semantic criteria
                if all(col in analysis_df.columns for col in ['f0_cv', 'sentiment_negative']):
                    # Create a composite score
                    analysis_df['composite_score'] = analysis_df['f0_cv'] * analysis_df['sentiment_negative']
                    
                    # Select based on high and low composite scores
                    high_composite = analysis_df.nlargest(self.num_case_studies // 2, 'composite_score')
                    low_composite = analysis_df.nsmallest(self.num_case_studies // 2, 'composite_score')
                    
                    selected = pd.concat([high_composite, low_composite])
                    selected = selected.drop_duplicates(subset=['file_id'])
                else:
                    # Fallback to random selection
                    selected = analysis_df.sample(min(self.num_case_studies, len(analysis_df)), random_state=rng)
        
        # Ensure we don't have more than requested
        if len(selected) > self.num_case_studies:
            selected = selected.sample(self.num_case_studies, random_state=rng)
        
        # Get the list of file_ids
        file_ids = selected['file_id'].tolist()
        
        logger.info(f"Selected {len(file_ids)} case studies using '{self.selection_criteria}' criteria")
        
        return file_ids
    
    def generate_case_study(self, 
                          file_id: str, 
                          analysis_df: pd.DataFrame,
                          audio_dir: Optional[str] = None) -> Dict:
        """
        Generate a detailed case study for a specific file
        
        Args:
            file_id: File ID for the case study
            analysis_df: Analysis DataFrame
            audio_dir: Optional directory containing processed audio
            
        Returns:
            Dictionary containing case study data
        """
        # Get data for this file
        file_data = analysis_df[analysis_df['file_id'] == file_id]
        
        if len(file_data) == 0:
            logger.warning(f"No data found for file_id {file_id}")
            return {'file_id': file_id, 'error': 'no_data_found'}
        
        # Extract row as dictionary
        row_dict = file_data.iloc[0].to_dict()
        
        # Create case study dictionary
        case_study = {
            'file_id': file_id,
            'metadata': {
                'sector': row_dict.get('sector', 'Unknown'),
                'duration_s': row_dict.get('duration_s', None),
                'rating_outcome': row_dict.get('composite_outcome', None),
                'time_gap_days': row_dict.get('time_gap_days', None),
                'rating_change_magnitude': row_dict.get('rating_change_magnitude', None),
                'communication_pattern': row_dict.get('communication_pattern', None),
                'nearest_action_agency': row_dict.get('nearest_action_agency', None),
                'earnings_call_date': row_dict.get('earnings_call_date', None)
            },
            'acoustic_features': {},
            'semantic_features': {},
            'combined_features': {},
            'rating_data': {},
            'percentile_rankings': {},
            'percentile_rankings_ci': {},
            'mad_effect_sizes': {}
        }
        
        # Calculate percentiles with confidence intervals for key features
        percentile_results = self._calculate_percentile_rankings_with_ci(file_id, analysis_df)
        case_study['percentile_rankings'] = percentile_results['percentiles']
        case_study['percentile_rankings_ci'] = percentile_results['confidence_intervals']
        
        # Calculate MAD-based effect sizes
        mad_effects = self._calculate_mad_effects(file_id, analysis_df)
        case_study['mad_effect_sizes'] = mad_effects
        
        # Add acoustic features
        for feature in self.key_acoustic_features:
            if feature in row_dict:
                case_study['acoustic_features'][feature] = row_dict[feature]
        
        # Add semantic features
        for feature in self.key_semantic_features:
            if feature in row_dict:
                case_study['semantic_features'][feature] = row_dict[feature]
        
        # Add combined features
        for feature in self.key_combined_features:
            if feature in row_dict:
                case_study['combined_features'][feature] = row_dict[feature]
        
        # Add PCA coordinates if available
        if self.pca_results is not None and 'file_id' in self.pca_results.columns:
            pca_data = self.pca_results[self.pca_results['file_id'] == file_id]
            if len(pca_data) > 0:
                case_study['pca_coordinates'] = {
                    'PC1': pca_data.iloc[0]['PC1'],
                    'PC2': pca_data.iloc[0]['PC2']
                }
        
        # Add rating-related data
        rating_columns = [
            'composite_outcome', 'sp_action', 'moodys_action', 'fitch_action',
            'time_gap_days', 'rating_change_magnitude', 'agency_count',
            'disagreement_flag', 'consensus_strength', 'information_content'
        ]
        
        for col in rating_columns:
            if col in row_dict:
                case_study['rating_data'][col] = row_dict[col]
        
        # Add audio features from metadata if available
        if self.audio_metadata is not None:
            audio_data = self.audio_metadata[self.audio_metadata['file_id'] == file_id]
            if len(audio_data) > 0:
                audio_row = audio_data.iloc[0].to_dict()
                case_study['audio_metadata'] = {
                    'original_sr': audio_row.get('original_sr', None),
                    'duration_s': audio_row.get('duration_s', None),
                    'vad_ratio': audio_row.get('vad_ratio', None),
                    'peak_amplitude': audio_row.get('peak_amplitude', None),
                    'rms_energy': audio_row.get('rms_energy', None),
                    'num_speakers': audio_row.get('num_speakers', None)
                }
        
        # Add short audio analysis if audio directory provided
        if audio_dir:
            audio_path = os.path.join(audio_dir, f"{file_id}.wav")
            if os.path.exists(audio_path):
                case_study['audio_analysis'] = self._analyze_audio_segments(audio_path)
        
        # Generate summary insights
        case_study['insights'] = self._generate_insights(case_study, analysis_df)
        
        return case_study
    
    def _calculate_percentile_rankings_with_ci(self, 
                                             file_id: str, 
                                             analysis_df: pd.DataFrame) -> Dict:
        """
        Calculate percentile rankings with bootstrap confidence intervals
        using the standard definition with proper handling of ties
        
        Args:
            file_id: File ID for the case study
            analysis_df: Analysis DataFrame
            
        Returns:
            Dictionary with percentiles and confidence intervals
        """
        percentiles = {}
        confidence_intervals = {}
        
        # Get data for this file
        file_data = analysis_df[analysis_df['file_id'] == file_id].iloc[0]
        
        # Combine all features
        all_features = (self.key_acoustic_features + 
                       self.key_semantic_features + 
                       self.key_combined_features)
        
        # Set local random state for bootstrap reproducibility
        rng = np.random.RandomState(self.random_seed)
        
        for feature in all_features:
            if feature in file_data and feature in analysis_df.columns:
                value = file_data[feature]
                all_values = analysis_df[feature].dropna().values
                
                if len(all_values) < 3:  # Not enough data for meaningful CI
                    # Standard definition with proper handling of ties
                    n = len(all_values)
                    less_than = np.sum(all_values < value)
                    equal_to = np.sum(all_values == value)
                    percentile = (less_than + 0.5 * equal_to) / n * 100
                    
                    percentiles[feature] = percentile
                    confidence_intervals[feature] = {
                        'ci_lower': percentile,
                        'ci_upper': percentile,
                        'ci_width': 0.0
                    }
                    continue
                
                # Calculate basic percentile rank using standard definition
                n = len(all_values)
                less_than = np.sum(all_values < value)
                equal_to = np.sum(all_values == value)
                percentile = (less_than + 0.5 * equal_to) / n * 100
                percentiles[feature] = percentile
                
                # Bootstrap to calculate confidence interval
                bootstrap_percentiles = []
                
                for _ in range(self.n_bootstrap):
                    # Sample with replacement
                    bootstrap_sample = rng.choice(
                        all_values, 
                        size=len(all_values), 
                        replace=True
                    )
                    
                    # Calculate percentile rank in bootstrap sample using standard definition
                    boot_less_than = np.sum(bootstrap_sample < value)
                    boot_equal_to = np.sum(bootstrap_sample == value)
                    boot_percentile = (boot_less_than + 0.5 * boot_equal_to) / len(bootstrap_sample) * 100
                    bootstrap_percentiles.append(boot_percentile)
                
                # Calculate confidence interval
                ci_lower = np.percentile(bootstrap_percentiles, 100 * self.alpha / 2)
                ci_upper = np.percentile(bootstrap_percentiles, 100 * (1 - self.alpha / 2))
                
                confidence_intervals[feature] = {
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper),
                    'ci_width': float(ci_upper - ci_lower)
                }
        
        return {
            'percentiles': percentiles,
            'confidence_intervals': confidence_intervals
        }
    
    def _calculate_mad_effects(self, 
                             file_id: str, 
                             analysis_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate MAD-based effect sizes for a case
        
        Args:
            file_id: File ID for the case study
            analysis_df: Analysis DataFrame
            
        Returns:
            Dictionary of MAD effect sizes
        """
        mad_effects = {}
        
        # Get data for this file
        file_data = analysis_df[analysis_df['file_id'] == file_id].iloc[0]
        
        # Combine all features
        all_features = (self.key_acoustic_features + 
                       self.key_semantic_features + 
                       self.key_combined_features)
        
        for feature in all_features:
            if feature in file_data and feature in analysis_df.columns:
                value = file_data[feature]
                all_values = analysis_df[feature].dropna().values
                
                if len(all_values) >= 3:  # Need at least 3 values for meaningful MAD
                    median = np.median(all_values)
                    mad = stats.median_abs_deviation(all_values)
                    
                    if mad == 0:
                        # All values are identical
                        if value == median:
                            mad_effect = 0.0
                        else:
                            mad_effect = np.sign(value - median)
                    else:
                        mad_effect = (value - median) / mad
                    
                    mad_effects[feature] = float(mad_effect)
        
        return mad_effects
    
    def _calculate_cohens_d(self, 
                          group1_values: np.ndarray, 
                          group2_values: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size between two groups
        
        Args:
            group1_values: Values for group 1
            group2_values: Values for group 2
            
        Returns:
            Cohen's d effect size
        """
        # Calculate means
        mean1 = np.mean(group1_values)
        mean2 = np.mean(group2_values)
        
        # Calculate pooled standard deviation
        n1, n2 = len(group1_values), len(group2_values)
        
        # Handle case where one group has only one observation
        if n1 <= 1 or n2 <= 1:
            return np.nan
        
        s1 = np.std(group1_values, ddof=1)
        s2 = np.std(group2_values, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        # Handle zero variance case
        if pooled_std == 0:
            return 0.0
        
        # Calculate Cohen's d
        d = (mean1 - mean2) / pooled_std
        
        return float(d)
    
    def _analyze_audio_segments(self, audio_path: str) -> Dict:
        """
        Analyze audio segments to identify sections with notable patterns
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of audio segment analysis
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=600)  # Limit to first 10 minutes for efficiency
            
            # Create 10-second segments
            segment_duration = 10  # seconds
            segment_samples = segment_duration * sr
            num_segments = len(audio) // segment_samples
            
            segments_analysis = []
            
            # Analyze each segment
            for i in range(min(10, num_segments)):  # Limit to 10 segments for efficiency
                start_sample = i * segment_samples
                end_sample = start_sample + segment_samples
                segment = audio[start_sample:end_sample]
                
                # Calculate segment features
                segment_features = self._extract_segment_features(segment, sr)
                
                # Add time information
                segment_features['start_time'] = i * segment_duration
                segment_features['end_time'] = (i + 1) * segment_duration
                segment_features['segment_id'] = i
                
                segments_analysis.append(segment_features)
            
            # Find segments with highest and lowest variability
            segments_df = pd.DataFrame(segments_analysis)
            segments_df.sort_values('f0_cv', ascending=False, inplace=True)
            
            high_variability_segments = segments_df.head(3).to_dict('records')
            low_variability_segments = segments_df.tail(3).to_dict('records')
            
            return {
                'high_variability_segments': high_variability_segments,
                'low_variability_segments': low_variability_segments,
                'num_segments_analyzed': len(segments_analysis)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing audio segments: {e}")
            return {'error': str(e)}
    
    def _extract_segment_features(self, 
                                segment: np.ndarray, 
                                sr: int) -> Dict[str, float]:
        """
        Extract features from an audio segment
        
        Args:
            segment: Audio segment
            sr: Sample rate
            
        Returns:
            Dictionary of segment features
        """
        features = {}
        
        try:
            # F0 features
            f0, voiced_flag, _ = librosa.pyin(
                segment,
                fmin=75,
                fmax=500,
                sr=sr
            )
            
            f0_voiced = f0[voiced_flag]
            if len(f0_voiced) > 0:
                features['f0_mean'] = float(np.mean(f0_voiced))
                features['f0_std'] = float(np.std(f0_voiced))
                features['f0_cv'] = float(np.std(f0_voiced) / np.mean(f0_voiced) if np.mean(f0_voiced) > 0 else 0)
            else:
                features['f0_mean'] = 0.0
                features['f0_std'] = 0.0
                features['f0_cv'] = 0.0
            
            # Temporal features
            energy = librosa.feature.rms(y=segment)[0]
            energy_threshold = np.mean(energy) * 0.5
            voiced = energy > energy_threshold
            
            features['speech_rate'] = float(librosa.beat.tempo(y=segment, sr=sr)[0] / 60.0)
            features['voice_activity_ratio'] = float(np.mean(voiced))
            features['pause_ratio'] = float(1 - np.mean(voiced))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Overall energy
            features['rms_energy'] = float(np.mean(energy))
            
        except Exception as e:
            logger.warning(f"Error extracting segment features: {e}")
            features['error'] = str(e)
        
        return features
    
    def _generate_insights(self, 
                         case_study: Dict, 
                         analysis_df: pd.DataFrame) -> List[str]:
        """
        Generate insights for the case study
        
        Args:
            case_study: Case study dictionary
            analysis_df: Analysis DataFrame
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Get key data
        f0_cv = case_study['acoustic_features'].get('f0_cv', None)
        sentiment_negative = case_study['semantic_features'].get('sentiment_negative', None)
        communication_pattern = case_study['metadata'].get('communication_pattern', None)
        rating_outcome = case_study['metadata'].get('rating_outcome', None)
        
        # F0 CV insights with confidence intervals
        if f0_cv is not None:
            f0_cv_percentile = case_study['percentile_rankings'].get('f0_cv', None)
            if f0_cv_percentile is not None:
                # Get confidence interval
                ci_data = case_study['percentile_rankings_ci'].get('f0_cv', {})
                ci_lower = ci_data.get('ci_lower', f0_cv_percentile)
                ci_upper = ci_data.get('ci_upper', f0_cv_percentile)
                
                # Get MAD effect
                mad_effect = case_study['mad_effect_sizes'].get('f0_cv', None)
                mad_str = f" (MAD effect: {mad_effect:.2f})" if mad_effect is not None else ""
                
                if f0_cv_percentile > 90:
                    insights.append(f"Very high F0 coefficient of variation: {f0_cv_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%){mad_str} - indicates potential executive stress")
                elif f0_cv_percentile > 75:
                    insights.append(f"Elevated F0 coefficient of variation: {f0_cv_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%){mad_str} - suggests moderate vocal stress")
                elif f0_cv_percentile < 25:
                    insights.append(f"Very low F0 coefficient of variation: {f0_cv_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%){mad_str} - indicates calm, controlled delivery")
        
        # Pause frequency insights
        pause_freq = case_study['acoustic_features'].get('pause_frequency', None)
        if pause_freq is not None:
            pause_percentile = case_study['percentile_rankings'].get('pause_frequency', None)
            if pause_percentile is not None:
                ci_data = case_study['percentile_rankings_ci'].get('pause_frequency', {})
                ci_lower = ci_data.get('ci_lower', pause_percentile)
                ci_upper = ci_data.get('ci_upper', pause_percentile)
                
                if pause_percentile > 75:
                    insights.append(f"High pause frequency: {pause_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%) - suggests hesitation or cognitive load")
                elif pause_percentile < 25:
                    insights.append(f"Low pause frequency: {pause_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%) - indicates fluent, confident speech")
        
        # Sentiment insights with confidence intervals
        if sentiment_negative is not None:
            sentiment_percentile = case_study['percentile_rankings'].get('sentiment_negative', None)
            if sentiment_percentile is not None:
                # Get confidence interval
                ci_data = case_study['percentile_rankings_ci'].get('sentiment_negative', {})
                ci_lower = ci_data.get('ci_lower', sentiment_percentile)
                ci_upper = ci_data.get('ci_upper', sentiment_percentile)
                
                # Get MAD effect
                mad_effect = case_study['mad_effect_sizes'].get('sentiment_negative', None)
                mad_str = f" (MAD effect: {mad_effect:.2f})" if mad_effect is not None else ""
                
                if sentiment_percentile > 90:
                    insights.append(f"Highly negative sentiment: {sentiment_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%){mad_str} - suggests challenging financial conditions")
                elif sentiment_percentile > 75:
                    insights.append(f"Elevated negative sentiment: {sentiment_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%){mad_str} - indicates potential concerns")
                elif sentiment_percentile < 25:
                    insights.append(f"Very low negative sentiment: {sentiment_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%){mad_str} - suggests positive financial outlook")
        
        # Acoustic-semantic alignment insights
        if f0_cv is not None and sentiment_negative is not None:
            alignment = case_study['combined_features'].get('acoustic_semantic_alignment', None)
            if alignment is not None:
                alignment_percentile = case_study['percentile_rankings'].get('acoustic_semantic_alignment', None)
                
                if alignment_percentile is not None:
                    ci_data = case_study['percentile_rankings_ci'].get('acoustic_semantic_alignment', {})
                    ci_lower = ci_data.get('ci_lower', alignment_percentile)
                    ci_upper = ci_data.get('ci_upper', alignment_percentile)
                    
                    if alignment_percentile > 75:
                        insights.append(f"Strong acoustic-semantic alignment: {alignment_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%) - consistent stress signals")
                    elif alignment_percentile < 25:
                        insights.append(f"Low acoustic-semantic alignment: {alignment_percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%) - divergent stress indicators")
        
        # Communication pattern insights
        if communication_pattern:
            if communication_pattern == 'high_stress':
                insights.append("High stress communication pattern detected - convergence of multiple stress indicators")
            elif communication_pattern == 'high_excitement':
                insights.append("High excitement pattern detected - suggests positive arousal rather than stress")
            elif communication_pattern == 'moderate_stress':
                insights.append("Moderate stress pattern detected - some stress indicators present")
            elif communication_pattern == 'moderate_excitement':
                insights.append("Moderate excitement pattern detected - mild positive arousal indicators")
            elif communication_pattern == 'baseline_stability':
                insights.append("Baseline stability pattern - typical earnings call communication pattern")
            elif communication_pattern == 'mixed_pattern':
                insights.append("Mixed communication pattern - inconsistent stress/arousal indicators")
        
        # Rating outcome insights
        if rating_outcome:
            if rating_outcome == 'downgrade':
                insights.append("Rating downgrade followed this earnings call - negative financial outcomes")
            elif rating_outcome == 'upgrade':
                insights.append("Rating upgrade followed this earnings call - positive financial outcomes")
            elif rating_outcome == 'affirm':
                insights.append("Rating affirmation followed this earnings call - stable financial outcomes")
        
        # Audio segment insights
        if 'audio_analysis' in case_study and 'high_variability_segments' in case_study['audio_analysis']:
            high_var_segments = case_study['audio_analysis']['high_variability_segments']
            if high_var_segments:
                highest_segment = high_var_segments[0]
                insights.append(f"Highest vocal stress detected at {highest_segment['start_time']}-{highest_segment['end_time']}s (F0 CV: {highest_segment['f0_cv']:.3f})")
        
        # If we have few insights, add a general one
        if len(insights) < 3:
            if case_study['metadata'].get('sector'):
                insights.append(f"This earnings call is from the {case_study['metadata']['sector']} sector")
            
            insights.append("Further investigation recommended to identify specific stress triggers")
        
        return insights
    
    def create_case_study_visualizations(self,
                                       case_studies: List[Dict],
                                       output_dir: str):
        """
        Create visualizations for case studies
        
        Args:
            case_studies: List of case study dictionaries
            output_dir: Output directory for visualizations
        """
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create summary visualization
        self._create_case_summary_plot(case_studies, viz_dir)
        
        # Create individual case visualizations
        for case in case_studies:
            self._create_individual_case_plot(case, viz_dir)
    
    def _create_case_summary_plot(self,
                                case_studies: List[Dict],
                                output_dir: str):
        """
        Create summary visualization of all case studies
        
        Args:
            case_studies: List of case study dictionaries
            output_dir: Output directory for visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(6, 5))
        axes = axes.flatten()
        
        # Plot 1: F0 CV percentiles
        ax = axes[0]
        file_ids = []
        f0_cv_percentiles = []
        colors = []
        
        for case in case_studies:
            if 'f0_cv' in case['percentile_rankings']:
                file_ids.append(case['file_id'])
                f0_cv_percentiles.append(case['percentile_rankings']['f0_cv'])
                
                # Color based on rating outcome
                outcome = case['metadata'].get('rating_outcome', 'unknown')
                if outcome == 'downgrade':
                    colors.append(self.color_palette[7])
                elif outcome == 'upgrade':
                    colors.append(self.color_palette[2])
                else:
                    colors.append(self.color_palette[3])
        
        bars = ax.barh(range(len(file_ids)), f0_cv_percentiles, color=colors, alpha=0.8)
        
        # Add percentile lines
        ax.axvline(x=50, color=self.color_palette[4], linestyle='--', alpha=0.5)
        ax.axvline(x=75, color=self.color_palette[5], linestyle='--', alpha=0.5)
        ax.axvline(x=90, color=self.color_palette[7], linestyle='--', alpha=0.5)
        
        # Add value labels positioned to avoid overlap
        for i, (bar, val) in enumerate(zip(bars, f0_cv_percentiles)):
            x_pos = val + 2 if val < 90 else val - 2
            ha = 'left' if val < 90 else 'right'
            ax.text(x_pos, i, f"{val:.0f}%", ha=ha, va='center', fontsize=6, fontweight='normal')
        
        ax.set_yticks(range(len(file_ids)))
        ax.set_yticklabels(file_ids, fontsize=6)
        ax.set_xlabel('Percentile Rank', fontsize=7, fontweight='normal')
        ax.set_title('F0 CV Percentiles', fontsize=8, fontweight='normal', pad=10)
        ax.set_xlim(0, 105)
        
        # Similar plots for other features...
        # (keeping it concise for space, but following same pattern)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'case_studies_summary.png'), dpi=300)
        plt.close()
    
    def _create_individual_case_plot(self,
                                   case: Dict,
                                   output_dir: str):
        """
        Create individual visualization for a case study
        
        Args:
            case: Case study dictionary
            output_dir: Output directory for visualization
        """
        # Create a radar plot for the case
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, polar=True)
        
        # Get features and percentiles
        features = []
        percentiles = []
        
        for feature in self.key_acoustic_features + self.key_semantic_features[:2]:  # Limit for clarity
            if feature in case['percentile_rankings']:
                features.append(feature.replace('_', ' ').title())
                percentiles.append(case['percentile_rankings'][feature])
        
        if not features:
            plt.close()
            return
        
        # Number of variables
        N = len(features)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Initialize the plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=7)
        
        # Draw y-axis labels
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=6)
        ax.set_ylim(0, 100)
        
        # Plot data
        percentiles += percentiles[:1]  # Complete the circle
        ax.plot(angles, percentiles, 'o-', linewidth=1.5, color=self.color_palette[5])
        ax.fill(angles, percentiles, alpha=0.25, color=self.color_palette[3])
        
        # Add title
        outcome = case['metadata'].get('rating_outcome', 'Unknown')
        plt.title(f"Case {case['file_id']} - {outcome.title()}", 
                 fontsize=9, fontweight='normal', pad=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"case_{case['file_id']}_radar.png"), dpi=300)
        plt.close()
    
    def export_case_studies(self, 
                          case_studies: List[Dict], 
                          output_dir: str):
        """
        Export case studies to various formats
        
        Args:
            case_studies: List of case study dictionaries
            output_dir: Output directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create main CSV file with key metrics
        self._export_case_studies_csv(case_studies, output_dir)
        
        # Create JSON file with full details
        self._export_case_studies_json(case_studies, output_dir)
        
        # Create individual case study files
        self._export_individual_case_studies(case_studies, output_dir)
        
        # Create comparative analysis CSV with effect sizes
        self._export_comparative_analysis(case_studies, output_dir)
        
        # Export segments data if available
        self._export_segment_analysis(case_studies, output_dir)
        
        # Create visualizations
        self.create_case_study_visualizations(case_studies, output_dir)
        
        # Generate summary report
        self._generate_summary_report(case_studies, output_dir)
        
        logger.info(f"Case studies exported to {output_dir}")
    
    def _generate_summary_report(self,
                               case_studies: List[Dict],
                               output_dir: str):
        """
        Generate a summary report of case studies
        
        Args:
            case_studies: List of case study dictionaries
            output_dir: Output directory
        """
        report = []
        
        # Add title
        report.append("# CASE STUDY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Add reproducibility note
        report.append("## REPRODUCIBILITY NOTE")
        report.append("-" * 50)
        report.append(f"Random seed used for case selection and bootstrap: {self.random_seed}")
        report.append(f"Bootstrap iterations for confidence intervals: {self.n_bootstrap}")
        report.append(f"Number of case studies: {len(case_studies)}")
        report.append(f"Selection criteria: {self.selection_criteria}")
        report.append("")
        
        # Add case study summaries
        report.append("## CASE STUDY SUMMARIES")
        report.append("-" * 50)
        
        for i, case in enumerate(case_studies, 1):
            report.append(f"\n### Case {i}: {case['file_id']}")
            
            # Metadata
            report.append(f"**Sector**: {case['metadata'].get('sector', 'Unknown')}")
            report.append(f"**Rating Outcome**: {case['metadata'].get('rating_outcome', 'Unknown')}")
            report.append(f"**Communication Pattern**: {case['metadata'].get('communication_pattern', 'Unknown')}")
            
            # Key percentiles with MAD effects
            report.append("\n**Key Feature Percentiles**:")
            
            for feature in ['f0_cv', 'pause_frequency', 'sentiment_negative']:
                if feature in case['percentile_rankings']:
                    percentile = case['percentile_rankings'][feature]
                    ci_data = case['percentile_rankings_ci'].get(feature, {})
                    ci_lower = ci_data.get('ci_lower', percentile)
                    ci_upper = ci_data.get('ci_upper', percentile)
                    
                    mad_effect = case['mad_effect_sizes'].get(feature, None)
                    mad_str = f" (MAD effect: {mad_effect:.2f})" if mad_effect is not None else ""
                    
                    feature_name = feature.replace('_', ' ').title()
                    report.append(f"- {feature_name}: {percentile:.1f}% (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%){mad_str}")
            
            # Insights
            if case.get('insights'):
                report.append("\n**Key Insights**:")
                for insight in case['insights'][:3]:  # Top 3 insights
                    report.append(f"- {insight}")
        
        # Write report
        report_path = os.path.join(output_dir, 'case_study_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Summary report saved to {report_path}")
    
    def _export_case_studies_csv(self, 
                               case_studies: List[Dict], 
                               output_dir: str):
        """
        Export case studies to CSV format
        
        Args:
            case_studies: List of case study dictionaries
            output_dir: Output directory
        """
        # Create list of dictionaries for DataFrame
        rows = []
        
        for case in case_studies:
            row = {
                'file_id': case['file_id'],
                'sector': case['metadata'].get('sector', 'Unknown'),
                'duration_s': case['metadata'].get('duration_s', None),
                'rating_outcome': case['metadata'].get('rating_outcome', None),
                'communication_pattern': case['metadata'].get('communication_pattern', None),
                'time_gap_days': case['metadata'].get('time_gap_days', None),
                'rating_change_magnitude': case['metadata'].get('rating_change_magnitude', None)
            }
            
            # Add acoustic features
            for feature, value in case['acoustic_features'].items():
                row[feature] = value
            
            # Add semantic features
            for feature, value in case['semantic_features'].items():
                row[feature] = value
            
            # Add combined features
            for feature, value in case['combined_features'].items():
                row[feature] = value
            
            # Add percentile rankings
            for feature, percentile in case['percentile_rankings'].items():
                row[f"{feature}_percentile"] = percentile
            
            # Add confidence intervals for percentiles
            for feature, ci_data in case['percentile_rankings_ci'].items():
                row[f"{feature}_percentile_ci_lower"] = ci_data.get('ci_lower', None)
                row[f"{feature}_percentile_ci_upper"] = ci_data.get('ci_upper', None)
                row[f"{feature}_percentile_ci_width"] = ci_data.get('ci_width', None)
            
            # Add MAD effect sizes
            for feature, mad_effect in case['mad_effect_sizes'].items():
                row[f"{feature}_mad_effect"] = mad_effect
            
            rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "case_studies_summary.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Case studies summary saved to {csv_path}")
    
    def _export_case_studies_json(self, 
                                case_studies: List[Dict], 
                                output_dir: str):
        """
        Export case studies to JSON format
        
        Args:
            case_studies: List of case study dictionaries
            output_dir: Output directory
        """
        # Convert numpy values to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convert case studies
        json_case_studies = [convert_numpy(case) for case in case_studies]
        
        # Save to JSON
        json_path = os.path.join(output_dir, "case_studies_full.json")
        with open(json_path, 'w') as f:
            json.dump(json_case_studies, f, indent=2)
        
        logger.info(f"Case studies saved to {json_path}")
    
    def _export_individual_case_studies(self, 
                                      case_studies: List[Dict], 
                                      output_dir: str):
        """
        Export individual case studies to separate files
        
        Args:
            case_studies: List of case study dictionaries
            output_dir: Output directory
        """
        # Create individual directory
        individual_dir = os.path.join(output_dir, "individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        # Save each case study as JSON and CSV
        for case in case_studies:
            file_id = case['file_id']
            
            # Save as JSON
            json_path = os.path.join(individual_dir, f"{file_id}_case_study.json")
            with open(json_path, 'w') as f:
                # Convert numpy values to Python native types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    else:
                        return obj
                
                json.dump(convert_numpy(case), f, indent=2)
            
            # Save key data as CSV
            csv_path = os.path.join(individual_dir, f"{file_id}_metrics.csv")
            
            # Create flattened dictionary for CSV
            flat_data = {
                'file_id': file_id,
                'sector': case['metadata'].get('sector', 'Unknown'),
                'rating_outcome': case['metadata'].get('rating_outcome', None),
                'communication_pattern': case['metadata'].get('communication_pattern', None)
            }
            
            # Add acoustic features
            for feature, value in case['acoustic_features'].items():
                flat_data[f"acoustic_{feature}"] = value
            
            # Add semantic features
            for feature, value in case['semantic_features'].items():
                flat_data[f"semantic_{feature}"] = value
            
            # Add combined features
            for feature, value in case['combined_features'].items():
                flat_data[f"combined_{feature}"] = value
            
            # Add percentile rankings with confidence intervals
            for feature, percentile in case['percentile_rankings'].items():
                flat_data[f"{feature}_percentile"] = percentile
                
                # Add CI if available
                if feature in case['percentile_rankings_ci']:
                    ci_data = case['percentile_rankings_ci'][feature]
                    flat_data[f"{feature}_percentile_ci_lower"] = ci_data.get('ci_lower', None)
                    flat_data[f"{feature}_percentile_ci_upper"] = ci_data.get('ci_upper', None)
                    flat_data[f"{feature}_percentile_ci_width"] = ci_data.get('ci_width', None)
                
                # Add MAD effect if available
                if feature in case['mad_effect_sizes']:
                    flat_data[f"{feature}_mad_effect"] = case['mad_effect_sizes'][feature]
            
            # Create DataFrame with single row and save to CSV
            pd.DataFrame([flat_data]).to_csv(csv_path, index=False)
            
            # Create insights text file
            if 'insights' in case and case['insights']:
                insights_path = os.path.join(individual_dir, f"{file_id}_insights.txt")
                with open(insights_path, 'w') as f:
                    f.write(f"INSIGHTS FOR CASE STUDY: {file_id}\n")
                    f.write("="*60 + "\n\n")
                    for i, insight in enumerate(case['insights'], 1):
                        f.write(f"{i}. {insight}\n")
        
        logger.info(f"Individual case studies saved to {individual_dir}")
    
    def _export_comparative_analysis(self, 
                                   case_studies: List[Dict], 
                                   output_dir: str):
        """
        Export comparative analysis of case studies with effect sizes
        
        Args:
            case_studies: List of case study dictionaries
            output_dir: Output directory
        """
        # Group by rating outcome
        rating_groups = {}
        
        for case in case_studies:
            outcome = case['metadata'].get('rating_outcome', 'unknown')
            if outcome not in rating_groups:
                rating_groups[outcome] = []
            rating_groups[outcome].append(case)
        
        # Calculate average metrics by group
        group_metrics = {}
        
        for outcome, cases in rating_groups.items():
            if not cases:
                continue
                
            # Calculate average acoustic features
            acoustic_features = {}
            acoustic_values = {}  # Store raw values for effect size calculation
            for feature in self.key_acoustic_features:
                values = [case['acoustic_features'].get(feature, None) for case in cases]
                values = [v for v in values if v is not None]
                if values:
                    acoustic_features[feature] = float(np.mean(values))
                    acoustic_values[feature] = values
            
            # Calculate average semantic features
            semantic_features = {}
            semantic_values = {}  # Store raw values for effect size calculation
            for feature in self.key_semantic_features:
                values = [case['semantic_features'].get(feature, None) for case in cases]
                values = [v for v in values if v is not None]
                if values:
                    semantic_features[feature] = float(np.mean(values))
                    semantic_values[feature] = values
            
            # Calculate average combined features
            combined_features = {}
            combined_values = {}  # Store raw values for effect size calculation
            for feature in self.key_combined_features:
                values = [case['combined_features'].get(feature, None) for case in cases]
                values = [v for v in values if v is not None]
                if values:
                    combined_features[feature] = float(np.mean(values))
                    combined_values[feature] = values
            
            # Calculate average percentile rankings
            percentile_rankings = {}
            for feature in self.key_acoustic_features + self.key_semantic_features + self.key_combined_features:
                values = [case['percentile_rankings'].get(feature, None) for case in cases]
                values = [v for v in values if v is not None]
                if values:
                    percentile_rankings[feature] = float(np.mean(values))
            
            # Calculate average MAD effects
            mad_effects = {}
            for feature in self.key_acoustic_features + self.key_semantic_features + self.key_combined_features:
                values = [case['mad_effect_sizes'].get(feature, None) for case in cases]
                values = [v for v in values if v is not None]
                if values:
                    mad_effects[feature] = float(np.mean(values))
            
            group_metrics[outcome] = {
                'count': len(cases),
                'acoustic_features': acoustic_features,
                'semantic_features': semantic_features,
                'combined_features': combined_features,
                'percentile_rankings': percentile_rankings,
                'mad_effects': mad_effects,
                'raw_values': {
                    'acoustic': acoustic_values,
                    'semantic': semantic_values,
                    'combined': combined_values
                }
            }
        
        # Calculate effect sizes between groups
        effect_sizes = self._calculate_group_effect_sizes(group_metrics)
        
        # Save as JSON
        json_path = os.path.join(output_dir, "comparative_analysis.json")
        
        # Remove raw values before saving (they're only needed for effect size calculation)
        save_metrics = {}
        for outcome, metrics in group_metrics.items():
            save_metrics[outcome] = {k: v for k, v in metrics.items() if k != 'raw_values'}
        
        # Add effect sizes to save data
        save_data = {
            'group_metrics': save_metrics,
            'effect_sizes': effect_sizes
        }
        
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Save as CSV for easier use
        rows = []
        
        for outcome, metrics in group_metrics.items():
            row = {
                'rating_outcome': outcome,
                'count': metrics['count']
            }
            
            # Add average acoustic features
            for feature, value in metrics['acoustic_features'].items():
                row[f"avg_{feature}"] = value
            
            # Add average semantic features
            for feature, value in metrics['semantic_features'].items():
                row[f"avg_{feature}"] = value
            
            # Add average combined features
            for feature, value in metrics['combined_features'].items():
                row[f"avg_{feature}"] = value
            
            # Add average percentile rankings
            for feature, value in metrics['percentile_rankings'].items():
                row[f"avg_{feature}_percentile"] = value
            
            # Add average MAD effects
            for feature, value in metrics['mad_effects'].items():
                row[f"avg_{feature}_mad_effect"] = value
            
            rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "comparative_analysis.csv")
        df.to_csv(csv_path, index=False)
        
        # Save effect sizes as separate CSV
        effect_size_rows = []
        for comparison, features in effect_sizes.items():
            for feature, d in features.items():
                effect_size_rows.append({
                    'comparison': comparison,
                    'feature': feature,
                    'cohens_d': d,
                    'effect_size_interpretation': self._interpret_effect_size(d)
                })
        
        if effect_size_rows:
            effect_size_df = pd.DataFrame(effect_size_rows)
            effect_size_csv_path = os.path.join(output_dir, "effect_sizes.csv")
            effect_size_df.to_csv(effect_size_csv_path, index=False)
            logger.info(f"Effect sizes saved to {effect_size_csv_path}")
        
        logger.info(f"Comparative analysis saved to {csv_path}")
    
    def _calculate_group_effect_sizes(self, group_metrics: Dict) -> Dict:
        """
        Calculate Cohen's d effect sizes between rating outcome groups
        
        Args:
            group_metrics: Dictionary of group metrics with raw values
            
        Returns:
            Dictionary of effect sizes
        """
        effect_sizes = {}
        
        # Get available groups
        groups = list(group_metrics.keys())
        
        # Calculate effect sizes for key comparisons
        comparisons = []
        
        # Always compare downgrade vs affirm if both exist
        if 'downgrade' in groups and 'affirm' in groups:
            comparisons.append(('downgrade', 'affirm'))
        
        # Compare upgrade vs affirm if both exist
        if 'upgrade' in groups and 'affirm' in groups:
            comparisons.append(('upgrade', 'affirm'))
        
        # Compare downgrade vs upgrade if both exist
        if 'downgrade' in groups and 'upgrade' in groups:
            comparisons.append(('downgrade', 'upgrade'))
        
        # Perform comparisons
        for group1, group2 in comparisons:
            comparison_key = f"{group1}_vs_{group2}"
            effect_sizes[comparison_key] = {}
            
            # Calculate effect sizes for acoustic features
            for feature in self.key_acoustic_features:
                if (feature in group_metrics[group1]['raw_values']['acoustic'] and 
                    feature in group_metrics[group2]['raw_values']['acoustic']):
                    
                    values1 = np.array(group_metrics[group1]['raw_values']['acoustic'][feature])
                    values2 = np.array(group_metrics[group2]['raw_values']['acoustic'][feature])
                    
                    if len(values1) > 0 and len(values2) > 0:
                        d = self._calculate_cohens_d(values1, values2)
                        if not np.isnan(d):
                            effect_sizes[comparison_key][feature] = d
            
            # Calculate effect sizes for semantic features
            for feature in self.key_semantic_features:
                if (feature in group_metrics[group1]['raw_values']['semantic'] and 
                    feature in group_metrics[group2]['raw_values']['semantic']):
                    
                    values1 = np.array(group_metrics[group1]['raw_values']['semantic'][feature])
                    values2 = np.array(group_metrics[group2]['raw_values']['semantic'][feature])
                    
                    if len(values1) > 0 and len(values2) > 0:
                        d = self._calculate_cohens_d(values1, values2)
                        if not np.isnan(d):
                            effect_sizes[comparison_key][feature] = d
            
            # Calculate effect sizes for combined features
            for feature in self.key_combined_features:
                if (feature in group_metrics[group1]['raw_values']['combined'] and 
                    feature in group_metrics[group2]['raw_values']['combined']):
                    
                    values1 = np.array(group_metrics[group1]['raw_values']['combined'][feature])
                    values2 = np.array(group_metrics[group2]['raw_values']['combined'][feature])
                    
                    if len(values1) > 0 and len(values2) > 0:
                        d = self._calculate_cohens_d(values1, values2)
                        if not np.isnan(d):
                            effect_sizes[comparison_key][feature] = d
        
        return effect_sizes
    
    def _interpret_effect_size(self, d: float) -> str:
        """
        Interpret Cohen's d effect size
        
        Args:
            d: Cohen's d value
            
        Returns:
            String interpretation
        """
        abs_d = abs(d)
        
        if abs_d < 0.2:
            size = "negligible"
        elif abs_d < 0.5:
            size = "small"
        elif abs_d < 0.8:
            size = "medium"
        else:
            size = "large"
        
        direction = "positive" if d > 0 else "negative"
        
        return f"{size} {direction}"
    
    def _export_segment_analysis(self, 
                               case_studies: List[Dict], 
                               output_dir: str):
        """
        Export segment analysis data if available
        
        Args:
            case_studies: List of case study dictionaries
            output_dir: Output directory
        """
        # Check if we have segment analysis
        has_segments = False
        for case in case_studies:
            if 'audio_analysis' in case and 'high_variability_segments' in case['audio_analysis']:
                has_segments = True
                break
        
        if not has_segments:
            return
        
        # Create segments directory
        segments_dir = os.path.join(output_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)
        
        # Collect all segments
        all_segments = []
        
        for case in case_studies:
            if 'audio_analysis' not in case or 'high_variability_segments' not in case['audio_analysis']:
                continue
                
            file_id = case['file_id']
            
            # Process high variability segments
            for segment in case['audio_analysis']['high_variability_segments']:
                segment_data = segment.copy()
                segment_data['file_id'] = file_id
                segment_data['segment_type'] = 'high_variability'
                all_segments.append(segment_data)
            
            # Process low variability segments
            if 'low_variability_segments' in case['audio_analysis']:
                for segment in case['audio_analysis']['low_variability_segments']:
                    segment_data = segment.copy()
                    segment_data['file_id'] = file_id
                    segment_data['segment_type'] = 'low_variability'
                    all_segments.append(segment_data)
        
        # Create DataFrame and save to CSV
        if all_segments:
            df = pd.DataFrame(all_segments)
            csv_path = os.path.join(segments_dir, "all_segments.csv")
            df.to_csv(csv_path, index=False)
            
            # Also save segments for each file
            for file_id in set(df['file_id']):
                file_segments = df[df['file_id'] == file_id]
                file_csv_path = os.path.join(segments_dir, f"{file_id}_segments.csv")
                file_segments.to_csv(file_csv_path, index=False)
            
            logger.info(f"Segment analysis saved to {segments_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate case studies for earnings call acoustic analysis"
    )
    parser.add_argument("--features_dir", type=str, required=True,
                       help="Directory containing combined features")
    parser.add_argument("--audio_dir", type=str, default=None,
                       help="Directory containing processed audio files")
    parser.add_argument("--ratings_file", type=str, default=None,
                       help="Path to ratings CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save case studies")
    parser.add_argument("--num_cases", type=int, default=5,
                       help="Number of case studies to generate")
    parser.add_argument("--selection", type=str, default="combined",
                       choices=["acoustic", "semantic", "combined", "rating_outcome"],
                       help="Criteria for selecting case studies")
    parser.add_argument("--bootstrap", type=int, default=10000,
                       help="Number of bootstrap iterations for confidence intervals")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for intervals")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize case study generator
    generator = CaseStudyGenerator(
        num_case_studies=args.num_cases,
        selection_criteria=args.selection,
        n_bootstrap=args.bootstrap,
        confidence_level=args.confidence,
        random_seed=args.random_seed
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        logger.info("Loading features and ratings data...")
        features_df, ratings_df = generator.load_data(
            args.features_dir,
            args.audio_dir,
            args.ratings_file
        )
        
        # Prepare analysis data
        logger.info("Preparing analysis data...")
        analysis_df = generator.prepare_analysis_data(features_df, ratings_df)
        
        # Select case studies
        logger.info("Selecting case studies...")
        case_study_ids = generator.select_case_studies(analysis_df)
        
        # Generate case studies
        logger.info("Generating case studies...")
        case_studies = []
        
        for file_id in case_study_ids:
            logger.info(f"Generating case study for {file_id}...")
            case_study = generator.generate_case_study(
                file_id,
                analysis_df,
                args.audio_dir
            )
            case_studies.append(case_study)
        
        # Export case studies
        logger.info("Exporting case studies...")
        generator.export_case_studies(case_studies, args.output_dir)
        
        logger.info("Case study generation complete!")
        logger.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error in case study generation: {e}", exc_info=True)


if __name__ == "__main__":
    main()