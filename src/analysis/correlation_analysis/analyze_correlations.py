#!/usr/bin/env python3
"""
Correlation Analysis for Earnings Call Acoustic Features
Analyzes relationships between acoustic features, semantic features, and rating outcomes
Following contemporary statistical framework for financial-acoustic research
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
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Correlation analysis for acoustic-semantic-rating relationships
    Implements methodological triangulation through correlation analysis
    """
    
    def __init__(self, 
                 significance_threshold: float = 0.05,
                 correction_method: str = 'fdr_bh',
                 random_seed: int = 42):
        """
        Initialize correlation analyzer
        
        Args:
            significance_threshold: p-value threshold for significance
            correction_method: Method for multiple comparisons correction
            random_seed: Random seed for reproducibility
        """
        self.significance_threshold = significance_threshold
        self.correction_method = correction_method
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Key acoustic features for correlation analysis (from thesis)
        self.key_acoustic_features = [
            'f0_cv',      # F0 Coefficient of Variation - Primary feature
            'f0_std',     # F0 Standard Deviation - Complementary pitch measure
            'pause_frequency',  # Pause Frequency - Temporal stress indicator
            'jitter_local'      # Jitter Local - Voice quality measure
        ]
        
        # Key semantic features for correlation analysis
        self.key_semantic_features = [
            'sentiment_positive',
            'sentiment_negative',
            'sentiment_neutral',
            'sentiment_variability'
        ]
        
        # Key combined features for correlation analysis
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
        
        # Acoustic-semantic convergence thresholds (from thesis)
        self.correlation_thresholds = {
            'strong': 0.6,
            'moderate': 0.3,
            'weak': 0.0
        }
        
        # Track constant features
        self.constant_features = set()
    
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
        logger.info(f"Loaded combined features: {len(features_df)} calls, {features_df.shape[1]} features")
        
        # Ensure file_id is string type
        if 'file_id' in features_df.columns:
            features_df['file_id'] = features_df['file_id'].astype(str)
        
        # Load pre-computed correlations if available
        corr_path = Path(features_dir) / "acoustic_semantic_correlations.csv"
        if corr_path.exists():
            logger.info(f"Found pre-computed correlations: {corr_path}")
            self.precomputed_correlations = pd.read_csv(corr_path, index_col=0)
        else:
            self.precomputed_correlations = None
        
        # Load PCA results if available
        pca_path = Path(features_dir) / "pca_results.csv"
        if pca_path.exists():
            logger.info(f"Found PCA results: {pca_path}")
            self.pca_results = pd.read_csv(pca_path)
        else:
            self.pca_results = None
        
        # Load feature importance if available
        importance_path = Path(features_dir) / "feature_importance.csv"
        if importance_path.exists():
            logger.info(f"Found feature importance: {importance_path}")
            self.feature_importance = pd.read_csv(importance_path, index_col=0)
        else:
            self.feature_importance = None
        
        # Load fusion summary if available
        summary_path = Path(features_dir) / "fusion_summary.json"
        if summary_path.exists():
            logger.info(f"Found fusion summary: {summary_path}")
            with open(summary_path, 'r') as f:
                self.fusion_summary = json.load(f)
        else:
            self.fusion_summary = None
        
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
        Prepare data for correlation analysis
        
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
        
        # Create binary rating outcome columns
        if 'composite_outcome' in merged_df.columns:
            # Create one-hot encoding for outcomes
            merged_df['is_downgrade'] = (merged_df['composite_outcome'] == 'downgrade').astype(int)
            merged_df['is_upgrade'] = (merged_df['composite_outcome'] == 'upgrade').astype(int)
            merged_df['is_affirm'] = (merged_df['composite_outcome'] == 'affirm').astype(int)
        
        return merged_df
    
    def is_constant_array(self, arr: np.ndarray) -> bool:
        """
        Check if an array is constant (all values are the same)
        
        Args:
            arr: Array to check
            
        Returns:
            True if array is constant, False otherwise
        """
        # Remove NaN values
        clean_arr = arr[~np.isnan(arr)]
        if len(clean_arr) == 0:
            return True
        return np.all(clean_arr == clean_arr[0])
    
    def classify_correlation_strength(self, r: float) -> str:
        """
        Classify correlation strength based on thesis thresholds
        
        Args:
            r: Correlation coefficient
            
        Returns:
            String classification of correlation strength
        """
        abs_r = abs(r)
        if abs_r >= self.correlation_thresholds['strong']:
            return 'strong'
        elif abs_r >= self.correlation_thresholds['moderate']:
            return 'moderate'
        else:
            return 'weak'
    
    def calculate_correlations_with_bootstrap(self, 
                                            x: np.ndarray, 
                                            y: np.ndarray, 
                                            n_bootstrap: int = 1000,
                                            feature_names: Optional[Tuple[str, str]] = None) -> Dict:
        """
        Calculate correlation with bootstrap confidence intervals
        
        Args:
            x: First variable
            y: Second variable
            n_bootstrap: Number of bootstrap iterations
            feature_names: Optional tuple of feature names for logging
            
        Returns:
            Dictionary with correlation, p-value, and confidence intervals
        """
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:  # Need at least 3 points
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }
        
        # Check for constant arrays
        if self.is_constant_array(x_clean) or self.is_constant_array(y_clean):
            if feature_names:
                if self.is_constant_array(x_clean):
                    self.constant_features.add(feature_names[0])
                if self.is_constant_array(y_clean):
                    self.constant_features.add(feature_names[1])
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }
        
        # Calculate observed correlation with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
            try:
                corr, p_value = stats.pearsonr(x_clean, y_clean)
            except:
                return {
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan
                }
        
        # Bootstrap for confidence intervals
        rng = np.random.RandomState(self.random_seed)
        bootstrap_corrs = []
        
        for _ in range(n_bootstrap):
            indices = rng.choice(len(x_clean), len(x_clean), replace=True)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
                try:
                    boot_corr, _ = stats.pearsonr(x_clean[indices], y_clean[indices])
                    bootstrap_corrs.append(boot_corr)
                except:
                    continue
        
        # Calculate confidence intervals
        if len(bootstrap_corrs) > 0:
            ci_lower = np.percentile(bootstrap_corrs, 2.5)
            ci_upper = np.percentile(bootstrap_corrs, 97.5)
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def calculate_correlations(self, analysis_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate comprehensive correlation matrices
        
        Args:
            analysis_df: Analysis DataFrame
            
        Returns:
            Dictionary of correlation matrices
        """
        correlation_matrices = {}
        
        # 1. Acoustic-Semantic Correlations
        if self.precomputed_correlations is not None:
            # Use pre-computed correlations if available
            acoustic_semantic_corr = self.precomputed_correlations
        else:
            # Calculate acoustic-semantic correlations
            acoustic_cols = [col for col in self.key_acoustic_features if col in analysis_df.columns]
            semantic_cols = [col for col in self.key_semantic_features if col in analysis_df.columns]
            
            if acoustic_cols and semantic_cols:
                # Calculate full correlation matrix
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
                    full_corr = analysis_df[acoustic_cols + semantic_cols].corr()
                
                # Extract acoustic-semantic correlations
                acoustic_semantic_corr = pd.DataFrame(
                    index=acoustic_cols,
                    columns=semantic_cols
                )
                
                for ac in acoustic_cols:
                    for sc in semantic_cols:
                        acoustic_semantic_corr.loc[ac, sc] = full_corr.loc[ac, sc]
            else:
                acoustic_semantic_corr = pd.DataFrame()
        
        correlation_matrices['acoustic_semantic'] = acoustic_semantic_corr
        
        # 2. Feature-Rating Correlations
        if 'is_downgrade' in analysis_df.columns:
            # Identify rating-related columns
            rating_cols = ['is_downgrade', 'is_upgrade', 'is_affirm']
            rating_cols = [col for col in rating_cols if col in analysis_df.columns]
            
            if rating_cols:
                # Identify feature columns
                feature_cols = self.key_acoustic_features + self.key_semantic_features + self.key_combined_features
                feature_cols = [col for col in feature_cols if col in analysis_df.columns]
                
                if feature_cols:
                    # Calculate correlation with rating outcomes
                    feature_rating_corr = pd.DataFrame(
                        index=feature_cols,
                        columns=rating_cols,
                        dtype=float  # Ensure float dtype
                    )
                    
                    for fc in feature_cols:
                        for rc in rating_cols:
                            # Calculate correlation with bootstrap CI
                            result = self.calculate_correlations_with_bootstrap(
                                analysis_df[fc].values,
                                analysis_df[rc].values,
                                feature_names=(fc, rc)
                            )
                            feature_rating_corr.loc[fc, rc] = result['correlation']
                    
                    correlation_matrices['feature_rating'] = feature_rating_corr
        
        # 3. Calculate full multi-factor correlation matrix
        # Select key columns for the full correlation matrix
        key_cols = (
            self.key_acoustic_features + 
            self.key_semantic_features + 
            self.key_combined_features
        )
        
        # Add rating columns if available
        if 'is_downgrade' in analysis_df.columns:
            key_cols += ['is_downgrade', 'is_upgrade', 'is_affirm']
        
        # Filter to columns that exist in the dataframe
        key_cols = [col for col in key_cols if col in analysis_df.columns]
        
        if key_cols:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
                full_corr = analysis_df[key_cols].corr()
            correlation_matrices['full'] = full_corr
        
        # 4. Calculate acoustic-acoustic and semantic-semantic correlations
        acoustic_cols = [col for col in self.key_acoustic_features if col in analysis_df.columns]
        semantic_cols = [col for col in self.key_semantic_features if col in analysis_df.columns]
        
        if acoustic_cols and len(acoustic_cols) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
                acoustic_corr = analysis_df[acoustic_cols].corr()
            correlation_matrices['acoustic_acoustic'] = acoustic_corr
        
        if semantic_cols and len(semantic_cols) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
                semantic_corr = analysis_df[semantic_cols].corr()
            correlation_matrices['semantic_semantic'] = semantic_corr
        
        # Log constant features if any were found
        if self.constant_features:
            logger.warning(f"Found constant features (all values identical): {', '.join(sorted(self.constant_features))}")
            logger.warning("Correlations involving constant features are set to NaN")
        
        return correlation_matrices
    
    def calculate_p_values(self, 
                         analysis_df: pd.DataFrame, 
                         correlation_matrices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate p-values for correlation matrices
        
        Args:
            analysis_df: Analysis DataFrame
            correlation_matrices: Dictionary of correlation matrices
            
        Returns:
            Dictionary of p-value matrices
        """
        p_value_matrices = {}
        
        for matrix_name, corr_matrix in correlation_matrices.items():
            if corr_matrix.empty:
                continue
            
            # Initialize p-value matrix
            p_value_matrix = pd.DataFrame(
                index=corr_matrix.index,
                columns=corr_matrix.columns,
                dtype=float  # Ensure float dtype
            )
            
            # Calculate p-values
            for i in corr_matrix.index:
                for j in corr_matrix.columns:
                    if i == j:
                        # Diagonal elements have p-value = 0
                        p_value_matrix.loc[i, j] = 0.0
                    else:
                        # Check if both columns exist in the dataframe
                        if i in analysis_df.columns and j in analysis_df.columns:
                            # Calculate correlation with bootstrap
                            result = self.calculate_correlations_with_bootstrap(
                                analysis_df[i].values,
                                analysis_df[j].values,
                                feature_names=(i, j)
                            )
                            p_value_matrix.loc[i, j] = result['p_value']
                        else:
                            p_value_matrix.loc[i, j] = np.nan
            
            p_value_matrices[matrix_name] = p_value_matrix
        
        return p_value_matrices
    
    def apply_multiple_testing_correction(self,
                                       p_value_matrices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply multiple testing correction to p-values
        
        Args:
            p_value_matrices: Dictionary of p-value matrices
            
        Returns:
            Dictionary of corrected p-value matrices
        """
        corrected_matrices = {}
        
        for matrix_name, p_matrix in p_value_matrices.items():
            if p_matrix.empty:
                continue
                
            # Create a copy for corrected values
            corrected_matrix = p_matrix.copy()
            
            # Extract all p-values (excluding diagonal)
            p_values = []
            indices = []
            
            for i in p_matrix.index:
                for j in p_matrix.columns:
                    if i != j and not pd.isna(p_matrix.loc[i, j]):
                        p_values.append(p_matrix.loc[i, j])
                        indices.append((i, j))
            
            if p_values:
                # Apply correction
                corrected_p_values = multipletests(
                    p_values,
                    alpha=self.significance_threshold,
                    method=self.correction_method
                )[1]
                
                # Update corrected matrix
                for (i, j), p_corr in zip(indices, corrected_p_values):
                    corrected_matrix.loc[i, j] = p_corr
                
                corrected_matrices[matrix_name] = corrected_matrix
        
        return corrected_matrices
    
    def identify_significant_correlations(self,
                                       correlation_matrices: Dict[str, pd.DataFrame],
                                       corrected_p_matrices: Dict[str, pd.DataFrame]) -> Dict:
        """
        Identify statistically significant correlations
        
        Args:
            correlation_matrices: Dictionary of correlation matrices
            corrected_p_matrices: Dictionary of corrected p-value matrices
            
        Returns:
            Dictionary of significant correlations
        """
        significant_correlations = {}
        
        for matrix_name, corr_matrix in correlation_matrices.items():
            if matrix_name not in corrected_p_matrices:
                continue
            
            p_matrix = corrected_p_matrices[matrix_name]
            
            # Find significant correlations
            significant_pairs = []
            
            for i in corr_matrix.index:
                for j in corr_matrix.columns:
                    if i != j:  # Skip diagonal
                        # Check if p-value is significant and correlation is not NaN
                        if (not pd.isna(corr_matrix.loc[i, j]) and
                            p_matrix.loc[i, j] <= self.significance_threshold and 
                            not pd.isna(p_matrix.loc[i, j])):
                            
                            correlation = corr_matrix.loc[i, j]
                            strength = self.classify_correlation_strength(correlation)
                            
                            significant_pairs.append({
                                'feature1': i,
                                'feature2': j,
                                'correlation': float(correlation),
                                'p_value': float(p_matrix.loc[i, j]),
                                'strength': strength,
                                'is_significant': True
                            })
            
            # Sort by absolute correlation
            significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            significant_correlations[matrix_name] = significant_pairs
        
        return significant_correlations
    
    def calculate_sector_specific_correlations(self, 
                                            analysis_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate sector-specific correlations
        
        Args:
            analysis_df: Analysis DataFrame
            
        Returns:
            Dictionary of sector-specific correlation matrices
        """
        if 'sector' not in analysis_df.columns:
            logger.warning("No sector column found. Skipping sector-specific correlations.")
            return {}
        
        sector_correlations = {}
        
        # Get unique sectors
        sectors = analysis_df['sector'].dropna().unique()
        
        for sector in sectors:
            # Get sector-specific data
            sector_df = analysis_df[analysis_df['sector'] == sector]
            
            if len(sector_df) < 3:  # Need at least 3 points for correlation
                logger.info(f"Insufficient data for sector '{sector}'. Skipping.")
                continue
            
            logger.info(f"Calculating correlations for sector '{sector}' (n={len(sector_df)})")
            
            # Define key columns for sector-specific correlation
            acoustic_cols = [col for col in self.key_acoustic_features if col in sector_df.columns]
            semantic_cols = [col for col in self.key_semantic_features if col in sector_df.columns]
            
            if acoustic_cols and semantic_cols:
                # Calculate acoustic-semantic correlations for sector
                acoustic_semantic_corr = pd.DataFrame(
                    index=acoustic_cols,
                    columns=semantic_cols,
                    dtype=float  # Ensure float dtype
                )
                
                for ac in acoustic_cols:
                    for sc in semantic_cols:
                        # Calculate correlation with bootstrap
                        result = self.calculate_correlations_with_bootstrap(
                            sector_df[ac].values,
                            sector_df[sc].values,
                            feature_names=(ac, sc)
                        )
                        acoustic_semantic_corr.loc[ac, sc] = result['correlation']
                
                sector_correlations[sector] = acoustic_semantic_corr
        
        return sector_correlations
    
    def create_correlation_tables(self,
                                correlation_matrices: Dict[str, pd.DataFrame],
                                p_value_matrices: Dict[str, pd.DataFrame],
                                corrected_p_matrices: Dict[str, pd.DataFrame],
                                significant_correlations: Dict,
                                output_dir: str):
        """
        Create correlation tables for reporting
        
        Args:
            correlation_matrices: Dictionary of correlation matrices
            p_value_matrices: Dictionary of p-value matrices
            corrected_p_matrices: Dictionary of corrected p-value matrices
            significant_correlations: Dictionary of significant correlations
            output_dir: Output directory for tables
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save correlation matrices
        for name, matrix in correlation_matrices.items():
            if not matrix.empty:
                matrix.to_csv(os.path.join(output_dir, f"{name}_correlations.csv"))
        
        # Save p-value matrices
        for name, matrix in p_value_matrices.items():
            if not matrix.empty:
                matrix.to_csv(os.path.join(output_dir, f"{name}_p_values.csv"))
        
        # Save corrected p-value matrices
        for name, matrix in corrected_p_matrices.items():
            if not matrix.empty:
                matrix.to_csv(os.path.join(output_dir, f"{name}_corrected_p_values.csv"))
        
        # Create significant correlations table
        for name, sig_corrs in significant_correlations.items():
            if sig_corrs:
                sig_df = pd.DataFrame(sig_corrs)
                sig_df.to_csv(os.path.join(output_dir, f"{name}_significant_correlations.csv"), index=False)
        
        # Create comprehensive correlation report
        self.create_correlation_report(
            correlation_matrices,
            p_value_matrices,
            corrected_p_matrices,
            significant_correlations,
            output_dir
        )
        
        logger.info(f"Correlation tables saved to {output_dir}")
    
    def create_correlation_report(self,
                                correlation_matrices: Dict[str, pd.DataFrame],
                                p_value_matrices: Dict[str, pd.DataFrame],
                                corrected_p_matrices: Dict[str, pd.DataFrame],
                                significant_correlations: Dict,
                                output_dir: str):
        """
        Create comprehensive correlation report
        
        Args:
            correlation_matrices: Dictionary of correlation matrices
            p_value_matrices: Dictionary of p-value matrices
            corrected_p_matrices: Dictionary of corrected p-value matrices
            significant_correlations: Dictionary of significant correlations
            output_dir: Output directory for report
        """
        report = []
        
        # Add title
        report.append("# CORRELATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Add reproducibility note
        report.append("## REPRODUCIBILITY NOTE")
        report.append("-" * 50)
        report.append(f"Random seed used for bootstrap procedures: {self.random_seed}")
        report.append(f"Bootstrap iterations for confidence intervals: 1000")
        report.append("")
        
        # Add constant features note if any
        if self.constant_features:
            report.append("## CONSTANT FEATURES WARNING")
            report.append("-" * 50)
            report.append("The following features have constant values (all samples identical):")
            for feat in sorted(self.constant_features):
                report.append(f"- {feat.replace('_', ' ').title()}")
            report.append("\nCorrelations involving these features are undefined (NaN).")
            report.append("")
        
        # Add summary of significant correlations
        report.append("## 1. SIGNIFICANT CORRELATIONS SUMMARY")
        report.append("-" * 50)
        
        for matrix_name, sig_corrs in significant_correlations.items():
            if not sig_corrs:
                continue
                
            report.append(f"\n### 1.1 {matrix_name.replace('_', '-').title()} Correlations")
            report.append(f"Total significant correlations: {len(sig_corrs)}")
            
            # Group by strength
            strong_corrs = [c for c in sig_corrs if c['strength'] == 'strong']
            moderate_corrs = [c for c in sig_corrs if c['strength'] == 'moderate']
            weak_corrs = [c for c in sig_corrs if c['strength'] == 'weak']
            
            if strong_corrs:
                report.append(f"\n**Strong correlations (|r| ≥ {self.correlation_thresholds['strong']}):**")
                for corr in strong_corrs[:5]:  # Top 5
                    feat1 = corr['feature1'].replace('_', ' ').title()
                    feat2 = corr['feature2'].replace('_', ' ').title()
                    r = corr['correlation']
                    p = corr['p_value']
                    direction = "positive" if r > 0 else "negative"
                    report.append(f"- {feat1} & {feat2}: r = {r:.3f} ({direction}, p = {p:.4f})")
            
            if moderate_corrs:
                report.append(f"\n**Moderate correlations ({self.correlation_thresholds['moderate']} ≤ |r| < {self.correlation_thresholds['strong']}):**")
                for corr in moderate_corrs[:5]:  # Top 5
                    feat1 = corr['feature1'].replace('_', ' ').title()
                    feat2 = corr['feature2'].replace('_', ' ').title()
                    r = corr['correlation']
                    p = corr['p_value']
                    direction = "positive" if r > 0 else "negative"
                    report.append(f"- {feat1} & {feat2}: r = {r:.3f} ({direction}, p = {p:.4f})")
        
        # Add acoustic-semantic correlations
        if 'acoustic_semantic' in correlation_matrices:
            report.append("\n## 2. ACOUSTIC-SEMANTIC CORRELATIONS")
            report.append("-" * 50)
            
            matrix = correlation_matrices['acoustic_semantic']
            
            if not matrix.empty:
                # Find convergent stress indicators (both acoustic and semantic negative)
                report.append("\n### 2.1 Convergent Stress Indicators")
                report.append("Looking for high correlations between stress-related acoustic features and negative sentiment:")
                
                # Check specific stress-related correlations
                stress_acoustic = ['f0_cv', 'f0_std', 'pause_frequency', 'jitter_local']
                stress_semantic = ['sentiment_negative', 'sentiment_variability']
                
                for ac in stress_acoustic:
                    for sc in stress_semantic:
                        if ac in matrix.index and sc in matrix.columns:
                            r = matrix.loc[ac, sc]
                            if not pd.isna(r):
                                strength = self.classify_correlation_strength(r)
                                ac_name = ac.replace('_', ' ').title()
                                sc_name = sc.replace('_', ' ').title()
                                direction = "positive" if r > 0 else "negative"
                                
                                # Check significance
                                sig_status = ""
                                if ('acoustic_semantic' in corrected_p_matrices and 
                                    ac in corrected_p_matrices['acoustic_semantic'].index and 
                                    sc in corrected_p_matrices['acoustic_semantic'].columns):
                                    p = corrected_p_matrices['acoustic_semantic'].loc[ac, sc]
                                    if not pd.isna(p) and p <= self.significance_threshold:
                                        sig_status = " **(significant)**"
                                
                                report.append(f"- {ac_name} & {sc_name}: r = {r:.3f} ({strength} {direction}){sig_status}")
        
        # Add methodological notes
        report.append("\n## 3. METHODOLOGICAL NOTES")
        report.append("-" * 50)
        report.append(f"""
This correlation analysis implements the methodology specified in the thesis:

1. **Acoustic-Semantic Convergence Thresholds** (as per Section 3.4):
   - Strong correlation: |r| ≥ {self.correlation_thresholds['strong']}
   - Moderate correlation: {self.correlation_thresholds['moderate']} ≤ |r| < {self.correlation_thresholds['strong']}
   - Weak correlation: |r| < {self.correlation_thresholds['moderate']}

2. **Multiple Comparison Correction**: P-values are adjusted using the {self.correction_method} method to control for false discovery rate.

3. **Significance Threshold**: Correlations are considered significant at p < {self.significance_threshold} after correction.

4. **Bootstrap Confidence Intervals**: Each correlation includes bootstrap-based confidence intervals (1000 iterations, seed={self.random_seed}).

5. **Sample Size Considerations**: Given n=24 with extreme class imbalance (21:2:1), correlations should be interpreted as exploratory baselines rather than definitive findings.

6. **Validation Approach**: FinBERT sentiment serves as a directional validator for acoustic stress markers, not as a fused feature.

7. **Constant Features**: Features with identical values across all samples cannot have meaningful correlations and are reported as NaN.
""")
        
        # Write report
        with open(os.path.join(output_dir, 'correlation_analysis_report.md'), 'w') as f:
            f.write('\n'.join(report))
    
    def create_visualizations(self,
                            correlation_matrices: Dict[str, pd.DataFrame],
                            significant_correlations: Dict,
                            output_dir: str):
        """
        Create correlation visualizations with consistent formatting
        
        Args:
            correlation_matrices: Dictionary of correlation matrices
            significant_correlations: Dictionary of significant correlations
            output_dir: Output directory for visualizations
        """
        # Create output directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Create heatmaps for correlation matrices
        for name, matrix in correlation_matrices.items():
            if matrix.empty:
                continue
            
            try:
                # Skip matrices with non-numeric data
                if matrix.dtypes.apply(lambda x: not np.issubdtype(x, np.number)).any():
                    logger.warning(f"Skipping heatmap for {name}: contains non-numeric data")
                    continue
                
                # Create heatmap with consistent formatting
                plt.figure(figsize=(6, 5))
                
                # Rename indices and columns for better readability
                matrix_plot = matrix.copy()
                matrix_plot.index = [idx.replace('_', ' ').title() for idx in matrix_plot.index]
                matrix_plot.columns = [col.replace('_', ' ').title() for col in matrix_plot.columns]
                
                # Replace any remaining non-numeric values with NaN
                matrix_plot = matrix_plot.apply(pd.to_numeric, errors='coerce')
                
                # Create mask for upper triangle if needed
                if name == 'full' or name == 'acoustic_acoustic' or name == 'semantic_semantic':
                    # Ensure matrix is square before creating mask
                    if matrix_plot.shape[0] == matrix_plot.shape[1]:
                        mask = np.zeros_like(matrix_plot, dtype=bool)
                        mask[np.triu_indices_from(mask, k=1)] = True
                    else:
                        mask = None
                else:
                    mask = None
                
                # Use consistent color palette
                cmap = sns.diverging_palette(220, 20, as_cmap=True)
                
                sns.heatmap(matrix_plot, annot=True, cmap=cmap, center=0, fmt='.2f',
                          mask=mask, vmin=-1, vmax=1, 
                          cbar_kws={'label': 'Correlation Coefficient'},
                          annot_kws={'fontsize': 7})
                
                plt.title(f"{name.replace('_', '-').title()} Correlations", 
                         fontsize=10, fontweight='normal', pad=10)
                
                # Adjust label sizes
                plt.xticks(fontsize=7, rotation=45, ha='right')
                plt.yticks(fontsize=7)
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"{name}_correlation_heatmap.png"), dpi=300)
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating heatmap for {name}: {e}")
        
        # 2. Create bar chart of top correlations
        for name, sig_corrs in significant_correlations.items():
            if not sig_corrs:
                continue
            
            try:
                # Take top correlations
                top_n = min(10, len(sig_corrs))
                top_corrs = sig_corrs[:top_n]
                
                # Create bar chart with consistent formatting
                plt.figure(figsize=(6, 5))
                
                # Prepare data
                labels = []
                values = []
                colors = []
                
                for c in top_corrs:
                    feat1 = c['feature1'].replace('_', ' ').title()
                    feat2 = c['feature2'].replace('_', ' ').title()
                    labels.append(f"{feat1}\n& {feat2}")
                    values.append(c['correlation'])
                    
                    # Color based on strength and direction
                    if c['strength'] == 'strong':
                        color = self.color_palette[7] if c['correlation'] < 0 else self.color_palette[5]
                    elif c['strength'] == 'moderate':
                        color = self.color_palette[6] if c['correlation'] < 0 else self.color_palette[3]
                    else:
                        color = self.color_palette[4]
                    colors.append(color)
                
                # Create horizontal bar chart
                bars = plt.barh(range(len(values)), values, color=colors, alpha=0.8)
                
                # Add labels
                plt.yticks(range(len(labels)), labels, fontsize=7)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add correlation values - position to avoid overlap
                for i, (bar, val) in enumerate(zip(bars, values)):
                    # Position text to avoid overlap with bar
                    if abs(val) < 0.1:
                        # For small values, place outside
                        x_pos = 0.15 if val > 0 else -0.15
                        ha = 'left' if val > 0 else 'right'
                    else:
                        # For larger values, place at the end of bar with padding
                        x_pos = val + (0.05 if val > 0 else -0.05)
                        ha = 'left' if val > 0 else 'right'
                    
                    plt.text(x_pos, i, f"{val:.3f}", ha=ha, va='center', 
                           fontsize=7, fontweight='normal')
                
                # Add threshold lines
                plt.axvline(x=self.correlation_thresholds['strong'], 
                          color=self.color_palette[5], linestyle='--', alpha=0.5, linewidth=1)
                plt.axvline(x=-self.correlation_thresholds['strong'], 
                          color=self.color_palette[7], linestyle='--', alpha=0.5, linewidth=1)
                plt.axvline(x=self.correlation_thresholds['moderate'], 
                          color=self.color_palette[3], linestyle='--', alpha=0.3, linewidth=1)
                plt.axvline(x=-self.correlation_thresholds['moderate'], 
                          color=self.color_palette[6], linestyle='--', alpha=0.3, linewidth=1)
                
                plt.xlabel('Correlation Coefficient (r)', fontsize=8, fontweight='normal')
                plt.title(f"Top {top_n} {name.replace('_', '-').title()} Correlations", 
                         fontsize=10, fontweight='normal', pad=10)
                plt.xlim(-1.1, 1.1)  # Extend limits for text
                plt.grid(True, axis='x', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"{name}_top_correlations.png"), dpi=300)
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating bar chart for {name}: {e}")
        
        # 3. Create scatter plot for key correlations
        if self.pca_results is not None and 'PC1' in self.pca_results.columns and 'PC2' in self.pca_results.columns:
            try:
                plt.figure(figsize=(6, 5))
                
                # Create scatter plot of PCA results
                if 'communication_pattern' in self.pca_results.columns:
                    # Define color mapping using palette
                    pattern_colors = {
                        'high_stress': self.color_palette[7],
                        'moderate_stress': self.color_palette[5],
                        'high_excitement': self.color_palette[2],
                        'moderate_excitement': self.color_palette[3],
                        'baseline_stability': self.color_palette[1],
                        'mixed_pattern': self.color_palette[4]
                    }
                    
                    # Filter palette to include only existing patterns
                    unique_patterns = self.pca_results['communication_pattern'].unique()
                    used_palette = {k: v for k, v in pattern_colors.items() if k in unique_patterns}
                    
                    # Create scatter plot with pattern colors
                    for pattern in unique_patterns:
                        if pattern in used_palette:
                            pattern_data = self.pca_results[self.pca_results['communication_pattern'] == pattern]
                            plt.scatter(pattern_data['PC1'], pattern_data['PC2'], 
                                      color=used_palette[pattern], 
                                      label=pattern.replace('_', ' ').title(),
                                      s=60, alpha=0.7, edgecolors=self.color_palette[6])
                    
                    # Get explained variance from fusion summary
                    if self.fusion_summary and 'pca_explained_variance' in self.fusion_summary:
                        explained_var = self.fusion_summary['pca_explained_variance']
                        if len(explained_var) >= 2:
                            plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)", 
                                     fontsize=8, fontweight='normal')
                            plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)", 
                                     fontsize=8, fontweight='normal')
                    
                    # Add file_id labels - position to avoid overlap
                    for i, row in self.pca_results.iterrows():
                        # Offset text slightly above point
                        plt.text(row['PC1'], row['PC2'] + 0.02, row['file_id'], 
                               fontsize=6, ha='center', va='bottom')
                    
                    plt.title("PCA of Combined Acoustic-Semantic Features", 
                            fontsize=10, fontweight='normal', pad=10)
                    plt.legend(loc='best', fontsize=7)
                    plt.grid(True, linestyle='--', alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, "pca_visualization.png"), dpi=300)
                    plt.close()
            except Exception as e:
                logger.warning(f"Error creating PCA visualization: {e}")
        
        logger.info(f"Visualizations saved to {viz_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Correlation analysis for earnings call features"
    )
    parser.add_argument("--features_dir", type=str, required=True,
                       help="Directory containing combined features")
    parser.add_argument("--ratings_file", type=str, default=None,
                       help="Path to ratings CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save correlation results")
    parser.add_argument("--significance", type=float, default=0.05,
                       help="Significance threshold for correlations")
    parser.add_argument("--correction", type=str, default="fdr_bh",
                       help="Method for multiple comparisons correction")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize correlation analyzer
    analyzer = CorrelationAnalyzer(
        significance_threshold=args.significance,
        correction_method=args.correction,
        random_seed=args.random_seed
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        logger.info("Loading features and ratings data...")
        features_df, ratings_df = analyzer.load_data(args.features_dir, args.ratings_file)
        
        # Prepare analysis data
        logger.info("Preparing analysis data...")
        analysis_df = analyzer.prepare_analysis_data(features_df, ratings_df)
        
        # Calculate correlations
        logger.info("Calculating correlation matrices...")
        correlation_matrices = analyzer.calculate_correlations(analysis_df)
        
        # Calculate p-values
        logger.info("Calculating p-values...")
        p_value_matrices = analyzer.calculate_p_values(analysis_df, correlation_matrices)
        
        # Apply multiple testing correction
        logger.info("Applying multiple testing correction...")
        corrected_p_matrices = analyzer.apply_multiple_testing_correction(p_value_matrices)
        
        # Identify significant correlations
        logger.info("Identifying significant correlations...")
        significant_correlations = analyzer.identify_significant_correlations(
            correlation_matrices, corrected_p_matrices
        )
        
        # Calculate sector-specific correlations
        logger.info("Calculating sector-specific correlations...")
        sector_correlations = analyzer.calculate_sector_specific_correlations(analysis_df)
        
        # Create correlation tables
        logger.info("Creating correlation tables...")
        analyzer.create_correlation_tables(
            correlation_matrices,
            p_value_matrices,
            corrected_p_matrices,
            significant_correlations,
            args.output_dir
        )
        
        # Create visualizations
        logger.info("Creating visualizations...")
        analyzer.create_visualizations(
            correlation_matrices,
            significant_correlations,
            args.output_dir
        )
        
        # Save sector-specific correlations
        if sector_correlations:
            sector_dir = os.path.join(args.output_dir, 'sectors')
            os.makedirs(sector_dir, exist_ok=True)
            
            for sector, matrix in sector_correlations.items():
                if not matrix.empty:
                    matrix.to_csv(os.path.join(sector_dir, f"{sector}_correlations.csv"))
            
            logger.info(f"Sector-specific correlations saved to {sector_dir}")
        
        logger.info("Correlation analysis complete!")
        logger.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}", exc_info=True)


if __name__ == "__main__":
    main()