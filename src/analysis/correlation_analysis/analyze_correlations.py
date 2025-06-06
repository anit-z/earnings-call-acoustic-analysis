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
                 correction_method: str = 'fdr_bh'):
        """
        Initialize correlation analyzer
        
        Args:
            significance_threshold: p-value threshold for significance
            correction_method: Method for multiple comparisons correction
        """
        self.significance_threshold = significance_threshold
        self.correction_method = correction_method
        
        # Key acoustic features for correlation analysis
        self.key_acoustic_features = [
            'f0_cv', 
            'f0_std', 
            'jitter_local', 
            'speech_rate', 
            'pause_frequency', 
            'acoustic_volatility_index'
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
                        columns=rating_cols
                    )
                    
                    for fc in feature_cols:
                        for rc in rating_cols:
                            # Calculate Pearson correlation if possible
                            if analysis_df[fc].notna().all() and analysis_df[rc].notna().all():
                                corr, p_value = stats.pearsonr(
                                    analysis_df[fc],
                                    analysis_df[rc]
                                )
                                feature_rating_corr.loc[fc, rc] = corr
                            else:
                                feature_rating_corr.loc[fc, rc] = np.nan
                    
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
            full_corr = analysis_df[key_cols].corr()
            correlation_matrices['full'] = full_corr
        
        # 4. Calculate acoustic-acoustic and semantic-semantic correlations
        acoustic_cols = [col for col in self.key_acoustic_features if col in analysis_df.columns]
        semantic_cols = [col for col in self.key_semantic_features if col in analysis_df.columns]
        
        if acoustic_cols and len(acoustic_cols) > 1:
            acoustic_corr = analysis_df[acoustic_cols].corr()
            correlation_matrices['acoustic_acoustic'] = acoustic_corr
        
        if semantic_cols and len(semantic_cols) > 1:
            semantic_corr = analysis_df[semantic_cols].corr()
            correlation_matrices['semantic_semantic'] = semantic_corr
        
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
                columns=corr_matrix.columns
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
                            # Remove rows with missing values in either column
                            valid_data = analysis_df[[i, j]].dropna()
                            
                            if len(valid_data) > 2:  # Need at least 3 points for correlation
                                try:
                                    corr, p_value = stats.pearsonr(
                                        valid_data[i],
                                        valid_data[j]
                                    )
                                    p_value_matrix.loc[i, j] = p_value
                                except:
                                    p_value_matrix.loc[i, j] = np.nan
                            else:
                                p_value_matrix.loc[i, j] = np.nan
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
                        # Check if p-value is significant
                        if (p_matrix.loc[i, j] <= self.significance_threshold and 
                            not pd.isna(p_matrix.loc[i, j])):
                            
                            significant_pairs.append({
                                'feature1': i,
                                'feature2': j,
                                'correlation': float(corr_matrix.loc[i, j]),
                                'p_value': float(p_matrix.loc[i, j]),
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
                    columns=semantic_cols
                )
                
                for ac in acoustic_cols:
                    for sc in semantic_cols:
                        # Calculate correlation if possible
                        valid_data = sector_df[[ac, sc]].dropna()
                        
                        if len(valid_data) > 2:  # Need at least 3 points for correlation
                            try:
                                corr, p_value = stats.pearsonr(
                                    valid_data[ac],
                                    valid_data[sc]
                                )
                                acoustic_semantic_corr.loc[ac, sc] = corr
                            except:
                                acoustic_semantic_corr.loc[ac, sc] = np.nan
                        else:
                            acoustic_semantic_corr.loc[ac, sc] = np.nan
                
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
        
        # Add summary of significant correlations
        report.append("## 1. SIGNIFICANT CORRELATIONS SUMMARY")
        report.append("-" * 50)
        
        for matrix_name, sig_corrs in significant_correlations.items():
            if not sig_corrs:
                continue
                
            report.append(f"\n### 1.1 {matrix_name.replace('_', '-').title()} Correlations")
            report.append(f"Total significant correlations: {len(sig_corrs)}")
            
            # Add top significant correlations
            top_n = min(10, len(sig_corrs))
            report.append(f"\nTop {top_n} significant correlations:")
            
            for i, corr in enumerate(sig_corrs[:top_n], 1):
                feat1 = corr['feature1'].replace('_', ' ').title()
                feat2 = corr['feature2'].replace('_', ' ').title()
                r = corr['correlation']
                p = corr['p_value']
                
                direction = "positive" if r > 0 else "negative"
                strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
                
                report.append(f"{i}. **{feat1}** and **{feat2}**: {r:.3f} ({strength} {direction}, p={p:.4f})")
        
        # Add acoustic-semantic correlations
        if 'acoustic_semantic' in correlation_matrices:
            report.append("\n## 2. ACOUSTIC-SEMANTIC CORRELATIONS")
            report.append("-" * 50)
            
            matrix = correlation_matrices['acoustic_semantic']
            
            if not matrix.empty:
                # Try to find strongest correlations safely
                try:
                    # Find strongest correlations
                    abs_matrix = matrix.abs()
                    unstacked = abs_matrix.unstack()
                    unstacked_sorted = unstacked.sort_values(ascending=False)
                    max_indices = unstacked_sorted.index[:5]
                    
                    report.append("\nStrongest acoustic-semantic correlations:")
                    
                    for idx in max_indices:
                        if not isinstance(idx, tuple) or len(idx) != 2:
                            continue
                            
                        feat1, feat2 = idx
                        
                        # Check if indices exist in matrix
                        if feat1 not in matrix.index or feat2 not in matrix.columns:
                            continue
                            
                        r = matrix.loc[feat1, feat2]
                        
                        # Safely access p-value if available
                        p = None
                        if ('acoustic_semantic' in p_value_matrices and 
                            feat1 in p_value_matrices['acoustic_semantic'].index and 
                            feat2 in p_value_matrices['acoustic_semantic'].columns):
                            p = p_value_matrices['acoustic_semantic'].loc[feat1, feat2]
                        
                        feat1_name = feat1.replace('_', ' ').title()
                        feat2_name = feat2.replace('_', ' ').title()
                        
                        direction = "positive" if r > 0 else "negative"
                        strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
                        
                        sig_status = ""
                        if p is not None and not pd.isna(p) and p <= self.significance_threshold:
                            sig_status = " (significant)"
                        
                        p_str = f", p={p:.4f}" if p is not None and not pd.isna(p) else ""
                        report.append(f"- **{feat1_name}** and **{feat2_name}**: {r:.3f} ({strength} {direction}{p_str}){sig_status}")
                except Exception as e:
                    logger.warning(f"Error finding strongest correlations: {e}")
                    report.append("\nError analyzing strongest correlations. See correlation matrices for details.")
        
        # Add feature-rating correlations
        if 'feature_rating' in correlation_matrices:
            report.append("\n## 3. FEATURE-RATING CORRELATIONS")
            report.append("-" * 50)
            
            matrix = correlation_matrices['feature_rating']
            
            if not matrix.empty:
                # Try to extract correlations with downgrades safely
                try:
                    # Extract correlations with downgrades
                    if 'is_downgrade' in matrix.columns:
                        downgrade_corrs = matrix['is_downgrade'].dropna().sort_values(ascending=False)
                        
                        report.append("\nFeatures most correlated with downgrades:")
                        for feat, r in downgrade_corrs.head(5).items():
                            # Safely access p-value if available
                            p = None
                            if ('feature_rating' in p_value_matrices and 
                                feat in p_value_matrices['feature_rating'].index and 
                                'is_downgrade' in p_value_matrices['feature_rating'].columns):
                                p = p_value_matrices['feature_rating'].loc[feat, 'is_downgrade']
                            
                            feat_name = feat.replace('_', ' ').title()
                            
                            direction = "positive" if r > 0 else "negative"
                            strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
                            
                            sig_status = ""
                            if p is not None and not pd.isna(p) and p <= self.significance_threshold:
                                sig_status = " (significant)"
                            
                            p_str = f", p={p:.4f}" if p is not None and not pd.isna(p) else ""
                            report.append(f"- **{feat_name}**: {r:.3f} ({strength} {direction}{p_str}){sig_status}")
                except Exception as e:
                    logger.warning(f"Error analyzing downgrade correlations: {e}")
                
                # Try to extract correlations with upgrades safely
                try:
                    # Extract correlations with upgrades
                    if 'is_upgrade' in matrix.columns:
                        upgrade_corrs = matrix['is_upgrade'].dropna().sort_values(ascending=False)
                        
                        report.append("\nFeatures most correlated with upgrades:")
                        for feat, r in upgrade_corrs.head(5).items():
                            # Safely access p-value if available
                            p = None
                            if ('feature_rating' in p_value_matrices and 
                                feat in p_value_matrices['feature_rating'].index and 
                                'is_upgrade' in p_value_matrices['feature_rating'].columns):
                                p = p_value_matrices['feature_rating'].loc[feat, 'is_upgrade']
                            
                            feat_name = feat.replace('_', ' ').title()
                            
                            direction = "positive" if r > 0 else "negative"
                            strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
                            
                            sig_status = ""
                            if p is not None and not pd.isna(p) and p <= self.significance_threshold:
                                sig_status = " (significant)"
                            
                            p_str = f", p={p:.4f}" if p is not None and not pd.isna(p) else ""
                            report.append(f"- **{feat_name}**: {r:.3f} ({strength} {direction}{p_str}){sig_status}")
                except Exception as e:
                    logger.warning(f"Error analyzing upgrade correlations: {e}")
        
        # Add pattern-based correlation interpretation
        if self.fusion_summary and 'communication_patterns' in self.fusion_summary:
            report.append("\n## 4. PATTERN-BASED CORRELATION INTERPRETATION")
            report.append("-" * 50)
            
            patterns = self.fusion_summary['communication_patterns']
            top_correlations = self.fusion_summary.get('top_acoustic_semantic_correlations', {})
            
            report.append("\nCommunication pattern distribution:")
            for pattern, count in patterns.items():
                pattern_name = pattern.replace('_', ' ').title()
                report.append(f"- {pattern_name}: {count}")
            
            report.append("\nTop acoustic-semantic correlations from pattern analysis:")
            for pair, corr in top_correlations.items():
                # Extract feature names from correlation pair string
                parts = pair.split('_vs_')
                if len(parts) == 2:
                    feat1 = parts[0].replace('_', ' ').title()
                    feat2 = parts[1].replace('_', ' ').title()
                    
                    direction = "positive" if corr > 0 else "negative"
                    strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
                    
                    report.append(f"- **{feat1}** and **{feat2}**: {corr:.3f} ({strength} {direction})")
        
        # Add methodological notes
        report.append("\n## 5. METHODOLOGICAL NOTES")
        report.append("-" * 50)
        report.append(f"""
This correlation analysis implements a comprehensive framework for examining relationships in acoustic-semantic-rating data:

1. Multiple Comparison Correction: P-values are adjusted using the {self.correction_method} method to control for false discovery rate.

2. Significance Threshold: Correlations are considered significant at p < {self.significance_threshold} after correction.

3. Effect Size Interpretation: Correlation strength is categorized as:
   - Strong: |r| > 0.5
   - Moderate: 0.3 < |r| < 0.5
   - Weak: |r| < 0.3

4. Sample Size Considerations: Given the small sample size, correlations should be interpreted cautiously, particularly for sector-specific analyses.

5. Convergent Evidence: Findings are most reliable when supported by multiple correlation indicators and aligned with theoretical expectations.
""")
        
        # Write report
        with open(os.path.join(output_dir, 'correlation_analysis_report.md'), 'w') as f:
            f.write('\n'.join(report))
    
    def create_visualizations(self,
                            correlation_matrices: Dict[str, pd.DataFrame],
                            significant_correlations: Dict,
                            output_dir: str):
        """
        Create correlation visualizations
        
        Args:
            correlation_matrices: Dictionary of correlation matrices
            significant_correlations: Dictionary of significant correlations
            output_dir: Output directory for visualizations
        """
        # Create output directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Create heatmaps for correlation matrices
        for name, matrix in correlation_matrices.items():
            if matrix.empty:
                continue
            
            try:
                # Create heatmap
                plt.figure(figsize=(12, 10))
                
                # Rename indices and columns for better readability
                matrix_plot = matrix.copy()
                matrix_plot.index = [idx.replace('_', ' ').title() for idx in matrix_plot.index]
                matrix_plot.columns = [col.replace('_', ' ').title() for col in matrix_plot.columns]
                
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
                
                sns.heatmap(matrix_plot, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                          mask=mask,
                          vmin=-1, vmax=1)
                
                plt.title(f"{name.replace('_', '-').title()} Correlations", fontsize=14)
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
                
                # Create bar chart
                plt.figure(figsize=(12, 8))
                
                # Prepare data
                labels = [f"{c['feature1'].replace('_', ' ').title()} & {c['feature2'].replace('_', ' ').title()}" 
                        for c in top_corrs]
                values = [c['correlation'] for c in top_corrs]
                colors = ['red' if v < 0 else 'blue' for v in values]
                
                # Sort by absolute correlation
                sorted_indices = np.argsort(np.abs(values))[::-1]
                labels = [labels[i] for i in sorted_indices]
                values = [values[i] for i in sorted_indices]
                colors = [colors[i] for i in sorted_indices]
                
                # Create horizontal bar chart
                bars = plt.barh(range(len(values)), values, color=colors, alpha=0.7)
                
                # Add labels
                plt.yticks(range(len(labels)), labels)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add correlation values
                for i, bar in enumerate(bars):
                    value = values[i]
                    color = 'white' if abs(value) > 0.5 else 'black'
                    ha = 'left' if value < 0 else 'right'
                    x_pos = value + (0.02 if value < 0 else -0.02)
                    plt.text(x_pos, i, f"{value:.3f}", ha=ha, va='center', color=color, fontsize=10)
                
                plt.xlabel('Correlation Coefficient (r)')
                plt.title(f"Top {top_n} {name.replace('_', '-').title()} Correlations", fontsize=14)
                plt.xlim(-1, 1)
                plt.grid(True, axis='x', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"{name}_top_correlations.png"), dpi=300)
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating bar chart for {name}: {e}")
        
        # 3. Create scatter plot for key correlations
        if self.pca_results is not None and 'PC1' in self.pca_results.columns and 'PC2' in self.pca_results.columns:
            try:
                plt.figure(figsize=(12, 10))
                
                # Create scatter plot of PCA results
                if 'communication_pattern' in self.pca_results.columns:
                    # Define color palette
                    palette = {
                        'high_stress': 'red',
                        'moderate_stress': 'orange',
                        'high_excitement': 'green',
                        'moderate_excitement': 'lightgreen',
                        'baseline_stability': 'blue',
                        'mixed_pattern': 'gray'
                    }
                    
                    # Filter palette to include only existing patterns
                    unique_patterns = self.pca_results['communication_pattern'].unique()
                    used_palette = {k: v for k, v in palette.items() if k in unique_patterns}
                    
                    # Create scatter plot with pattern colors
                    sns.scatterplot(
                        data=self.pca_results, 
                        x='PC1', y='PC2', 
                        hue='communication_pattern',
                        palette=used_palette,
                        s=100, alpha=0.7
                    )
                    
                    # Get explained variance from fusion summary
                    if self.fusion_summary and 'pca_explained_variance' in self.fusion_summary:
                        explained_var = self.fusion_summary['pca_explained_variance']
                        if len(explained_var) >= 2:
                            plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
                            plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
                    
                    # Add file_id labels
                    for i, row in self.pca_results.iterrows():
                        plt.text(row['PC1'], row['PC2'], row['file_id'], fontsize=8)
                    
                    plt.title("PCA of Combined Acoustic-Semantic Features", fontsize=14)
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
    
    args = parser.parse_args()
    
    # Initialize correlation analyzer
    analyzer = CorrelationAnalyzer(
        significance_threshold=args.significance,
        correction_method=args.correction
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