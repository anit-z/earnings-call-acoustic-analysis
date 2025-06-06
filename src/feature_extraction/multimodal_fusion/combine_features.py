#!/usr/bin/env python3
"""
Multimodal Feature Fusion for Earnings Call Analysis
Combines acoustic and semantic features for directional validation
Implements acoustic-semantic correlation following circumplex model of emotion
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultimodalFeatureFusion:
    """
    Combines acoustic and semantic features for earnings call analysis
    Implements methodological triangulation through correlation rather than fusion
    """
    
    def __init__(self, normalize_features: bool = True):
        """
        Initialize multimodal feature fusion
        
        Args:
            normalize_features: Whether to normalize features before fusion
        """
        self.normalize_features = normalize_features
        
    def load_features(self, 
                     acoustic_dir: str, 
                     linguistic_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load acoustic and linguistic features
        
        Args:
            acoustic_dir: Directory containing acoustic features
            linguistic_dir: Directory containing linguistic features
            
        Returns:
            Tuple of (acoustic_features, linguistic_features) DataFrames
        """
        # Load acoustic features
        acoustic_path = Path(acoustic_dir) / "all_features.csv"
        if not acoustic_path.exists():
            logger.error(f"Acoustic features file not found: {acoustic_path}")
            raise FileNotFoundError(f"Acoustic features file not found: {acoustic_path}")
        
        acoustic_df = pd.read_csv(acoustic_path)
        logger.info(f"Loaded acoustic features: {len(acoustic_df)} calls, {acoustic_df.shape[1]} features")
        
        # Load linguistic features
        linguistic_path = Path(linguistic_dir) / "all_sentiment_results.csv"
        if not linguistic_path.exists():
            logger.error(f"Linguistic features file not found: {linguistic_path}")
            raise FileNotFoundError(f"Linguistic features file not found: {linguistic_path}")
        
        linguistic_df = pd.read_csv(linguistic_path)
        logger.info(f"Loaded linguistic features: {len(linguistic_df)} calls, {linguistic_df.shape[1]} features")
        
        return acoustic_df, linguistic_df
    
    def preprocess_features(self, 
                          acoustic_df: pd.DataFrame, 
                          linguistic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess features: handle missing values, outliers, normalization
        
        Args:
            acoustic_df: Acoustic features DataFrame
            linguistic_df: Linguistic features DataFrame
            
        Returns:
            Preprocessed (acoustic_df, linguistic_df)
        """
        # Filter out error cases
        if 'error' in acoustic_df.columns:
            acoustic_df = acoustic_df[acoustic_df['error'].isna()]
        
        if 'error' in linguistic_df.columns:
            linguistic_df = linguistic_df[linguistic_df['error'].isna()]
        
        # Ensure file_id is string type for proper matching
        acoustic_df['file_id'] = acoustic_df['file_id'].astype(str)
        linguistic_df['file_id'] = linguistic_df['file_id'].astype(str)
        
        # Select key features for fusion
        acoustic_features = [
            'file_id', 'f0_cv', 'f0_mean', 'f0_std', 'jitter_local', 'shimmer_local', 
            'hnr_mean', 'speech_rate', 'pause_frequency', 'spectral_centroid_mean',
            'acoustic_volatility_index', 'duration_s'
        ]
        
        linguistic_features = [
            'file_id', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'sentiment_variability', 'dominant_sentiment', 'dominant_sentiment_score',
            'confidence_mean'
        ]
        
        # Keep only columns that exist in both dataframes
        acoustic_features = [col for col in acoustic_features if col in acoustic_df.columns]
        linguistic_features = [col for col in linguistic_features if col in linguistic_df.columns]
        
        # Select features
        acoustic_df = acoustic_df[acoustic_features]
        linguistic_df = linguistic_df[linguistic_features]
        
        # Handle missing values
        acoustic_df = acoustic_df.dropna(subset=['file_id'])
        linguistic_df = linguistic_df.dropna(subset=['file_id'])
        
        # Normalize features if requested
        if self.normalize_features:
            # Get numerical columns excluding file_id and other non-numeric
            acoustic_num_cols = acoustic_df.select_dtypes(include=np.number).columns
            linguistic_num_cols = linguistic_df.select_dtypes(include=np.number).columns
            
            # Apply min-max scaling
            scaler = MinMaxScaler()
            
            acoustic_df[acoustic_num_cols] = scaler.fit_transform(acoustic_df[acoustic_num_cols])
            linguistic_df[linguistic_num_cols] = scaler.fit_transform(linguistic_df[linguistic_num_cols])
        
        return acoustic_df, linguistic_df
    
    def combine_features(self, 
                       acoustic_df: pd.DataFrame, 
                       linguistic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine acoustic and linguistic features
        
        Args:
            acoustic_df: Preprocessed acoustic features
            linguistic_df: Preprocessed linguistic features
            
        Returns:
            Combined features DataFrame
        """
        # Merge on file_id
        combined_df = pd.merge(
            acoustic_df,
            linguistic_df,
            on='file_id',
            how='inner'
        )
        
        logger.info(f"Combined features: {len(combined_df)} calls with both acoustic and linguistic data")
        
        if len(combined_df) == 0:
            logger.error("No matching file_ids between acoustic and linguistic features")
            logger.error(f"Acoustic file_ids: {acoustic_df['file_id'].tolist()}")
            logger.error(f"Linguistic file_ids: {linguistic_df['file_id'].tolist()}")
            raise ValueError("No matching file_ids between acoustic and linguistic features")
        
        # Add multimodal metrics
        combined_df = self.add_multimodal_metrics(combined_df)
        
        return combined_df
    
    def add_multimodal_metrics(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add multimodal metrics based on combined features
        
        Args:
            combined_df: Combined features DataFrame
            
        Returns:
            DataFrame with additional multimodal metrics
        """
        # Calculate acoustic-semantic alignment
        if all(col in combined_df.columns for col in ['acoustic_volatility_index', 'sentiment_negative']):
            # Scale both metrics to 0-1 range if not already
            if self.normalize_features == False:
                combined_df['acoustic_volatility_scaled'] = combined_df['acoustic_volatility_index'] / combined_df['acoustic_volatility_index'].max()
                combined_df['sentiment_negative_scaled'] = combined_df['sentiment_negative'] / combined_df['sentiment_negative'].max()
            else:
                combined_df['acoustic_volatility_scaled'] = combined_df['acoustic_volatility_index']
                combined_df['sentiment_negative_scaled'] = combined_df['sentiment_negative']
            
            # Calculate alignment score (closer to 1 = better alignment)
            combined_df['acoustic_semantic_alignment'] = 1 - abs(
                combined_df['acoustic_volatility_scaled'] - combined_df['sentiment_negative_scaled']
            )
        
        # Add pattern classification based on circumplex model of emotion
        if all(col in combined_df.columns for col in ['acoustic_volatility_index', 'sentiment_negative', 'sentiment_positive']):
            combined_df['communication_pattern'] = combined_df.apply(
                self._classify_communication_pattern, axis=1
            )
        
        # Calculate acoustic-semantic distance metrics
        if all(col in combined_df.columns for col in ['f0_cv', 'sentiment_variability']):
            combined_df['acoustic_semantic_variability_distance'] = abs(
                combined_df['f0_cv'] - combined_df['sentiment_variability']
            )
        
        return combined_df
    
    def _classify_communication_pattern(self, row: pd.Series) -> str:
        """
        Classify communication pattern based on acoustic and semantic features
        Follows Russell's Circumplex Model framework
        
        Args:
            row: DataFrame row with combined features
            
        Returns:
            Communication pattern classification
        """
        # Get key metrics
        acoustic_vol = row['acoustic_volatility_index']
        neg_sentiment = row['sentiment_negative']
        pos_sentiment = row['sentiment_positive']
        
        # Define thresholds (assuming normalized 0-1 scale)
        high_threshold = 0.7
        moderate_threshold = 0.5
        
        # Pattern classification logic
        if acoustic_vol > high_threshold and neg_sentiment > high_threshold:
            return "high_stress"
        elif acoustic_vol > high_threshold and pos_sentiment > high_threshold:
            return "high_excitement"
        elif acoustic_vol > moderate_threshold and neg_sentiment > moderate_threshold:
            return "moderate_stress"
        elif acoustic_vol > moderate_threshold and pos_sentiment > moderate_threshold:
            return "moderate_excitement"
        elif acoustic_vol < moderate_threshold and max(neg_sentiment, pos_sentiment) < moderate_threshold:
            return "baseline_stability"
        else:
            return "mixed_pattern"
    
    def calculate_correlations(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between acoustic and semantic features
        
        Args:
            combined_df: Combined features DataFrame
            
        Returns:
            Correlation matrix DataFrame
        """
        # Define key acoustic and semantic features for correlation
        acoustic_keys = ['f0_cv', 'f0_std', 'jitter_local', 'speech_rate', 
                        'pause_frequency', 'acoustic_volatility_index']
        semantic_keys = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 
                        'sentiment_variability']
        
        # Filter to columns that exist
        acoustic_keys = [col for col in acoustic_keys if col in combined_df.columns]
        semantic_keys = [col for col in semantic_keys if col in combined_df.columns]
        
        # Select relevant columns for correlation
        correlation_cols = acoustic_keys + semantic_keys
        
        # Calculate correlation matrix
        correlation_matrix = combined_df[correlation_cols].corr()
        
        # Extract only cross-modal correlations
        cross_modal_corr = pd.DataFrame()
        
        for a_feat in acoustic_keys:
            for s_feat in semantic_keys:
                cross_modal_corr.loc[a_feat, s_feat] = correlation_matrix.loc[a_feat, s_feat]
        
        return cross_modal_corr
    
    def perform_dimensionality_reduction(self, 
                                      combined_df: pd.DataFrame, 
                                      n_components: int = 2) -> pd.DataFrame:
        """
        Perform PCA dimensionality reduction on combined features
        
        Args:
            combined_df: Combined features DataFrame
            n_components: Number of principal components
            
        Returns:
            DataFrame with principal components
        """
        # Get numerical columns excluding identifiers
        exclude_cols = ['file_id', 'dominant_sentiment', 'communication_pattern']
        numeric_cols = combined_df.select_dtypes(include=np.number).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(combined_df[feature_cols])
        
        # Create PCA DataFrame
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Add identifiers and category columns
        pca_df['file_id'] = combined_df['file_id']
        
        if 'dominant_sentiment' in combined_df.columns:
            pca_df['dominant_sentiment'] = combined_df['dominant_sentiment']
        
        if 'communication_pattern' in combined_df.columns:
            pca_df['communication_pattern'] = combined_df['communication_pattern']
        
        # Add explained variance
        explained_variance = pca.explained_variance_ratio_
        logger.info(f"PCA explained variance: {explained_variance}")
        
        return pca_df, explained_variance
    
    def create_visualizations(self, 
                            combined_df: pd.DataFrame, 
                            correlations: pd.DataFrame,
                            pca_df: pd.DataFrame,
                            explained_variance: np.ndarray,
                            output_dir: str):
        """
        Create visualizations of multimodal features
        
        Args:
            combined_df: Combined features DataFrame
            correlations: Correlation matrix DataFrame
            pca_df: PCA results DataFrame
            explained_variance: PCA explained variance ratio
            output_dir: Output directory for visualizations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Acoustic-Semantic Correlation Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Acoustic-Semantic Feature Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'acoustic_semantic_correlations.png'), dpi=300)
        plt.close()
        
        # 2. PCA Visualization
        if 'PC1' in pca_df.columns and 'PC2' in pca_df.columns:
            plt.figure(figsize=(12, 10))
            
            # Color points by communication pattern if available
            if 'communication_pattern' in pca_df.columns:
                palette = {
                    'high_stress': 'red',
                    'moderate_stress': 'orange',
                    'high_excitement': 'green',
                    'moderate_excitement': 'lightgreen',
                    'baseline_stability': 'blue',
                    'mixed_pattern': 'gray'
                }
                
                scatter = sns.scatterplot(
                    data=pca_df,
                    x='PC1',
                    y='PC2',
                    hue='communication_pattern',
                    palette=palette,
                    s=100,
                    alpha=0.7
                )
                
                # Add legend outside plot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                scatter = sns.scatterplot(
                    data=pca_df,
                    x='PC1',
                    y='PC2',
                    s=100,
                    alpha=0.7
                )
            
            # Add file_id labels
            for i, row in pca_df.iterrows():
                plt.text(row['PC1'], row['PC2'], row['file_id'], fontsize=8)
            
            # Add axis labels with explained variance
            plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)')
            plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)')
            plt.title('PCA of Combined Acoustic-Semantic Features')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=300)
            plt.close()
        
        # 3. Acoustic-Semantic Alignment Scatter Plot
        if all(col in combined_df.columns for col in ['acoustic_volatility_index', 'sentiment_negative']):
            plt.figure(figsize=(10, 8))
            
            # Color points by communication pattern if available
            if 'communication_pattern' in combined_df.columns:
                palette = {
                    'high_stress': 'red',
                    'moderate_stress': 'orange',
                    'high_excitement': 'green',
                    'moderate_excitement': 'lightgreen',
                    'baseline_stability': 'blue',
                    'mixed_pattern': 'gray'
                }
                
                scatter = sns.scatterplot(
                    data=combined_df,
                    x='acoustic_volatility_index',
                    y='sentiment_negative',
                    hue='communication_pattern',
                    palette=palette,
                    s=100,
                    alpha=0.7
                )
                
                # Add legend outside plot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                scatter = sns.scatterplot(
                    data=combined_df,
                    x='acoustic_volatility_index',
                    y='sentiment_negative',
                    s=100,
                    alpha=0.7
                )
            
            # Add file_id labels
            for i, row in combined_df.iterrows():
                plt.text(row['acoustic_volatility_index'], row['sentiment_negative'], row['file_id'], fontsize=8)
            
            # Add quadrant lines
            mid_x = combined_df['acoustic_volatility_index'].median()
            mid_y = combined_df['sentiment_negative'].median()
            
            plt.axvline(x=mid_x, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(y=mid_y, color='gray', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            plt.text(combined_df['acoustic_volatility_index'].max() * 0.8, 
                    combined_df['sentiment_negative'].max() * 0.8, 
                    'High Stress\nQuadrant', 
                    fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))
            
            plt.text(combined_df['acoustic_volatility_index'].min() * 1.2, 
                    combined_df['sentiment_negative'].min() * 1.2, 
                    'Baseline\nQuadrant', 
                    fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))
            
            plt.xlabel('Acoustic Volatility Index')
            plt.ylabel('Negative Sentiment Score')
            plt.title('Acoustic-Semantic Alignment')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'acoustic_semantic_alignment.png'), dpi=300)
            plt.close()
        
        # 4. Communication Pattern Distribution
        if 'communication_pattern' in combined_df.columns:
            plt.figure(figsize=(10, 6))
            
            pattern_counts = combined_df['communication_pattern'].value_counts()
            
            # Define color palette
            colors = {
                'high_stress': 'red',
                'moderate_stress': 'orange',
                'high_excitement': 'green',
                'moderate_excitement': 'lightgreen',
                'baseline_stability': 'blue',
                'mixed_pattern': 'gray'
            }
            
            # Filter palette to include only existing patterns
            plot_colors = [colors.get(pattern, 'gray') for pattern in pattern_counts.index]
            
            bars = plt.bar(pattern_counts.index, pattern_counts.values, color=plot_colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height}', ha='center', va='bottom')
            
            plt.xlabel('Communication Pattern')
            plt.ylabel('Count')
            plt.title('Distribution of Communication Patterns')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'communication_patterns.png'), dpi=300)
            plt.close()
            
        logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_feature_importance(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate feature importance based on correlation with patterns
        
        Args:
            combined_df: Combined features DataFrame
            
        Returns:
            DataFrame with feature importance scores
        """
        if 'communication_pattern' not in combined_df.columns:
            logger.warning("Communication pattern not found, skipping feature importance analysis")
            return pd.DataFrame()
        
        # One-hot encode communication patterns
        pattern_dummies = pd.get_dummies(combined_df['communication_pattern'], prefix='pattern')
        combined_with_dummies = pd.concat([combined_df, pattern_dummies], axis=1)
        
        # Get numerical features
        exclude_cols = ['file_id', 'dominant_sentiment', 'communication_pattern']
        exclude_cols.extend(pattern_dummies.columns)
        
        numeric_cols = combined_df.select_dtypes(include=np.number).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate correlation with each pattern
        importance_df = pd.DataFrame(index=feature_cols)
        
        for pattern in pattern_dummies.columns:
            correlations = []
            p_values = []
            
            for feature in feature_cols:
                corr, p_value = stats.pearsonr(
                    combined_with_dummies[feature],
                    combined_with_dummies[pattern]
                )
                correlations.append(corr)
                p_values.append(p_value)
            
            importance_df[f'{pattern}_corr'] = correlations
            importance_df[f'{pattern}_p'] = p_values
        
        # Add absolute importance (average absolute correlation)
        abs_corr_cols = [col for col in importance_df.columns if col.endswith('_corr')]
        importance_df['abs_importance'] = importance_df[abs_corr_cols].abs().mean(axis=1)
        
        # Sort by importance
        importance_df = importance_df.sort_values('abs_importance', ascending=False)
        
        return importance_df


def main():
    parser = argparse.ArgumentParser(
        description="Combine acoustic and linguistic features for multimodal analysis"
    )
    parser.add_argument("--acoustic_dir", type=str, required=True,
                       help="Directory containing acoustic features")
    parser.add_argument("--linguistic_dir", type=str, required=True,
                       help="Directory containing linguistic features")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save combined features and visualizations")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize features before combining")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualizations of combined features")
    
    args = parser.parse_args()
    
    # Initialize feature fusion
    fusion = MultimodalFeatureFusion(normalize_features=args.normalize)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load features
        logger.info("Loading acoustic and linguistic features...")
        acoustic_df, linguistic_df = fusion.load_features(args.acoustic_dir, args.linguistic_dir)
        
        # Preprocess features
        logger.info("Preprocessing features...")
        acoustic_df, linguistic_df = fusion.preprocess_features(acoustic_df, linguistic_df)
        
        # Combine features
        logger.info("Combining features...")
        combined_df = fusion.combine_features(acoustic_df, linguistic_df)
        
        # Calculate correlations
        logger.info("Calculating acoustic-semantic correlations...")
        correlations = fusion.calculate_correlations(combined_df)
        
        # Perform dimensionality reduction
        logger.info("Performing dimensionality reduction...")
        pca_df, explained_variance = fusion.perform_dimensionality_reduction(combined_df)
        
        # Generate feature importance
        logger.info("Calculating feature importance...")
        importance_df = fusion.generate_feature_importance(combined_df)
        
        # Save results
        combined_df.to_csv(os.path.join(args.output_dir, "combined_features.csv"), index=False)
        correlations.to_csv(os.path.join(args.output_dir, "acoustic_semantic_correlations.csv"))
        pca_df.to_csv(os.path.join(args.output_dir, "pca_results.csv"), index=False)
        
        if not importance_df.empty:
            importance_df.to_csv(os.path.join(args.output_dir, "feature_importance.csv"))
        
        # Create summary statistics
        pattern_counts = combined_df['communication_pattern'].value_counts().to_dict() if 'communication_pattern' in combined_df.columns else {}
        
        # Get top correlations as strings instead of tuples to avoid JSON serialization issues
        top_corrs = {}
        if not correlations.empty:
            # Unstack to get (feature1, feature2) pairs
            unstacked = correlations.unstack()
            # Sort by absolute value
            sorted_corrs = unstacked.abs().sort_values(ascending=False).head(5)
            
            # Convert to dictionary with string keys
            for i, ((feat1, feat2), value) in enumerate(sorted_corrs.items()):
                corr_value = unstacked.loc[feat1, feat2]
                top_corrs[f"{feat1}_vs_{feat2}"] = float(corr_value)
        
        summary = {
            'n_calls_analyzed': len(combined_df),
            'average_acoustic_volatility': float(combined_df['acoustic_volatility_index'].mean()) if 'acoustic_volatility_index' in combined_df.columns else None,
            'average_negative_sentiment': float(combined_df['sentiment_negative'].mean()) if 'sentiment_negative' in combined_df.columns else None,
            'communication_patterns': pattern_counts,
            'top_acoustic_semantic_correlations': top_corrs,
            'pca_explained_variance': explained_variance.tolist(),
            'top_features_by_importance': importance_df.index[:5].tolist() if not importance_df.empty else []
        }
        
        with open(os.path.join(args.output_dir, "fusion_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create visualizations if requested
        if args.visualize:
            logger.info("Creating visualizations...")
            fusion.create_visualizations(
                combined_df, 
                correlations, 
                pca_df, 
                explained_variance,
                os.path.join(args.output_dir, "visualizations")
            )
        
        logger.info("Multimodal feature fusion complete!")
        logger.info(f"Results saved to {args.output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("MULTIMODAL FEATURE FUSION SUMMARY")
        print("="*60)
        print(f"Total calls analyzed: {summary['n_calls_analyzed']}")
        
        if 'communication_patterns' in summary and summary['communication_patterns']:
            print("\nCommunication pattern distribution:")
            for pattern, count in summary['communication_patterns'].items():
                print(f"  - {pattern}: {count}")
        
        if 'top_acoustic_semantic_correlations' in summary and summary['top_acoustic_semantic_correlations']:
            print("\nTop acoustic-semantic correlations:")
            for i, (pair, corr) in enumerate(summary['top_acoustic_semantic_correlations'].items(), 1):
                print(f"  {i}. {pair}: {corr:.3f}")
        
    except Exception as e:
        logger.error(f"Error in multimodal feature fusion: {e}", exc_info=True)


if __name__ == "__main__":
    main()