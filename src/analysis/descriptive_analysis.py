# src/analysis/descriptive_analysis.py
"""Main analysis script for acoustic-rating correlations"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import (FEATURES_DIR, RESULTS_DIR, BOOTSTRAP_ITERATIONS, 
                          CONFIDENCE_LEVEL, PERCENTILE_THRESHOLDS)

class DescriptiveAnalysis:
    def __init__(self):
        # Load all data
        self.acoustic_features = pd.read_csv(FEATURES_DIR / "acoustic_features.csv")
        self.sentiment_features = pd.read_csv(FEATURES_DIR / "finbert_sentiment.csv")
        
        # Merge datasets
        self.data = pd.merge(
            self.acoustic_features,
            self.sentiment_features,
            on=['company', 'call_date'],
            how='inner'
        )
        
        # Separate by rating outcome
        self.affirmations = self.data[self.data['composite_outcome'] == 'affirm']
        self.downgrades = self.data[self.data['composite_outcome'] == 'downgrade']
        self.upgrades = self.data[self.data['composite_outcome'] == 'upgrade']
        
    def calculate_percentile_rank(self, value, baseline_distribution):
        """Calculate percentile rank with bootstrap CI"""
        percentile_rank = (np.sum(baseline_distribution <= value) / 
                          len(baseline_distribution)) * 100
        
        # Bootstrap for CI
        bootstrap_ranks = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            bootstrap_sample = np.random.choice(
                baseline_distribution,
                size=len(baseline_distribution),
                replace=True
            )
            boot_rank = (np.sum(bootstrap_sample <= value) / 
                        len(bootstrap_sample)) * 100
            bootstrap_ranks.append(boot_rank)
        
        ci_lower = np.percentile(bootstrap_ranks, (1 - CONFIDENCE_LEVEL) * 100 / 2)
        ci_upper = np.percentile(bootstrap_ranks, (1 + CONFIDENCE_LEVEL) * 100 / 2)
        
        return {
            'percentile': percentile_rank,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def analyze_acoustic_patterns(self):
        """Analyze acoustic patterns across rating outcomes"""
        acoustic_cols = ['f0_mean', 'f0_std', 'f0_cv', 'jitter', 
                        'speech_rate', 'pause_ratio']
        
        results = []
        
        for feature in acoustic_cols:
            baseline = self.affirmations[feature].values
            
            # Calculate baseline statistics
            baseline_stats = {
                'feature': feature,
                'baseline_mean': np.mean(baseline),
                'baseline_std': np.std(baseline),
                'baseline_median': np.median(baseline),
                'baseline_mad': stats.median_abs_deviation(baseline)
            }
            
            # Analyze non-affirmation cases
            for idx, row in self.downgrades.iterrows():
                case_value = row[feature]
                percentile_result = self.calculate_percentile_rank(case_value, baseline)
                
                results.append({
                    **baseline_stats,
                    'company': row['company'],
                    'outcome': 'downgrade',
                    'value': case_value,
                    'standardized_distance': (case_value - np.median(baseline)) / 
                                           stats.median_abs_deviation(baseline),
                    **percentile_result
                })
            
            for idx, row in self.upgrades.iterrows():
                case_value = row[feature]
                percentile_result = self.calculate_percentile_rank(case_value, baseline)
                
                results.append({
                    **baseline_stats,
                    'company': row['company'],
                    'outcome': 'upgrade',
                    'value': case_value,
                    'standardized_distance': (case_value - np.median(baseline)) / 
                                           stats.median_abs_deviation(baseline),
                    **percentile_result
                })
        
        return pd.DataFrame(results)
    
    def analyze_acoustic_sentiment_correlation(self):
        """Analyze correlation between acoustic stress and semantic negativity"""
        # Define stress indicators
        stress_features = ['f0_cv', 'jitter', 'pause_ratio']
        
        correlations = {}
        
        for feature in stress_features:
            # Calculate correlation with sentiment
            corr_negative = stats.pearsonr(
                self.data[feature], 
                self.data['negative_score']
            )
            corr_polarity = stats.pearsonr(
                self.data[feature],
                self.data['sentiment_polarity']
            )
            
            correlations[feature] = {
                'corr_negative': corr_negative[0],
                'p_negative': corr_negative[1],
                'corr_polarity': corr_polarity[0],
                'p_polarity': corr_polarity[1]
            }
        
        return pd.DataFrame(correlations).T
    
    def create_visualizations(self, results_df):
        """Create comprehensive visualizations"""
        # 1. Box plots with case overlays
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        features = ['f0_cv', 'jitter', 'speech_rate', 'pause_ratio', 
                   'spectral_centroid_mean', 'mfcc_0_mean']
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            # Box plot for baseline
            baseline_data = self.affirmations[feature].values
            bp = ax.boxplot([baseline_data], positions=[1], widths=0.6,
                           patch_artist=True, showmeans=True)
            bp['boxes'][0].set_facecolor('lightblue')
            
            # Overlay individual cases
            if len(self.downgrades) > 0:
                for _, row in self.downgrades.iterrows():
                    ax.scatter([1.2], [row[feature]], color='red', s=100, 
                             alpha=0.8, label='Downgrade' if _ == 0 else "")
            
            if len(self.upgrades) > 0:
                for _, row in self.upgrades.iterrows():
                    ax.scatter([1.3], [row[feature]], color='green', s=100,
                             alpha=0.8, label='Upgrade' if _ == 0 else "")
            
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([1])
            ax.set_xticklabels(['Affirmations'])
            ax.set_title(f'{feature}')
            
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'figures' / 'acoustic_features_boxplots.png', dpi=300)
        plt.close()
        
        # 2. Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select key features for correlation
        corr_features = ['f0_cv', 'jitter', 'speech_rate', 'pause_ratio',
                        'negative_score', 'positive_score', 'sentiment_polarity']
        corr_data = self.data[corr_features].corr()
        
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax)
        ax.set_title('Acoustic-Sentiment Feature Correlations')
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'figures' / 'correlation_heatmap.png', dpi=300)
        plt.close()
        
        # 3. Percentile ranking visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter results for key features
        key_features = ['f0_cv', 'jitter', 'pause_ratio']
        plot_data = results_df[results_df['feature'].isin(key_features)]
        
        # Create grouped bar plot
        x = np.arange(len(key_features))
        width = 0.35
        
        for i, (outcome, color) in enumerate([('downgrade', 'red'), ('upgrade', 'green')]):
            outcome_data = plot_data[plot_data['outcome'] == outcome]
            if len(outcome_data) > 0:
                percentiles = []
                errors = []
                
                for feature in key_features:
                    feat_data = outcome_data[outcome_data['feature'] == feature]
                    if len(feat_data) > 0:
                        percentiles.append(feat_data['percentile'].mean())
                        errors.append([
                            feat_data['percentile'].mean() - feat_data['ci_lower'].mean(),
                            feat_data['ci_upper'].mean() - feat_data['percentile'].mean()
                        ])
                    else:
                        percentiles.append(0)
                        errors.append([0, 0])
                
                ax.bar(x + i*width, percentiles, width, label=outcome.capitalize(),
                      color=color, alpha=0.7, yerr=np.array(errors).T)
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Acoustic Features')
        ax.set_ylabel('Percentile Rank')
        ax.set_title('Non-Affirmation Cases: Percentile Rankings Against Baseline')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(key_features)
        ax.legend()
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'figures' / 'percentile_rankings.png', dpi=300)
        plt.close()
    
    def generate_summary_statistics(self, results_df):
        """Generate summary statistics table"""
        summary = []
        
        for feature in results_df['feature'].unique():
            feat_data = results_df[results_df['feature'] == feature]
            
            # Baseline statistics
            baseline_row = {
                'Feature': feature,
                'Baseline Mean (SD)': f"{feat_data['baseline_mean'].iloc[0]:.3f} ({feat_data['baseline_std'].iloc[0]:.3f})",
                'Baseline Median (MAD)': f"{feat_data['baseline_median'].iloc[0]:.3f} ({feat_data['baseline_mad'].iloc[0]:.3f})"
            }
            
            # Add case statistics
            for outcome in ['downgrade', 'upgrade']:
                outcome_data = feat_data[feat_data['outcome'] == outcome]
                if len(outcome_data) > 0:
                    baseline_row[f'{outcome.capitalize()} Value'] = f"{outcome_data['value'].mean():.3f}"
                    baseline_row[f'{outcome.capitalize()} Percentile'] = f"{outcome_data['percentile'].mean():.1f} [{outcome_data['ci_lower'].mean():.1f}, {outcome_data['ci_upper'].mean():.1f}]"
                else:
                    baseline_row[f'{outcome.capitalize()} Value'] = "N/A"
                    baseline_row[f'{outcome.capitalize()} Percentile'] = "N/A"
            
            summary.append(baseline_row)
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(RESULTS_DIR / 'tables' / 'summary_statistics.csv', index=False)
        
        return summary_df

def main():
    """Run complete analysis"""
    print("Starting descriptive analysis...")
    
    # Initialize analyzer
    analyzer = DescriptiveAnalysis()
    
    # Run analyses
    print("Analyzing acoustic patterns...")
    acoustic_results = analyzer.analyze_acoustic_patterns()
    acoustic_results.to_csv(RESULTS_DIR / 'tables' / 'acoustic_analysis_results.csv', index=False)
    
    print("Analyzing acoustic-sentiment correlations...")
    correlation_results = analyzer.analyze_acoustic_sentiment_correlation()
    correlation_results.to_csv(RESULTS_DIR / 'tables' / 'correlation_results.csv', index=False)
    
    print("Creating visualizations...")
    analyzer.create_visualizations(acoustic_results)
    
    print("Generating summary statistics...")
    summary_stats = analyzer.generate_summary_statistics(acoustic_results)
    
    print("Analysis complete! Results saved to:", RESULTS_DIR)
    
if __name__ == "__main__":
    main()