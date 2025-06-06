#!/usr/bin/env python3
"""
FinBERT Sentiment Classification for Earnings Call Analysis
Implements call-level semantic aggregation for directional validation of acoustic patterns
Following contemporary scientific framework for acoustic-semantic correlation analysis
"""

import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CallLevelFinBERTAnalyzer:
    """
    Call-level FinBERT sentiment analysis for organizational communication climate
    Implements methodological triangulation through correlation rather than fusion
    """
    
    def __init__(self, 
                 model_name: str = "yiyanghkust/finbert-tone",
                 max_sequence_length: int = 512,
                 overlap_tokens: int = 50,
                 confidence_threshold: float = 0.0):  # Include all predictions
        """
        Initialize FinBERT analyzer
        
        Args:
            model_name: FinBERT model identifier
            max_sequence_length: Maximum token sequence length
            overlap_tokens: Token overlap for chunk processing
            confidence_threshold: Minimum confidence (set to 0 for transparency)
        """
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.overlap_tokens = overlap_tokens
        self.confidence_threshold = confidence_threshold
        
        # FIXED: Set sentiment labels BEFORE loading model
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Load sector mappings for stratification
        self.sector_mappings = self._load_sector_mappings()
        
    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        logger.info(f"Loading FinBERT model: {self.model_name}")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("FinBERT model loaded successfully")
        
        # Verify label mapping with test
        self._verify_label_mapping()
    
    def _verify_label_mapping(self):
        """Verify FinBERT label mapping with test sentences"""
        test_texts = [
            ("The company delivered exceptional results with record profits", "positive"),
            ("We face significant challenges and declining revenues", "negative"), 
            ("The quarterly report shows standard metrics", "neutral")
        ]
        
        logger.info("Verifying FinBERT label mapping...")
        
        for text, expected in test_texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                  max_length=512, padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
                predicted_idx = np.argmax(probs)
                predicted_label = self.sentiment_labels[predicted_idx]
                
                logger.info(f"Test: '{text[:50]}...' -> Predicted: {predicted_label} (confidence: {probs[predicted_idx]:.3f})")
    
    def _load_sector_mappings(self) -> Dict[str, str]:
        """Load sector mappings for stratified analysis"""
        try:
            metadata_path = Path("data/raw/earnings21/earnings21-file-metadata.csv")
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                return dict(zip(df['file_id'].astype(str), df['sector']))
            else:
                logger.warning("Sector metadata not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading sector mappings: {e}")
            return {}
    
    def analyze_call_transcript(self, transcript_path: str) -> Dict:
        """
        Analyze entire call transcript for sentiment
        
        Args:
            transcript_path: Path to call transcript
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Load transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Extract call text
            if 'call_text' in transcript_data:
                call_text = transcript_data['call_text']
            else:
                # Fallback to concatenating tokens if needed
                call_text = self._reconstruct_text_from_tokens(transcript_data)
            
            file_id = transcript_data.get('file_id', Path(transcript_path).stem)
            logger.info(f"Analyzing transcript: {file_id}")
            
            # Validate text content
            if not call_text or len(call_text.strip()) < 50:
                logger.warning(f"Insufficient text content for {file_id}")
                return {
                    'file_id': file_id,
                    'error': 'insufficient_text_content'
                }
            
            # Split text into overlapping chunks
            chunks = self._split_text_into_chunks(call_text)
            logger.info(f"Processing {len(chunks)} text chunks for {file_id}")
            
            # Analyze each chunk
            chunk_sentiments = []
            chunk_confidences = []
            
            for chunk in chunks:
                sentiment_scores, confidence = self._analyze_chunk(chunk)
                chunk_sentiments.append(sentiment_scores)
                chunk_confidences.append(confidence)
            
            # Aggregate results
            results = self._aggregate_chunk_sentiments(
                chunk_sentiments, 
                chunk_confidences,
                file_id
            )
            
            # Add sector information
            results['sector'] = self.sector_mappings.get(file_id, 'unknown')
            
            # Calculate distributional statistics for uncertainty quantification
            results.update(self._calculate_distributional_statistics(chunk_sentiments))
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing {transcript_path}: {e}")
            return {
                'file_id': Path(transcript_path).stem,
                'error': str(e)
            }
    
    def _reconstruct_text_from_tokens(self, transcript_data: Dict) -> str:
        """Reconstruct text from token data if needed"""
        if 'normalized_tokens' in transcript_data:
            tokens = transcript_data['normalized_tokens']
            words = []
            for token in tokens:
                word = token.get('word', '')
                punct = token.get('punctuation', '')
                words.append(word + punct)
            return ' '.join(words)
        return ""
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for BERT processing
        Implements sliding window approach for complete coverage
        """
        # Tokenize full text
        tokens = self.tokenizer.tokenize(text)
        
        chunks = []
        chunk_size = self.max_sequence_length - 2  # Account for [CLS] and [SEP]
        stride = chunk_size - self.overlap_tokens
        
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
            
            # Stop if we've reached the end
            if i + chunk_size >= len(tokens):
                break
        
        # Ensure at least one chunk
        if not chunks and text:
            chunks = [text[:5000]]  # Fallback for very short texts
        
        return chunks
    
    def _analyze_chunk(self, text: str) -> Tuple[Dict[str, float], float]:
        """
        Analyze a single text chunk with FinBERT
        
        Returns:
            Tuple of (sentiment_scores, confidence)
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_sequence_length,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Calculate probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probs = probabilities.cpu().numpy()[0]
            
            # Calculate confidence (max probability)
            confidence = float(np.max(probs))
            
            # FIXED: Correct sentiment score mapping for yiyanghkust/finbert-tone
            # Model outputs: [negative, neutral, positive]
            sentiment_scores = {
                'negative': float(probs[0]),   # Index 0 = negative
                'neutral': float(probs[1]),    # Index 1 = neutral
                'positive': float(probs[2])    # Index 2 = positive
            }
        
        return sentiment_scores, confidence
    
    def _aggregate_chunk_sentiments(self, 
                                  chunk_sentiments: List[Dict],
                                  chunk_confidences: List[float],
                                  file_id: str) -> Dict:
        """
        Aggregate chunk-level sentiments to call-level
        Implements weighted aggregation based on confidence
        """
        # FIXED: Convert to arrays with correct order [negative, neutral, positive]
        sentiments_array = np.array([
            [s['negative'], s['neutral'], s['positive']] 
            for s in chunk_sentiments
        ])
        
        confidences_array = np.array(chunk_confidences)
        
        # Calculate weighted averages
        if self.confidence_threshold > 0:
            # Filter by confidence
            valid_mask = confidences_array >= self.confidence_threshold
            if np.any(valid_mask):
                valid_sentiments = sentiments_array[valid_mask]
                valid_confidences = confidences_array[valid_mask]
                weights = valid_confidences / np.sum(valid_confidences)
                weighted_sentiments = np.average(valid_sentiments, weights=weights, axis=0)
            else:
                # If no chunks meet threshold, use simple average
                weighted_sentiments = np.mean(sentiments_array, axis=0)
        else:
            # Use confidence-weighted average
            weights = confidences_array / np.sum(confidences_array)
            weighted_sentiments = np.average(sentiments_array, weights=weights, axis=0)
        
        # FIXED: Calculate additional statistics with correct indices
        results = {
            'file_id': file_id,
            'num_chunks': len(chunk_sentiments),
            
            # Primary sentiment scores (weighted) - CORRECTED ORDER
            'sentiment_negative': float(weighted_sentiments[0]),   # Index 0 = negative
            'sentiment_neutral': float(weighted_sentiments[1]),    # Index 1 = neutral
            'sentiment_positive': float(weighted_sentiments[2]),   # Index 2 = positive
            
            # Simple averages for comparison - CORRECTED ORDER
            'sentiment_negative_mean': float(np.mean(sentiments_array[:, 0])),
            'sentiment_neutral_mean': float(np.mean(sentiments_array[:, 1])),
            'sentiment_positive_mean': float(np.mean(sentiments_array[:, 2])),
            
            # Confidence metrics
            'confidence_mean': float(np.mean(confidences_array)),
            'confidence_std': float(np.std(confidences_array)),
            'confidence_min': float(np.min(confidences_array)),
            
            # FIXED: Dominant sentiment with correct label mapping
            'dominant_sentiment': self.sentiment_labels[np.argmax(weighted_sentiments)],
            'dominant_sentiment_score': float(np.max(weighted_sentiments))
        }
        
        # Add chunk-level details for transparency
        results['chunk_sentiments'] = chunk_sentiments
        results['chunk_confidences'] = chunk_confidences if isinstance(chunk_confidences, list) else chunk_confidences.tolist()
        
        return results
    
    def _calculate_distributional_statistics(self, 
                                           chunk_sentiments: List[Dict]) -> Dict:
        """
        Calculate distributional statistics for uncertainty quantification
        Aligns with bootstrap methodology from acoustic analysis
        """
        # FIXED: Array construction with correct order
        sentiments_array = np.array([
            [s['negative'], s['neutral'], s['positive']] 
            for s in chunk_sentiments
        ])
        
        stats = {}
        
        # Standard deviations (sentiment variability) - CORRECTED ORDER
        stats['sentiment_negative_std'] = float(np.std(sentiments_array[:, 0]))
        stats['sentiment_neutral_std'] = float(np.std(sentiments_array[:, 1]))
        stats['sentiment_positive_std'] = float(np.std(sentiments_array[:, 2]))
        
        # Percentiles for extreme sentiment detection - CORRECTED ORDER
        for i, label in enumerate(['negative', 'neutral', 'positive']):
            stats[f'sentiment_{label}_p05'] = float(np.percentile(sentiments_array[:, i], 5))
            stats[f'sentiment_{label}_p25'] = float(np.percentile(sentiments_array[:, i], 25))
            stats[f'sentiment_{label}_p75'] = float(np.percentile(sentiments_array[:, i], 75))
            stats[f'sentiment_{label}_p95'] = float(np.percentile(sentiments_array[:, i], 95))
        
        # Overall sentiment variability (for correlation with acoustic volatility)
        stats['sentiment_variability'] = float(np.mean([
            stats['sentiment_negative_std'],
            stats['sentiment_neutral_std'],
            stats['sentiment_positive_std']
        ]))
        
        # Sentiment entropy (distribution uncertainty)
        mean_sentiments = np.mean(sentiments_array, axis=0)
        mean_sentiments = mean_sentiments / np.sum(mean_sentiments)  # Normalize
        entropy = -np.sum(mean_sentiments * np.log(mean_sentiments + 1e-10))
        stats['sentiment_entropy'] = float(entropy)
        
        return stats
    
    def calculate_acoustic_semantic_correlation(self,
                                              sentiment_results: pd.DataFrame,
                                              acoustic_features: pd.DataFrame) -> Dict:
        """
        Calculate correlations between acoustic features and sentiment
        Implements methodological triangulation for pattern validation
        """
        # Merge dataframes on file_id
        merged_df = pd.merge(
            sentiment_results,
            acoustic_features,
            on='file_id',
            how='inner'
        )
        
        if len(merged_df) < 3:
            logger.warning("Insufficient data for correlation analysis")
            return {}
        
        # Define key acoustic-semantic pairs for correlation
        correlation_pairs = [
            # Convergent stress evidence pairs
            ('f0_cv', 'sentiment_negative'),
            ('acoustic_volatility_index', 'sentiment_negative'),
            ('jitter_local', 'sentiment_negative'),
            
            # Divergent arousal pattern pairs
            ('f0_cv', 'sentiment_positive'),
            ('acoustic_volatility_index', 'sentiment_positive'),
            
            # Stability baseline pairs
            ('f0_cv', 'sentiment_neutral'),
            ('pause_frequency', 'sentiment_variability')
        ]
        
        correlations = {}
        
        for acoustic_feat, sentiment_feat in correlation_pairs:
            if acoustic_feat in merged_df.columns and sentiment_feat in merged_df.columns:
                # Calculate Pearson correlation
                valid_mask = merged_df[[acoustic_feat, sentiment_feat]].notna().all(axis=1)
                if valid_mask.sum() >= 3:
                    corr, p_value = stats.pearsonr(
                        merged_df.loc[valid_mask, acoustic_feat],
                        merged_df.loc[valid_mask, sentiment_feat]
                    )
                    
                    correlations[f'{acoustic_feat}_vs_{sentiment_feat}'] = {
                        'correlation': float(corr),
                        'p_value': float(p_value),
                        'n_samples': int(valid_mask.sum()),
                        'interpretation': self._interpret_correlation(corr, acoustic_feat, sentiment_feat)
                    }
        
        # Calculate pattern classification based on correlations
        pattern_classification = self._classify_correlation_patterns(correlations)
        
        return {
            'correlations': correlations,
            'pattern_classification': pattern_classification,
            'n_calls_analyzed': len(merged_df)
        }
    
    def _interpret_correlation(self, corr: float, 
                             acoustic_feat: str, 
                             sentiment_feat: str) -> str:
        """
        Interpret correlation strength and direction
        Based on contemporary emotion-voice literature
        """
        abs_corr = abs(corr)
        
        if abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.6:
            strength = "moderate"
        else:
            strength = "strong"
        
        # Specific interpretations for key pairs
        if acoustic_feat in ['f0_cv', 'acoustic_volatility_index'] and sentiment_feat == 'sentiment_negative':
            if corr > 0.3:
                return f"{strength} convergent stress evidence"
            else:
                return f"{strength} lack of stress convergence"
        
        elif acoustic_feat in ['f0_cv', 'acoustic_volatility_index'] and sentiment_feat == 'sentiment_positive':
            if corr > 0.3:
                return f"{strength} divergent arousal pattern (excitement)"
            else:
                return f"{strength} lack of positive arousal"
        
        else:
            direction = "positive" if corr > 0 else "negative"
            return f"{strength} {direction} correlation"
    
    def _classify_correlation_patterns(self, correlations: Dict) -> Dict:
        """
        Classify overall acoustic-semantic pattern
        Following Russell's Circumplex Model framework
        """
        # Extract key correlations
        stress_corr = correlations.get('acoustic_volatility_index_vs_sentiment_negative', {}).get('correlation', 0)
        excitement_corr = correlations.get('acoustic_volatility_index_vs_sentiment_positive', {}).get('correlation', 0)
        
        # Pattern classification logic
        if stress_corr > 0.3 and excitement_corr < 0.3:
            pattern = "convergent_stress_evidence"
            confidence = "high" if stress_corr > 0.6 else "moderate"
            
        elif excitement_corr > 0.3 and stress_corr < 0.3:
            pattern = "divergent_excitement_pattern"
            confidence = "high" if excitement_corr > 0.6 else "moderate"
            
        elif abs(stress_corr) < 0.3 and abs(excitement_corr) < 0.3:
            pattern = "stable_communication_baseline"
            confidence = "high"
            
        else:
            pattern = "mixed_ambiguous_pattern"
            confidence = "low"
        
        return {
            'pattern': pattern,
            'confidence': confidence,
            'stress_correlation': float(stress_corr),
            'excitement_correlation': float(excitement_corr),
            'evidence': self._generate_pattern_evidence(pattern, correlations)
        }
    
    def _generate_pattern_evidence(self, pattern: str, correlations: Dict) -> List[str]:
        """Generate evidence statements for pattern classification"""
        evidence = []
        
        if pattern == "convergent_stress_evidence":
            evidence.append("High acoustic volatility correlates with negative sentiment")
            evidence.append("Voice quality degradation aligns with semantic negativity")
            
        elif pattern == "divergent_excitement_pattern":
            evidence.append("High acoustic volatility correlates with positive sentiment")
            evidence.append("Arousal indicators suggest excitement rather than stress")
            
        elif pattern == "stable_communication_baseline":
            evidence.append("Low correlations between acoustic and semantic features")
            evidence.append("Consistent with routine financial communication")
            
        else:
            evidence.append("Mixed correlation patterns detected")
            evidence.append("Further investigation needed for clear interpretation")
        
        return evidence
    
    def perform_sector_stratified_analysis(self, 
                                         sentiment_results: pd.DataFrame) -> Dict:
        """
        Analyze sentiment patterns by sector
        Controls for industry-specific communication norms
        """
        # Add sector information
        sentiment_results['sector'] = sentiment_results['file_id'].map(self.sector_mappings)
        
        sector_analysis = {}
        
        for sector in sentiment_results['sector'].unique():
            if pd.isna(sector) or sector == 'unknown':
                continue
            
            sector_data = sentiment_results[sentiment_results['sector'] == sector]
            
            if len(sector_data) >= 1:  # Include single-call sectors for completeness
                sector_analysis[sector] = {
                    'n_calls': len(sector_data),
                    'sentiment_negative_mean': float(sector_data['sentiment_negative'].mean()),
                    'sentiment_negative_std': float(sector_data['sentiment_negative'].std()) if len(sector_data) > 1 else 0.0,
                    'sentiment_positive_mean': float(sector_data['sentiment_positive'].mean()),
                    'sentiment_neutral_mean': float(sector_data['sentiment_neutral'].mean()),
                    'sentiment_variability_mean': float(sector_data['sentiment_variability'].mean()),
                    'dominant_sentiment_distribution': sector_data['dominant_sentiment'].value_counts().to_dict()
                }
        
        # Identify sectors with distinctive patterns
        if sector_analysis:
            # Find sectors with high negative sentiment
            negative_scores = {s: d['sentiment_negative_mean'] 
                             for s, d in sector_analysis.items()}
            most_negative_sector = max(negative_scores, key=negative_scores.get) if negative_scores else 'none'
            
            # Find sectors with high variability
            variability_scores = {s: d['sentiment_variability_mean'] 
                                for s, d in sector_analysis.items()}
            most_variable_sector = max(variability_scores, key=variability_scores.get) if variability_scores else 'none'
            
            sector_analysis['summary'] = {
                'most_negative_sector': most_negative_sector,
                'most_variable_sector': most_variable_sector,
                'sector_differences_detected': len(set(negative_scores.values())) > 1 if negative_scores else False
            }
        
        return sector_analysis
    
    def save_results(self, results: Dict, output_path: str):
        """Save sentiment analysis results"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Remove chunk-level details for main save (too verbose)
            save_data = {k: v for k, v in results.items() 
                        if k not in ['chunk_sentiments', 'chunk_confidences']}
            json.dump(save_data, f, indent=2)
        
        # Save chunk details separately if needed
        if 'chunk_sentiments' in results:
            chunk_path = output_path.replace('.json', '_chunks.json')
            with open(chunk_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'file_id': results['file_id'],
                    'chunk_sentiments': results['chunk_sentiments'],
                    'chunk_confidences': results['chunk_confidences']
                }, f, indent=2)
    
    def create_visualization(self, 
                           sentiment_results: pd.DataFrame,
                           output_dir: str):
        """
        Create visualizations for sentiment analysis results
        """
        # Check if we have valid data
        if len(sentiment_results) == 0:
            logger.error("No sentiment results available for visualization")
            return
            
        sentiment_cols = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
        
        # Check if sentiment columns exist
        if not all(col in sentiment_results.columns for col in sentiment_cols):
            logger.error("Required sentiment columns missing for visualization")
            logger.error(f"Available columns: {list(sentiment_results.columns)}")
            return
        
        # Set up plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sentiment distribution
        ax = axes[0, 0]
        sentiment_means = sentiment_results[sentiment_cols].mean()
        bars = ax.bar(range(len(sentiment_means)), sentiment_means.values, 
                     color=['lightgreen', 'lightcoral', 'lightgray'])
        ax.set_xticks(range(len(sentiment_means)))
        ax.set_xticklabels(['Positive', 'Negative', 'Neutral'])
        ax.set_title('Average Sentiment Distribution Across Calls')
        ax.set_ylabel('Mean Sentiment Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, sentiment_means.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Sentiment variability by call
        ax = axes[0, 1]
        if 'sentiment_variability' in sentiment_results.columns:
            scatter = ax.scatter(range(len(sentiment_results)), 
                               sentiment_results['sentiment_variability'],
                               alpha=0.6, c=sentiment_results['sentiment_negative'], 
                               cmap='RdYlBu_r')
            ax.set_xlabel('Call Index')
            ax.set_ylabel('Sentiment Variability')
            ax.set_title('Sentiment Variability Across Calls')
            plt.colorbar(scatter, ax=ax, label='Negative Sentiment')
        else:
            ax.text(0.5, 0.5, 'Sentiment variability data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sentiment Variability Across Calls')
        
        # 3. Confidence distribution
        ax = axes[1, 0]
        if 'confidence_mean' in sentiment_results.columns:
            n, bins, patches = ax.hist(sentiment_results['confidence_mean'], bins=15, alpha=0.7, 
                                     color='skyblue', edgecolor='black')
            mean_conf = sentiment_results['confidence_mean'].mean()
            ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                      label=f"Mean: {mean_conf:.3f}")
            ax.set_xlabel('Mean Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of FinBERT Confidence Scores')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Confidence data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribution of FinBERT Confidence Scores')
        
        # 4. Dominant sentiment pie chart
        ax = axes[1, 1]
        if 'dominant_sentiment' in sentiment_results.columns:
            dominant_counts = sentiment_results['dominant_sentiment'].value_counts()
            colors = ['lightgreen' if x == 'positive' else 'lightcoral' if x == 'negative' else 'lightgray' 
                     for x in dominant_counts.index]
            wedges, texts, autotexts = ax.pie(dominant_counts.values, 
                                            labels=dominant_counts.index,
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90)
            ax.set_title('Distribution of Dominant Sentiments')
            
            # Add count information
            for i, (label, count) in enumerate(zip(dominant_counts.index, dominant_counts.values)):
                ax.text(0, -1.3 - i*0.1, f"{label}: {count} calls", 
                       ha='center', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Dominant sentiment data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribution of Dominant Sentiments')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_analysis_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="FinBERT sentiment analysis for earnings call transcripts"
    )
    parser.add_argument("--transcript_dir", type=str, required=True,
                       help="Directory containing processed transcripts")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save sentiment analysis results")
    parser.add_argument("--acoustic_features", type=str,
                       help="Path to acoustic features CSV for correlation analysis")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CallLevelFinBERTAnalyzer()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get transcript files
    transcript_files = list(Path(args.transcript_dir).glob("*_call_level.json"))
    logger.info(f"Found {len(transcript_files)} transcript files to analyze")
    
    if len(transcript_files) == 0:
        logger.error(f"No transcript files found in {args.transcript_dir}")
        logger.error("Expected files with pattern: *_call_level.json")
        return
    
    # Analyze all transcripts
    all_results = []
    successful_analyses = 0
    
    for transcript_file in tqdm(transcript_files, desc="Analyzing sentiments"):
        results = analyzer.analyze_call_transcript(str(transcript_file))
        
        # Check if analysis was successful
        if 'error' not in results:
            successful_analyses += 1
            
            # Save individual results
            output_path = Path(args.output_dir) / f"{results['file_id']}_sentiment.json"
            analyzer.save_results(results, str(output_path))
        else:
            logger.warning(f"Failed to analyze {transcript_file}: {results.get('error', 'unknown_error')}")
        
        # Collect for aggregate analysis
        all_results.append(results)
    
    logger.info(f"Successfully analyzed {successful_analyses} out of {len(transcript_files)} transcripts")
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Filter out error cases
    if 'error' in results_df.columns:
        valid_results = results_df[results_df['error'].isna()]
        error_results = results_df[results_df['error'].notna()]
        
        if len(error_results) > 0:
            logger.warning(f"Errors in {len(error_results)} transcripts:")
            for _, row in error_results.iterrows():
                logger.warning(f"  {row['file_id']}: {row['error']}")
    else:
        valid_results = results_df
    
    if len(valid_results) == 0:
        logger.error("No successful sentiment analyses completed")
        return
    
    results_df = valid_results
    logger.info(f"Using {len(results_df)} valid results for analysis")
    
    # Save aggregate results
    results_df.to_csv(
        Path(args.output_dir) / "all_sentiment_results.csv",
        index=False
    )
    
    # Perform sector-stratified analysis
    logger.info("Performing sector-stratified analysis...")
    sector_analysis = analyzer.perform_sector_stratified_analysis(results_df)
    
    with open(Path(args.output_dir) / "sector_analysis.json", 'w') as f:
        json.dump(sector_analysis, f, indent=2)
    
    # Perform acoustic-semantic correlation if acoustic features provided
    if args.acoustic_features and os.path.exists(args.acoustic_features):
        logger.info("Calculating acoustic-semantic correlations...")
        
        try:
            acoustic_df = pd.read_csv(args.acoustic_features)
            correlation_results = analyzer.calculate_acoustic_semantic_correlation(
                results_df, acoustic_df
            )
            
            if correlation_results:
                with open(Path(args.output_dir) / "acoustic_semantic_correlations.json", 'w') as f:
                    json.dump(correlation_results, f, indent=2)
                
                # Log pattern classification
                pattern = correlation_results.get('pattern_classification', {})
                logger.info(f"Pattern Classification: {pattern.get('pattern', 'unknown')}")
                logger.info(f"Confidence: {pattern.get('confidence', 'unknown')}")
                logger.info(f"Evidence: {pattern.get('evidence', [])}")
            else:
                logger.warning("No correlation results generated - check data compatibility")
                
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
    elif args.acoustic_features:
        logger.warning(f"Acoustic features file not found: {args.acoustic_features}")
    
    # Create visualizations if requested
    if args.visualize:
        try:
            analyzer.create_visualization(results_df, args.output_dir)
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    # Generate summary report
    summary = {
        'total_calls_analyzed': len(results_df),
        'sentiment_distribution': {},
        'confidence_metrics': {},
        'dominant_sentiments': {},
        'sector_analysis_summary': sector_analysis.get('summary', {})
    }
    
    # Calculate summary statistics if columns exist
    if all(col in results_df.columns for col in ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']):
        summary['sentiment_distribution'] = {
            'positive_mean': float(results_df['sentiment_positive'].mean()),
            'negative_mean': float(results_df['sentiment_negative'].mean()),
            'neutral_mean': float(results_df['sentiment_neutral'].mean())
        }
    
    if 'confidence_mean' in results_df.columns:
        summary['confidence_metrics'] = {
            'mean': float(results_df['confidence_mean'].mean()),
            'std': float(results_df['confidence_mean'].std()),
            'min': float(results_df['confidence_mean'].min())
        }
    
    if 'dominant_sentiment' in results_df.columns:
        summary['dominant_sentiments'] = results_df['dominant_sentiment'].value_counts().to_dict()
    
    with open(Path(args.output_dir) / "sentiment_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("FinBERT sentiment analysis complete!")
    logger.info(f"Results saved to {args.output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("FINBERT SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total calls analyzed: {summary['total_calls_analyzed']}")
    
    if summary['sentiment_distribution']:
        print(f"Average sentiment scores:")
        print(f"  - Positive: {summary['sentiment_distribution']['positive_mean']:.3f}")
        print(f"  - Negative: {summary['sentiment_distribution']['negative_mean']:.3f}")
        print(f"  - Neutral: {summary['sentiment_distribution']['neutral_mean']:.3f}")
    
    if summary['dominant_sentiments']:
        print(f"Dominant sentiments: {summary['dominant_sentiments']}")
    
    if args.acoustic_features and 'correlation_results' in locals() and correlation_results:
        pattern = correlation_results.get('pattern_classification', {})
        print(f"\nAcoustic-Semantic Pattern: {pattern.get('pattern', 'unknown')}")
        print(f"Pattern Confidence: {pattern.get('confidence', 'unknown')}")


if __name__ == "__main__":
    main()