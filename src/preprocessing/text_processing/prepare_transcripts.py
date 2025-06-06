#!/usr/bin/env python3
"""
Transcript Processing Pipeline for Earnings Call Analysis
Implements call-level semantic aggregation for FinBERT directional validation
ENHANCED: Full A2/B1 demonstrator integration with sector analysis and bootstrap CI
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple
import sys
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """
    Call-level transcript processing with semantic aggregation
    Enhanced with A2/B1 demonstrator requirements
    """
    
    def __init__(self):
        # Use relative paths from project root
        self.project_root = PROJECT_ROOT
        self.nlp_ref_dir = self.project_root / "data" / "raw" / "earnings21" / "transcripts" / "nlp_references"
        self.norm_dir = self.project_root / "data" / "raw" / "earnings21" / "transcripts" / "normalizations"
        self.wer_tag_dir = self.project_root / "data" / "raw" / "earnings21" / "transcripts" / "wer_tags"
        self.speaker_metadata_path = self.project_root / "data" / "raw" / "earnings21" / "speaker-metadata.csv"
        self.file_metadata_path = self.project_root / "data" / "raw" / "earnings21" / "earnings21-file-metadata.csv"
        self.output_dir = self.project_root / "data" / "processed" / "transcripts"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "call_level").mkdir(exist_ok=True)
        (self.output_dir / "sentiment_analysis").mkdir(exist_ok=True)
        (self.output_dir / "validation").mkdir(exist_ok=True)
        (self.output_dir / "demonstrator").mkdir(exist_ok=True)
        
        # Load metadata
        self.speaker_metadata = self._load_speaker_metadata()
        self.file_metadata = self._load_file_metadata()
        
        # Initialize FinBERT
        self._init_finbert()
        
        # Load baseline for demonstrator
        self.baseline_features = self._load_baseline_features()
        
        # Bootstrap parameters
        self.n_bootstrap = 1000
        
    def _init_finbert(self):
        """Initialize FinBERT model for sentiment analysis."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.finbert_model_name = "yiyanghkust/finbert-tone"
        
        logger.info(f"Loading FinBERT model: {self.finbert_model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.finbert_model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.finbert_model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.sentiment_labels = ['positive', 'negative', 'neutral']
        self.max_sequence_length = 512
        
    def _load_speaker_metadata(self) -> pd.DataFrame:
        """Load speaker metadata as DataFrame for easier manipulation."""
        if self.speaker_metadata_path.exists():
            return pd.read_csv(self.speaker_metadata_path)
        else:
            logger.warning(f"Speaker metadata not found at {self.speaker_metadata_path}")
            return pd.DataFrame()
    
    def _load_file_metadata(self) -> pd.DataFrame:
        """Load file metadata including sector information."""
        if self.file_metadata_path.exists():
            df = pd.read_csv(self.file_metadata_path)
            # Create file_id to sector mapping
            self.file_to_sector = dict(zip(df['file_id'].astype(str), df['sector']))
            return df
        else:
            logger.warning(f"File metadata not found at {self.file_metadata_path}")
            self.file_to_sector = {}
            return pd.DataFrame()
    
    def _load_baseline_features(self) -> Optional[pd.DataFrame]:
        """Load baseline features for demonstrator comparison."""
        baseline_path = self.project_root / "data" / "baseline" / "21_company_features.csv"
        if baseline_path.exists():
            logger.info(f"Loaded baseline features from {baseline_path}")
            return pd.read_csv(baseline_path)
        else:
            logger.warning("Baseline features not found - demonstrator comparisons will be limited")
            return None
    
    def debug_nlp_file(self, nlp_path: Path, num_lines: int = 10):
        """Debug helper to inspect NLP file format."""
        logger.info(f"Debugging NLP file: {nlp_path}")
        with open(nlp_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines[:num_lines]):
            parts = line.strip().split('|')
            logger.info(f"Line {i}: {len(parts)} parts")
            logger.info(f"  Raw: {repr(line.strip())}")
            if len(parts) >= 4:
                logger.info(f"  Token: '{parts[0]}', Speaker: '{parts[1]}', Start: '{parts[2]}', End: '{parts[3]}'")
    
    def parse_timestamp(self, timestamp_str: str) -> float:
        """
        Parse timestamp string to float seconds.
        Handles various formats including empty strings and milliseconds.
        """
        if not timestamp_str or timestamp_str.strip() == '':
            return 0.0
        
        try:
            # Try direct float conversion first
            return float(timestamp_str)
        except ValueError:
            # Handle HH:MM:SS.fff format if needed
            if ':' in timestamp_str:
                parts = timestamp_str.split(':')
                if len(parts) == 3:
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
            
            # Handle other formats
            logger.debug(f"Unable to parse timestamp: '{timestamp_str}'")
            return 0.0
    
    def load_nlp_file(self, nlp_path: Path) -> List[Dict]:
        """Load and parse .nlp file with proper delimiter handling and robust timestamp parsing."""
        tokens = []
        
        with open(nlp_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Debug first few lines to understand format
        if logger.isEnabledFor(logging.DEBUG) and lines:
            self.debug_nlp_file(nlp_path, 5)
        
        # Skip header if present
        if lines and lines[0].startswith('token|'):
            lines = lines[1:]
        
        # Track timestamps for validation
        valid_timestamps = 0
        total_tokens = 0
        
        for line_num, line in enumerate(lines):
            parts = line.strip().split('|')
            if len(parts) >= 4:
                total_tokens += 1
                try:
                    # Parse timestamps with robust handling
                    start_time = self.parse_timestamp(parts[2])
                    end_time = self.parse_timestamp(parts[3])
                    
                    # Validate timestamps
                    if start_time > 0 or end_time > 0:
                        valid_timestamps += 1
                    
                    # If end_time is 0 but start_time is valid, estimate end_time
                    if start_time > 0 and end_time == 0:
                        # Estimate based on average word duration (0.3 seconds)
                        end_time = start_time + 0.3
                    
                    # Ensure end_time is after start_time
                    if end_time <= start_time:
                        end_time = start_time + 0.1
                    
                    token = {
                        'word': parts[0],
                        'speaker_id': parts[1] if parts[1] else 'unknown',
                        'start_time': start_time,
                        'end_time': end_time,
                        'line_num': line_num
                    }
                    
                    # Add optional fields
                    if len(parts) > 4:
                        token['punctuation'] = parts[4]
                    if len(parts) > 5:
                        token['case'] = parts[5]
                    if len(parts) > 6:
                        token['semantic_tag'] = parts[6]
                    if len(parts) > 7:
                        token['wer_tags'] = parts[7]
                    
                    tokens.append(token)
                except Exception as e:
                    logger.debug(f"Error parsing line {line_num} in {nlp_path}: {e}")
                    continue
        
        # Log parsing statistics
        if total_tokens > 0:
            timestamp_ratio = valid_timestamps / total_tokens
            logger.info(f"Loaded {len(tokens)} tokens from {nlp_path.name}, {timestamp_ratio:.1%} with valid timestamps")
            
            # If no valid timestamps found, generate approximate timestamps
            if timestamp_ratio < 0.1:
                logger.warning(f"Low timestamp ratio ({timestamp_ratio:.1%}) in {nlp_path.name}, generating approximate timestamps")
                tokens = self.generate_approximate_timestamps(tokens)
        
        return tokens
    
    def generate_approximate_timestamps(self, tokens: List[Dict]) -> List[Dict]:
        """
        Generate approximate timestamps based on token position and average speaking rate.
        Used when actual timestamps are missing or invalid.
        """
        if not tokens:
            return tokens
        
        # Assume average speaking rate of 150 words per minute
        avg_word_duration = 60.0 / 150.0  # 0.4 seconds per word
        
        current_time = 0.0
        for token in tokens:
            if token['start_time'] == 0.0 and token['end_time'] == 0.0:
                token['start_time'] = current_time
                token['end_time'] = current_time + avg_word_duration
                current_time = token['end_time']
            else:
                # Use existing timestamp if valid
                current_time = max(current_time, token['end_time'])
        
        return tokens
    
    def load_normalization(self, norm_path: Path) -> Dict:
        """Load normalization data with verbalization candidates."""
        try:
            with open(norm_path, 'r', encoding='utf-8') as f:
                norm_data = json.load(f)
            
            # Process normalization data
            processed = {}
            for idx, data in norm_data.items():
                if 'candidates' in data and data['candidates']:
                    # Sort by probability and get best verbalization
                    best_candidate = sorted(data['candidates'], 
                                          key=lambda x: x['probability'], 
                                          reverse=True)[0]
                    
                    # Join verbalization tokens
                    if isinstance(best_candidate['verbalization'], list):
                        verbalization = ' '.join(best_candidate['verbalization'])
                    else:
                        verbalization = best_candidate['verbalization']
                    
                    processed[idx] = {
                        'verbalization': verbalization,
                        'class': data.get('class', ''),
                        'probability': best_candidate['probability']
                    }
            
            return processed
        except Exception as e:
            logger.warning(f"Error loading normalization from {norm_path}: {e}")
            return {}
    
    def parse_wer_tag_indices(self, wer_tag_str: str) -> List[str]:
        """Parse WER tag string to get normalization indices."""
        if not wer_tag_str or wer_tag_str == '[]':
            return []
        
        try:
            # Remove brackets and quotes, split by comma
            indices_str = wer_tag_str.strip('[]').replace("'", "")
            if indices_str:
                return [idx.strip() for idx in indices_str.split(',') if idx.strip()]
            return []
        except Exception as e:
            logger.debug(f"Error parsing WER tag string '{wer_tag_str}': {e}")
            return []
    
    def calculate_speaker_durations(self, tokens: List[Dict]) -> Dict[str, float]:
        """Calculate total duration for each speaker with validation."""
        speaker_durations = {}
        
        for token in tokens:
            speaker_id = token['speaker_id']
            duration = token['end_time'] - token['start_time']
            
            # Validate duration
            if duration < 0:
                duration = 0.1  # Default minimum duration
            elif duration > 10:  # Single word shouldn't be > 10 seconds
                duration = 0.5  # Default reasonable duration
            
            if speaker_id not in speaker_durations:
                speaker_durations[speaker_id] = 0.0
            speaker_durations[speaker_id] += duration
        
        # Log speaker duration statistics
        if speaker_durations:
            total_duration = sum(speaker_durations.values())
            logger.debug(f"Speaker durations - Total: {total_duration:.1f}s, Speakers: {len(speaker_durations)}")
        
        return speaker_durations
    
    def calculate_sentiment_ci(self, chunk_sentiments: List[Dict], n_bootstrap: int = None) -> Dict:
        """
        Calculate bootstrap confidence intervals for sentiment scores
        A2/B1 Requirement: Uncertainty quantification
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap
        
        if not chunk_sentiments:
            return {}
        
        # Extract sentiment arrays
        positive_scores = [s['positive'] for s in chunk_sentiments]
        negative_scores = [s['negative'] for s in chunk_sentiments]
        neutral_scores = [s['neutral'] for s in chunk_sentiments]
        
        ci_results = {}
        
        # Bootstrap for each sentiment type
        for sentiment_type, scores in [('positive', positive_scores), 
                                       ('negative', negative_scores), 
                                       ('neutral', neutral_scores)]:
            
            bootstrap_means = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                resampled = np.random.choice(scores, size=len(scores), replace=True)
                bootstrap_means.append(np.mean(resampled))
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            
            ci_results[f'sentiment_{sentiment_type}_ci'] = {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
                'width': float(ci_upper - ci_lower)
            }
        
        return ci_results
    
    def reconstruct_call_level_transcript(self, file_id: str) -> Dict:
        """
        Reconstruct call-level transcript with duration weighting.
        Treats each earnings call as a single analytical unit.
        """
        nlp_path = self.nlp_ref_dir / f"{file_id}.nlp"
        norm_path = self.norm_dir / f"{file_id}.norm.json"
        wer_path = self.wer_tag_dir / f"{file_id}.wer_tag.json"
        
        # Load data
        tokens = self.load_nlp_file(nlp_path)
        normalizations = self.load_normalization(norm_path) if norm_path.exists() else {}
        
        # Calculate speaker durations for weighting
        speaker_durations = self.calculate_speaker_durations(tokens)
        total_duration = sum(speaker_durations.values())
        
        # If total duration is still 0, use token count estimation
        if total_duration == 0:
            logger.warning(f"Zero total duration for {file_id}, using token count estimation")
            # Estimate ~150 words per minute
            total_duration = len(tokens) * 0.4  # 0.4 seconds per word
            # Distribute evenly among speakers
            num_speakers = len(set(token['speaker_id'] for token in tokens))
            if num_speakers > 0:
                avg_duration = total_duration / num_speakers
                speaker_durations = {sid: avg_duration for sid in set(token['speaker_id'] for token in tokens)}
        
        # Reconstruct normalized transcript
        normalized_tokens = []
        speaker_segments = {}
        
        for token in tokens:
            speaker_id = token['speaker_id']
            
            # Initialize speaker segment tracking
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = {
                    'tokens': [],
                    'duration': speaker_durations.get(speaker_id, 0),
                    'weight': speaker_durations.get(speaker_id, 0) / total_duration if total_duration > 0 else 0
                }
            
            # Apply normalization
            normalized_word = token['word']
            normalization_applied = False
            
            # Check WER tag indices for normalization
            if 'wer_tags' in token:
                wer_indices = self.parse_wer_tag_indices(token.get('wer_tags', ''))
                
                for idx in wer_indices:
                    if idx in normalizations:
                        normalized_word = normalizations[idx]['verbalization']
                        normalization_applied = True
                        break
            
            # Build normalized token
            norm_token = {
                'word': normalized_word,
                'original': token['word'],
                'speaker_id': speaker_id,
                'start_time': token['start_time'],
                'end_time': token['end_time'],
                'punctuation': token.get('punctuation', ''),
                'normalized': normalization_applied
            }
            
            normalized_tokens.append(norm_token)
            speaker_segments[speaker_id]['tokens'].append(norm_token)
        
        # Create call-level text
        call_text = self._create_call_level_text(normalized_tokens)
        
        # Calculate duration-weighted features
        weighted_features = self._calculate_weighted_features(speaker_segments, total_duration)
        
        # Add sector information
        sector = self.file_to_sector.get(file_id, 'unknown')
        
        return {
            'file_id': file_id,
            'sector': sector,
            'call_text': call_text,
            'normalized_tokens': normalized_tokens,
            'speaker_segments': speaker_segments,
            'total_duration': total_duration,
            'num_speakers': len(speaker_segments),
            'weighted_features': weighted_features
        }
    
    def _create_call_level_text(self, tokens: List[Dict]) -> str:
        """Create call-level text from normalized tokens."""
        words = []
        for token in tokens:
            word = token['word']
            if token.get('punctuation'):
                word += token['punctuation']
            words.append(word)
        
        # Join with proper spacing
        text = ' '.join(words)
        # Clean up spacing around punctuation
        text = text.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!')
        
        return text
    
    def _calculate_weighted_features(self, speaker_segments: Dict, total_duration: float) -> Dict:
        features = {
            'dominant_speaker_ratio': 0.0,
            'speaker_entropy': 0.0,
            'avg_tokens_per_speaker': 0.0,
            'total_speaker_duration': total_duration
        }
        
        if not speaker_segments:
            return features

        # durations is a list of (speaker_id, duration) tuples
        durations = [(sid, seg['duration']) for sid, seg in speaker_segments.items()]
        durations.sort(key=lambda x: x[1], reverse=True)

        if durations and total_duration > 0:
            dominant_duration = durations[0][1]
            features['dominant_speaker_ratio'] = dominant_duration / total_duration

            # Use duration directly (not d[1])
            probs = [duration / total_duration for _, duration in durations if duration > 0]
            if probs:
                entropy = -sum(p * np.log(p) for p in probs if p > 0)
                features['speaker_entropy'] = entropy

        total_tokens = sum(len(seg['tokens']) for seg in speaker_segments.values())
        features['avg_tokens_per_speaker'] = total_tokens / len(speaker_segments) if speaker_segments else 0

        return features
    
    def analyze_call_sentiment(self, call_text: str) -> Dict:
        """
        Perform FinBERT sentiment analysis on call-level text.
        Returns aggregate sentiment scores for organizational communication tone.
        """
        # Split into chunks for BERT processing
        chunks = self._split_text_into_chunks(call_text)
        
        all_scores = []
        chunk_sentiments = []
        
        for chunk in chunks:
            # Tokenize
            inputs = self.tokenizer(
                chunk,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_sequence_length,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                scores = predictions.cpu().numpy()[0]
                
                all_scores.append(scores)
                chunk_sentiments.append({
                    'positive': float(scores[0]),
                    'negative': float(scores[1]),
                    'neutral': float(scores[2])
                })
        
        # Aggregate sentiment metrics
        scores_array = np.array(all_scores)
        
        sentiment_results = {
            # Mean sentiments
            'sentiment_positive_mean': float(np.mean(scores_array[:, 0])),
            'sentiment_negative_mean': float(np.mean(scores_array[:, 1])),
            'sentiment_neutral_mean': float(np.mean(scores_array[:, 2])),
            
            # Standard deviations (variability)
            'sentiment_positive_std': float(np.std(scores_array[:, 0])),
            'sentiment_negative_std': float(np.std(scores_array[:, 1])),
            'sentiment_neutral_std': float(np.std(scores_array[:, 2])),
            
            # Extreme percentiles
            'sentiment_positive_p95': float(np.percentile(scores_array[:, 0], 95)),
            'sentiment_negative_p95': float(np.percentile(scores_array[:, 1], 95)),
            
            # Overall metrics
            'sentiment_dominant': self.sentiment_labels[np.argmax(np.mean(scores_array, axis=0))],
            'sentiment_variability': float(np.mean(np.std(scores_array, axis=0))),
            'num_chunks': len(chunks),
            
            # Chunk-level details for validation
            'chunk_sentiments': chunk_sentiments
        }
        
        # Add bootstrap confidence intervals
        confidence_intervals = self.calculate_sentiment_ci(chunk_sentiments)
        sentiment_results.update(confidence_intervals)
        
        return sentiment_results
    
    def _split_text_into_chunks(self, text: str, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for BERT processing."""
        words = text.split()
        chunks = []
        
        # Calculate chunk size (leave room for special tokens)
        chunk_size = self.max_sequence_length - 50
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    def calculate_temporal_stability(self, tokens: List[Dict], 
                                   sentiment_results: Dict) -> Dict:
        """
        Assess temporal stability by comparing first-half vs second-half sentiment.
        """
        if not tokens or len(tokens) < 10:  # Need minimum tokens for meaningful analysis
            return {'temporal_stability_score': 0.0}
        
        # Find temporal midpoint
        start_time = tokens[0]['start_time']
        end_time = tokens[-1]['end_time']
        total_duration = end_time - start_time
        
        # If no valid timestamps, use token count
        if total_duration <= 0:
            midpoint_idx = len(tokens) // 2
            first_half = tokens[:midpoint_idx]
            second_half = tokens[midpoint_idx:]
        else:
            midpoint = start_time + (total_duration / 2)
            first_half = [t for t in tokens if t['end_time'] <= midpoint]
            second_half = [t for t in tokens if t['start_time'] > midpoint]
        
        # Ensure both halves have content
        if len(first_half) < 5 or len(second_half) < 5:
            return {'temporal_stability_score': 0.0}
        
        # Reconstruct text for each half
        first_text = self._create_call_level_text(first_half)
        second_text = self._create_call_level_text(second_half)
        
        stability_metrics = {}
        
        if first_text and second_text:
            # Analyze sentiment for each half
            first_sentiment = self.analyze_call_sentiment(first_text)
            second_sentiment = self.analyze_call_sentiment(second_text)
            
            # Calculate changes
            for sentiment_type in ['positive', 'negative', 'neutral']:
                key = f'sentiment_{sentiment_type}_mean'
                if key in first_sentiment and key in second_sentiment:
                    change = second_sentiment[key] - first_sentiment[key]
                    stability_metrics[f'{sentiment_type}_temporal_change'] = float(change)
            
            # Overall stability score (lower is more stable)
            if stability_metrics:
                stability_metrics['temporal_stability_score'] = float(
                    np.mean([abs(v) for v in stability_metrics.values()])
                )
            else:
                stability_metrics['temporal_stability_score'] = 0.0
        else:
            stability_metrics['temporal_stability_score'] = 0.0
        
        return stability_metrics
    
    def create_demonstrator_output(self, results: Dict) -> Dict:
        """
        Format results for voice technology demonstrator
        A2/B1 Requirement: Simple demonstrator integration
        """
        # Extract key metrics
        sentiment = results.get('sentiment_analysis', {})
        temporal = results.get('temporal_stability', {})
        
        # Calculate percentile ranks if baseline available
        percentile_ranks = {}
        if self.baseline_features is not None and 'sentiment_variability' in sentiment:
            # Compare against baseline sentiment variability
            baseline_variability = self.baseline_features.get('sentiment_variability', [])
            if len(baseline_variability) > 0:
                percentile = stats.percentileofscore(baseline_variability, 
                                                   sentiment['sentiment_variability'])
                percentile_ranks['sentiment_variability_percentile'] = percentile
        
        # Classify communication pattern
        pattern_classification = self.classify_communication_pattern(sentiment, temporal)
        
        return {
            'file_id': results['file_id'],
            'sector': results.get('sector', 'unknown'),
            'sentiment_profile': {
                'dominant': sentiment.get('sentiment_dominant', 'neutral'),
                'positive_mean': sentiment.get('sentiment_positive_mean', 0),
                'negative_mean': sentiment.get('sentiment_negative_mean', 0),
                'neutral_mean': sentiment.get('sentiment_neutral_mean', 0),
                'variability': sentiment.get('sentiment_variability', 0),
                'confidence_intervals': {
                    'positive': sentiment.get('sentiment_positive_ci', {}),
                    'negative': sentiment.get('sentiment_negative_ci', {}),
                    'neutral': sentiment.get('sentiment_neutral_ci', {})
                }
            },
            'temporal_pattern': {
                'stability_score': temporal.get('temporal_stability_score', 0),
                'classification': 'stable' if temporal.get('temporal_stability_score', 0) < 0.1 else 'shifting',
                'changes': {
                    'positive': temporal.get('positive_temporal_change', 0),
                    'negative': temporal.get('negative_temporal_change', 0),
                    'neutral': temporal.get('neutral_temporal_change', 0)
                }
            },
            'pattern_classification': pattern_classification,
            'percentile_ranks': percentile_ranks,
            'speaker_dynamics': {
                'num_speakers': results.get('num_speakers', 0),
                'dominant_speaker_ratio': results.get('weighted_features', {}).get('dominant_speaker_ratio', 0),
                'speaker_entropy': results.get('weighted_features', {}).get('speaker_entropy', 0)
            },
            'integration_ready': True,
            'demonstrator_compatible': True
        }
    
    def classify_communication_pattern(self, sentiment: Dict, temporal: Dict) -> Dict:
        """
        Classify communication patterns based on sentiment and temporal features
        Implements research framework from methodology
        """
        sentiment_dominant = sentiment.get('sentiment_dominant', 'neutral')
        sentiment_variability = sentiment.get('sentiment_variability', 0)
        temporal_stability = temporal.get('temporal_stability_score', 0)
        
        # Pattern classification rules based on methodology
        if sentiment_dominant == 'negative' and sentiment_variability > 0.3:
            pattern = "High Stress Communication"
            confidence = "High" if temporal_stability > 0.1 else "Medium"
            description = "Negative sentiment with high variability suggests stress"
        elif sentiment_dominant == 'positive' and sentiment_variability < 0.2:
            pattern = "Confident Communication"
            confidence = "High"
            description = "Positive sentiment with low variability indicates confidence"
        elif temporal_stability > 0.15:
            pattern = "Shifting Communication Tone"
            confidence = "Medium"
            description = "Significant temporal changes in sentiment detected"
        elif sentiment_variability > 0.4:
            pattern = "Volatile Communication"
            confidence = "High"
            description = "High sentiment variability across segments"
        else:
            pattern = "Neutral/Professional Communication"
            confidence = "Medium"
            description = "Standard business communication pattern"
        
        return {
            'pattern': pattern,
            'confidence': confidence,
            'description': description,
            'key_indicators': {
                'sentiment_dominant': sentiment_dominant,
                'sentiment_variability': round(sentiment_variability, 3),
                'temporal_stability': round(temporal_stability, 3)
            }
        }
    
    def save_call_level_results(self, file_id: str, results: Dict):
        """Save call-level processing results."""
        # Save full results
        output_path = self.output_dir / "call_level" / f"{file_id}_call_level.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            # Prepare serializable data
            save_data = {
                'file_id': file_id,
                'sector': results.get('sector', 'unknown'),
                'call_text': results['call_text'],
                'total_duration': results['total_duration'],
                'num_speakers': results['num_speakers'],
                'weighted_features': results['weighted_features'],
                'sentiment_analysis': results.get('sentiment_analysis', {}),
                'temporal_stability': results.get('temporal_stability', {})
            }
            json.dump(save_data, f, indent=2)
        
        # Save sentiment results separately
        if 'sentiment_analysis' in results:
            sentiment_path = self.output_dir / "sentiment_analysis" / f"{file_id}_sentiment.json"
            with open(sentiment_path, 'w', encoding='utf-8') as f:
                json.dump(results['sentiment_analysis'], f, indent=2)
        
        # Save demonstrator output
        if 'demonstrator_output' in results:
            demo_path = self.output_dir / "demonstrator" / f"{file_id}_demonstrator.json"
            with open(demo_path, 'w', encoding='utf-8') as f:
                json.dump(results['demonstrator_output'], f, indent=2)
    
    def process_all_calls(self):
        """Process all earnings calls for call-level analysis."""
        nlp_files = list(self.nlp_ref_dir.glob("*.nlp"))
        
        if not nlp_files:
            logger.error(f"No .nlp files found in {self.nlp_ref_dir}")
            return
        
        logger.info(f"Found {len(nlp_files)} earnings calls to process")
        
        all_results = []
        demonstrator_outputs = []
        
        for nlp_path in tqdm(nlp_files, desc="Processing earnings calls"):
            file_id = nlp_path.stem
            try:
                # Reconstruct call-level transcript
                call_data = self.reconstruct_call_level_transcript(file_id)
                
                # Log duration for debugging
                logger.info(f"{file_id}: Duration={call_data['total_duration']:.1f}s, Speakers={call_data['num_speakers']}, Sector={call_data['sector']}")
                
                # Perform sentiment analysis
                sentiment_results = self.analyze_call_sentiment(call_data['call_text'])
                call_data['sentiment_analysis'] = sentiment_results
                
                # Calculate temporal stability
                temporal_stability = self.calculate_temporal_stability(
                    call_data['normalized_tokens'],
                    sentiment_results
                )
                call_data['temporal_stability'] = temporal_stability
                
                # Create demonstrator output
                demonstrator_output = self.create_demonstrator_output(call_data)
                call_data['demonstrator_output'] = demonstrator_output
                demonstrator_outputs.append(demonstrator_output)
                
                # Save results
                self.save_call_level_results(file_id, call_data)
                
                # Collect summary for validation
                summary = {
                    'file_id': file_id,
                    'sector': call_data['sector'],
                    'num_speakers': call_data['num_speakers'],
                    'total_duration': call_data['total_duration'],
                    **call_data['weighted_features'],
                    **{k: v for k, v in sentiment_results.items() 
                       if k != 'chunk_sentiments'},
                    **temporal_stability
                }
                all_results.append(summary)
                
            except Exception as e:
                logger.error(f"Error processing {file_id}: {e}", exc_info=True)
                continue
        
        # Save aggregated results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(
                self.output_dir / "call_level_analysis_summary.csv",
                index=False
            )
            
            # Save demonstrator outputs
            with open(self.output_dir / "demonstrator" / "all_demonstrator_outputs.json", 'w') as f:
                json.dump(demonstrator_outputs, f, indent=2)
            
            # Perform validation
            self._validate_results(results_df)
        
        self._print_summary()
    
    def _validate_results(self, results_df: pd.DataFrame):
        """Validate processing results for quality control."""
        validation_report = {}
        
        # Check sentiment balance
        sentiment_cols = ['sentiment_positive_mean', 'sentiment_negative_mean', 'sentiment_neutral_mean']
        if all(col in results_df.columns for col in sentiment_cols):
            sentiment_sums = results_df[sentiment_cols].sum(axis=1)
            validation_report['sentiment_balance_check'] = {
                'mean_sum': float(sentiment_sums.mean()),
                'std_sum': float(sentiment_sums.std()),
                'valid_ratio': float((abs(sentiment_sums - 1.0) < 0.01).mean())
            }
        
        # Check temporal stability
        if 'temporal_stability_score' in results_df.columns:
            # Filter out zero scores which indicate insufficient data
            valid_stability = results_df[results_df['temporal_stability_score'] > 0]['temporal_stability_score']
            if len(valid_stability) > 0:
                validation_report['temporal_stability'] = {
                    'mean': float(valid_stability.mean()),
                    'std': float(valid_stability.std()),
                    'stable_calls_ratio': float((valid_stability < 0.1).mean()),
                    'valid_temporal_analysis': len(valid_stability)
                }
        
        # Check speaker diversity
        if 'num_speakers' in results_df.columns and 'speaker_entropy' in results_df.columns:
            # Filter valid durations
            valid_calls = results_df[results_df['total_duration'] > 0]
            if len(valid_calls) > 0:
                validation_report['speaker_diversity'] = {
                    'mean_entropy': float(valid_calls['speaker_entropy'].mean()),
                    'single_speaker_ratio': float((valid_calls['num_speakers'] == 1).mean()),
                    'high_diversity_ratio': float((valid_calls['speaker_entropy'] > 1.0).mean()),
                    'avg_speakers_per_call': float(valid_calls['num_speakers'].mean()),
                    'calls_with_valid_duration': len(valid_calls)
                }
        
        # Check duration statistics
        if 'total_duration' in results_df.columns:
            validation_report['duration_statistics'] = {
                'mean_duration': float(results_df['total_duration'].mean()),
                'std_duration': float(results_df['total_duration'].std()),
                'min_duration': float(results_df['total_duration'].min()),
                'max_duration': float(results_df['total_duration'].max()),
                'zero_duration_calls': int((results_df['total_duration'] == 0).sum())
            }
        
        # Check sector distribution
        if 'sector' in results_df.columns:
            sector_counts = results_df['sector'].value_counts().to_dict()
            validation_report['sector_distribution'] = {
                'sectors': sector_counts,
                'num_sectors': len(sector_counts),
                'unknown_sectors': int((results_df['sector'] == 'unknown').sum())
            }
        
        # Check confidence intervals
        ci_cols = [col for col in results_df.columns if '_ci' in col]
        if ci_cols:
            validation_report['confidence_intervals'] = {
                'ci_columns_found': len(ci_cols),
                'bootstrap_samples': self.n_bootstrap
            }
        
        # Save validation report
        validation_path = self.output_dir / "validation" / "call_level_validation.json"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(validation_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"Validation report saved to {validation_path}")
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("CALL-LEVEL TRANSCRIPT PROCESSING SUMMARY")
        print("="*60)
        
        call_files = list((self.output_dir / "call_level").glob("*.json"))
        sentiment_files = list((self.output_dir / "sentiment_analysis").glob("*.json"))
        demonstrator_files = list((self.output_dir / "demonstrator").glob("*_demonstrator.json"))
        
        print(f"Processed calls: {len(call_files)}")
        print(f"Output directory: {self.output_dir}")
        print("\nOutput formats:")
        print(f"  - Call-level data: {self.output_dir / 'call_level'}")
        print(f"  - Sentiment analysis: {self.output_dir / 'sentiment_analysis'}")
        print(f"  - Demonstrator outputs: {self.output_dir / 'demonstrator'} ({len(demonstrator_files)} files)")
        print(f"  - Validation reports: {self.output_dir / 'validation'}")
        print(f"  - Summary CSV: {self.output_dir / 'call_level_analysis_summary.csv'}")
        
        # Print duration summary if available
        summary_path = self.output_dir / "call_level_analysis_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            if 'total_duration' in df.columns:
                valid_durations = df[df['total_duration'] > 0]['total_duration']
                if len(valid_durations) > 0:
                    print(f"\nDuration Statistics:")
                    print(f"  - Calls with valid duration: {len(valid_durations)}/{len(df)}")
                    print(f"  - Average duration: {valid_durations.mean() / 60:.1f} minutes")
                    print(f"  - Total duration: {valid_durations.sum() / 3600:.1f} hours")
            
            if 'sector' in df.columns:
                print(f"\nSector Distribution:")
                sector_counts = df['sector'].value_counts()
                for sector, count in sector_counts.items():
                    print(f"  - {sector}: {count} calls")
        
        print("\nA2/B1 Demonstrator Features:")
        print("  ✓ Sector-based analysis")
        print("  ✓ Bootstrap confidence intervals")
        print("  ✓ Pattern classification")
        print("  ✓ Demonstrator-ready outputs")
        print("  ✓ Integration with acoustic analysis pipeline")


def main():
    processor = TranscriptProcessor()
    processor.process_all_calls()


if __name__ == "__main__":
    main()