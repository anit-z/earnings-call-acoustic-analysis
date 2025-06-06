#!/usr/bin/env python3
"""
Acoustic Feature Extraction for Earnings Call Analysis
Implements call-level aggregation for organizational communication climate analysis
Following contemporary scientific framework with bootstrap uncertainty quantification
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

import librosa
import opensmile
from scipy import stats
from scipy.stats import median_abs_deviation
from tqdm import tqdm
import parselmouth
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CallLevelAcousticExtractor:
    """
    Call-level acoustic feature extraction with duration weighting
    Implements organizational communication climate analysis
    """
    
    def __init__(self, 
                 frame_duration_ms: int = 50,
                 step_size_ms: int = 20,
                 f0_min: float = 75,
                 f0_max: float = 500,
                 n_bootstrap: int = 10000):
        """
        Initialize acoustic feature extractor
        
        Args:
            frame_duration_ms: Frame duration in milliseconds
            step_size_ms: Step size in milliseconds
            f0_min: Minimum F0 frequency (Hz)
            f0_max: Maximum F0 frequency (Hz)
            n_bootstrap: Number of bootstrap samples for CI
        """
        self.frame_duration_ms = frame_duration_ms
        self.step_size_ms = step_size_ms
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.n_bootstrap = n_bootstrap
        
        # Initialize openSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals
        )
        
        # Load metadata for sector normalization
        self.sector_mappings = self._load_sector_mappings()
        
    def _load_sector_mappings(self) -> Dict[str, str]:
        """Load sector mappings from metadata"""
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
    
    def extract_call_level_features(self, audio_path: str) -> Dict:
        """
        Extract call-level acoustic features with duration weighting
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of acoustic features and statistics
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            logger.info(f"Processing {audio_path}: {len(audio)/sr:.1f}s duration")
            
            # Extract fundamental frequency (F0) features
            f0_features = self._extract_f0_features(audio, sr)
            
            # Extract voice quality features
            voice_quality = self._extract_voice_quality_features(audio, sr)
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features(audio, sr)
            
            # Extract spectral features
            spectral_features = self._extract_spectral_features(audio, sr)
            
            # Extract openSMILE features
            smile_features = self._extract_opensmile_features(audio_path)
            
            # Combine all features
            features = {
                **f0_features,
                **voice_quality,
                **temporal_features,
                **spectral_features,
                **smile_features,
                'duration_s': len(audio) / sr,
                'file_id': Path(audio_path).stem
            }
            
            # Add bootstrap confidence intervals for key features
            features_with_ci = self._add_bootstrap_ci(features, audio, sr)
            
            return features_with_ci
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return {'file_id': Path(audio_path).stem, 'error': str(e)}
    
    def _extract_f0_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract fundamental frequency features using Parselmouth"""
        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Extract pitch
        pitch = sound.to_pitch(
            time_step=self.step_size_ms / 1000,
            pitch_floor=self.f0_min,
            pitch_ceiling=self.f0_max
        )
        
        # Get F0 values (excluding unvoiced)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]  # Remove unvoiced frames
        
        if len(f0_values) > 0:
            # Calculate distributional statistics
            f0_features = {
                'f0_mean': float(np.mean(f0_values)),
                'f0_std': float(np.std(f0_values)),
                'f0_median': float(np.median(f0_values)),
                'f0_mad': float(median_abs_deviation(f0_values)),
                'f0_cv': float(np.std(f0_values) / np.mean(f0_values)) if np.mean(f0_values) > 0 else 0,
                'f0_min': float(np.min(f0_values)),
                'f0_max': float(np.max(f0_values)),
                'f0_range': float(np.max(f0_values) - np.min(f0_values)),
                'f0_p05': float(np.percentile(f0_values, 5)),
                'f0_p25': float(np.percentile(f0_values, 25)),
                'f0_p75': float(np.percentile(f0_values, 75)),
                'f0_p95': float(np.percentile(f0_values, 95)),
                'f0_iqr': float(np.percentile(f0_values, 75) - np.percentile(f0_values, 25)),
                'f0_skewness': float(stats.skew(f0_values)),
                'f0_kurtosis': float(stats.kurtosis(f0_values)),
                'voiced_fraction': float(len(f0_values) / len(pitch.selected_array['frequency']))
            }
        else:
            f0_features = {f'f0_{stat}': 0.0 for stat in 
                         ['mean', 'std', 'median', 'mad', 'cv', 'min', 'max', 
                          'range', 'p05', 'p25', 'p75', 'p95', 'iqr', 'skewness', 'kurtosis']}
            f0_features['voiced_fraction'] = 0.0
        
        return f0_features
    
    def _extract_voice_quality_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract voice quality features (jitter, shimmer, HNR)"""
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Point process for jitter calculation
        pitch = sound.to_pitch()
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 
                                              self.f0_min, self.f0_max)
        
        # Jitter measurements
        jitter_features = {}
        try:
            jitter_features['jitter_local'] = float(
                parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            )
            jitter_features['jitter_ddp'] = float(
                parselmouth.praat.call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            )
        except:
            jitter_features = {'jitter_local': 0.0, 'jitter_ddp': 0.0}
        
        # Shimmer measurements
        shimmer_features = {}
        try:
            shimmer_features['shimmer_local'] = float(
                parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 
                                      0, 0, 0.0001, 0.02, 1.3, 1.6)
            )
            shimmer_features['shimmer_apq3'] = float(
                parselmouth.praat.call([sound, point_process], "Get shimmer (apq3)", 
                                      0, 0, 0.0001, 0.02, 1.3, 1.6)
            )
        except:
            shimmer_features = {'shimmer_local': 0.0, 'shimmer_apq3': 0.0}
        
        # Harmonics-to-Noise Ratio
        try:
            harmonicity = sound.to_harmonicity(time_step=0.01, minimum_pitch=self.f0_min)
            hnr_values = harmonicity.values[harmonicity.values != -200]  # Remove silent frames
            if len(hnr_values) > 0:
                hnr_features = {
                    'hnr_mean': float(np.mean(hnr_values)),
                    'hnr_std': float(np.std(hnr_values)),
                    'hnr_p05': float(np.percentile(hnr_values, 5))
                }
            else:
                hnr_features = {'hnr_mean': 0.0, 'hnr_std': 0.0, 'hnr_p05': 0.0}
        except:
            hnr_features = {'hnr_mean': 0.0, 'hnr_std': 0.0, 'hnr_p05': 0.0}
        
        return {**jitter_features, **shimmer_features, **hnr_features}
    
    def _extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract temporal features (speech rate, pauses)"""
        # Energy-based voice activity detection
        hop_length = int(self.step_size_ms * sr / 1000)
        frame_length = int(self.frame_duration_ms * sr / 1000)
        
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                    hop_length=hop_length)[0]
        
        # Adaptive threshold
        energy_threshold = np.percentile(energy, 30)
        speech_frames = energy > energy_threshold
        
        # Calculate pause statistics
        pause_lengths = []
        pause_start = None
        
        for i, is_speech in enumerate(speech_frames):
            if not is_speech and pause_start is None:
                pause_start = i
            elif is_speech and pause_start is not None:
                pause_length = (i - pause_start) * hop_length / sr
                if pause_length > 0.1:  # Consider pauses > 100ms
                    pause_lengths.append(pause_length)
                pause_start = None
        
        # Calculate temporal features
        total_duration = len(audio) / sr
        speech_duration = np.sum(speech_frames) * hop_length / sr
        
        temporal_features = {
            'speech_rate': float(speech_duration / total_duration) if total_duration > 0 else 0,
            'pause_frequency': float(len(pause_lengths) / total_duration) if total_duration > 0 else 0,
            'pause_duration_mean': float(np.mean(pause_lengths)) if pause_lengths else 0,
            'pause_duration_std': float(np.std(pause_lengths)) if pause_lengths else 0,
            'pause_duration_cv': float(np.std(pause_lengths) / np.mean(pause_lengths)) 
                                if pause_lengths and np.mean(pause_lengths) > 0 else 0,
            'speaking_time_ratio': float(speech_duration / total_duration) if total_duration > 0 else 0
        }
        
        return temporal_features
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract spectral features"""
        hop_length = int(self.step_size_ms * sr / 1000)
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        spectral_features = {
            # Spectral centroid statistics
            'spectral_centroid_mean': float(np.mean(centroid)),
            'spectral_centroid_std': float(np.std(centroid)),
            'spectral_centroid_cv': float(np.std(centroid) / np.mean(centroid)) 
                                   if np.mean(centroid) > 0 else 0,
            
            # Spectral rolloff statistics
            'spectral_rolloff_mean': float(np.mean(rolloff)),
            'spectral_rolloff_std': float(np.std(rolloff)),
            
            # Spectral bandwidth statistics
            'spectral_bandwidth_mean': float(np.mean(bandwidth)),
            'spectral_bandwidth_std': float(np.std(bandwidth)),
            
            # MFCC statistics (mean and std for each coefficient)
            **{f'mfcc{i}_mean': float(np.mean(mfccs[i])) for i in range(13)},
            **{f'mfcc{i}_std': float(np.std(mfccs[i])) for i in range(13)}
        }
        
        return spectral_features
    
    def _extract_opensmile_features(self, audio_path: str) -> Dict:
        """Extract ComParE features using openSMILE"""
        try:
            # Extract features
            features_df = self.smile.process_file(audio_path)
            
            # Convert to dictionary with proper naming
            smile_features = {}
            for col in features_df.columns:
                # Simplify feature names
                feature_name = col.replace(' ', '_').lower()
                if 'f0' in feature_name or 'pitch' in feature_name:
                    smile_features[f'smile_{feature_name}'] = float(features_df[col].iloc[0])
            
            return smile_features
        except Exception as e:
            logger.warning(f"OpenSMILE extraction failed: {e}")
            return {}
    
    def _add_bootstrap_ci(self, features: Dict, audio: np.ndarray, sr: int) -> Dict:
        """Add bootstrap confidence intervals for key features"""
        # Extract raw F0 values for bootstrap
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]
        
        if len(f0_values) > 10:  # Need sufficient data for bootstrap
            # Bootstrap F0 variability
            f0_cv_bootstrap = []
            for _ in range(self.n_bootstrap):
                bootstrap_sample = np.random.choice(f0_values, size=len(f0_values), replace=True)
                cv = np.std(bootstrap_sample) / np.mean(bootstrap_sample) if np.mean(bootstrap_sample) > 0 else 0
                f0_cv_bootstrap.append(cv)
            
            features['f0_cv_ci_lower'] = float(np.percentile(f0_cv_bootstrap, 2.5))
            features['f0_cv_ci_upper'] = float(np.percentile(f0_cv_bootstrap, 97.5))
            features['f0_cv_ci_width'] = features['f0_cv_ci_upper'] - features['f0_cv_ci_lower']
        
        return features
    
    def normalize_features_by_sector(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sector-specific normalization
        
        Args:
            features_df: DataFrame with features and file_id
            
        Returns:
            Normalized features DataFrame
        """
        # Add sector information
        features_df['sector'] = features_df['file_id'].map(self.sector_mappings)
        
        # Define features to normalize (exclude metadata columns)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['file_id', 'sector', 'error', 'duration_s']]
        
        # Apply sector-specific normalization
        normalized_df = features_df.copy()
        
        for sector in features_df['sector'].unique():
            if pd.isna(sector):
                continue
                
            sector_mask = features_df['sector'] == sector
            sector_data = features_df.loc[sector_mask, feature_cols]
            
            if len(sector_data) > 1:
                # Min-max scaling within sector
                scaler = MinMaxScaler()
                normalized_values = scaler.fit_transform(sector_data)
                normalized_df.loc[sector_mask, feature_cols] = normalized_values
            else:
                # If only one sample in sector, center at 0.5
                normalized_df.loc[sector_mask, feature_cols] = 0.5
        
        # Apply winsorization to handle extreme values
        for col in feature_cols:
            normalized_df[col] = stats.mstats.winsorize(normalized_df[col], limits=[0.05, 0.05])
        
        return normalized_df
    
    def calculate_acoustic_volatility_index(self, features: Dict) -> float:
        """
        Calculate composite acoustic volatility index
        
        Args:
            features: Dictionary of acoustic features
            
        Returns:
            Acoustic volatility index
        """
        # Key volatility indicators
        volatility_features = [
            'f0_cv',
            'f0_iqr',
            'spectral_centroid_cv',
            'pause_duration_cv',
            'jitter_local',
            'shimmer_local'
        ]
        
        volatility_scores = []
        for feat in volatility_features:
            if feat in features and not pd.isna(features[feat]):
                volatility_scores.append(features[feat])
        
        if volatility_scores:
            # Normalize each score to 0-1 range based on typical values
            normalized_scores = []
            for i, score in enumerate(volatility_scores):
                # Use feature-specific normalization ranges
                if volatility_features[i] == 'f0_cv':
                    normalized = np.clip(score / 0.3, 0, 1)  # CV typically 0-0.3
                elif volatility_features[i] == 'jitter_local':
                    normalized = np.clip(score / 0.02, 0, 1)  # Jitter typically 0-2%
                elif volatility_features[i] == 'shimmer_local':
                    normalized = np.clip(score / 0.05, 0, 1)  # Shimmer typically 0-5%
                else:
                    normalized = score  # Already normalized
                normalized_scores.append(normalized)
            
            return float(np.mean(normalized_scores))
        else:
            return 0.0
    
    def validate_features(self, features_df: pd.DataFrame) -> Dict:
        """
        Validate extracted features for quality control
        
        Args:
            features_df: DataFrame of extracted features
            
        Returns:
            Validation report
        """
        validation = {
            'total_files': len(features_df),
            'successful_extractions': 0,
            'feature_completeness': {},
            'feature_ranges': {},
            'internal_consistency': {}
        }
        
        # Check successful extractions - Fixed error handling
        if 'error' in features_df.columns:
            successful_mask = features_df['error'].isna()
            validation['successful_extractions'] = successful_mask.sum()
        else:
            # If no error column, all are successful
            validation['successful_extractions'] = len(features_df)
        
        # Check feature completeness
        feature_cols = [col for col in features_df.columns 
                    if col not in ['file_id', 'sector', 'error', 'duration_s']]
        
        for col in feature_cols:
            if col in features_df.columns:
                valid_values = features_df[col].notna().sum()
                validation['feature_completeness'][col] = float(valid_values / len(features_df))
        
        # Check feature ranges for key features
        key_features = ['f0_mean', 'f0_cv', 'f0_std', 'jitter_local', 'shimmer_local', 
                    'hnr_mean', 'acoustic_volatility_index']
        
        for col in key_features:
            if col in features_df.columns and features_df[col].notna().any():
                validation['feature_ranges'][col] = {
                    'min': float(features_df[col].min()),
                    'max': float(features_df[col].max()),
                    'mean': float(features_df[col].mean()),
                    'std': float(features_df[col].std()),
                    'median': float(features_df[col].median())
                }
        
        # Check internal consistency (correlations between related features)
        consistency_pairs = [
            ('f0_std', 'f0_cv'),
            ('jitter_local', 'shimmer_local'),
            ('f0_mean', 'f0_median'),
            ('spectral_centroid_mean', 'spectral_centroid_std')
        ]
        
        for feat1, feat2 in consistency_pairs:
            if feat1 in features_df.columns and feat2 in features_df.columns:
                valid_mask = features_df[[feat1, feat2]].notna().all(axis=1)
                if valid_mask.sum() > 1:
                    corr = features_df.loc[valid_mask, [feat1, feat2]].corr().iloc[0, 1]
                    validation['internal_consistency'][f'{feat1}_vs_{feat2}'] = float(corr)
        
        # Check for outliers in key features
        validation['outlier_detection'] = {}
        for col in ['f0_cv', 'acoustic_volatility_index']:
            if col in features_df.columns:
                q1 = features_df[col].quantile(0.25)
                q3 = features_df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((features_df[col] < (q1 - 1.5 * iqr)) | 
                        (features_df[col] > (q3 + 1.5 * iqr))).sum()
                validation['outlier_detection'][col] = {
                    'n_outliers': int(outliers),
                    'outlier_ratio': float(outliers / len(features_df))
                }
        
        return validation


def main():
    parser = argparse.ArgumentParser(
        description="Extract call-level acoustic features for earnings calls"
    )
    parser.add_argument("--audio_dir", type=str, required=True,
                       help="Directory containing processed audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save extracted features")
    parser.add_argument("--features", type=str, 
                       default="pitch,voice_quality,temporal,spectral,opensmile",
                       help="Comma-separated list of feature types to extract")
    parser.add_argument("--normalize", action="store_true",
                       help="Apply sector-specific normalization")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation after extraction")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = CallLevelAcousticExtractor()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get audio files
    audio_files = list(Path(args.audio_dir).glob("*.wav"))
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Extract features for all files
    all_features = []
    
    for audio_file in tqdm(audio_files, desc="Extracting acoustic features"):
        features = extractor.extract_call_level_features(str(audio_file))
        
        # Add acoustic volatility index
        features['acoustic_volatility_index'] = extractor.calculate_acoustic_volatility_index(features)
        
        all_features.append(features)
        
        # Save individual feature file
        output_path = Path(args.output_dir) / f"{audio_file.stem}_features.json"
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=2)
    
    # Create features DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Apply normalization if requested
    if args.normalize:
        logger.info("Applying sector-specific normalization...")
        features_df = extractor.normalize_features_by_sector(features_df)
    
    # Save aggregated features
    features_df.to_csv(
        Path(args.output_dir) / "all_features.csv",
        index=False
    )
    
    # Run validation if requested
    if args.validate:
        logger.info("Running feature validation...")
        validation_report = extractor.validate_features(features_df)
        
        with open(Path(args.output_dir) / "validation_report.json", 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"Validation complete: {validation_report['successful_extractions']}/{validation_report['total_files']} files processed successfully")
    
    # Generate summary statistics
    summary_stats = {
        'num_calls_processed': len(features_df),
        'average_duration': float(features_df['duration_s'].mean()),
        'acoustic_volatility_stats': {
            'mean': float(features_df['acoustic_volatility_index'].mean()),
            'std': float(features_df['acoustic_volatility_index'].std()),
            'p25': float(features_df['acoustic_volatility_index'].quantile(0.25)),
            'p50': float(features_df['acoustic_volatility_index'].quantile(0.50)),
            'p75': float(features_df['acoustic_volatility_index'].quantile(0.75)),
            'p95': float(features_df['acoustic_volatility_index'].quantile(0.95))
        }
    }
    
    with open(Path(args.output_dir) / "extraction_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info("Acoustic feature extraction complete!")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()