# src/feature_extraction/acoustic_features.py
"""Extract acoustic features from earnings call audio"""

import numpy as np
import pandas as pd
import librosa
import opensmile
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import ACOUSTIC_CONFIG, PROCESSED_DATA_DIR, FEATURES_DIR

class AcousticFeatureExtractor:
    def __init__(self, config=ACOUSTIC_CONFIG):
        self.config = config
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        
    def extract_f0_features(self, audio, sr):
        """Extract F0-related features"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=self.config['f0_min'], 
            fmax=self.config['f0_max'],
            sr=sr
        )
        
        # Remove unvoiced segments
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            features = {
                'f0_mean': np.mean(f0_voiced),
                'f0_std': np.std(f0_voiced),
                'f0_cv': np.std(f0_voiced) / np.mean(f0_voiced) if np.mean(f0_voiced) > 0 else 0,
                'f0_min': np.min(f0_voiced),
                'f0_max': np.max(f0_voiced),
                'f0_range': np.max(f0_voiced) - np.min(f0_voiced),
                'f0_percentile_95': np.percentile(f0_voiced, 95),
                'jitter': self.calculate_jitter(f0_voiced)
            }
        else:
            features = {k: 0 for k in ['f0_mean', 'f0_std', 'f0_cv', 'f0_min', 
                                       'f0_max', 'f0_range', 'f0_percentile_95', 'jitter']}
        
        return features
    
    def calculate_jitter(self, f0_voiced):
        """Calculate jitter (F0 perturbation)"""
        if len(f0_voiced) < 2:
            return 0
        
        period_differences = np.abs(np.diff(1/f0_voiced))
        mean_period = np.mean(1/f0_voiced)
        
        return np.mean(period_differences) / mean_period if mean_period > 0 else 0
    
    def extract_temporal_features(self, audio, sr):
        """Extract speech rate and pause features"""
        # Simple voice activity detection
        energy = librosa.feature.rms(y=audio)[0]
        threshold = np.mean(energy) * 0.5
        voiced = energy > threshold
        
        # Calculate speaking rate (syllables per second approximation)
        # This is simplified - real implementation would use syllable detection
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Pause detection
        pause_lengths = []
        in_pause = False
        pause_start = 0
        
        for i, is_voiced in enumerate(voiced):
            if not is_voiced and not in_pause:
                in_pause = True
                pause_start = i
            elif is_voiced and in_pause:
                in_pause = False
                pause_length = (i - pause_start) * self.config['hop_length'] / sr
                if pause_length > 0.2:  # Consider pauses > 200ms
                    pause_lengths.append(pause_length)
        
        features = {
            'speech_rate': tempo / 60.0,  # Convert BPM to Hz
            'pause_count': len(pause_lengths),
            'pause_mean': np.mean(pause_lengths) if pause_lengths else 0,
            'pause_std': np.std(pause_lengths) if pause_lengths else 0,
            'pause_ratio': np.sum(pause_lengths) / (len(audio) / sr) if pause_lengths else 0
        }
        
        return features
    
    def extract_spectral_features(self, audio, sr):
        """Extract MFCC and spectral features"""
        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=self.config['n_mfcc'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length']
        )
        
        # Calculate statistics for each MFCC coefficient
        mfcc_features = {}
        for i in range(self.config['n_mfcc']):
            mfcc_features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            mfcc_features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Additional spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        spectral_features = {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff)
        }
        
        return {**mfcc_features, **spectral_features}
    
    def extract_call_level_features(self, audio_path):
        """Extract all features for a complete earnings call"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
        
        # Extract feature groups
        f0_features = self.extract_f0_features(audio, sr)
        temporal_features = self.extract_temporal_features(audio, sr)
        spectral_features = self.extract_spectral_features(audio, sr)
        
        # Combine all features
        all_features = {**f0_features, **temporal_features, **spectral_features}
        
        return all_features

def process_all_calls():
    """Process all earnings calls and extract features"""
    # Load metadata
    metadata = pd.read_csv(PROCESSED_DATA_DIR / "metadata.csv")
    
    # Initialize extractor
    extractor = AcousticFeatureExtractor()
    
    # Process each call
    all_features = []
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        audio_path = PROCESSED_DATA_DIR / "audio" / row['audio_file']
        
        if audio_path.exists():
            features = extractor.extract_call_level_features(audio_path)
            features['company'] = row['company']
            features['call_date'] = row['call_date']
            features['composite_outcome'] = row['composite_outcome']
            all_features.append(features)
    
    # Save features
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(FEATURES_DIR / "acoustic_features.csv", index=False)
    
if __name__ == "__main__":
    process_all_calls()