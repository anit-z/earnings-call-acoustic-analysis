# demonstrator/backend/services/realtime_processor.py
"""Real-time audio processing for streaming analysis"""

import numpy as np
import librosa
from collections import deque
from typing import Dict, Optional
import time

class RealtimeAudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, window_size=5):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.window_size = window_size  # seconds
        
        # Buffer for sliding window analysis
        self.audio_buffer = deque(
            maxlen=int(window_size * sample_rate)
        )
        
        # Feature history for smoothing
        self.feature_history = deque(maxlen=10)
        
        self.last_analysis_time = time.time()
        self.analysis_interval = 0.5  # seconds
        
    def process_chunk(self, audio_chunk: bytes) -> Optional[Dict[str, float]]:
        """Process incoming audio chunk"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Check if enough time has passed for analysis
        current_time = time.time()
        if current_time - self.last_analysis_time < self.analysis_interval:
            return None
        
        self.last_analysis_time = current_time
        
        # Analyze current buffer
        if len(self.audio_buffer) >= self.sample_rate:  # At least 1 second
            features = self._extract_realtime_features()
            
            # Smooth features
            smoothed_features = self._smooth_features(features)
            
            return smoothed_features
        
        return None
    
    def _extract_realtime_features(self) -> Dict[str, float]:
        """Extract features from current buffer"""
        audio_array = np.array(self.audio_buffer)
        
        features = {'timestamp': time.time()}
        
        # F0 analysis
        f0, voiced_flag, _ = librosa.pyin(
            audio_array,
            fmin=75,
            fmax=500,
            sr=self.sample_rate
        )
        
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            features['f0_mean'] = float(np.mean(f0_voiced))
            features['f0_std'] = float(np.std(f0_voiced))
            features['f0_cv'] = features['f0_std'] / features['f0_mean'] if features['f0_mean'] > 0 else 0
        else:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_cv'] = 0
        
        # Energy-based features
        energy = librosa.feature.rms(y=audio_array)[0]
        features['energy_mean'] = float(np.mean(energy))
        features['energy_std'] = float(np.std(energy))
        
        # Simple speech rate estimation
        tempo, _ = librosa.beat.beat_track(y=audio_array, sr=self.sample_rate)
        features['speech_rate'] = float(tempo / 60.0)
        
        # Voice activity
        threshold = np.mean(energy) * 0.5
        voice_activity = np.mean(energy > threshold)
        features['voice_activity_ratio'] = float(voice_activity)
        features['pause_ratio'] = float(1 - voice_activity)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_array, 
            sr=self.sample_rate
        )[0]
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        
        return features
    
    def _smooth_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply smoothing to reduce noise in real-time features"""
        self.feature_history.append(features)
        
        if len(self.feature_history) < 3:
            return features
        
        # Compute rolling average for key features
        smoothed = features.copy()
        smooth_keys = ['f0_cv', 'speech_rate', 'pause_ratio', 'spectral_centroid']
        
        for key in smooth_keys:
            if key in features:
                values = [f.get(key, 0) for f in self.feature_history]
                smoothed[key] = float(np.mean(values))
        
        return smoothed
    
    def reset(self):
        """Reset buffers for new session"""
        self.audio_buffer.clear()
        self.feature_history.clear()
        self.last_analysis_time = time.time()