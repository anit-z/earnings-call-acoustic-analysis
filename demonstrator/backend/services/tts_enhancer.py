# demonstrator/backend/services/tts_enhancer.py
"""Text-to-Speech enhancement with stress prosody modeling"""

import numpy as np
from typing import Dict, Any
import torch
from TTS.api import TTS
import soundfile as sf
import io
import base64

class StressAwareTTS:
    def __init__(self):
        # Initialize TTS model (using Coqui TTS as example)
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def map_stress_to_prosody(
        self, 
        stress_profile: Dict[str, float]
    ) -> Dict[str, Any]:
        """Map stress indicators to prosody parameters"""
        # Extract stress level
        stress_level = stress_profile.get('composite_stress_score', 50) / 100
        
        prosody_params = {
            'pitch_shift': 0,
            'pitch_variability': 1.0,
            'speaking_rate': 1.0,
            'pause_frequency': 1.0,
            'volume_variability': 1.0
        }
        
        if stress_level > 0.75:  # High stress
            prosody_params.update({
                'pitch_shift': 0.1,  # Slightly higher pitch
                'pitch_variability': 1.5,  # More F0 variation
                'speaking_rate': 1.1,  # Slightly faster
                'pause_frequency': 1.3,  # More pauses
                'volume_variability': 1.4  # More volume changes
            })
        elif stress_level > 0.5:  # Moderate stress
            prosody_params.update({
                'pitch_shift': 0.05,
                'pitch_variability': 1.2,
                'speaking_rate': 1.05,
                'pause_frequency': 1.1,
                'volume_variability': 1.2
            })
        
        return prosody_params
    
    def synthesize(
        self, 
        text: str, 
        prosody_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate speech with stress-aware prosody"""
        # Basic synthesis
        wav = self.tts.tts(text)
        
        # Apply prosody modifications
        modified_wav = self._apply_prosody_modifications(
            wav, 
            prosody_params
        )
        
        # Convert to base64 for API response
        buffer = io.BytesIO()
        sf.write(buffer, modified_wav, 22050, format='WAV')
        audio_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'audio_data': audio_base64,
            'url': f"data:audio/wav;base64,{audio_base64}",
            'sample_rate': 22050
        }
    
    def _apply_prosody_modifications(
        self, 
        audio: np.ndarray, 
        prosody_params: Dict[str, float]
    ) -> np.ndarray:
        """Apply prosody modifications to synthesized speech"""
        import parselmouth
        
        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=22050)
        
        # Pitch modification
        if prosody_params['pitch_shift'] != 0:
            pitch = sound.to_pitch()
            pitch_tier = pitch.to_pitch_tier()
            
            # Shift pitch
            for i in range(len(pitch_tier.points)):
                pitch_tier.points[i].value *= (1 + prosody_params['pitch_shift'])
            
            # Add variability
            if prosody_params['pitch_variability'] > 1:
                for i in range(len(pitch_tier.points)):
                    variation = np.random.normal(0, 0.05 * prosody_params['pitch_variability'])
                    pitch_tier.points[i].value *= (1 + variation)
        
        # Duration modification for speaking rate
        if prosody_params['speaking_rate'] != 1.0:
            sound = sound.lengthen(1 / prosody_params['speaking_rate'])
        
        # Add pauses (simplified)
        if prosody_params['pause_frequency'] > 1:
            # Insert silence at sentence boundaries
            # This is a simplified implementation
            pass
        
        return sound.values[0]
    
    def create_training_data(
        self, 
        earnings_calls_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Create training data for stress-aware TTS from earnings calls
        """
        training_data = {
            'stress_profiles': [],
            'prosody_mappings': []
        }
        
        for _, call in earnings_calls_features.iterrows():
            stress_profile = {
                'f0_cv': call['f0_cv'],
                'jitter': call['jitter'],
                'speech_rate': call['speech_rate'],
                'pause_ratio': call['pause_ratio']
            }
            
            prosody_mapping = self.map_stress_to_prosody(stress_profile)
            
            training_data['stress_profiles'].append(stress_profile)
            training_data['prosody_mappings'].append(prosody_mapping)
        
        return training_data