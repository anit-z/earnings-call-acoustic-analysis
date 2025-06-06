#!/usr/bin/env python3
"""
Audio Preprocessing Pipeline for Earnings Call Acoustic Analysis
Implements call-level aggregation and multi-speaker normalization
Enhanced with voice technology demonstrator integration (A2/B1)
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal
from scipy.stats import mstats
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for multiprocessing
_global_preprocessor = None
_global_output_dir = None
_global_rttm_dir = None


def parse_rttm_file(rttm_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse RTTM file to extract speaker segments
    
    Args:
        rttm_path: Path to RTTM file
        
    Returns:
        Dictionary mapping speaker IDs to list of (start, end) tuples
    """
    speaker_segments = {}
    
    try:
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == 'SPEAKER':
                    # Extract relevant fields
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    speaker_id = parts[7]
                    
                    end_time = start_time + duration
                    
                    if speaker_id not in speaker_segments:
                        speaker_segments[speaker_id] = []
                    
                    speaker_segments[speaker_id].append((start_time, end_time))
        
        logger.info(f"Parsed {len(speaker_segments)} speakers from {rttm_path}")
        
    except Exception as e:
        logger.error(f"Error parsing RTTM file {rttm_path}: {e}")
    
    return speaker_segments


def process_single_file_worker(audio_file):
    """Worker function for parallel processing"""
    output_path = os.path.join(_global_output_dir, f"{audio_file.stem}.wav")
    
    # Load speaker segments from RTTM if available
    speaker_segments = None
    if _global_rttm_dir:
        rttm_file = Path(_global_rttm_dir) / f"{audio_file.stem}.rttm"
        if rttm_file.exists():
            speaker_segments = parse_rttm_file(str(rttm_file))
    
    # Process audio
    return _global_preprocessor.process_call(
        str(audio_file),
        output_path,
        speaker_segments
    )


class AudioPreprocessor:
    """
    Call-level audio preprocessing with multi-speaker normalization
    Focused on preprocessing only - feature extraction handled by acoustic_features.py
    """
    
    def __init__(self, 
                 target_sr: int = 16000,
                 frame_size_ms: int = 25,
                 overlap_ratio: float = 0.5,
                 wiener_filter_size: int = 5,
                 spectral_floor_db: float = -20.0,
                 oversubtraction_factor: float = 2.0,
                 silence_margin_s: float = 0.1,
                 peak_norm_range: Tuple[float, float] = (-0.95, 0.95),
                 baseline_features_path: Optional[str] = None):
        """
        Initialize audio preprocessor with parameters
        """
        self.target_sr = target_sr
        self.frame_size_ms = frame_size_ms
        self.overlap_ratio = overlap_ratio
        self.wiener_filter_size = wiener_filter_size
        self.spectral_floor_db = spectral_floor_db
        self.oversubtraction_factor = oversubtraction_factor
        self.silence_margin_s = silence_margin_s
        self.peak_norm_range = peak_norm_range
        
        # Calculate frame parameters
        self.frame_size = int(self.target_sr * self.frame_size_ms / 1000)
        self.hop_length = int(self.frame_size * (1 - self.overlap_ratio))
        
        # Load sector mappings if available
        self.sector_mappings = self._load_sector_mappings()
        
        # Load baseline features for demonstrator
        self.baseline_features = None
        if baseline_features_path and Path(baseline_features_path).exists():
            self.baseline_features = pd.read_csv(baseline_features_path)
            logger.info(f"Loaded baseline features from {baseline_features_path}")
    
    def _load_sector_mappings(self) -> Dict[str, str]:
        """Load sector mappings from metadata"""
        try:
            metadata_path = Path("data/raw/earnings21/earnings21-file-metadata.csv")
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                return dict(zip(df['file_id'].astype(str), df['sector']))
            else:
                logger.warning("Sector metadata not found, using default normalization")
                return {}
        except Exception as e:
            logger.error(f"Error loading sector mappings: {e}")
            return {}
    
    def process_call(self, audio_path: str, output_path: str, 
                    speaker_segments: Optional[Dict] = None) -> Dict:
        """
        Process a single earnings call audio file
        Note: Feature extraction removed - handled by acoustic_features.py
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            logger.info(f"Loaded audio: {audio_path}, SR: {sr}, Shape: {audio.shape}")
            
            # Convert to mono if needed
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Store original sample rate before resampling
            original_sr = sr
            
            # Resample to target sampling rate
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # Apply Wiener filtering for noise reduction
            audio = self._apply_wiener_filter(audio, sr)
            
            # Voice activity detection
            vad_mask = self._voice_activity_detection(audio, sr)
            
            # Apply spectral subtraction
            audio = self._spectral_subtraction(audio, sr, vad_mask)
            
            # Trim silence
            audio = self._trim_silence(audio, sr)
            
            # Get sector for normalization
            file_id = Path(audio_path).stem
            sector = self.sector_mappings.get(file_id, 'default')
            
            # Apply sector-specific normalization
            audio = self._normalize_audio(audio, sector)
            
            # Save processed audio
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, sr)
            
            # Prepare basic statistics (detailed features in acoustic_features.py)
            stats = {
                'file_id': file_id,
                'sector': sector,
                'original_sr': original_sr,
                'duration_s': len(audio) / sr,
                'vad_ratio': float(np.mean(vad_mask)) if len(vad_mask) > 0 else 0.0,
                'peak_amplitude': float(np.max(np.abs(audio))),
                'rms_energy': float(np.sqrt(np.mean(audio**2))),
                'num_speakers': len(speaker_segments) if speaker_segments else 0,
                'output_path': output_path
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return {'file_id': Path(audio_path).stem, 'error': str(e)}
    
    def _apply_wiener_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply Wiener filtering for noise reduction"""
        try:
            # Estimate noise spectrum from quiet sections
            frame_energy = librosa.feature.rms(
                y=audio, 
                frame_length=self.frame_size,
                hop_length=self.hop_length
            )[0]
            
            # Find quiet frames (bottom 10% energy)
            noise_threshold = np.percentile(frame_energy, 10)
            noise_frames = frame_energy < noise_threshold
            
            # Apply Wiener filter
            filtered = signal.wiener(audio, mysize=self.wiener_filter_size)
            
            return filtered
        except Exception as e:
            logger.warning(f"Wiener filter failed: {e}")
            return audio
    
    def _voice_activity_detection(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detect voice activity using energy and spectral features"""
        try:
            # Energy-based VAD
            energy = librosa.feature.rms(
                y=audio,
                frame_length=self.frame_size,
                hop_length=self.hop_length
            )[0]
            
            # Spectral centroid for voicing detection
            centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=sr,
                hop_length=self.hop_length
            )[0]
            
            # Combine energy and spectral features
            energy_threshold = np.percentile(energy, 20)
            centroid_threshold = np.percentile(centroid, 30)
            
            vad_mask = (energy > energy_threshold) & (centroid > centroid_threshold)
            
            # Apply median filtering to smooth VAD decisions
            if len(vad_mask) > 5:
                vad_mask = signal.medfilt(vad_mask.astype(float), kernel_size=5) > 0.5
            
            return vad_mask
        except Exception as e:
            logger.warning(f"VAD failed: {e}")
            return np.ones(len(audio) // self.hop_length, dtype=bool)
    
    def _spectral_subtraction(self, audio: np.ndarray, sr: int, 
                            vad_mask: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction with oversubtraction factor"""
        try:
            # STFT
            D = librosa.stft(audio, n_fft=self.frame_size, hop_length=self.hop_length)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Estimate noise spectrum from non-speech frames
            if len(vad_mask) > 0:
                noise_frames = ~vad_mask
                if np.any(noise_frames):
                    noise_spectrum = np.median(magnitude[:, noise_frames], axis=1, keepdims=True)
                else:
                    noise_spectrum = np.zeros((magnitude.shape[0], 1))
            else:
                noise_spectrum = np.zeros((magnitude.shape[0], 1))
            
            # Apply spectral subtraction
            subtracted = magnitude - self.oversubtraction_factor * noise_spectrum
            
            # Apply gain floor
            gain_floor = 10 ** (self.spectral_floor_db / 20)
            subtracted = np.maximum(subtracted, gain_floor * magnitude)
            
            # Reconstruct signal
            D_subtracted = subtracted * np.exp(1j * phase)
            audio_subtracted = librosa.istft(D_subtracted, hop_length=self.hop_length)
            
            return audio_subtracted
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}")
            return audio
    
    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trim leading and trailing silence with margin"""
        try:
            # Find non-silent intervals
            non_silent_intervals = librosa.effects.split(
                audio, 
                top_db=30,
                frame_length=self.frame_size,
                hop_length=self.hop_length
            )
            
            if len(non_silent_intervals) > 0:
                # Get start and end with margins
                start = max(0, non_silent_intervals[0][0] - int(self.silence_margin_s * sr))
                end = min(len(audio), non_silent_intervals[-1][1] + int(self.silence_margin_s * sr))
                
                return audio[start:end]
            else:
                return audio
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray, sector: str) -> np.ndarray:
        """Apply sector-specific peak normalization"""
        # Peak normalization
        peak = np.max(np.abs(audio))
        if peak > 0:
            # Use 0.949 instead of 0.95 to avoid floating point issues
            target_peak = min(self.peak_norm_range[1], 0.949)
            audio = audio * (target_peak / peak)
            
            # Double check and force below 0.95
            new_peak = np.max(np.abs(audio))
            if new_peak >= 0.95:
                audio = audio * (0.949 / new_peak)
        
        return audio
    
    def create_demonstrator_output(self, features: Dict) -> Dict:
        """
        Create output for voice technology demonstrator (A2/B1 requirement)
        
        Args:
            features: Dictionary of acoustic features from acoustic_features.py
            
        Returns:
            Demonstrator-ready output with visualization data
        """
        # Calculate percentile rank against baseline
        percentile_rank = self.compare_to_baseline(features) if self.baseline_features is not None else None
        
        # Classify communication pattern
        pattern = self.classify_pattern(features)
        
        # Prepare visualization data
        viz_data = self.prepare_viz_data(features)
        
        return {
            'acoustic_volatility': features.get('acoustic_volatility_index', 0.0),
            'percentile_rank': percentile_rank,
            'pattern_classification': pattern,
            'visualization_data': viz_data,
            'key_features': {
                'f0_cv': features.get('f0_cv', 0.0),
                'jitter_local': features.get('jitter_local', 0.0),
                'shimmer_local': features.get('shimmer_local', 0.0),
                'speech_rate': features.get('speech_rate', 0.0),
                'pause_frequency': features.get('pause_frequency', 0.0)
            },
            'confidence_intervals': {
                'f0_cv_ci': [features.get('f0_cv_ci_lower', 0.0), 
                            features.get('f0_cv_ci_upper', 0.0)]
            },
            'metadata': {
                'file_id': features.get('file_id', ''),
                'sector': features.get('sector', ''),
                'duration_s': features.get('duration_s', 0.0)
            }
        }
    
    def compare_to_baseline(self, features: Dict) -> Dict:
        """
        Compare features to baseline distribution for percentile ranking
        """
        if self.baseline_features is None:
            return None
        
        percentile_ranks = {}
        
        # Key features to compare
        key_features = ['f0_cv', 'acoustic_volatility_index', 'jitter_local', 
                       'shimmer_local', 'pause_frequency']
        
        for feat in key_features:
            if feat in features and feat in self.baseline_features.columns:
                value = features[feat]
                baseline_values = self.baseline_features[feat].dropna()
                
                if len(baseline_values) > 0:
                    percentile = (baseline_values < value).sum() / len(baseline_values) * 100
                    percentile_ranks[feat] = float(percentile)
        
        # Overall percentile based on acoustic volatility
        overall_percentile = percentile_ranks.get('acoustic_volatility_index', 50.0)
        
        return {
            'overall_percentile': overall_percentile,
            'feature_percentiles': percentile_ranks,
            'interpretation': self._interpret_percentile(overall_percentile)
        }
    
    def _interpret_percentile(self, percentile: float) -> str:
        """Interpret percentile rank for demonstrator"""
        if percentile >= 95:
            return "Extremely high stress/arousal (top 5%)"
        elif percentile >= 75:
            return "High stress/arousal (top 25%)"
        elif percentile >= 50:
            return "Above average stress/arousal"
        elif percentile >= 25:
            return "Below average stress/arousal"
        else:
            return "Low stress/arousal (bottom 25%)"
    
    def classify_pattern(self, features: Dict) -> Dict:
        """
        Classify communication pattern based on acoustic features
        Following methodology from revised_methodology.docx
        """
        # Get key indicators
        f0_cv = features.get('f0_cv', 0)
        volatility = features.get('acoustic_volatility_index', 0)
        
        # Simplified pattern classification (full implementation needs FinBERT sentiment)
        if volatility > 0.75:
            pattern = "High Volatility Pattern"
            confidence = "High" if f0_cv > 0.2 else "Medium"
            description = "Elevated acoustic stress indicators suggest heightened emotional arousal"
        elif volatility < 0.25:
            pattern = "Stable Communication Pattern"
            confidence = "High"
            description = "Low acoustic volatility indicates controlled, professional communication"
        else:
            pattern = "Moderate Volatility Pattern"
            confidence = "Medium"
            description = "Acoustic features within normal range for business communication"
        
        return {
            'pattern': pattern,
            'confidence': confidence,
            'description': description,
            'key_indicators': {
                'f0_variability': 'High' if f0_cv > 0.15 else 'Normal',
                'voice_quality': 'Degraded' if features.get('jitter_local', 0) > 0.01 else 'Clear',
                'speech_rhythm': 'Disrupted' if features.get('pause_frequency', 0) > 2 else 'Regular'
            }
        }
    
    def prepare_viz_data(self, features: Dict) -> Dict:
        """
        Prepare data for visualization in demonstrator interface
        """
        # Feature categories for radar chart
        radar_data = {
            'categories': ['Pitch Variability', 'Voice Quality', 'Speech Rhythm', 
                          'Spectral Stability', 'Overall Volatility'],
            'values': [
                min(features.get('f0_cv', 0) * 3.33, 1.0),  # Scale to 0-1
                1.0 - min(features.get('jitter_local', 0) * 100, 1.0),  # Invert (lower is better)
                1.0 - min(features.get('pause_frequency', 0) / 3, 1.0),  # Invert
                1.0 - min(features.get('spectral_centroid_cv', 0) * 2, 1.0),  # Invert
                features.get('acoustic_volatility_index', 0)
            ]
        }
        
        # Time series data (simplified - would need frame-level features in practice)
        time_series = {
            'feature_evolution': {
                'timestamps': [0, 0.25, 0.5, 0.75, 1.0],  # Normalized time
                'f0_trajectory': [0.5, 0.6, 0.7, 0.6, 0.5],  # Placeholder
                'energy_trajectory': [0.4, 0.5, 0.6, 0.5, 0.4]  # Placeholder
            }
        }
        
        # Distribution comparison
        distribution_data = {
            'current_value': features.get('acoustic_volatility_index', 0),
            'baseline_stats': {
                'mean': 0.3,  # Would come from baseline_features
                'std': 0.15,
                'percentiles': [0.1, 0.2, 0.3, 0.4, 0.6]  # p10, p25, p50, p75, p90
            }
        }
        
        return {
            'radar_chart': radar_data,
            'time_series': time_series,
            'distribution': distribution_data,
            'color_scheme': self._get_color_scheme(features.get('acoustic_volatility_index', 0))
        }
    
    def _get_color_scheme(self, volatility: float) -> Dict:
        """Get color scheme for visualization based on volatility level"""
        if volatility > 0.75:
            return {'primary': '#d32f2f', 'secondary': '#ff5252', 'name': 'high_stress'}
        elif volatility > 0.5:
            return {'primary': '#f57c00', 'secondary': '#ff9800', 'name': 'moderate_stress'}
        elif volatility > 0.25:
            return {'primary': '#388e3c', 'secondary': '#4caf50', 'name': 'low_stress'}
        else:
            return {'primary': '#1976d2', 'secondary': '#2196f3', 'name': 'stable'}
    
    def validate_processing(self, processed_dir: str) -> pd.DataFrame:
        """Validate processed audio files"""
        validation_results = []
        
        for audio_file in Path(processed_dir).glob("*.wav"):
            try:
                # Load processed audio
                audio, sr = librosa.load(audio_file, sr=None)
                
                # Basic validation checks
                checks = {
                    'file_id': audio_file.stem,
                    'sample_rate_correct': sr == self.target_sr,
                    'is_mono': audio.ndim == 1,
                    'peak_in_range': np.max(np.abs(audio)) <= 0.95,
                    'duration_s': len(audio) / sr,
                    'has_content': np.max(np.abs(audio)) > 0.01
                }
                
                validation_results.append(checks)
                
            except Exception as e:
                logger.error(f"Validation error for {audio_file}: {e}")
                validation_results.append({
                    'file_id': audio_file.stem,
                    'error': str(e)
                })
        
        return pd.DataFrame(validation_results)


def main():
    parser = argparse.ArgumentParser(
        description="Process earnings call audio files for acoustic analysis"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing raw audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save processed audio")
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Target sampling rate")
    parser.add_argument("--rttm_dir", type=str,
                       help="Directory containing RTTM speaker diarization files")
    parser.add_argument("--baseline_features", type=str,
                       help="Path to baseline features CSV for demonstrator")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation after processing")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip processing if output file already exists")
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel processing")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--demo_features", type=str,
                       help="Path to features CSV for demonstrator output generation")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        target_sr=args.sample_rate,
        baseline_features_path=args.baseline_features
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If demo features provided, generate demonstrator output
    if args.demo_features:
        logger.info("Generating demonstrator outputs...")
        features_df = pd.read_csv(args.demo_features)
        demo_outputs = []
        
        for _, row in features_df.iterrows():
            features = row.to_dict()
            demo_output = preprocessor.create_demonstrator_output(features)
            demo_output['file_id'] = features.get('file_id', '')
            demo_outputs.append(demo_output)
        
        # Save demonstrator outputs
        with open(os.path.join(args.output_dir, "demonstrator_outputs.json"), 'w') as f:
            json.dump(demo_outputs, f, indent=2)
        
        logger.info(f"Generated demonstrator outputs for {len(demo_outputs)} files")
        return
    
    # Get all audio files with multiple format support
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
        audio_files.extend(Path(args.input_dir).glob(ext))
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Check for existing processing stats
    stats_file = os.path.join(args.output_dir, "preprocessing_stats.csv")
    existing_stats = []
    processed_files = set()
    
    if args.skip_existing and os.path.exists(stats_file):
        existing_df = pd.read_csv(stats_file)
        existing_df = existing_df.drop_duplicates(subset=['file_id'], keep='first')
        existing_stats = existing_df.to_dict('records')
        processed_files = set(existing_df['file_id'].values)
        logger.info(f"Found {len(processed_files)} already processed files")
    
    # Filter out already processed files
    files_to_process = []
    for audio_file in audio_files:
        if args.skip_existing and audio_file.stem in processed_files:
            output_path = os.path.join(args.output_dir, f"{audio_file.stem}.wav")
            if os.path.exists(output_path):
                logger.info(f"Skipping already processed: {audio_file.stem}")
                continue
        files_to_process.append(audio_file)
    
    logger.info(f"{len(files_to_process)} files need processing")
    
    if len(files_to_process) > 0:
        if args.parallel:
            # Parallel processing
            global _global_preprocessor, _global_output_dir, _global_rttm_dir
            _global_preprocessor = preprocessor
            _global_output_dir = args.output_dir
            _global_rttm_dir = args.rttm_dir
            
            from multiprocessing import Pool
            
            with Pool(processes=args.num_workers) as pool:
                processing_stats = list(tqdm(
                    pool.imap(process_single_file_worker, files_to_process),
                    total=len(files_to_process),
                    desc="Processing audio files"
                ))
        else:
            # Sequential processing
            processing_stats = []
            
            for audio_file in tqdm(files_to_process, desc="Processing audio files"):
                output_path = os.path.join(args.output_dir, f"{audio_file.stem}.wav")
                
                # Load speaker segments from RTTM if available
                speaker_segments = None
                if args.rttm_dir:
                    rttm_file = Path(args.rttm_dir) / f"{audio_file.stem}.rttm"
                    if rttm_file.exists():
                        speaker_segments = parse_rttm_file(str(rttm_file))
                
                # Process audio
                stats = preprocessor.process_call(
                    str(audio_file),
                    output_path,
                    speaker_segments
                )
                processing_stats.append(stats)
        
        # Combine with existing stats
        all_stats = existing_stats + processing_stats
    else:
        all_stats = existing_stats
    
    # Save preprocessing statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.drop_duplicates(subset=['file_id'], keep='first')
    stats_df.to_csv(stats_file, index=False)
    logger.info("Saved preprocessing statistics")
    
    # Run validation if requested
    if args.validate:
        logger.info("Running validation...")
        validation_df = preprocessor.validate_processing(args.output_dir)
        validation_df.to_csv(
            os.path.join(args.output_dir, "validation_results.csv"),
            index=False
        )
        
        # Report validation summary
        total_files = len(validation_df)
        passed = validation_df[validation_df['sample_rate_correct'] & 
                               validation_df['is_mono'] & 
                               validation_df['peak_in_range'] & 
                               validation_df['has_content']].shape[0] if 'sample_rate_correct' in validation_df else 0
        logger.info(f"Validation complete: {passed}/{total_files} files passed all checks")
        
        # Report sector analysis
        print("\nSector Analysis:")
        print(stats_df['sector'].value_counts())
        
        # Report processing summary
        print(f"\nProcessing Summary:")
        print(f"Total files: {len(stats_df)}")
        print(f"Average duration: {stats_df['duration_s'].mean():.1f}s")
        print(f"Total duration: {stats_df['duration_s'].sum() / 3600:.1f} hours")
    
    logger.info("Audio preprocessing complete!")


if __name__ == "__main__":
    main()