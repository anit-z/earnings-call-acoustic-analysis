# config/config.py
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Acoustic feature extraction parameters
ACOUSTIC_CONFIG = {
    'sample_rate': 16000,
    'frame_size': 0.025,  # 25ms
    'frame_stride': 0.010,  # 10ms
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,
    'f0_min': 75,
    'f0_max': 500,
}

# Rating classification rules
RATING_PRIORITY = {
    'downgrade': 3,
    'upgrade': 2,
    'affirm': 1,
    'no action': 0,
    'unrated': -1
}

# FinBERT configuration
FINBERT_MODEL = "yiyanghkust/finbert-tone"
FINBERT_BATCH_SIZE = 32

# Analysis parameters
BOOTSTRAP_ITERATIONS = 10000
CONFIDENCE_LEVEL = 0.95
PERCENTILE_THRESHOLDS = [5, 10, 25, 50, 75, 90, 95]