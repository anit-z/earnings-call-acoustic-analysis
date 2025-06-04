# demonstrator/backend/api/main.py
"""Main FastAPI application for Voice Technology Demonstrator"""

from fastapi import FastAPI, UploadFile, File, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from typing import Dict, Any, List
import numpy as np
import torch
import librosa
import io
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.feature_extraction.acoustic_features import AcousticFeatureExtractor
from src.feature_extraction.finbert_sentiment import FinBERTSentimentAnalyzer
from demonstrator.backend.services.stress_analyzer import StressAnalyzer
from demonstrator.backend.services.realtime_processor import RealtimeAudioProcessor
from demonstrator.backend.services.tts_enhancer import StressAwareTTS
from demonstrator.backend.models.schemas import (
    StressAnalysisResponse, 
    TTSRequest,
    RealtimeStreamConfig
)

app = FastAPI(
    title="Financial Speech Stress Analyzer API",
    description="Voice technology demonstrator for acoustic stress detection in earnings calls",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
acoustic_extractor = AcousticFeatureExtractor()
sentiment_analyzer = FinBERTSentimentAnalyzer()
stress_analyzer = StressAnalyzer()
tts_enhancer = StressAwareTTS()

# WebSocket connections for real-time streaming
active_connections: List[WebSocket] = []

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Financial Speech Stress Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/v1/analyze/stress-level",
            "real-time": "/api/v1/stream/analyze",
            "synthesis": "/api/v1/synthesize/stress-aware-tts",
            "baseline": "/api/v1/baseline/statistics"
        }
    }

@app.post("/api/v1/analyze/stress-level", response_model=StressAnalysisResponse)
async def analyze_stress_level(
    audio: UploadFile = File(...),
    transcript: UploadFile = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """
    Analyze stress level in uploaded audio file
    
    Args:
        audio: Audio file (WAV, MP3, FLAC)
        transcript: Optional transcript file for FinBERT analysis
    
    Returns:
        Stress analysis results with confidence intervals
    """
    try:
        # Read audio file
        audio_bytes = await audio.read()
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Extract acoustic features
        acoustic_features = acoustic_extractor.extract_call_level_features_from_array(
            audio_data, sr
        )
        
        # Analyze against baseline
        stress_metrics = stress_analyzer.analyze_features(acoustic_features)
        
        # FinBERT analysis if transcript provided
        sentiment_results = None
        if transcript:
            transcript_text = (await transcript.read()).decode('utf-8')
            sentiment_results = sentiment_analyzer.analyze_text(transcript_text)
            
            # Calculate acoustic-semantic correlation
            correlation = stress_analyzer.calculate_correlation(
                acoustic_features, sentiment_results
            )
            stress_metrics['acoustic_semantic_correlation'] = correlation
        
        # Background task for logging
        background_tasks.add_task(
            log_analysis, 
            audio.filename, 
            stress_metrics
        )
        
        return {
            "status": "success",
            "stress_indicators": stress_metrics,
            "acoustic_features": acoustic_features,
            "sentiment_analysis": sentiment_results,
            "recommendations": generate_recommendations(stress_metrics)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.websocket("/api/v1/stream/analyze")
async def websocket_realtime_analysis(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming and analysis
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    processor = RealtimeAudioProcessor()
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Process chunk
            features = processor.process_chunk(data)
            
            if features:
                # Analyze stress indicators
                stress_level = stress_analyzer.quick_analysis(features)
                
                # Send results back
                await websocket.send_json({
                    "timestamp": features['timestamp'],
                    "stress_level": stress_level,
                    "f0_volatility": features.get('f0_cv', 0),
                    "speech_rate": features.get('speech_rate', 0)
                })
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)

@app.post("/api/v1/synthesize/stress-aware-tts")
async def synthesize_stress_aware_speech(request: TTSRequest):
    """
    Generate speech synthesis with stress-aware prosody modeling
    
    Args:
        request: Text and stress profile for synthesis
    
    Returns:
        Audio file with stress-modeled prosody
    """
    try:
        # Map stress indicators to prosody parameters
        prosody_params = tts_enhancer.map_stress_to_prosody(
            request.stress_profile
        )
        
        # Generate speech with stress modeling
        audio_output = tts_enhancer.synthesize(
            request.text,
            prosody_params
        )
        
        return {
            "status": "success",
            "audio_url": audio_output['url'],
            "prosody_parameters": prosody_params
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/api/v1/baseline/statistics")
async def get_baseline_statistics():
    """
    Get baseline statistics from affirmation earnings calls
    """
    baseline_stats = stress_analyzer.get_baseline_statistics()
    
    return {
        "total_baseline_calls": baseline_stats['n_samples'],
        "feature_distributions": baseline_stats['distributions'],
        "percentile_thresholds": baseline_stats['percentiles'],
        "last_updated": baseline_stats['timestamp']
    }

@app.post("/api/v1/multilingual/analyze")
async def analyze_multilingual_stress(
    audio: UploadFile = File(...),
    language: str = "en"
):
    """
    Analyze stress in multilingual earnings calls
    """
    # Language-agnostic acoustic features
    audio_bytes = await audio.read()
    audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    
    universal_features = acoustic_extractor.extract_language_agnostic_features(
        audio_data, sr
    )
    
    # Multilingual FinBERT if available
    multilingual_sentiment = None
    if language != "en":
        # Use multilingual model
        pass  # Placeholder for multilingual implementation
    
    return {
        "language": language,
        "universal_stress_indicators": universal_features,
        "language_specific_analysis": multilingual_sentiment
    }

def generate_recommendations(stress_metrics: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on stress analysis"""
    recommendations = []
    
    if stress_metrics.get('f0_volatility_percentile', 0) > 90:
        recommendations.append(
            "High vocal stress detected. Consider additional due diligence."
        )
    
    if stress_metrics.get('acoustic_semantic_correlation', 0) > 0.6:
        recommendations.append(
            "Strong alignment between vocal stress and negative sentiment."
        )
    
    return recommendations

async def log_analysis(filename: str, results: Dict[str, Any]):
    """Background task for logging analysis results"""
    # Implementation for logging to database or file
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)