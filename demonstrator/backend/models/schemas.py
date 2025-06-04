# demonstrator/backend/models/schemas.py
"""Pydantic models for API request/response validation"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class StressProfile(BaseModel):
    composite_stress_score: float = Field(..., ge=0, le=100)
    f0_volatility: float = Field(default=0.15)
    speech_rate: float = Field(default=3.5)
    pause_ratio: float = Field(default=0.2)

class TTSRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    stress_profile: StressProfile
    voice: str = Field(default="default")
    language: str = Field(default="en")

class FeaturePercentile(BaseModel):
    percentile: float
    ci_lower: float
    ci_upper: float
    uncertainty_range: float

class StressIndicators(BaseModel):
    composite_stress_score: float
    feature_percentiles: Dict[str, FeaturePercentile]
    pattern_classification: str
    confidence_intervals: Optional[Dict[str, List[float]]]

class StressAnalysisResponse(BaseModel):
    status: str
    stress_indicators: StressIndicators
    acoustic_features: Dict[str, float]
    sentiment_analysis: Optional[Dict[str, float]]
    recommendations: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)

class RealtimeStreamConfig(BaseModel):
    sample_rate: int = Field(default=16000)
    chunk_size: int = Field(default=1024)
    analysis_interval: float = Field(default=0.5)