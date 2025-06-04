// demonstrator/frontend/src/App.js
import React, { useState, useEffect, useRef } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  Box,
  CircularProgress,
  Alert,
  Slider,
  Card,
  CardContent
} from '@mui/material';
import { Line, Bar } from 'react-chartjs-2';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState(null);
  const [realtimeData, setRealtimeData] = useState([]);
  
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const mediaRecorder = useRef(null);
  const websocket = useRef(null);

  useEffect(() => {
    // Initialize WaveSurfer
    if (waveformRef.current && !wavesurfer.current) {
      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#1976d2',
        progressColor: '#4caf50',
        cursorColor: '#ff5722',
        responsive: true,
        height: 100
      });
    }

    return () => {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
      }
      if (websocket.current) {
        websocket.current.close();
      }
    };
  }, []);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
      const url = URL.createObjectURL(file);
      wavesurfer.current.load(url);
    }
  };

  const analyzeAudio = async () => {
    if (!audioFile) {
      setError('Please select an audio file');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('audio', audioFile);

    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/v1/analyze/stress-level`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );

      setAnalysisResults(response.data);
    } catch (err) {
      setError('Analysis failed: ' + err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const startRealTimeAnalysis = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Setup WebSocket connection
      websocket.current = new WebSocket(`ws://localhost:8000/api/v1/stream/analyze`);
      
      websocket.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setRealtimeData(prev => [...prev.slice(-50), data]);
      };

      // Setup MediaRecorder
      mediaRecorder.current = new MediaRecorder(stream);
      
      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0 && websocket.current.readyState === WebSocket.OPEN) {
          websocket.current.send(event.data);
        }
      };

      mediaRecorder.current.start(100); // Send chunks every 100ms
      setIsRecording(true);
    } catch (err) {
      setError('Failed to start recording: ' + err.message);
    }
  };

  const stopRealTimeAnalysis = () => {
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach(track => track.stop());
    }
    if (websocket.current) {
      websocket.current.close();
    }
    setIsRecording(false);
  };

  const renderAnalysisResults = () => {
    if (!analysisResults) return null;

    const { stress_indicators, acoustic_features, recommendations } = analysisResults;

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Stress Analysis Results
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Composite Stress Score
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <Slider
                      value={stress_indicators.composite_stress_score}
                      valueLabelDisplay="on"
                      disabled
                      sx={{
                        color: stress_indicators.composite_stress_score > 75 ? 'error.main' : 
                               stress_indicators.composite_stress_score > 50 ? 'warning.main' : 
                               'success.main'
                      }}
                    />
                  </Box>
                </Box>
              </Box>
              <Typography variant="body2">
                Pattern: {stress_indicators.pattern_classification}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Feature Percentiles
              </Typography>
              <Bar
                data={{
                  labels: Object.keys(stress_indicators.feature_percentiles),
                  datasets: [{
                    label: 'Percentile Rank',
                    data: Object.values(stress_indicators.feature_percentiles).map(
                      f => f.percentile
                    ),
                    backgroundColor: 'rgba(25, 118, 210, 0.6)',
                    borderColor: 'rgba(25, 118, 210, 1)',
                    borderWidth: 1
                  }]
                }}
                options={{
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100
                    }
                  }
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recommendations
              </Typography>
              {recommendations && recommendations.map((rec, index) => (
                <Alert key={index} severity="info" sx={{ mb: 1 }}>
                  {rec}
                </Alert>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderRealtimeChart = () => {
    if (realtimeData.length === 0) return null;

    const chartData = {
      labels: realtimeData.map(d => new Date(d.timestamp * 1000).toLocaleTimeString()),
      datasets: [
        {
          label: 'Stress Level',
          data: realtimeData.map(d => d.stress_level),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
        },
        {
          label: 'F0 Volatility',
          data: realtimeData.map(d => d.f0_volatility),
          borderColor: 'rgb(53, 162, 235)',
          backgroundColor: 'rgba(53, 162, 235, 0.5)',
        }
      ]
    };

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Real-Time Stress Analysis
          </Typography>
          <Line data={chartData} options={{ responsive: true }} />
        </CardContent>
      </Card>
    );
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        Financial Speech Stress Analyzer
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="h5" gutterBottom>
              Upload Audio File
            </Typography>
            <input
              accept="audio/*"
              style={{ display: 'none' }}
              id="audio-file-input"
              type="file"
              onChange={handleFileUpload}
            />
            <label htmlFor="audio-file-input">
              <Button
                variant="contained"
                component="span"
                startIcon={<UploadFileIcon />}
              >
                Choose Audio File
              </Button>
            </label>
            {audioFile && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                Selected: {audioFile.name}
              </Typography>
            )}
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="h5" gutterBottom>
              Real-Time Analysis
            </Typography>
            <Button
              variant="contained"
              color={isRecording ? "secondary" : "primary"}
              startIcon={isRecording ? <StopIcon /> : <MicIcon />}
              onClick={isRecording ? stopRealTimeAnalysis : startRealTimeAnalysis}
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </Button>
          </Grid>
        </Grid>

        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Waveform
          </Typography>
          <div ref={waveformRef} />
        </Box>

        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Button
            variant="contained"
            size="large"
            onClick={analyzeAudio}
            disabled={!audioFile || isAnalyzing}
          >
            {isAnalyzing ? <CircularProgress size={24} /> : 'Analyze Audio'}
          </Button>
        </Box>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {analysisResults && renderAnalysisResults()}
      
      {isRecording && renderRealtimeChart()}
    </Container>
  );
}

export default App;