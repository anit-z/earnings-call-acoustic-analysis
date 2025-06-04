# scripts/run_feature_extraction.slurm
#!/bin/bash
#SBATCH --job-name=acoustic_features
#SBATCH --output=logs/feature_extraction_%j.out
#SBATCH --error=logs/feature_extraction_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load modules
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load FFmpeg/4.4.2-GCCcore-11.3.0

# Activate virtual environment
source /scratch/s6055702/earnings_call_acoustic_analysis/venv/bin/activate

# Change to project directory
cd /scratch/s6055702/earnings_call_acoustic_analysis

# Run feature extraction
echo "Starting acoustic feature extraction..."
python src/feature_extraction/acoustic_features.py

echo "Starting FinBERT sentiment analysis..."
python src/feature_extraction/finbert_sentiment.py

echo "Feature extraction complete!"