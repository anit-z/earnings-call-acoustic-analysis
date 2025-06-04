# scripts/run_complete_pipeline.sh
#!/bin/bash

# Setup environment (run once)
if [ ! -d "venv" ]; then
    echo "Setting up environment..."
    bash scripts/setup_environment.sh
fi

# Activate environment
source venv/bin/activate

# Download and prepare data
echo "Preparing data..."
python src/preprocessing/download_earnings21.py

# Submit feature extraction job
echo "Submitting feature extraction job..."
FE_JOB=$(sbatch scripts/run_feature_extraction.slurm | awk '{print $4}')
echo "Feature extraction job ID: $FE_JOB"

# Submit analysis job with dependency
echo "Submitting analysis job (dependent on feature extraction)..."
sbatch --dependency=afterok:$FE_JOB scripts/run_analysis.slurm

echo "Pipeline submitted! Check logs/ directory for progress."