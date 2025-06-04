# scripts/run_analysis.slurm
#!/bin/bash
#SBATCH --job-name=acoustic_analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=regular
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load modules
module load Python/3.10.4-GCCcore-11.3.0

# Activate virtual environment
source /scratch/s6055702/earnings_call_acoustic_analysis/venv/bin/activate

# Change to project directory
cd /scratch/s6055702/earnings_call_acoustic_analysis

# Run analysis
echo "Starting descriptive analysis..."
python src/analysis/descriptive_analysis.py

echo "Analysis complete!"