# Earnings Call Acoustic Analysis

Statistical analysis of vocal stress patterns in earnings conference calls and their relationship with subsequent rating agency actions.

## Project Status
- [x] Directory structure created
- [x] Virtual environment setup
- [ ] Earnings-21 dataset downloaded
- [ ] Feature extraction pipeline implemented
- [ ] Statistical analysis completed
- [ ] Voice technology demonstrator built

## Quick Start
```bash
# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python src/preprocessing/download_earnings21.py
Directory Structure
data/: Raw and processed data
src/: Source code for analysis
demonstrator/: Voice technology demonstrator
notebooks/: Jupyter notebooks for exploration
results/: Figures and tables
scripts/: Utility and SLURM scripts

## Experiments

### Comprehensive Acoustic Features
Focus on primary vocal stress indicators:
- F0 Coefficient of Variation (f0_cv)
- F0 Standard Deviation (f0_std)
- Pause Frequency
- Jitter Local