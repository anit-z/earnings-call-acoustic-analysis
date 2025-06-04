# scripts/setup_environment.sh
#!/bin/bash

# Load necessary modules on Hábrók
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load FFmpeg/4.4.2-GCCcore-11.3.0

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers datasets
pip install librosa soundfile
pip install opensmile
pip install jupyter notebook
pip install tqdm
pip install statsmodels
pip install python-dotenv