#!/bin/bash
# build_demonstrator.sh - Build the complete demonstrator

set -e

echo "=== Building Voice Technology Demonstrator ==="

# Backend setup
echo "Setting up backend..."
cd demonstrator/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup
echo "Setting up frontend..."
cd ../frontend
npm install
npm run build

# Database setup
echo "Setting up database..."
cd ../../
python scripts/setup_database.py

# Docker build
echo "Building Docker images..."
cd deployment/docker
docker-compose build

echo "=== Build complete! ==="
echo "To run the demonstrator:"
echo "  cd deployment/docker"
echo "  docker-compose up"
