#!/usr/bin/env python3
"""Setup database for the demonstrator"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_database():
    """Initialize database with schema and baseline data"""
    print("Setting up database...")
    
    # Create database directory
    db_dir = Path("demonstrator/database")
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # Create migrations directory
    migrations_dir = db_dir / "migrations"
    migrations_dir.mkdir(exist_ok=True)
    
    # Create initial migration
    with open(migrations_dir / "001_initial_schema.sql", "w") as f:
        f.write("""
-- Initial database schema
CREATE TABLE IF NOT EXISTS earnings_calls (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    call_date DATE NOT NULL,
    audio_path VARCHAR(500),
    transcript_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS acoustic_features (
    id SERIAL PRIMARY KEY,
    call_id INTEGER REFERENCES earnings_calls(id),
    f0_mean FLOAT,
    f0_std FLOAT,
    f0_cv FLOAT,
    jitter FLOAT,
    speech_rate FLOAT,
    pause_ratio FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id SERIAL PRIMARY KEY,
    call_id INTEGER REFERENCES earnings_calls(id),
    composite_stress_score FLOAT,
    pattern_classification VARCHAR(100),
    sentiment_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
    
    print("Database setup complete!")
    print("Migration file created at:", migrations_dir / "001_initial_schema.sql")

if __name__ == "__main__":
    setup_database()
