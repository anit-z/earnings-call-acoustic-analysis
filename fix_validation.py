import pandas as pd
import json
from pathlib import Path

# Load the already extracted features
features_df = pd.read_csv("data/features/acoustic/all_features.csv")

# Simple validation
validation = {
    'total_files': len(features_df),
    'successful_extractions': len(features_df) if 'error' not in features_df.columns else len(features_df[features_df['error'].isna()]),
    'feature_completeness': {},
    'acoustic_volatility_stats': {}
}

# Get acoustic volatility stats if the column exists
if 'acoustic_volatility_index' in features_df.columns:
    validation['acoustic_volatility_stats'] = {
        'mean': float(features_df['acoustic_volatility_index'].mean()),
        'std': float(features_df['acoustic_volatility_index'].std()),
        'p25': float(features_df['acoustic_volatility_index'].quantile(0.25)),
        'p50': float(features_df['acoustic_volatility_index'].quantile(0.50)),
        'p75': float(features_df['acoustic_volatility_index'].quantile(0.75))
    }

# Check feature completeness
feature_cols = [col for col in features_df.columns 
               if col not in ['file_id', 'sector', 'error', 'duration_s']]

for col in feature_cols[:10]:  # Just check first 10 features for summary
    valid_values = features_df[col].notna().sum()
    validation['feature_completeness'][col] = float(valid_values / len(features_df))

# Save validation report
with open("data/features/acoustic/validation_report.json", 'w') as f:
    json.dump(validation, f, indent=2)

print(f"Validation complete: {validation['successful_extractions']}/{validation['total_files']} files processed successfully")
print(f"\nFirst few features completeness:")
for feat, comp in list(validation['feature_completeness'].items())[:5]:
    print(f"  {feat}: {comp:.1%}")

if validation['acoustic_volatility_stats']:
    print(f"\nAcoustic volatility stats:")
    print(f"  Mean: {validation['acoustic_volatility_stats']['mean']:.3f}")
    print(f"  Std:  {validation['acoustic_volatility_stats']['std']:.3f}")
    print(f"  P50:  {validation['acoustic_volatility_stats']['p50']:.3f}")
