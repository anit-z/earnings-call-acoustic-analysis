import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import requests
from io import StringIO

# Handle scipy import gracefully
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("scipy not available. Statistical tests will be limited.")

# Page config
st.set_page_config(
    page_title="Earnings Call Acoustic Analysis",
    layout="wide"
)

# Title
st.title("Earnings Call Acoustic Analysis Demonstrator")
st.markdown("**Exploring correlations between acoustic stress indicators and credit rating actions**")

# Define paths for Streamlit Cloud and local environments
@st.cache_resource
def get_project_root():
    """Determine project root directory for both local and Streamlit Cloud"""
    # Check if we're on Streamlit Cloud
    if 'STREAMLIT_SHARING_MODE' in os.environ or '/mount/src/' in str(Path.cwd()):
        # We're on Streamlit Cloud
        return Path("/mount/src/earnings-call-acoustic-analysis")
    else:
        # Local development - try to find project root
        current_dir = Path.cwd()
        possible_roots = [
            current_dir,
            current_dir.parent,
            current_dir.parent.parent,
            Path(__file__).parent.parent if '__file__' in globals() else current_dir,
        ]
        
        for root in possible_roots:
            if (root / "data").exists() or (root / "demonstrator").exists():
                return root
        
        return current_dir

# Get project root
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# GitHub raw URL for ratings data
RATINGS_GITHUB_URL = "https://raw.githubusercontent.com/anit-z/earnings-call-acoustic-analysis/main/data/raw/ratings/ratings_metadata.csv"

# Load ratings data from GitHub
@st.cache_data
def load_ratings_from_github():
    """Load ratings data directly from GitHub"""
    try:
        response = requests.get(RATINGS_GITHUB_URL)
        if response.status_code == 200:
            ratings_df = pd.read_csv(StringIO(response.text))
            st.success("Successfully loaded ratings data from GitHub!")
            return ratings_df
        else:
            st.warning("Could not load ratings data from GitHub")
            return None
    except Exception as e:
        st.warning(f"Error loading ratings from GitHub: {e}")
        return None

# Create sample data function
@st.cache_data
def create_sample_data():
    """Create sample data for demonstration based on actual data structure"""
    np.random.seed(42)
    
    # Load ratings from GitHub or create sample
    ratings_df = load_ratings_from_github()
    
    if ratings_df is not None:
        # Use actual file_ids and sectors from ratings
        n_samples = len(ratings_df)
        file_ids = ratings_df['file_id'].astype(str).tolist()
        sectors = ratings_df['sector'].tolist()
        outcomes = ratings_df['composite_outcome'].tolist()
    else:
        # Create sample data
        n_samples = 24
        file_ids = [f'{4340000 + i}' for i in range(n_samples)]
        sectors = np.random.choice(['Financial', 'Technology', 'Healthcare', 'Energy', 'Consumer Goods'], n_samples)
        outcomes = np.random.choice(['affirm', 'downgrade', 'upgrade'], n_samples, p=[0.7, 0.2, 0.1])
    
    # Create sample combined features
    sample_data = pd.DataFrame({
        'file_id': file_ids,
        'f0_cv': np.random.normal(0.15, 0.05, n_samples),
        'f0_std': np.random.normal(25, 8, n_samples),
        'f0_mean': np.random.normal(150, 30, n_samples),
        'pause_frequency': np.random.normal(1.5, 0.5, n_samples),
        'jitter_local': np.random.normal(0.01, 0.003, n_samples),
        'shimmer_local': np.random.normal(0.03, 0.01, n_samples),
        'hnr_mean': np.random.normal(15, 3, n_samples),
        'speech_rate': np.random.normal(0.7, 0.1, n_samples),
        'spectral_centroid_mean': np.random.normal(1500, 300, n_samples),
        'acoustic_volatility_index': np.random.normal(0.5, 0.2, n_samples),
        'sentiment_negative': np.random.normal(0.3, 0.15, n_samples),
        'sentiment_positive': np.random.normal(0.4, 0.15, n_samples),
        'sentiment_neutral': np.random.normal(0.3, 0.1, n_samples),
        'sentiment_variability': np.random.normal(0.2, 0.08, n_samples),
        'dominant_sentiment': np.random.choice(['positive', 'negative', 'neutral'], n_samples),
        'sector': sectors,
        'communication_pattern': np.random.choice(['high_stress', 'moderate_stress', 'baseline_stability', 'mixed_pattern', 'high_excitement'], n_samples),
        'composite_outcome': outcomes
    })
    
    # Add realistic correlations based on outcomes
    sample_data.loc[sample_data['composite_outcome'] == 'downgrade', 'acoustic_volatility_index'] *= 1.3
    sample_data.loc[sample_data['composite_outcome'] == 'downgrade', 'sentiment_negative'] *= 1.2
    sample_data.loc[sample_data['composite_outcome'] == 'downgrade', 'f0_cv'] *= 1.25
    sample_data.loc[sample_data['composite_outcome'] == 'upgrade', 'sentiment_positive'] *= 1.2
    sample_data.loc[sample_data['composite_outcome'] == 'upgrade', 'acoustic_volatility_index'] *= 0.8
    
    # Ensure values are in reasonable ranges
    sample_data['sentiment_negative'] = sample_data['sentiment_negative'].clip(0, 1)
    sample_data['sentiment_positive'] = sample_data['sentiment_positive'].clip(0, 1)
    sample_data['sentiment_neutral'] = sample_data['sentiment_neutral'].clip(0, 1)
    sample_data['acoustic_volatility_index'] = sample_data['acoustic_volatility_index'].clip(0, 1)
    
    # If we have ratings data, merge additional columns
    if ratings_df is not None:
        # Merge additional rating information
        sample_data = sample_data.merge(
            ratings_df[['file_id', 'sp_action', 'moodys_action', 'fitch_action', 
                        'time_gap_days', 'case_study_flag']],
            on='file_id',
            how='left'
        )
    
    return sample_data

# Load data with fallback to sample data
@st.cache_data
def load_data():
    """Load combined features and ratings data with fallback to sample data"""
    features_path = DATA_DIR / "features/combined/combined_features.csv"
    
    try:
        if features_path.exists():
            # Load actual data
            features_df = pd.read_csv(features_path)
            
            # Try to load ratings from GitHub first
            ratings_df = load_ratings_from_github()
            
            if ratings_df is None:
                # Try local ratings file
                ratings_path = DATA_DIR / "raw/ratings/ratings_metadata.csv"
                if ratings_path.exists():
                    ratings_df = pd.read_csv(ratings_path)
            
            if ratings_df is not None:
                # Merge ratings data
                features_df['file_id'] = features_df['file_id'].astype(str)
                ratings_df['file_id'] = ratings_df['file_id'].astype(str)
                features_df = pd.merge(features_df, ratings_df, on='file_id', how='left')
                st.success("Loaded actual data successfully!")
            else:
                st.warning("Ratings data not found. Some analyses will be limited.")
            
            return features_df
        else:
            # Use sample data
            st.info("Using sample data for demonstration. Upload your processed data to see actual results.")
            return create_sample_data()
    except Exception as e:
        st.warning(f"Error loading data: {str(e)}. Using sample data for demonstration.")
        return create_sample_data()

# Create sample case studies
@st.cache_data
def create_sample_case_studies():
    """Create sample case studies for demonstration"""
    return {
        "4384683": {
            "file_id": "4384683",
            "rating_outcome": "downgrade",
            "pattern_classification": "High Stress",
            "confidence": "High",
            "key_insights": [
                "Significantly elevated acoustic volatility (95th percentile)",
                "High negative sentiment (88th percentile)",
                "F0 coefficient of variation in top quartile",
                "Rating downgrade by both S&P and Moody's within 210 days"
            ],
            "acoustic_features": {
                "f0_cv": {"value": 0.237, "percentile": 95},
                "acoustic_volatility_index": {"value": 0.85, "percentile": 93},
                "pause_frequency": {"value": 2.5, "percentile": 90},
                "jitter_local": {"value": 0.018, "percentile": 85}
            },
            "semantic_features": {
                "sentiment_negative": {"value": 0.65, "percentile": 88},
                "sentiment_positive": {"value": 0.15, "percentile": 12},
                "sentiment_variability": {"value": 0.35, "percentile": 82}
            }
        },
        "4368670": {
            "file_id": "4368670",
            "rating_outcome": "upgrade",
            "pattern_classification": "Mixed Pattern",
            "confidence": "Medium",
            "key_insights": [
                "High F0 coefficient of variation (100th percentile)",
                "But positive rating action by Fitch",
                "Shows complexity of acoustic-rating relationships",
                "Disagreement between rating agencies (Moody's affirm vs Fitch upgrade)"
            ],
            "acoustic_features": {
                "f0_cv": {"value": 0.35, "percentile": 100},
                "acoustic_volatility_index": {"value": 0.55, "percentile": 65},
                "pause_frequency": {"value": 1.2, "percentile": 40}
            },
            "semantic_features": {
                "sentiment_negative": {"value": 0.25, "percentile": 35},
                "sentiment_positive": {"value": 0.55, "percentile": 78}
            }
        }
    }

# Load case studies with fallback
@st.cache_data
def load_case_studies():
    """Load case study data from multiple possible locations"""
    possible_paths = [
        RESULTS_DIR / "analysis/case_studies.json",
        RESULTS_DIR / "tables/case_studies/case_studies_full.json",
        RESULTS_DIR / "analysis/descriptive/case_studies.json"
    ]
    
    for case_study_path in possible_paths:
        try:
            if case_study_path.exists():
                with open(case_study_path, 'r') as f:
                    return json.load(f)
        except:
            continue
    
    # Return sample case studies if none found
    st.info("Using sample case studies for demonstration.")
    return create_sample_case_studies()

# Main app
def main():
    # Show info box for Streamlit Cloud
    if 'STREAMLIT_SHARING_MODE' in os.environ or '/mount/src/' in str(Path.cwd()):
        with st.expander("ℹ️ About this Demo", expanded=False):
            st.info("""
            **Welcome to the Earnings Call Acoustic Analysis Demonstrator!**
            
            This demo is running with sample data. To use your own data:
            1. Fork the repository
            2. Add your processed data files to the repository
            3. Deploy your own instance on Streamlit Cloud
            
            Ratings data is loaded from: [GitHub Repository](https://github.com/anit-z/earnings-call-acoustic-analysis/blob/main/data/raw/ratings/ratings_metadata.csv)
            """)
    
    # Show current paths in sidebar for debugging
    with st.sidebar:
        st.header("Navigation")
        page = st.sidebar.radio("Select View", 
            ["Overview", "Individual Analysis", "Group Comparison", "Case Studies", "Settings"])
        
        if page == "Settings":
            st.subheader("Current Paths")
            st.text(f"Project Root: {PROJECT_ROOT}")
            st.text(f"Data Dir: {DATA_DIR}")
            st.text(f"Results Dir: {RESULTS_DIR}")
            st.text(f"scipy available: {SCIPY_AVAILABLE}")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data loaded. Please check the data paths and ensure the pipeline has been run.")
        return
    
    case_studies = load_case_studies()
    
    # Display selected page
    if page == "Overview":
        show_overview(df)
    elif page == "Individual Analysis":
        show_individual_analysis(df)
    elif page == "Group Comparison":
        show_group_comparison(df)
    elif page == "Case Studies":
        show_case_studies(case_studies, df)

def show_overview(df):
    st.header("Dataset Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", len(df))
    with col2:
        if 'composite_outcome' in df.columns:
            st.metric("Downgrades", len(df[df['composite_outcome'] == 'downgrade']))
        else:
            st.metric("Downgrades", "N/A")
    with col3:
        if 'composite_outcome' in df.columns:
            st.metric("Upgrades", len(df[df['composite_outcome'] == 'upgrade']))
        else:
            st.metric("Upgrades", "N/A")
    with col4:
        if 'sector' in df.columns:
            st.metric("Sectors", df['sector'].nunique())
        else:
            st.metric("Sectors", "N/A")
    
    # Add instructions for using the app
    st.markdown("""
    ### Instructions
    - Use the sidebar to navigate between views
    - Select "Individual Analysis" to explore specific calls
    - Select "Group Comparison" to compare features across different groups
    - Select "Case Studies" to examine notable examples
    """)

def show_individual_analysis(df):
    st.header("Individual Call Analysis")
    
    # Select a file
    file_id = st.selectbox("Select Call ID", sorted(df['file_id'].unique()))
    
    # Get data for selected file
    file_data = df[df['file_id'] == file_id].iloc[0]
    
    # Display basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'sector' in file_data:
            st.info(f"**Sector:** {file_data['sector']}")
    with col2:
        if 'composite_outcome' in file_data:
            outcome_color = {'upgrade': '🟢', 'downgrade': '🔴', 'affirm': '🔵'}.get(
                file_data['composite_outcome'], '⚪')
            st.markdown(f"**Rating Outcome:** {outcome_color} {file_data['composite_outcome']}")
    with col3:
        if 'communication_pattern' in file_data:
            st.info(f"**Pattern:** {file_data['communication_pattern']}")
    
    # Additional rating info if available
    if all(col in file_data.index for col in ['sp_action', 'moodys_action', 'fitch_action']):
        st.subheader("Rating Agency Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("S&P", file_data['sp_action'])
        with col2:
            st.metric("Moody's", file_data['moodys_action'])
        with col3:
            st.metric("Fitch", file_data['fitch_action'])
    
    # Display metrics in tabs
    tab1, tab2, tab3 = st.tabs(["Acoustic Features", "Semantic Features", "Feature Comparison"])
    
    with tab1:
        st.subheader("Acoustic Features")
        acoustic_features = ['f0_cv', 'f0_std', 'f0_mean', 'pause_frequency', 
                           'jitter_local', 'shimmer_local', 'acoustic_volatility_index']
        
        # Create two columns for features
        col1, col2 = st.columns(2)
        for i, feature in enumerate(acoustic_features):
            if feature in file_data and feature in df.columns:
                value = file_data[feature]
                if pd.notna(value):
                    # Calculate percentile
                    percentile = (df[feature] < value).sum() / len(df) * 100
                    
                    # Determine which column to use
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        # Color code based on percentile
                        delta_color = "normal" if 25 <= percentile <= 75 else "inverse" if percentile < 25 else "off"
                        st.metric(
                            feature.replace('_', ' ').title(),
                            f"{value:.3f}",
                            f"{percentile:.1f}%ile",
                            delta_color=delta_color
                        )
    
    with tab2:
        st.subheader("Semantic Features")
        semantic_features = ['sentiment_negative', 'sentiment_positive', 'sentiment_neutral',
                           'sentiment_variability', 'dominant_sentiment']
        
        col1, col2 = st.columns(2)
        for i, feature in enumerate(semantic_features):
            if feature in file_data:
                value = file_data[feature]
                if pd.notna(value):
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        if feature == 'dominant_sentiment':
                            st.metric(feature.replace('_', ' ').title(), value)
                        elif feature in df.columns and df[feature].dtype in ['float64', 'int64']:
                            percentile = (df[feature] < value).sum() / len(df) * 100
                            st.metric(
                                feature.replace('_', ' ').title(),
                                f"{value:.3f}",
                                f"{percentile:.1f}%ile"
                            )
    
    with tab3:
        st.subheader("Feature Radar Chart")
        
        # Select key features for radar chart
        radar_features = ['f0_cv', 'pause_frequency', 'sentiment_negative', 
                         'sentiment_positive', 'acoustic_volatility_index']
        available_radar = [f for f in radar_features if f in df.columns]
        
        if len(available_radar) >= 3:
            # Calculate percentiles for each feature
            percentiles = []
            labels = []
            for feature in available_radar:
                if feature in file_data and pd.notna(file_data[feature]):
                    percentile = (df[feature] < file_data[feature]).sum() / len(df) * 100
                    percentiles.append(percentile)
                    labels.append(feature.replace('_', ' ').title())
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Number of variables
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            percentiles += percentiles[:1]  # Complete the circle
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, percentiles, 'o-', linewidth=2, color='blue')
            ax.fill(angles, percentiles, alpha=0.25, color='blue')
            
            # Fix axis to go in the right order and start at 12 o'clock
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Percentile')
            ax.set_title(f'Feature Profile for {file_id}', y=1.08)
            
            # Add grid
            ax.grid(True)
            
            st.pyplot(fig)
            plt.close()

def show_group_comparison(df):
    st.header("Group Comparison Analysis")
    
    # Select grouping variable
    grouping_options = []
    if 'composite_outcome' in df.columns:
        grouping_options.append('Rating Outcome')
    if 'sector' in df.columns:
        grouping_options.append('Sector')
    if 'communication_pattern' in df.columns:
        grouping_options.append('Communication Pattern')
    
    if not grouping_options:
        st.warning("No grouping variables available for comparison")
        return
    
    group_by = st.selectbox("Group by", grouping_options)
    
    # Map display name to column name
    group_col_map = {
        'Rating Outcome': 'composite_outcome',
        'Sector': 'sector',
        'Communication Pattern': 'communication_pattern'
    }
    group_col = group_col_map[group_by]
    
    # Feature selection
    all_features = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    feature = st.selectbox("Select Feature to Compare", all_features)
    
    if feature not in df.columns:
        st.error(f"Feature {feature} not found in data")
        return
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    df_clean = df.dropna(subset=[group_col, feature])
    groups = df_clean.groupby(group_col)[feature]
    
    data_to_plot = [group.values for name, group in groups]
    labels = [name for name, group in groups]
    
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_ylabel(feature.replace('_', ' ').title())
    ax1.set_title(f'{feature.replace("_", " ").title()} by {group_by}')
    ax1.tick_params(axis='x', rotation=45)
    
    # Violin plot
    positions = range(len(labels))
    vp = ax2.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
    
    # Color violins
    for pc, color in zip(vp['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel(feature.replace('_', ' ').title())
    ax2.set_title(f'{feature.replace("_", " ").title()} Distribution by {group_by}')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Summary statistics
    st.subheader("Summary Statistics")
    summary = df_clean.groupby(group_col)[feature].agg(['count', 'mean', 'std', 'median', 
                                                        ('25%', lambda x: x.quantile(0.25)),
                                                        ('75%', lambda x: x.quantile(0.75))])
    summary = summary.round(4)
    st.dataframe(summary)
    
    # Statistical tests
    if SCIPY_AVAILABLE:
        if len(labels) == 2:
            # Two groups - use t-test
            group1, group2 = list(groups)
            t_stat, p_value = stats.ttest_ind(group1[1], group2[1])
            st.info(f"T-test between {group1[0]} and {group2[0]}: t={t_stat:.3f}, p={p_value:.4f}")
        elif len(labels) > 2:
            # Multiple groups - use ANOVA
            f_stat, p_value = stats.f_oneway(*data_to_plot)
            st.info(f"One-way ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
    else:
        st.info("Statistical tests not available. Install scipy for this functionality.")

def show_case_studies(case_studies, df):
    st.header("Case Studies Analysis")
    
    if not case_studies:
        st.warning("No case studies available. Please run the case study analysis first.")
        return
    
    # Select case study
    case_ids = list(case_studies.keys())
    case_id = st.selectbox("Select Case Study", case_ids)
    case = case_studies[case_id]
    
    # Display case information
    st.subheader(f"Case Study: {case_id}")
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'rating_outcome' in case:
            outcome_color = {'upgrade': '🟢', 'downgrade': '🔴', 'affirm': '🔵'}.get(
                case.get('rating_outcome', ''), '⚪')
            st.markdown(f"### {outcome_color} Rating: {case['rating_outcome'].upper()}")
    with col2:
        if 'pattern_classification' in case:
            st.info(f"**Pattern:** {case['pattern_classification']}")
    with col3:
        if 'confidence' in case:
            st.info(f"**Confidence:** {case['confidence']}")
    
    # Key insights
    if 'key_insights' in case:
        st.subheader("Key Insights")
        for insight in case['key_insights']:
            st.write(f"• {insight}")
    
    # Feature analysis
    tab1, tab2, tab3 = st.tabs(["Feature Percentiles", "Acoustic-Semantic Alignment", "Comparison"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Acoustic Features")
            if 'acoustic_features' in case:
                for feature, data in case['acoustic_features'].items():
                    if isinstance(data, dict) and 'percentile' in data:
                        value = data.get('value', 0)
                        percentile = data.get('percentile', 50)
                        
                        # Color based on extremity
                        if percentile >= 90 or percentile <= 10:
                            st.markdown(f"**:red[{feature.replace('_', ' ').title()}]**")
                            st.metric("", f"{value:.3f}", f"{percentile:.1f}%ile")
                        elif percentile >= 75 or percentile <= 25:
                            st.markdown(f"**:orange[{feature.replace('_', ' ').title()}]**")
                            st.metric("", f"{value:.3f}", f"{percentile:.1f}%ile")
                        else:
                            st.markdown(f"**{feature.replace('_', ' ').title()}**")
                            st.metric("", f"{value:.3f}", f"{percentile:.1f}%ile")
        
        with col2:
            st.subheader("Semantic Features")
            if 'semantic_features' in case:
                for feature, data in case['semantic_features'].items():
                    if isinstance(data, dict) and 'percentile' in data:
                        value = data.get('value', 0)
                        percentile = data.get('percentile', 50)
                        
                        # Color based on extremity
                        if percentile >= 90 or percentile <= 10:
                            st.markdown(f"**:red[{feature.replace('_', ' ').title()}]**")
                            st.metric("", f"{value:.3f}", f"{percentile:.1f}%ile")
                        elif percentile >= 75 or percentile <= 25:
                            st.markdown(f"**:orange[{feature.replace('_', ' ').title()}]**")
                            st.metric("", f"{value:.3f}", f"{percentile:.1f}%ile")
                        else:
                            st.markdown(f"**{feature.replace('_', ' ').title()}**")
                            st.metric("", f"{value:.3f}", f"{percentile:.1f}%ile")
    
    with tab2:
        if st.checkbox("Show Acoustic-Semantic Alignment Plot"):
            show_alignment_plot(case, df)
    
    with tab3:
        # Compare with baseline group
        baseline_group = st.selectbox("Compare with", ['affirm', 'all calls'])
        
        if baseline_group == 'affirm' and 'composite_outcome' in df.columns:
            baseline_df = df[df['composite_outcome'] == 'affirm']
        else:
            baseline_df = df
        
        # Select features to compare
        features_to_compare = ['f0_cv', 'acoustic_volatility_index', 'sentiment_negative', 'sentiment_positive']
        available_features = [f for f in features_to_compare if f in df.columns]
        
        if available_features and ('acoustic_features' in case or 'semantic_features' in case):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            case_values = []
            baseline_means = []
            baseline_stds = []
            labels = []
            
            for feature in available_features:
                # Get case value
                case_val = None
                if feature in case.get('acoustic_features', {}):
                    case_val = case['acoustic_features'][feature].get('value', 0)
                elif feature in case.get('semantic_features', {}):
                    case_val = case['semantic_features'][feature].get('value', 0)
                
                if case_val is not None:
                    case_values.append(case_val)
                    baseline_means.append(baseline_df[feature].mean())
                    baseline_stds.append(baseline_df[feature].std())
                    labels.append(feature.replace('_', ' ').title())
            
            if case_values:
                # Normalize to z-scores for comparison
                z_scores = [(case_values[i] - baseline_means[i]) / baseline_stds[i] 
                           for i in range(len(case_values))]
                
                # Create bar plot
                bars = ax.bar(labels, z_scores)
                
                # Color bars based on direction
                for bar, z in zip(bars, z_scores):
                    if z > 0:
                        bar.set_color('red')
                    else:
                        bar.set_color('blue')
                
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_ylabel('Z-Score (standard deviations from baseline)')
                ax.set_title(f'Feature Comparison: {case_id} vs {baseline_group}')
                plt.xticks(rotation=45, ha='right')
                
                # Add significance lines
                ax.axhline(y=2, color='red', linestyle='--', alpha=0.5)
                ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

def show_alignment_plot(case, df):
    """Show acoustic-semantic alignment plot for a case study"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define feature pairs to plot
    if 'acoustic_volatility_index' in df.columns and 'sentiment_negative' in df.columns:
        x_feature = 'acoustic_volatility_index'
        y_feature = 'sentiment_negative'
    elif 'f0_cv' in df.columns and 'sentiment_negative' in df.columns:
        x_feature = 'f0_cv'
        y_feature = 'sentiment_negative'
    else:
        st.warning("Required features not available for alignment plot")
        return
    
    # Plot all points colored by outcome
    if 'composite_outcome' in df.columns:
        outcomes = df['composite_outcome'].unique()
        colors = {'upgrade': 'green', 'downgrade': 'red', 'affirm': 'gray'}
        
        for outcome in outcomes:
            mask = df['composite_outcome'] == outcome
            ax.scatter(df.loc[mask, x_feature], df.loc[mask, y_feature], 
                      alpha=0.5, s=50, color=colors.get(outcome, 'gray'), 
                      label=outcome)
    else:
        ax.scatter(df[x_feature], df[y_feature], 
                  alpha=0.5, s=50, color='gray', label='All calls')
    
    # Highlight case
    case_x = None
    case_y = None
    
    if 'acoustic_features' in case and x_feature in case['acoustic_features']:
        case_x = case['acoustic_features'][x_feature].get('value')
    if 'semantic_features' in case and y_feature in case['semantic_features']:
        case_y = case['semantic_features'][y_feature].get('value')
    
    if case_x is not None and case_y is not None:
        ax.scatter([case_x], [case_y], color='black', s=200, edgecolors='red', 
                  linewidth=3, label='Selected case', zorder=5)
        
        # Add annotation
        ax.annotate(case.get('file_id', 'Case'), 
                   xy=(case_x, case_y), 
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add quadrant lines
    ax.axvline(x=df[x_feature].median(), color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=df[y_feature].median(), color='black', linestyle='--', alpha=0.3)
    
    # Add quadrant labels
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    
    ax.text(x_lim[1] * 0.8, y_lim[1] * 0.9, 'High Stress\nQuadrant', 
           fontsize=12, ha='center', bbox=dict(boxstyle='round,pad=0.5', 
                                              facecolor='red', alpha=0.2))
    ax.text(x_lim[0] * 1.2, y_lim[0] * 1.1, 'Low Stress\nQuadrant', 
           fontsize=12, ha='center', bbox=dict(boxstyle='round,pad=0.5', 
                                              facecolor='green', alpha=0.2))
    
    ax.set_xlabel(x_feature.replace('_', ' ').title())
    ax.set_ylabel(y_feature.replace('_', ' ').title())
    ax.set_title('Acoustic-Semantic Feature Space')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

if __name__ == "__main__":
    # Add path configuration option
    if st.sidebar.checkbox("Configure Paths"):
        custom_root = st.sidebar.text_input("Project Root Path", str(PROJECT_ROOT))
        if custom_root and Path(custom_root).exists():
            PROJECT_ROOT = Path(custom_root)
            DATA_DIR = PROJECT_ROOT / "data"
            RESULTS_DIR = PROJECT_ROOT / "results"
            st.sidebar.success(f"Updated project root to: {PROJECT_ROOT}")
    
    main()