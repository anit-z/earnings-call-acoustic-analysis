import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Earnings Call Acoustic Analysis",
    layout="wide"
)

# Title
st.title("Earnings Call Acoustic Analysis Demonstrator")
st.markdown("**Exploring correlations between acoustic stress indicators and credit rating actions**")

# Define paths
BASE_DIR = Path("/scratch/s6055702/earnings_call_acoustic_analysis")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Load data
@st.cache_data
def load_data():
    # Load combined features
    features_df = pd.read_csv(DATA_DIR / "features/combined/combined_features.csv")
    
    # Load ratings if available
    ratings_path = DATA_DIR / "raw/ratings/ratings_metadata.csv"
    if ratings_path.exists():
        ratings_df = pd.read_csv(ratings_path)
        features_df = pd.merge(features_df, ratings_df, on='file_id', how='left')
    
    return features_df

# Load case studies
@st.cache_data
def load_case_studies():
    case_study_path = RESULTS_DIR / "analysis/case_studies.json"
    if case_study_path.exists():
        with open(case_study_path, 'r') as f:
            return json.load(f)
    return {}

# Main app
def main():
    # Load data
    df = load_data()
    case_studies = load_case_studies()
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select View", 
        ["Overview", "Individual Analysis", "Group Comparison", "Case Studies"])
    
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Calls", len(df))
    with col2:
        if 'composite_outcome' in df.columns:
            st.metric("Downgrades", len(df[df['composite_outcome'] == 'downgrade']))
    with col3:
        if 'composite_outcome' in df.columns:
            st.metric("Upgrades", len(df[df['composite_outcome'] == 'upgrade']))
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select features to display
    acoustic_features = ['f0_cv', 'f0_std', 'pause_frequency', 'jitter_local']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, feature in enumerate(acoustic_features):
        if feature in df.columns:
            ax = axes[i]
            df[feature].hist(ax=ax, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(feature.replace('_', ' ').title())
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_individual_analysis(df):
    st.header("Individual Call Analysis")
    
    # Select a file
    file_id = st.selectbox("Select Call ID", df['file_id'].unique())
    
    # Get data for selected file
    file_data = df[df['file_id'] == file_id].iloc[0]
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Acoustic Features")
        acoustic_features = ['f0_cv', 'f0_std', 'pause_frequency', 'jitter_local']
        for feature in acoustic_features:
            if feature in file_data:
                value = file_data[feature]
                percentile = (df[feature] < value).sum() / len(df) * 100
                st.metric(
                    feature.replace('_', ' ').title(),
                    f"{value:.3f}",
                    f"{percentile:.1f}%ile"
                )
    
    with col2:
        st.subheader("Semantic Features")
        if 'sentiment_negative' in file_data:
            st.metric("Negative Sentiment", f"{file_data['sentiment_negative']:.3f}")
        if 'sentiment_positive' in file_data:
            st.metric("Positive Sentiment", f"{file_data['sentiment_positive']:.3f}")
        if 'composite_outcome' in file_data:
            st.metric("Rating Outcome", file_data['composite_outcome'])

def show_group_comparison(df):
    st.header("Group Comparison")
    
    if 'composite_outcome' not in df.columns:
        st.warning("No rating outcomes available for comparison")
        return
    
    # Feature selection
    feature = st.selectbox(
        "Select Feature to Compare",
        ['f0_cv', 'f0_std', 'pause_frequency', 'jitter_local', 'sentiment_negative']
    )
    
    if feature not in df.columns:
        st.error(f"Feature {feature} not found in data")
        return
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Group data
    groups = df.groupby('composite_outcome')[feature]
    
    # Box plot
    data_to_plot = [group.dropna().values for name, group in groups]
    labels = [name for name, group in groups]
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = ['lightgreen', 'lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax.set_ylabel(feature.replace('_', ' ').title())
    ax.set_title(f'{feature.replace("_", " ").title()} by Rating Outcome')
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    summary = df.groupby('composite_outcome')[feature].agg(['mean', 'std', 'median', 'count'])
    st.dataframe(summary)

def show_case_studies(case_studies, df):
    st.header("Case Studies")
    
    if not case_studies:
        st.warning("No case studies available")
        return
    
    # Select case study
    case_id = st.selectbox("Select Case Study", list(case_studies.keys()))
    case = case_studies[case_id]
    
    # Display case information
    st.subheader(f"Case: {case_id}")
    
    if 'rating_outcome' in case:
        st.info(f"Rating Outcome: **{case['rating_outcome'].upper()}**")
    
    # Show percentile ranks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Acoustic Feature Percentiles")
        if 'acoustic_features' in case:
            for feature, data in case['acoustic_features'].items():
                if 'percentile' in data and data['percentile'] is not None:
                    st.metric(
                        feature.replace('_', ' ').title(),
                        f"{data['value']:.3f}",
                        f"{data['percentile']:.1f}%ile"
                    )
    
    with col2:
        st.subheader("Semantic Feature Percentiles")
        if 'semantic_features' in case:
            for feature, data in case['semantic_features'].items():
                if 'percentile' in data and data['percentile'] is not None:
                    st.metric(
                        feature.replace('_', ' ').title(),
                        f"{data['value']:.3f}",
                        f"{data['percentile']:.1f}%ile"
                    )
    
    # Visualize case position
    if st.checkbox("Show Acoustic-Semantic Alignment"):
        show_alignment_plot(case, df)

def show_alignment_plot(case, df):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot all points
    if 'f0_cv' in df.columns and 'sentiment_negative' in df.columns:
        ax.scatter(df['f0_cv'], df['sentiment_negative'], 
                  alpha=0.5, s=30, color='gray', label='All calls')
        
        # Highlight case
        if ('acoustic_features' in case and 'f0_cv' in case['acoustic_features'] and
            'semantic_features' in case and 'sentiment_negative' in case['semantic_features']):
            
            x = case['acoustic_features']['f0_cv']['value']
            y = case['semantic_features']['sentiment_negative']['value']
            ax.scatter([x], [y], color='red', s=100, edgecolors='black', 
                      linewidth=2, label='Selected case')
        
        # Add quadrant lines
        ax.axvline(x=df['f0_cv'].median(), color='black', linestyle='--', alpha=0.3)
        ax.axhline(y=df['sentiment_negative'].median(), color='black', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('F0 Coefficient of Variation')
        ax.set_ylabel('Negative Sentiment')
        ax.set_title('Acoustic-Semantic Feature Space')
        ax.legend()
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()