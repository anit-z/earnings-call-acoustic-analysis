import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Page config
st.set_page_config(
    page_title="Earnings Call Acoustic Analysis",
    layout="wide"
)

# Title
st.title("Earnings Call Acoustic Analysis Demonstrator")
st.markdown("**Exploring correlations between acoustic stress indicators and credit rating actions**")

# Define paths relative to the script location or allow override
@st.cache_resource
def get_project_root():
    """Determine project root directory"""
    # Check if running from within project structure
    current_dir = Path.cwd()
    
    # Try to find project root by looking for key directories
    possible_roots = [
        current_dir,  # Current directory
        current_dir.parent,  # One level up
        current_dir.parent.parent,  # Two levels up
        Path(__file__).parent.parent if '__file__' in globals() else current_dir,  # Relative to script
    ]
    
    for root in possible_roots:
        if (root / "data").exists() and (root / "results").exists():
            return root
    
    # If not found, use current directory
    st.warning("Project root not found. Using current directory. You may need to adjust paths.")
    return current_dir

# Get project root
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Load data
@st.cache_data
def load_data():
    """Load combined features and ratings data"""
    features_path = DATA_DIR / "features/combined/combined_features.csv"
    
    if not features_path.exists():
        st.error(f"Combined features file not found at: {features_path}")
        st.info("Please ensure you have run the feature extraction and combination pipeline.")
        return pd.DataFrame()
    
    # Load combined features
    features_df = pd.read_csv(features_path)
    
    # Load ratings if available
    ratings_path = DATA_DIR / "raw/ratings/ratings_metadata.csv"
    if ratings_path.exists():
        ratings_df = pd.read_csv(ratings_path)
        # Ensure file_id is string type for proper merging
        features_df['file_id'] = features_df['file_id'].astype(str)
        ratings_df['file_id'] = ratings_df['file_id'].astype(str)
        features_df = pd.merge(features_df, ratings_df, on='file_id', how='left')
    else:
        st.warning("Ratings data not found. Some analyses will be limited.")
    
    return features_df

# Load case studies
@st.cache_data
def load_case_studies():
    """Load case study data from multiple possible locations"""
    # Try multiple locations for case studies
    possible_paths = [
        RESULTS_DIR / "analysis/case_studies.json",
        RESULTS_DIR / "tables/case_studies/case_studies_full.json",
        RESULTS_DIR / "analysis/descriptive/case_studies.json"
    ]
    
    for case_study_path in possible_paths:
        if case_study_path.exists():
            with open(case_study_path, 'r') as f:
                return json.load(f)
    
    st.warning("Case studies not found. Run case study analysis first.")
    return {}

# Load analysis results
@st.cache_data
def load_analysis_results():
    """Load analysis results if available"""
    analysis_path = RESULTS_DIR / "analysis/analysis_results.json"
    if analysis_path.exists():
        with open(analysis_path, 'r') as f:
            return json.load(f)
    return {}

# Main app
def main():
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
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data loaded. Please check the data paths and ensure the pipeline has been run.")
        return
    
    case_studies = load_case_studies()
    analysis_results = load_analysis_results()
    
    # Display selected page
    if page == "Overview":
        show_overview(df, analysis_results)
    elif page == "Individual Analysis":
        show_individual_analysis(df)
    elif page == "Group Comparison":
        show_group_comparison(df)
    elif page == "Case Studies":
        show_case_studies(case_studies, df)

def show_overview(df, analysis_results):
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
    
    # Communication patterns if available
    if 'communication_pattern' in df.columns:
        st.subheader("Communication Patterns")
        pattern_counts = df['communication_pattern'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {
            'high_stress': 'red',
            'moderate_stress': 'orange',
            'high_excitement': 'green',
            'moderate_excitement': 'lightgreen',
            'baseline_stability': 'blue',
            'mixed_pattern': 'gray'
        }
        bar_colors = [colors.get(x, 'gray') for x in pattern_counts.index]
        
        bars = ax.bar(pattern_counts.index, pattern_counts.values, color=bar_colors)
        ax.set_xlabel('Communication Pattern')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Communication Patterns')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Feature distributions
    st.subheader("Key Feature Distributions")
    
    # Select features to display
    acoustic_features = ['f0_cv', 'acoustic_volatility_index', 'pause_frequency', 'jitter_local']
    semantic_features = ['sentiment_negative', 'sentiment_positive', 'sentiment_variability']
    
    # Check which features are available
    available_acoustic = [f for f in acoustic_features if f in df.columns]
    available_semantic = [f for f in semantic_features if f in df.columns]
    
    if available_acoustic:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_acoustic[:4]):
            ax = axes[i]
            df[feature].hist(ax=ax, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(feature.replace('_', ' ').title())
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
            # Add mean line
            mean_val = df[feature].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.text(mean_val, ax.get_ylim()[1]*0.9, f'μ={mean_val:.3f}', 
                   ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        st.pyplot(fig)

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
            outcome_color = {'upgrade': 'green', 'downgrade': 'red', 'affirm': 'blue'}.get(
                file_data['composite_outcome'], 'gray')
            st.markdown(f"**Rating Outcome:** :{outcome_color}[{file_data['composite_outcome']}]")
    with col3:
        if 'communication_pattern' in file_data:
            st.info(f"**Pattern:** {file_data['communication_pattern']}")
    
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
    
    # Summary statistics
    st.subheader("Summary Statistics")
    summary = df_clean.groupby(group_col)[feature].agg(['count', 'mean', 'std', 'median', 
                                                        ('25%', lambda x: x.quantile(0.25)),
                                                        ('75%', lambda x: x.quantile(0.75))])
    summary = summary.round(4)
    st.dataframe(summary)
    
    # Statistical tests
    if len(labels) == 2:
        # Two groups - use t-test
        from scipy import stats
        group1, group2 = list(groups)
        t_stat, p_value = stats.ttest_ind(group1[1], group2[1])
        st.info(f"T-test between {group1[0]} and {group2[0]}: t={t_stat:.3f}, p={p_value:.4f}")
    elif len(labels) > 2:
        # Multiple groups - use ANOVA
        from scipy import stats
        f_stat, p_value = stats.f_oneway(*data_to_plot)
        st.info(f"One-way ANOVA: F={f_stat:.3f}, p={p_value:.4f}")

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
        
        if available_features and 'acoustic_features' in case:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            case_values = []
            baseline_means = []
            baseline_stds = []
            labels = []
            
            for feature in available_features:
                # Get case value
                if feature in case.get('acoustic_features', {}):
                    case_val = case['acoustic_features'][feature].get('value', 0)
                elif feature in case.get('semantic_features', {}):
                    case_val = case['semantic_features'][feature].get('value', 0)
                else:
                    continue
                
                case_values.append(case_val)
                baseline_means.append(baseline_df[feature].mean())
                baseline_stds.append(baseline_df[feature].std())
                labels.append(feature.replace('_', ' ').title())
            
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