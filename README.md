# **Earnings Call Acoustic Analysis: Correlation with Credit Ratings**

\[[Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)\]([https://earnings-call-acoustic-analysis-qryia9evgh5zebp5eysjkg.streamlit.app/](https://earnings-call-acoustic-analysis-qryia9evgh5zebp5eysjkg.streamlit.app/))  
 \[[Python](https://img.shields.io/badge/python-3.8+-blue.svg)\]([https://www.python.org/downloads/](https://www.python.org/downloads/))  
 \[[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)\]([https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))

## **Overview**

This repository includes the implementation for a master's thesis that examined the relationship between acoustic features of earnings calls and subsequent credit rating actions by Fitch, Moody's, and S\&P Global. To validate acoustic stress indicators in financial communications, the study uses FinBERT-based semantic scores. Results indicate while some downgrade and upgrade cases exhibited elevated vocal stress and negative sentiments, the patterns are heterogeneous and occasionally divergent, underscoring the complexity of financial communication. The single upgrade case presents both high acoustic variability and notably, negative semantic tone, suggesting that tone and language signals do not always converge with rating outcomes. Transparent and replicable methodology is developed to support future research, given the exploration nature of this study.

The study provides an empirical method for integrating multimodal acoustic semantic analysis with financial outcome indicators, while openly acknowledging the limitations imposed by small sample and data imbalance. Future research is recommended to use larger datasets and multimodal fusion mechanisms. The findings highlight the potential for voice technology to integrate into credit risk analysis and capital market domains. 

**Live Demo**: [Earnings Call Acoustic Analysis Demonstrator](https://earnings-call-acoustic-analysis-qryia9evgh5zebp5eysjkg.streamlit.app/)

## **Key Features**

* **Acoustic Feature Extraction**: F0 coefficient of variation, F0 standard deviation, pause frequency, and jitter analysis  
* **Semantic Validation**: FinBERT integration for sentiment analysis and directional validation  
* **Statistical Analysis**: Percentile ranking, bootstrap confidence intervals, and MAD-based effect estimation  
* **Interactive Dashboard**: Streamlit-based demonstrator for exploring acoustic-semantic-rating relationships  
* **Case Study Framework**: Detailed analysis of upgrade/downgrade cases against affirmation baseline

## **Installation**

### **Prerequisites**

* Python 3.8+  
* pip package manager  
* Virtual environment (recommended)

### **Setup**

1. Clone the repository:

git clone https://github.com/anit-z/earnings-call-acoustic-analysis.git  
cd earnings-call-acoustic-analysis

2. Create and activate virtual environment:

python \-m venv venv  
source venv/bin/activate  \# On Windows: venv\\Scripts\\activate

3. Install dependencies:

pip install \-r requirements.txt

### **Dependencies**

Core dependencies include:

* `streamlit` \- Interactive dashboard  
* `pandas`, `numpy` \- Data processing  
* `matplotlib`, `seaborn` \- Visualization  
* `scipy` \- Statistical analysis  
* `librosa` \- Audio processing  
* `parselmouth` \- Praat integration  
* `transformers` \- FinBERT implementation  
* `requests` \- Data fetching

## **Dataset**

This study uses the **Earnings-21** dataset:

* 44 earnings calls from 2020  
* 39 hours of audio recordings  
* 24 companies with credit ratings (21 affirmations, 2 downgrades, 1 upgrade)  
* 9 industry sectors  
* Available at: [Earnings-21 Dataset](https://github.com/revdotcom/speech-datasets)

Credit rating metadata is available at:
data/raw/ratings/ratings\_metadata.csv


## **Usage**

\# Generate case studies  
python src/analysis/case\_studies/generate\_case\_studies.py \\  
    \--features\_dir data/features/combined \\  
    \--audio\_dir data/processed/audio \\  
    \--ratings\_file data/raw/ratings/ratings\_metadata.csv \\  
    \--output\_dir results/analysis/case\_studies \\  
    \--num\_cases 5 \\  
    \--selection combined \\  
    \--bootstrap 10000 \\  
    \--confidence 0.95 \\  
    \--random\_seed 42

\# Analyze correlations  
python src/analysis/correlation\_analysis/analyze\_correlations.py \\  
    \--features\_dir data/features/combined \\  
    \--ratings\_file data/raw/ratings/ratings\_metadata.csv \\  
    \--output\_dir results/analysis/correlations \\  
    \--significance 0.05 \\  
    \--correction fdr\_bh \\  
    \--random\_seed 42

\# Descriptive statistics  
python src/analysis/descriptive\_stats/descriptive\_analysis.py \\  
    \--features\_file data/features/combined/combined\_features.csv \\  
    \--ratings\_file data/raw/ratings/ratings\_metadata.csv \\  
    \--output\_dir results/analysis/descriptive \\  
    \--bootstrap 10000 \\  
    \--confidence 0.95 \\  
    \--random\_seed 42

### **4\. Run Interactive Dashboard**

streamlit run demonstrator/demonstrator.py

Or visit the [live demo](https://earnings-call-acoustic-analysis-qryia9evgh5zebp5eysjkg.streamlit.app/)

## **Methodology**

### **Acoustic Features**

* **F0\_cv**: Fundamental frequency coefficient of variation  
* **F0\_std**: F0 standard deviation (normalized)  
* **Pause Frequency**: Proportion of silence/pause rate  
* **Jitter Local**: Voice micro-instability measure

### **Statistical Approach**

* Percentile ranking against affirmation baseline  
* Bootstrap resampling (10,000 iterations)  
* 95% confidence intervals  
* Effect size estimation using MAD scaling

### **Validation Framework**

* FinBERT sentiment as directional validator  
* Acoustic-semantic convergence/divergence patterns  
* Case-by-case profiling for non-affirmation events

## **Limitations**

* Small sample size (n=24) with extreme class imbalance  
* Single-year data (2020) limiting generalizability  
* Call-level aggregation obscuring speaker-specific patterns  
* Limited to four core acoustic features

## **Citation**

If you use this code or methodology in your research, please cite:

@mastersthesis{zhang2025earnings,  
  title={Case Study on Correlation between Earnings Call Acoustic Features and Credit Ratings: A FinBERT Validation Approach},  
  author={Zhang, Tiantian},  
  year={2025},  
  school={[University of Groningen](https://www.rug.nl/cf/?lang=en), Campus Fryslân},  
  type={MSc Thesis}  
}

## **Contributing**

Contributions are welcome\! Please:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit changes (`git commit -m 'Add amazing feature'`)  
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## **License**

This project is licensed under the MIT License \- see LICENSE file for details.

## **Acknowledgments**

* Earnings-21 dataset creators   
  Del Rio, M., Delworth, N., Westerman, R., Huang, M., Bhandari, N., Palakapilly, J., McNamara, Q., Dong, J., Żelasko, P., & Jetté, M. (2021). Earnings-21: A practical benchmark for ASR in the wild. In *Proceedings of Interspeech 2021* (pp. 3465–3469). [https://doi.org/10.21437/Interspeech.2021-1915](https://doi.org/10.21437/Interspeech.2021-1915)  
* FinBERT developers  
  Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A large language model for extracting information from financial text. Contemporary Accounting Research, 40(2), 806-841.  
* University of Groningen, Campus Fryslân MSc VT supervisory team  
* Open-source speech processing community

## **Contact**

* **Author**: Tiantian Zhang  
* **Email**: \[t.zhang.28@rug.student.nl\]  
* **GitHub**: [@anit-z](https://github.com/anit-z)

## **Future Work**

* Expand to larger datasets (SPGISpeech, MAEC)  
* Implement speaker-level differentiation  
* Develop real-time processing capabilities  
* Integrate additional acoustic features  
* Apply multimodal fusion techniques

---

**Note**: This is research code. While efforts have been made to ensure reproducibility, results may vary based on computational environment and random seed settings.