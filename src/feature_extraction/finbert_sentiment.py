# src/feature_extraction/finbert_sentiment.py
"""Extract FinBERT sentiment from earnings call transcripts"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import FINBERT_MODEL, FINBERT_BATCH_SIZE, PROCESSED_DATA_DIR, FEATURES_DIR

class FinBERTSentimentAnalyzer:
    def __init__(self, model_name=FINBERT_MODEL):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def analyze_text(self, text, max_length=512):
        """Analyze sentiment of text using FinBERT"""
        # Split text into chunks if too long
        chunks = self._split_text(text, max_length)
        
        all_sentiments = []
        
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # FinBERT returns [positive, negative, neutral]
            sentiment_scores = predictions.cpu().numpy()[0]
            all_sentiments.append(sentiment_scores)
        
        # Average sentiments across chunks
        avg_sentiment = np.mean(all_sentiments, axis=0)
        
        return {
            'positive_score': avg_sentiment[0],
            'negative_score': avg_sentiment[1],
            'neutral_score': avg_sentiment[2],
            'sentiment_polarity': avg_sentiment[0] - avg_sentiment[1],  # Range: [-1, 1]
            'sentiment_subjectivity': 1 - avg_sentiment[2]  # Higher = more subjective
        }
    
    def _split_text(self, text, max_length):
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_tokens = len(self.tokenizer.tokenize(word))
            if current_length + word_tokens > max_length - 2:  # Account for [CLS] and [SEP]
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_tokens
            else:
                current_chunk.append(word)
                current_length += word_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
def process_all_transcripts():
    """Process all transcripts and extract sentiment"""
    # Load metadata
    metadata = pd.read_csv(PROCESSED_DATA_DIR / "metadata.csv")
    
    # Initialize analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    # Process each transcript
    all_sentiments = []
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        transcript_path = PROCESSED_DATA_DIR / "transcripts" / row['transcript_file']
        
        if transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            sentiment = analyzer.analyze_text(transcript)
            sentiment['company'] = row['company']
            sentiment['call_date'] = row['call_date']
            all_sentiments.append(sentiment)
    
    # Save sentiments
    sentiment_df = pd.DataFrame(all_sentiments)
    sentiment_df.to_csv(FEATURES_DIR / "finbert_sentiment.csv", index=False)
    
if __name__ == "__main__":
    process_all_transcripts()