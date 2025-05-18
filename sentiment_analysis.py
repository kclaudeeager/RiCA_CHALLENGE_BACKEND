import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import argparse
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
import time
import backoff
from transformers import pipeline
from datetime import datetime
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


class SentimentAnalyzer:
    def __init__(self, model_path: Optional[str] = None, use_llm: bool = False):
        """Initialize the sentiment analyzer"""
        self.use_llm = use_llm
        self.model = None
        self.last_request_time = 0
        self.min_request_interval = 3.0  # Increased to 3 seconds
        
        # Initialize transformer models for fallback
        print("Loading transformer models...")
        self.fallback_classifier = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=-1
        )
        
        # Load custom model if provided
        if not use_llm and model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)


    def train(self, data: List[Dict[str, Any]], model_output_path: Optional[str] = None) -> None:
        """
        Train a sentiment analysis model
        
        Args:
            data: List of data items with "content" and "sentiment" fields
            model_output_path: Path to save the trained model
        """
        if self.use_llm:
            print("Using LLM for analysis, no training needed")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['content'], df['sentiment'], test_size=0.2, random_state=42
        )
        
        # Create and train pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save model if path is provided
        if model_output_path:
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            joblib.dump(self.model, model_output_path)
            print(f"Model saved to {model_output_path}")
    
    
    
    def analyze_with_transformers(self, text: str) -> Dict[str, Any]:
        """Analyze using transformer models as fallback"""
        try:
            # Get sentiment
            sentiment_result = self.fallback_classifier(text)[0]
            sentiment = sentiment_result["label"].lower()
            
            # Map sentiment labels
            sentiment_map = {
                "neg": "negative",
                "pos": "positive",
                "neu": "neutral"
            }
            sentiment = sentiment_map.get(sentiment, sentiment)
            
            # Topic detection using keywords
            topics = {
                "price": "Price Fairness",
                "cost": "Price Fairness",
                "expensive": "Price Fairness",
                "payment": "Payment Methods",
                "card": "Tap Card Issues",
                "tap": "Tap Card Issues",
                "rural": "Rural Impact",
                "village": "Rural Impact",
                "calculate": "Fare Calculation",
                "distance": "Distance Accuracy",
                "reliable": "System Reliability",
                "system": "System Reliability",
                "route": "Route Changes",
                "implement": "Implementation Pace",
                "economy": "Economic Impact"
            }
            
            text_lower = text.lower()
            topic = "Unknown"
            for keyword, topic_name in topics.items():
                if keyword in text_lower:
                    topic = topic_name
                    break
            
            return {
                "sentiment": sentiment,
                "topic": topic,
                "confidence": sentiment_result["score"],
                "analyzed_by": "transformer"
            }
            
        except Exception as e:
            print(f"Transformer analysis failed: {str(e)}")
            return {
                "sentiment": "neutral",
                "topic": "Unknown",
                "confidence": 0.0,
                "analyzed_by": "fallback"
            }
            
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, ValueError),
        max_tries=3,
        max_time=30
    )
    def analyze_with_llm(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Groq LLM API with fallback"""
        if not GROQ_API_KEY:
            return self.analyze_with_transformers(text)
        
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        
        prompt = f"""You are analyzing feedback about Rwanda's distance-based transport fare system.
For the following text, provide:
1. Sentiment probabilities for positive, neutral, and negative.
2. Main topic from: Price Fairness, Fare Calculation, Rural Impact, Payment Methods, System Reliability, 
    Tap Card Issues, Route Changes, Implementation Pace, Distance Accuracy, Economic Impact

Respond ONLY with a JSON object in this exact format:
{{"sentiment_probabilities": {{"positive": <probability>, "neutral": <probability>, "negative": <probability>}}, "topic": "<topic>"}}
where <probability> is a float between 0 and 1 summing to 1, and <topic> is one from the list above.

Text: "{text}"
"""
        
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 150
            }

            response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Try to extract JSON if it's embedded in other text
            try:
                # Find JSON-like structure
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON structure found in response")
                
                # Validate required fields and values
                if not isinstance(analysis, dict):
                    raise ValueError("Response is not a dictionary")
                    
                sentiment_probs = analysis.get("sentiment_probabilities", {})
                if not all(key in sentiment_probs for key in ["positive", "neutral", "negative"]):
                    raise ValueError("Missing sentiment probabilities")
                    
                if "topic" not in analysis:
                    raise ValueError("Missing topic field")
                    
                analysis["analyzed_by"] = "llm"
                print(f"LLM analysis result: {analysis}")
                return analysis
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse LLM response: {str(e)}\nResponse was: {content}")
                return self.analyze_with_transformers(text)
                
        except Exception as e:
            print(f"LLM analysis failed: {str(e)}")
            return self.analyze_with_transformers(text)
        
    def analyze(self, text: str) -> Dict[str, Any]:
        """Main analysis method"""
        if not text or not isinstance(text, str):
            return {
                "sentiment": "neutral",
                "topic": "Unknown",
                "confidence": 0.0,
                "analyzed_by": "invalid_input"
            }
            
        try:
            if self.use_llm:
                result = self.analyze_with_llm(text)
            else:
                result = self.analyze_with_transformers(text)
                
            # Add timestamp
            result["analyzed_at"] = datetime.now().isoformat()
            return result
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return {
                "sentiment": "neutral",
                "topic": "Unknown",
                "confidence": 0.0,
                "analyzed_by": "error",
                "analyzed_at": datetime.now().isoformat()
            }
    
    def batch_analyze(self, data: List[Dict[str, Any]], content_field: str = "content") -> List[Dict[str, Any]]:
        """Process a batch of items"""
        results = []
        total = len(data)
        
        for i, item in enumerate(data):
            if (i + 1) % 10 == 0:
                print(f"Processing item {i + 1}/{total}...")
                
            text = item.get(content_field, "")
            analysis = self.analyze(text)
            
            new_item = item.copy()
            new_item.update(analysis)
            results.append(new_item)
            
        return results

def process_dataset(input_file: str, output_file: str, model_path: Optional[str] = None, use_llm: bool = False, train: bool = False):
    """
    Process a dataset with sentiment analysis
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        model_path: Path to model file (for loading or saving)
        use_llm: Whether to use LLM-based analysis
        train: Whether to train a new model
    """
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(model_path, use_llm)
    
    # Train if requested
    if train and not use_llm:
        print(f"Training model on {len(data)} examples...")
        analyzer.train(data, model_path)
    
    # Process data
    print(f"Analyzing sentiment for {len(data)} items...")
    results = analyzer.batch_analyze(data)
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment analysis for transport fare feedback")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--model", type=str, help="Model file path (for loading or saving)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for sentiment analysis")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.model, args.use_llm, args.train)
