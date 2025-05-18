import os
import json
import random
import datetime
import argparse
from typing import List, Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Define sentiment categories and sources
SENTIMENTS = ["positive", "neutral", "negative"]
SOURCES = ["Twitter", "Facebook", "News Comments", "Forums", "SMS Feedback"]
REGIONS = ["Kigali", "Eastern", "Western", "Northern", "Southern"]
DEMOGRAPHICS = ["Students", "Workers", "Elderly", "Business"]

# Common themes and concerns about distance-based fare
THEMES = {
    "positive": [
        "The new fare system is more fair based on actual travel distance",
        "I save money on my commute with the new system",
        "Distance-based fares have improved service quality",
        "It's more transparent knowing exactly what I'm paying for",
        "The tap card system is convenient and easy to use",
        "I appreciate the government modernizing our transport system",
        "Buses seem less crowded now with the new fare structure",
        "Digital payments make transport more efficient"
    ],
    "neutral": [
        "Still trying to understand how the new fare calculation works",
        "The system seems okay but needs some improvements",
        "I haven't noticed much difference in what I pay",
        "Wondering how this will affect transport in the long run",
        "Some routes seem better value than others",
        "The change makes sense but the implementation is confusing",
        "Need more information about how distances are calculated",
        "Not sure if this benefits everyone equally"
    ],
    "negative": [
        "Prices are too high for short distances",
        "Rural areas are unfairly affected by distance-based pricing",
        "The tap card system often fails or has technical issues",
        "The fare calculation seems inaccurate for my route",
        "Transport has become unaffordable for daily commuters",
        "The change was implemented too quickly without proper explanation",
        "I'm paying almost double what I used to pay",
        "The system disadvantages those living far from city centers",
        "Many elderly people find the new system confusing",
        "There's not enough infrastructure in rural areas for this system",
        "Payment methods are too limited for many people"
    ]
}

# Languages to include for multi-language simulation
LANGUAGES = ["English", "Kinyarwanda", "French"]

def generate_prompt(source: str, month: int, sentiment_bias: Dict[str, float]) -> str:
    """
    Generate a prompt for the LLM to create synthetic data
    """
    # Adjust sentiment distribution based on the month (showing trend over time)
    adjusted_bias = sentiment_bias.copy()
    
    # Make some trends over time (more negative in earlier months, improving slightly)
    if month <= 2:  # March and April (right after implementation)
        adjusted_bias["negative"] += 0.2
        adjusted_bias["positive"] -= 0.1
    elif month >= 4:  # May and later (slight improvement)
        adjusted_bias["negative"] -= 0.1
        adjusted_bias["positive"] += 0.1
        
    # Normalize to ensure probabilities sum to 1
    total = sum(adjusted_bias.values())
    for k in adjusted_bias:
        adjusted_bias[k] /= total
        
    # Different prompt templates based on source
    source_contexts = {
        "Twitter": "short tweets (max 280 characters)",
        "Facebook": "Facebook posts (medium length, somewhat conversational)",
        "News Comments": "comments on news articles (can be more detailed and reference the article)",
        "Forums": "forum posts discussing the transport system in detail",
        "SMS Feedback": "brief SMS feedback messages (short and to the point)"
    }
    
    sentiment_dist = ", ".join([f"{k}: {v*100:.1f}%" for k, v in adjusted_bias.items()])
    
    prompt = f"""Generate 10 realistic {source_contexts[source]} about Rwanda's new distance-based public transport fare system that was implemented in March 2025.

The sentiment distribution should be approximately: {sentiment_dist}

Include a mix of the following languages: 70% English, 20% Kinyarwanda, 10% French.

For each item, include:
1. The text content (in the appropriate style for {source})
2. The sentiment (positive, neutral, or negative)
3. The language used (English, Kinyarwanda, or French)
4. A region in Rwanda (Kigali, Eastern, Western, Northern, Southern)
5. A demographic group (Students, Workers, Elderly, Business)
6. The specific concern or topic being discussed
7. A date in {month_to_name(month)} 2025

Format as a JSON array with objects containing fields: content, sentiment, language, region, demographic, topic, date
"""
    return prompt

def month_to_name(month_num: int) -> str:
    """Convert month number to name"""
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    return months[month_num - 1]

def call_groq_api(prompt: str) -> List[Dict[str, Any]]:
    """Call the Groq API to generate synthetic data"""
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY not set. Using mock data instead.")
        return generate_mock_data()
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192",  # Or another appropriate model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Extract the JSON data from the response
        try:
            # Try to parse the entire response as JSON
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            import re
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data
            else:
                print(f"Failed to parse JSON from response: {content[:100]}...")
                return generate_mock_data()
                
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return generate_mock_data()

def generate_mock_data() -> List[Dict[str, Any]]:
    """Generate mock data when API is not available"""
    mock_data = []
    
    for _ in range(10):
        sentiment = random.choice(SENTIMENTS)
        language = random.choices(LANGUAGES, weights=[0.7, 0.2, 0.1])[0]
        region = random.choice(REGIONS)
        demographic = random.choice(DEMOGRAPHICS)
        
        mock_data.append({
            "content": random.choice(THEMES[sentiment]),
            "sentiment": sentiment,
            "language": language,
            "region": region,
            "demographic": demographic,
            "topic": "Price Fairness" if sentiment == "negative" else "System Implementation",
            "date": f"2025-05-{random.randint(1, 28):02d}"
        })
    
    return mock_data

def generate_dataset(output_dir: str, num_months: int = 3):
    """Generate a complete dataset and save to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize sentiment biases for different sources
    sentiment_biases = {
        "Twitter": {"positive": 0.2, "neutral": 0.3, "negative": 0.5},
        "Facebook": {"positive": 0.25, "neutral": 0.35, "negative": 0.4},
        "News Comments": {"positive": 0.3, "neutral": 0.3, "negative": 0.4},
        "Forums": {"positive": 0.15, "neutral": 0.25, "negative": 0.6},
        "SMS Feedback": {"positive": 0.2, "neutral": 0.2, "negative": 0.6}
    }
    
    all_data = []
    
    # Generate data for multiple months
    for month in range(3, 3 + num_months):  # Starting from March (implementation)
        for source in SOURCES:
            print(f"Generating {source} data for {month_to_name(month)}...")
            
            prompt = generate_prompt(source, month, sentiment_biases[source])
            data = call_groq_api(prompt)
            
            # Add source and format dates properly
            for item in data:
                item["source"] = source
                
                # Ensure date is in YYYY-MM-DD format
                if "date" in item and not item["date"].startswith("2025-"):
                    day = random.randint(1, 28)
                    item["date"] = f"2025-{month:02d}-{day:02d}"
                
                # Add a unique ID
                item["id"] = f"{len(all_data) + 1:06d}"
            
            all_data.extend(data)
    
    # Save the complete dataset
    with open(os.path.join(output_dir, "transport_fare_feedback_dataset.json"), "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Generated dataset with {len(all_data)} entries")
    
    # Create separate files by source
    for source in SOURCES:
        source_data = [item for item in all_data if item["source"] == source]
        filename = f"{source.lower().replace(' ', '_')}_data.json"
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(source_data, f, indent=2)
        print(f"  - {len(source_data)} entries for {source}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for Rwanda transport fare sentiment analysis")
    parser.add_argument("--output", type=str, default="data", help="Output directory for the generated data")
    parser.add_argument("--months", type=int, default=3, help="Number of months to generate data for (starting from March 2025)")
    args = parser.parse_args()
    
    generate_dataset(args.output, args.months)
