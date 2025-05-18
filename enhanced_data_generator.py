import os
import json
import random
import datetime
import argparse
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm  # For progress bars

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Define sentiment categories and sources
SENTIMENTS = ["positive", "neutral", "negative"]
SOURCES = ["Twitter", "Facebook", "News Comments", "Forums", "SMS Feedback"]
REGIONS = ["Kigali", "Eastern", "Western", "Northern", "Southern"]
DEMOGRAPHICS = ["Students", "Workers", "Elderly", "Business"]

# Common themes based on transport fare system
TOPICS = [
    "Price Fairness", 
    "Fare Calculation", 
    "Rural Impact", 
    "Payment Methods", 
    "System Reliability",
    "Tap Card Issues",
    "Route Changes",
    "Implementation Pace",
    "Distance Accuracy",
    "Economic Impact"
]

# Richer multilingual themes and concerns about distance-based fare
THEMES = {
    "positive": {
        "English": [
            "The new fare system is more fair based on actual travel distance",
            "I save money on my commute with the new system",
            "Distance-based fares have improved service quality",
            "It's more transparent knowing exactly what I'm paying for",
            "The tap card system is convenient and easy to use",
            "I appreciate the government modernizing our transport system",
            "Buses seem less crowded now with the new fare structure",
            "Digital payments make transport more efficient"
        ],
        "Kinyarwanda": [
            "Uburyo bushya bwo kwishyura urugendo ni bwiza bwubahiriza intera",
            "Ndakora ubuzigame ku mafaranga y'urugendo kubera iyi sisteme nshya",
            "Amafaranga ashingiye ku ntera yatumye serivisi itangwa neza",
            "Ni byiza kumenya neza ibyo nishyura",
            "Ikarita yo kwishyuza ni yoroshye gukoreshwa",
            "Ndashimira guverinoma kubera kuvugurura uburyo bw'ubutwererero",
            "Amavatiri ntakiri yuzuye cyane ubu",
            "Kwishyura hakoreshejwe ikoranabuhanga birashimishije"
        ],
        "French": [
            "Le nouveau système de tarification est plus équitable en fonction de la distance réelle",
            "J'économise de l'argent sur mes déplacements avec le nouveau système",
            "Les tarifs basés sur la distance ont amélioré la qualité du service",
            "C'est plus transparent de savoir exactement ce que je paie",
            "Le système de carte à puce est pratique et facile à utiliser",
            "J'apprécie que le gouvernement modernise notre système de transport",
            "Les bus semblent moins bondés maintenant avec la nouvelle structure tarifaire",
            "Les paiements numériques rendent le transport plus efficace"
        ]
    },
    "neutral": {
        "English": [
            "Still trying to understand how the new fare calculation works",
            "The system seems okay but needs some improvements",
            "I haven't noticed much difference in what I pay",
            "Wondering how this will affect transport in the long run",
            "Some routes seem better value than others",
            "The change makes sense but the implementation is confusing",
            "Need more information about how distances are calculated",
            "Not sure if this benefits everyone equally"
        ],
        "Kinyarwanda": [
            "Ngerageza gusobanukirwa neza uburyo bushya bwo kubara amafaranga",
            "Iyi sisteme igaragara neza ariko ikeneye kunozwa",
            "Ntabwo nabonye itandukaniro rinini ku mafaranga nishyura",
            "Ndibaza uko bizagira ingaruka ku buryo bw'ubutwarere mu gihe kirekire",
            "Zimwe mu nzira zigaragara nk'izifite agaciro karushijeho",
            "Ihinduka rigaragara rifite ishingiro ariko ishyirwa mu bikorwa riratanga urujijo",
            "Hakenewe amakuru arushijeho ku buryo intera zibarwa",
            "Simeze neza niba bizagirira buri wese akamaro mu rugero rumwe"
        ],
        "French": [
            "J'essaie toujours de comprendre comment fonctionne le nouveau calcul des tarifs",
            "Le système semble correct mais nécessite quelques améliorations",
            "Je n'ai pas remarqué beaucoup de différence dans ce que je paie",
            "Je me demande comment cela affectera les transports à long terme",
            "Certains itinéraires semblent offrir un meilleur rapport qualité-prix que d'autres",
            "Le changement est logique mais la mise en œuvre est déroutante",
            "Besoin de plus d'informations sur la façon dont les distances sont calculées",
            "Pas sûr que cela profite à tous de manière égale"
        ]
    },
    "negative": {
        "English": [
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
        ],
        "Kinyarwanda": [
            "Ibiciro ni bihanitse cyane ku ntera ngufi",
            "Uturere tw'icyaro tubangamiwe n'uburyo bwo kugena ibiciro bishingiye ku ntera",
            "Ikarita yo kwishyura intera ijya inaniranwa cyangwa ifite ibibazo bya tekiniki",
            "Uburyo bwo kubara amafaranga y'urugendo ntabwo buboneye ku nzira yanjye",
            "Ubutware bwahindutse budashoboka ku bakozi bajya ku kazi buri munsi",
            "Ihinduka ryashyizwe mu bikorwa vuba cyane nta gusobanura bihagije",
            "Ndishyura hafi ibikubye kabiri ibyo nahoranishyura",
            "Iyi sisiteme ibangamiye ababa kure y'umujyi",
            "Abantu benshi bakuze basanga uburyo bushya bubatera urujijo",
            "Nta ibikorwa remezo bihagije mu duce tw'icyaro kugira ngo hakoreshwe iyi sisiteme",
            "Uburyo bwo kwishyura ni bucye ku bantu benshi"
        ],
        "French": [
            "Les prix sont trop élevés pour les courtes distances",
            "Les zones rurales sont injustement affectées par la tarification basée sur la distance",
            "Le système de carte à puce tombe souvent en panne ou présente des problèmes techniques",
            "Le calcul du tarif semble inexact pour mon itinéraire",
            "Le transport est devenu inabordable pour les navetteurs quotidiens",
            "Le changement a été mis en œuvre trop rapidement sans explication adéquate",
            "Je paie presque le double de ce que je payais avant",
            "Le système désavantage ceux qui vivent loin des centres-villes",
            "De nombreuses personnes âgées trouvent le nouveau système déroutant",
            "Il n'y a pas assez d'infrastructure dans les zones rurales pour ce système",
            "Les méthodes de paiement sont trop limitées pour beaucoup de personnes"
        ]
    }
}

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
    topics_str = ", ".join(TOPICS)
    
    prompt = f"""Generate 10 realistic {source_contexts[source]} about Rwanda's new distance-based public transport fare system that was implemented in March 2025.

The sentiment distribution should be approximately: {sentiment_dist}

Include a mix of the following languages: 70% English, 20% Kinyarwanda, 10% French.

For each item, include:
1. The text content (in the appropriate style for {source})
2. The sentiment (positive, neutral, or negative)
3. The language used (English, Kinyarwanda, or French)
4. A region in Rwanda (Kigali, Eastern, Western, Northern, Southern)
5. A demographic group (Students, Workers, Elderly, Business)
6. The specific concern or topic being discussed (choose from: {topics_str})
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
        return generate_mock_data(10)
    
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
                # Attempt to extract JSON-like content without code block markers
                try:
                    json_start = content.find("[")
                    json_end = content.rfind("]")
                    if json_start != -1 and json_end != -1:
                        data = json.loads(content[json_start:json_end + 1])
                        return data
                except json.JSONDecodeError:
                    pass
                print(f"Failed to parse JSON from response: {content[:100]}...")
                return generate_mock_data(10)
                
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return generate_mock_data(10)

def generate_mock_text(sentiment: str, language: str) -> str:
    """Generate realistic mock text based on sentiment and language"""
    if language in THEMES[sentiment] and THEMES[sentiment][language]:
        return random.choice(THEMES[sentiment][language])
    
    # Fallback to English if the specific language is not available
    return random.choice(THEMES[sentiment]["English"])

def generate_mock_data(count: int = 10) -> List[Dict[str, Any]]:
    """Generate mock data when API is not available"""
    mock_data = []
    
    today = datetime.date.today()
    
    for _ in range(count):
        sentiment = random.choices(SENTIMENTS, weights=[0.25, 0.25, 0.5])[0]
        language = random.choices(list(LANGUAGES.keys()), weights=list(LANGUAGES.values()))[0]
        region = random.choice(REGIONS)
        demographic = random.choice(DEMOGRAPHICS)
        topic = random.choice(TOPICS)
        
        # Generate a random date in March-May 2025
        month = random.randint(3, 5)
        day = random.randint(1, 28)
        
        mock_data.append({
            "content": generate_mock_text(sentiment, language),
            "sentiment": sentiment,
            "language": language,
            "region": region,
            "demographic": demographic,
            "topic": topic,
            "date": f"2025-{month:02d}-{day:02d}"
        })
    
    return mock_data

def generate_realistic_mock_data(batch_size: int = 50, total_items: int = 1000) -> List[Dict[str, Any]]:
    """Generate more realistic mock data with proper distributions"""
    all_data = []
    
    # Set up source distributions
    source_weights = {
        "Twitter": 0.35,
        "Facebook": 0.25,
        "News Comments": 0.20,
        "Forums": 0.12,
        "SMS Feedback": 0.08
    }
    
    # Set up language distributions
    language_weights = {
        "English": 0.7,
        "Kinyarwanda": 0.2,
        "French": 0.1
    }
    
    # Create batches with progress bar
    with tqdm(total=total_items, desc="Generating mock data") as pbar:
        while len(all_data) < total_items:
            # Generate a batch
            current_batch_size = min(batch_size, total_items - len(all_data))
            
            for _ in range(current_batch_size):
                # Select source based on distribution
                source = random.choices(list(source_weights.keys()), 
                                       weights=list(source_weights.values()))[0]
                
                # Different sentiment distributions for different sources
                sentiment_weights = {"positive": 0.2, "neutral": 0.3, "negative": 0.5}
                if source == "Twitter":
                    sentiment_weights = {"positive": 0.2, "neutral": 0.3, "negative": 0.5}
                elif source == "Forums":
                    sentiment_weights = {"positive": 0.15, "neutral": 0.25, "negative": 0.6}
                
                sentiment = random.choices(list(sentiment_weights.keys()),
                                          weights=list(sentiment_weights.values()))[0]
                
                language = random.choices(list(language_weights.keys()), 
                                         weights=list(language_weights.values()))[0]
                
                # Demographics with realistic distribution per region
                region = random.choice(REGIONS)
                if region == "Kigali":
                    demographic = random.choices(DEMOGRAPHICS, weights=[0.3, 0.4, 0.1, 0.2])[0]
                elif region in ["Eastern", "Western"]:
                    demographic = random.choices(DEMOGRAPHICS, weights=[0.2, 0.3, 0.4, 0.1])[0]
                else:
                    demographic = random.choices(DEMOGRAPHICS, weights=[0.25, 0.35, 0.3, 0.1])[0]
                
                # Topic distribution based on sentiment
                if sentiment == "negative":
                    topic_weights = [0.25, 0.20, 0.15, 0.15, 0.1, 0.05, 0.05, 0.02, 0.02, 0.01]
                elif sentiment == "neutral":
                    topic_weights = [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
                else:  # positive
                    topic_weights = [0.1, 0.1, 0.05, 0.2, 0.2, 0.05, 0.1, 0.1, 0.05, 0.05]
                
                topic = random.choices(TOPICS, weights=topic_weights)[0]
                
                # Generate a random date with more recent dates more likely
                month = random.choices([3, 4, 5], weights=[0.3, 0.3, 0.4])[0]
                day = random.randint(1, 28)
                
                # For more realism in the dataset, create actual content that varies by source type
                content = generate_mock_text(sentiment, language)
                
                # Modify content based on source
                if source == "Twitter":
                    # Add hashtags sometimes
                    if random.random() < 0.4:
                        hashtags = ["#RwandaTransport", "#DistanceFare", "#PublicTransport", 
                                   "#RwandaBuses", "#NewFares", "#TapCard"]
                        content += " " + random.choice(hashtags)
                        
                    # Truncate if too long for Twitter
                    if len(content) > 260:
                        content = content[:260] + "..."
                        
                elif source == "Facebook":
                    # Make it more conversational
                    if random.random() < 0.3 and language == "English":
                        prefixes = ["Just experienced ", "My thoughts on ", "Has anyone else noticed ", 
                                   "I think ", "Today I realized ", "Question: "]
                        content = random.choice(prefixes) + content.lower()
                        
                elif source == "Forums":
                    # More detailed and formal
                    if language == "English" and random.random() < 0.4:
                        content += f" This is particularly important for {demographic} in the {region} region."
                
                all_data.append({
                    "id": f"{len(all_data) + 1:06d}",
                    "content": content,
                    "sentiment": sentiment,
                    "language": language,
                    "region": region,
                    "demographic": demographic,
                    "topic": topic,
                    "source": source,
                    "date": f"2025-{month:02d}-{day:02d}"
                })
            
            pbar.update(current_batch_size)
    
    return all_data

def add_language_detection_errors(data: List[Dict[str, Any]], error_rate: float = 0.05) -> List[Dict[str, Any]]:
    """Add some language detection 'errors' to simulate real-world detection issues"""
    languages = ["English", "Kinyarwanda", "French"]
    
    for item in data:
        if random.random() < error_rate:
            # Randomly assign an incorrect language
            actual_lang = item["language"]
            available_langs = [lang for lang in languages if lang != actual_lang]
            item["detected_language"] = random.choice(available_langs)
        else:
            item["detected_language"] = item["language"]
    
    return data

def generate_dataset(output_dir: str, num_months: int = 3, use_api: bool = True, total_entries: Optional[int] = None):
    """Generate a complete dataset and save to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use API or generate mock data
    all_data = []
    
    if use_api and GROQ_API_KEY:
        # Initialize sentiment biases for different sources
        sentiment_biases = {
            "Twitter": {"positive": 0.2, "neutral": 0.3, "negative": 0.5},
            "Facebook": {"positive": 0.25, "neutral": 0.35, "negative": 0.4},
            "News Comments": {"positive": 0.3, "neutral": 0.3, "negative": 0.4},
            "Forums": {"positive": 0.15, "neutral": 0.25, "negative": 0.6},
            "SMS Feedback": {"positive": 0.2, "neutral": 0.2, "negative": 0.6}
        }
        
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
    else:
        # Generate synthetic data without API
        entries_count = total_entries or 1000  # Default to 1000 entries
        all_data = generate_realistic_mock_data(total_items=entries_count)
        
        # Add some language detection "errors" to simulate real-world scenarios
        all_data = add_language_detection_errors(all_data)
    
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
    
    # Create separate files by month for time-series analysis
    for month in range(3, 3 + num_months):
        month_data = [item for item in all_data if item["date"].startswith(f"2025-{month:02d}")]
        filename = f"month_{month:02d}_data.json"
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(month_data, f, indent=2)
        print(f"  - {len(month_data)} entries for month {month:02d}")
    
    # Create summary statistics
    generate_summary_stats(all_data, output_dir)

def generate_summary_stats(data: List[Dict[str, Any]], output_dir: str):
    """Generate summary statistics from the dataset"""
    stats = {
        "total_entries": len(data),
        "by_sentiment": {s: sum(1 for item in data if item["sentiment"] == s) for s in SENTIMENTS},
        "by_language": {l: sum(1 for item in data if item["language"] == l) for l in ["English", "Kinyarwanda", "French"]},
        "by_source": {s: sum(1 for item in data if item["source"] == s) for s in SOURCES},
        "by_region": {r: sum(1 for item in data if item["region"] == r) for r in REGIONS},
        "by_demographic": {d: sum(1 for item in data if item["demographic"] == d) for d in DEMOGRAPHICS},
        "by_topic": {t: sum(1 for item in data if item["topic"] == t) for t in TOPICS},
        "by_month": {
            "March": sum(1 for item in data if item["date"].startswith("2025-03")),
            "April": sum(1 for item in data if item["date"].startswith("2025-04")),
            "May": sum(1 for item in data if item["date"].startswith("2025-05"))
        },
        "sentiment_by_month": {
            "March": {
                "positive": sum(1 for item in data if item["date"].startswith("2025-03") and item["sentiment"] == "positive"),
                "neutral": sum(1 for item in data if item["date"].startswith("2025-03") and item["sentiment"] == "neutral"),
                "negative": sum(1 for item in data if item["date"].startswith("2025-03") and item["sentiment"] == "negative")
            },
            "April": {
                "positive": sum(1 for item in data if item["date"].startswith("2025-04") and item["sentiment"] == "positive"),
                "neutral": sum(1 for item in data if item["date"].startswith("2025-04") and item["sentiment"] == "neutral"),
                "negative": sum(1 for item in data if item["date"].startswith("2025-04") and item["sentiment"] == "negative")
            },
            "May": {
                "positive": sum(1 for item in data if item["date"].startswith("2025-05") and item["sentiment"] == "positive"),
                "neutral": sum(1 for item in data if item["date"].startswith("2025-05") and item["sentiment"] == "neutral"),
                "negative": sum(1 for item in data if item["date"].startswith("2025-05") and item["sentiment"] == "negative")
            }
        },
        "sentiment_by_region": {
            region: {
                "positive": sum(1 for item in data if item["region"] == region and item["sentiment"] == "positive"),
                "neutral": sum(1 for item in data if item["region"] == region and item["sentiment"] == "neutral"),
                "negative": sum(1 for item in data if item["region"] == region and item["sentiment"] == "negative")
            } for region in REGIONS
        }
    }
    
    # Save summary statistics
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print("Generated dataset statistics")

# Define language distribution for language generation
LANGUAGES = {
    "English": 0.7,
    "Kinyarwanda": 0.2,
    "French": 0.1
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for Rwanda transport fare sentiment analysis")
    parser.add_argument("--output", type=str, default="data", help="Output directory for the generated data")
    parser.add_argument("--months", type=int, default=3, help="Number of months to generate data for (starting from March 2025)")
    parser.add_argument("--no-api", action="store_true", help="Don't use Groq API, generate mock data instead")
    parser.add_argument("--entries", type=int, default=1000, help="Number of entries to generate for mock data")
    args = parser.parse_args()
    
    generate_dataset(args.output, args.months, not args.no_api, args.entries)