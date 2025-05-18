import os
import json
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import time
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Translation function using Groq API
def translate_text(text: str, source_language: str, target_language: str = "English") -> str:
    """
    Translate text using Groq API
    
    Args:
        text: The text to translate
        source_language: The language of the input text
        target_language: The language to translate to (default: English)
        
    Returns:
        Translated text
    """
    if source_language == target_language:
        return text
        
    if not GROQ_API_KEY:
        print(f"Warning: GROQ_API_KEY not set. Cannot translate from {source_language} to {target_language}")
        return f"[Translation missing: {text}]"
    
    prompt = f"""Translate the following text from {source_language} to {target_language}:
    
"{text}"

Provide only the translated text without any additional context or explanation.
"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-8b-8192",  # Using a smaller model for efficiency
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    try:
        for attempt in range(5):  # Retry up to 5 times
            try:
                response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                translated_text = result["choices"][0]["message"]["content"]
                
                # Remove any quotation marks that might be included
                translated_text = translated_text.strip().strip('"\'')
                
                return translated_text
            except requests.exceptions.HTTPError as http_err:
                if response.status_code == 429 and attempt < 4:  # Too Many Requests
                    print(f"Rate limit hit. Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise http_err
        
        # Remove any quotation marks that might be included
        translated_text = translated_text.strip().strip('"\'')
        
        return translated_text
        
    except Exception as e:
        print(f"Error translating text: {e}")
        return f"[Translation error: {text}]"

def process_feedback_data(input_file: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Process feedback data by translating non-English content
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the processed data (optional)
        
    Returns:
        Processed data with translations
    """
    # Load the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    for i, item in enumerate(data):
        content = item.get("content", "")
        language = item.get("language", "English")
        
        # Create a new processed item
        processed_item = item.copy()
        
        # Only translate non-English content
        if language != "English":
            print(f"Translating item {i+1}/{len(data)} from {language} to English...")
            translation = translate_text(content, language, "English")
            processed_item["content_original"] = content
            processed_item["content"] = translation
            processed_item["language_original"] = language
            processed_item["language"] = "English (Translated)"
        
        processed_data.append(processed_item)
    
    # Save the processed data if an output file is specified
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
    
    return processed_data

def batch_translate_dataset(data_dir: str):
    """
    Translate all dataset files in a directory
    
    Args:
        data_dir: Directory containing the dataset files
    """
    # Process main dataset
    main_file = os.path.join(data_dir, "transport_fare_feedback_dataset.json")
    if os.path.exists(main_file):
        output_file = os.path.join(data_dir, "translated", "transport_fare_feedback_dataset_translated.json")
        process_feedback_data(main_file, output_file)
    
    # Process individual source files
    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and filename != "transport_fare_feedback_dataset.json":
            input_file = os.path.join(data_dir, filename)
            output_file = os.path.join(data_dir, "translated", filename.replace(".json", "_translated.json"))
            process_feedback_data(input_file, output_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Translate non-English content in feedback data")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file or directory")
    parser.add_argument("--output", type=str, help="Output JSON file (optional)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        batch_translate_dataset(args.input)
    else:
        output_file = args.output or args.input.replace(".json", "_translated.json")
        process_feedback_data(args.input, output_file)
