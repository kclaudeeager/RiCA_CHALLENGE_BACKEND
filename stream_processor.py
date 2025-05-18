import os
import json
import time
from typing import Dict, List, Any, Callable
from dotenv import load_dotenv
import argparse
import glob

# Import our modules
from sentiment_analysis import SentimentAnalyzer
from translation_pipeline import translate_text

load_dotenv()

class StreamProcessor:
    def __init__(
        self,
        sentiment_model_path: str = None,
        use_llm: bool = True
    ):
        """
        Initialize the stream processor
        
        Args:
            sentiment_model_path: Path to sentiment analysis model
            use_llm: Whether to use LLM for sentiment analysis
        """
        # Initialize sentiment analyzer
        self.analyzer = SentimentAnalyzer(sentiment_model_path, use_llm)

    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single message
        
        Args:
            message: Input message
            
        Returns:
            Processed message
        """
        result = message.copy()
        
        if "content" not in message:
            return result  # Return unmodified for non-content messages
    
        # Step 1: Translation if needed
        content = message.get("content", "")
        language = message.get("language", "English")
        
        if language != "English":
            try:
                translated = translate_text(content, language, "English")
                result["content_original"] = content
                result["content"] = translated
                result["language_original"] = language
                result["language"] = "English (Translated)"
            except Exception as e:
                print(f"Translation error: {e}")
        
        # Step 2: Sentiment analysis
        try:
            analysis = self.analyzer.analyze(result["content"])
            result.update(analysis)
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            result["sentiment"] = "unknown"
            result["topic"] = "unknown"
        
        # Add processing timestamp
        result["processed_at"] = int(time.time())
        
        return result
    
    def batch_process_file(self, input_file: str, output_file: str = None, batch_size: int = 100):
        """
        Process messages from a file in batches
        
        Args:
            input_file: Input JSON file
            output_file: Output JSON file (optional)
            batch_size: Batch size for processing
        """
        # Load data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        
        # Handle both list and dictionary inputs
        if isinstance(data, dict):
            # Process dictionary data
            processed = self.process_message(data)
            results.append(processed)
        else:
            # Process list data in batches
            total = len(data)
            for i in range(0, total, batch_size):
                batch = data[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} items)...")
                
                for item in batch:
                    processed = self.process_message(item)
                    results.append(processed)
        
        # Save results if output file is specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results
    
    
    def batch_process_directory(self, input_dir: str, output_dir: str = None, batch_size: int = 100):
        """
        Process all JSON files in a directory
        
        Args:
            input_dir: Input directory containing JSON files
            output_dir: Output directory for processed files (optional)
            batch_size: Batch size for processing
        """
        # Get all JSON files in directory
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return
        
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            filename = os.path.basename(json_file)
            if filename=="dataset_stats.json":
                print("Skipping the stats file")
                continue
            
            print(f"\nProcessing {filename}...")
            
            if output_dir:
                output_file = os.path.join(output_dir, f"processed_{filename}")
            else:
                output_file = None
                
            self.batch_process_file(json_file, output_file, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processor for transport fare feedback")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing JSON files")
    parser.add_argument("--output", type=str, help="Output directory for processed files")
    parser.add_argument("--model", type=str, help="Path to sentiment analysis model")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for sentiment analysis")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Create processor
    processor = StreamProcessor(
        sentiment_model_path=args.model,
        use_llm=args.use_llm
    )
    
    # Process directory
    processor.batch_process_directory(args.input, args.output, args.batch_size)