import os
import json
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import uuid
from datetime import datetime
from pathlib import Path

# Import our modules
from sentiment_analysis import SentimentAnalyzer
from translation_pipeline import translate_text
from stream_processor import StreamProcessor
from enhanced_data_generator import generate_dataset

app = FastAPI(
    title="Rwanda Transport Fare Sentiment Analysis API",
    description="API for analyzing sentiment of feedback on Rwanda's distance-based transport fare system",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the sentiment analyzer and stream processor
sentiment_analyzer = SentimentAnalyzer(use_llm=True)
stream_processor = StreamProcessor(use_llm=True)

# Create data directories if they don't exist
os.makedirs("data/input", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/translated", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

# Background processing tasks
ongoing_tasks = {}

@app.get("/")
async def root():
    return {"message": "Rwanda Transport Fare Sentiment Analysis API is running"}

@app.post("/api/analyze")
async def analyze_text(text: str, language: str = "English"):
    """Analyze sentiment of a single text input"""
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Translate if needed
    if language != "English":
        translated = translate_text(text, language, "English")
        original_text = text
        text = translated
    else:
        original_text = None
    
    # Analyze sentiment
    result = sentiment_analyzer.analyze(text)
    
    # Format response
    response = {
        "text": text,
        "sentiment": max(result.get("sentiment_probabilities", {}).items(), key=lambda x: x[1])[0] if "sentiment_probabilities" in result else result.get("sentiment", "neutral"),
        "topic": result.get("topic", "Unknown"),
        "confidence": max(result.get("sentiment_probabilities", {}).values()) if "sentiment_probabilities" in result else result.get("confidence", 0.0),
        "probabilities": result.get("sentiment_probabilities"),
        "analyzed_by": result.get("analyzed_by", "unknown")
    }
    
    if original_text:
        response["original_text"] = original_text
        response["original_language"] = language
    
    return response

@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_type: str = Form("sentiment")
):
    """Upload a JSON file for batch processing"""
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are accepted")
    
    # Generate a unique ID for this job
    job_id = str(uuid.uuid4())
    input_path = f"data/input/{job_id}.json"
    output_path = f"data/processed/{job_id}.json"
    
    # Save the uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Start processing in the background
    background_tasks.add_task(process_file, job_id, input_path, output_path, process_type)
    
    ongoing_tasks[job_id] = {
        "status": "processing",
        "started_at": datetime.now().isoformat(),
        "file_name": file.filename,
        "process_type": process_type
    }
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in ongoing_tasks:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ongoing_tasks[job_id]

@app.get("/api/result/{job_id}")
async def get_job_result(job_id: str):
    """Get the result of a completed job"""
    if job_id not in ongoing_tasks:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if ongoing_tasks[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    
    output_path = f"data/processed/{job_id}.json"
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    with open(output_path, "r") as f:
        data = json.load(f)
    
    return data

@app.post("/api/generate-dataset")
async def generate_synthetic_dataset(
    background_tasks: BackgroundTasks,
    entries: int = 1000,
    months: int = 3
):
    """Generate a synthetic dataset for testing"""
    job_id = str(uuid.uuid4())
    output_dir = f"data/output/{job_id}"
    
    background_tasks.add_task(
        generate_dataset_task, 
        job_id, 
        output_dir, 
        entries, 
        months
    )
    
    ongoing_tasks[job_id] = {
        "status": "processing",
        "started_at": datetime.now().isoformat(),
        "process_type": "generate_dataset",
        "params": {
            "entries": entries,
            "months": months
        }
    }
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/api/datasets")
async def list_datasets():
    """List all available datasets"""
    datasets = []
    
    for task_id, task in ongoing_tasks.items():
        if task["process_type"] == "generate_dataset" and task["status"] == "completed":
            datasets.append({
                "id": task_id,
                "created_at": task["started_at"],
                "completed_at": task.get("completed_at"),
                "entries": task["params"]["entries"],
                "months": task["params"]["months"]
            })
    
    return {"datasets": datasets}

@app.get("/api/stats/{dataset_id}")
async def get_dataset_stats(dataset_id: str):
    """Get statistics for a dataset"""
    stats_path = f"data/output/{dataset_id}/dataset_stats.json"
    
    if not os.path.exists(stats_path):
        raise HTTPException(status_code=404, detail="Dataset statistics not found")
    
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    return stats


@app.get("/api/sentiments")
async def get_sentiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sentiment: Optional[str] = None,
    topic: Optional[str] = None
):
    """Get paginated sentiment analysis results with filtering options"""
    try:
        # Calculate pagination offsets
        offset = (page - 1) * page_size
        
        # Get all processed files
        processed_files = []
        for job_id in ongoing_tasks:
            if (ongoing_tasks[job_id]["status"] == "completed" and 
                ongoing_tasks[job_id].get("process_type") == "sentiment"):
                file_path = f"data/processed/{job_id}.json"
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        processed_files.extend(data)

        # Apply filters
        filtered_results = processed_files
        
        if date_from:
            filtered_results = [r for r in filtered_results 
                              if r.get("processed_at", "") >= date_from]
        
        if date_to:
            filtered_results = [r for r in filtered_results 
                              if r.get("processed_at", "") <= date_to]
        
        if sentiment:
            filtered_results = [r for r in filtered_results 
                              if r.get("sentiment") == sentiment]
        
        if topic:
            filtered_results = [r for r in filtered_results 
                              if r.get("topic") == topic]

        # Get total count for pagination
        total_results = len(filtered_results)
        total_pages = (total_results + page_size - 1) // page_size

        # Paginate results
        paginated_results = filtered_results[offset:offset + page_size]

        return {
            "results": paginated_results,
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_results": total_results
            },
            "filters": {
                "date_from": date_from,
                "date_to": date_to,
                "sentiment": sentiment,
                "topic": topic
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiments/summary")
async def get_sentiment_summary(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
):
    """Get summary statistics of sentiment analysis results"""
    try:
        # Aggregate results from all processed files
        all_results = []
        for job_id in ongoing_tasks:
            if (ongoing_tasks[job_id]["status"] == "completed" and 
                ongoing_tasks[job_id].get("process_type") == "sentiment"):
                file_path = f"data/processed/{job_id}.json"
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        all_results.extend(data)

        # Apply date filters
        if date_from:
            all_results = [r for r in all_results 
                          if r.get("processed_at", "") >= date_from]
        
        if date_to:
            all_results = [r for r in all_results 
                          if r.get("processed_at", "") <= date_to]

        # Calculate summaries
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        topic_counts = {}
        total_entries = len(all_results)

        for result in all_results:
            # Count sentiments
            sentiment = result.get("sentiment", "neutral")
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

            # Count topics
            topic = result.get("topic", "Unknown")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        return {
            "total_entries": total_entries,
            "sentiment_distribution": {
                k: (v / total_entries if total_entries > 0 else 0) 
                for k, v in sentiment_counts.items()
            },
            "topic_distribution": {
                k: (v / total_entries if total_entries > 0 else 0) 
                for k, v in topic_counts.items()
            },
            "time_range": {
                "from": date_from,
                "to": date_to
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/all-sentiments")
async def get_all_sentiments():
    """Retrieve all sentiments from processed JSON files"""
    try:
        all_sentiments = []
        processed_dir = Path("data/processed")
        
        # Iterate through all JSON files in the processed directory
        for file_path in processed_dir.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                all_sentiments.extend(data)
        
        return {"sentiments": all_sentiments, "total_count": len(all_sentiments)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background processing tasks
def process_file(job_id: str, input_path: str, output_path: str, process_type: str):
    """Process a file in the background"""
    try:
        if process_type == "sentiment":
            # Use stream processor's batch processing
            results = stream_processor.batch_process_file(input_path, output_path)
            
        elif process_type == "translate":
            with open(input_path, "r") as f:
                data = json.load(f)
                
            results = []
            for item in data:
                content = item.get("content", "")
                language = item.get("language", "English")
                
                # Only translate non-English content
                if language != "English":
                    translation = translate_text(content, language, "English")
                    item["content_original"] = content
                    item["content"] = translation
                    item["language_original"] = language
                    item["language"] = "English (Translated)"
                
                results.append(item)
                
            # Save results for translation
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        else:
            # For other process types, just copy the file
            with open(input_path, "r") as f:
                data = json.load(f)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        
        ongoing_tasks[job_id]["status"] = "completed"
        ongoing_tasks[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        ongoing_tasks[job_id]["status"] = "failed"
        ongoing_tasks[job_id]["error"] = str(e)


def generate_dataset_task(job_id: str, output_dir: str, entries: int, months: int):
    """Generate a synthetic dataset in the background"""
    try:
        generate_dataset(output_dir, months, use_api=False, total_entries=entries)
        
        ongoing_tasks[job_id]["status"] = "completed"
        ongoing_tasks[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        ongoing_tasks[job_id]["status"] = "failed"
        ongoing_tasks[job_id]["error"] = str(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000)
