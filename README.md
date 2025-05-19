# Rwanda Transport Fare Sentiment Analysis Backend

A FastAPI-based backend service for analyzing public sentiment regarding Rwanda's shift from flat-rate to distance-based transport fares. The service processes multilingual feedback, performs sentiment analysis, and provides RESTful APIs for data access and analysis.

## Features

- **Sentiment Analysis API**: Analyze text sentiment using both ML and LLM approaches
- **Multilingual Support**: Handles content in English, Kinyarwanda, and French
- **Synthetic Data Generation**: Generate test datasets with configurable parameters
- **Background Processing**: Asynchronous processing for large datasets
- **RESTful APIs**: Comprehensive endpoints for data management and analysis
- **Real-time Processing**: Stream processing capabilities with Kafka integration

## Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

## API Endpoints

### Core Endpoints

- `GET /`: Health check endpoint
- `POST /api/analyze`: Analyze sentiment of a single text input
- `POST /api/upload`: Upload JSON file for batch processing
- `GET /api/status/{job_id}`: Check processing job status
- `GET /api/result/{job_id}`: Get processed results

### Dataset Management

- `POST /api/generate-dataset`: Generate synthetic dataset
- `GET /api/datasets`: List available datasets
- `GET /api/stats/{dataset_id}`: Get dataset statistics

### Sentiment Analysis

- `GET /api/sentiments`: Get paginated sentiment analysis results
- `GET /api/sentiments/summary`: Get sentiment analysis summary
- `GET /api/all-sentiments`: Retrieve all processed sentiments

## Usage

### Running the Server

```bash
python backend_api.py
```

Or using Docker:

```bash
docker build -t transport-fare-backend .
docker run -p 8000:8000 transport-fare-backend
```

### Example API Requests

Analyze single text:
```bash
curl -X POST "http://localhost:8000/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "The new fare system is great!", "language": "English"}'
```

Generate synthetic dataset:
```bash
curl -X POST "http://localhost:8000/api/generate-dataset" \
     -H "Content-Type: application/json" \
     -d '{"entries": 1000, "months": 3}'
```

## Project Structure

```
├── backend_api.py           # Main FastAPI application
├── sentiment_analysis.py    # Sentiment analysis implementation
├── translation_pipeline.py  # Translation service
├── stream_processor.py      # Kafka stream processing
├── enhanced_data_generator.py # Synthetic data generation
├── requirements.txt         # Python dependencies
└── Dockerfile              # Docker configuration
```

## Development

The backend is built with FastAPI and follows REST principles. To extend the functionality:

1. Add new endpoints in `backend_api.py`
2. Implement new processing logic in relevant modules
3. Update the API documentation using FastAPI's built-in docs

## Docker Support

The included Dockerfile sets up the environment and runs the API server. Build and run using:

```bash
docker-compose up --build
```

## License

This project is licensed under the MIT License.
