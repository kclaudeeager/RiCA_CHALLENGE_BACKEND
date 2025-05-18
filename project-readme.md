# Rwanda Transport Fare Sentiment Analysis

This project analyzes public sentiment regarding Rwanda's shift from flat-rate to distance-based transport fares. It processes multilingual feedback from various sources, performs sentiment analysis, and presents insights via an interactive dashboard.

## Project Structure

```
├── data/                  # Data storage
│   ├── raw/               # Raw generated data
│   ├── translated/        # Translated data
│   └── processed/         # Processed data with sentiment analysis
├── models/                # Trained sentiment analysis models
├── dashboard/             # React dashboard application
├── scripts/
│   ├── enhanced_data_generator.py     # Generate synthetic data
│   ├── translation_pipeline.py        # Translate multilingual content
│   ├── sentiment_analysis.py          # Perform sentiment analysis
│   └── stream_processor.py            # Process data streams with Kafka
├── docker-compose.yml     # Docker setup for Kafka and dashboard
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Features

- **Multilingual Support**: Handles content in English, Kinyarwanda, and French
- **Diverse Data Sources**: Processes feedback from social media, forums, SMS, and news comments
- **Real-time Processing**: Uses Kafka for stream processing
- **Sentiment Analysis**: Uses both traditional ML and LLM approaches
- **Interactive Dashboard**: Visualizes sentiment trends, key concerns, and demographic insights
- **Recommendation System**: Generates policy recommendations based on sentiment analysis

## Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- Docker and Docker Compose (optional, for Kafka setup)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/rwanda-transport-sentiment.git
cd rwanda-transport-sentiment
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file with the following:

```
GROQ_API_KEY=your_groq_api_key
KAFKA_BROKER=localhost:9092
```

4. **Start Kafka (if using stream processing)**

```bash
docker-compose up -d
```

5. **Install and build the dashboard**

```bash
cd dashboard
npm install
npm run build
```

## Usage

### 1. Generate Synthetic Data

```bash
python scripts/enhanced_data_generator.py --output data/raw --entries 1000 --no-api
```

### 2. Translate Non-English Content

```bash
python scripts/translation_pipeline.py --input data/raw/transport_fare_feedback_dataset.json --output data/translated/transport_fare_feedback_dataset.json
```

### 3. Perform Sentiment Analysis

Using traditional ML:

```bash
python scripts/sentiment_analysis.py --input data/translated/transport_fare_feedback_dataset.json --output data/processed/sentiment_analysis_results.json --model models/sentiment_model.joblib --train
```

Or using LLM:

```bash
python scripts/sentiment_analysis.py --input data/translated/transport_fare_feedback_dataset.json --output data/processed/sentiment_analysis_results.json --use-llm
```

### 4. Stream Processing with Kafka

Simulate a data stream:

```bash
python scripts/stream_processor.py --mode simulate --input data/raw/transport_fare_feedback_dataset.json --delay 0.5
```

Process the stream:

```bash
python scripts/stream_processor.py --mode stream --use-llm
```

Or process a file directly:

```bash
python scripts/stream_processor.py --mode file --input data/translated/transport_fare_feedback_dataset.json --output data/processed/stream_processed_results.json --use-llm
```

### 5. Run the Dashboard

```bash
cd dashboard
npm start
```

The dashboard will be available at http://localhost:3000

## Data Flow

1. **Data Generation/Collection**:
   - Generate synthetic data or collect real data
   - Store raw data in `data/raw/`

2. **Data Processing**:
   - Translate non-English content
   - Clean and normalize text
   - Perform sentiment analysis
   - Store processed data in `data/processed/`

3. **Stream Processing** (optional):
   - Stream data through Kafka
   - Process in real-time
   - Update dashboard with latest insights

4. **Visualization**:
   - Display sentiment trends over time
   - Show top concerns by demographic
   - Generate actionable insights and recommendations

## Dashboard Features

- **Overview**: Overall sentiment trends, top concerns, data sources, and demographics
- **Trends**: Detailed sentiment analysis by source, region, and demographic
- **Alerts**: Flagged issues and misinformation tracking
- **Recommendations**: Policy recommendations and community engagement strategies

## Extending the Project

- **Add Real Data Sources**: Integrate with Twitter API, Facebook Graph API, etc.
- **Enhance Analysis**: Add topic modeling, entity recognition, and aspect-based sentiment analysis
- **Improve Visualization**: Add more interactive elements and drill-down capabilities
- **Deploy**: Set up CI/CD pipeline for continuous deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Rwanda Transport Authority (hypothetical) for the challenge statement
- Groq for the LLM API access
