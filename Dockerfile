FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY translation_pipeline.py .
COPY sentiment_analysis.py .
COPY stream_processor.py .
COPY enhanced_data_generator.py .
COPY backend_api.py .

# Create directories for data
RUN mkdir -p /app/data /app/data/translated /app/data/processed /app/data/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Copy .env file
COPY .env .

# Load environment variables from .env
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dotenv \
    && rm -rf /var/lib/apt/lists/*

# Ensure sensitive data is securely loaded at runtime
RUN export GROQ_API_KEY=$(grep GROQ_API_KEY .env | cut -d '=' -f2) && echo "GROQ_API_KEY=$GROQ_API_KEY" >> /app/.env

# Expose the port for the API
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000"]