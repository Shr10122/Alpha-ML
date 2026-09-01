FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies via apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy dependencies file first (optimizes Docker layer caching)
COPY requirements.txt .

# 3. Install Python dependencies with increased timeout
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 4. Copy application source code
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]