FROM python:3.11-slim

# Install system dependencies (build tools for compiling certain python packages like C extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure critical directories exist
RUN mkdir -p /app/cache /app/models /app/logs /app/plots

# Expose backend port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=5000

# Run with Gunicorn — strictly 1 worker to preserve in-memory shared Flask locks/SSE client state arrays
# 8 threads allows parallel SSE clients + API hits without wiping State.
CMD ["gunicorn", "app:app", "--workers=1", "--threads=8", "--worker-class=gthread", "--bind=0.0.0.0:5000", "--timeout=120"]
