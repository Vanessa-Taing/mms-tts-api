# Dockerfile
FROM python:3.10.11-slim-buster

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    perl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install requirements (required for VITS build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Clone and setup VITS
RUN git clone https://github.com/jaywalnut310/vits.git && \
    cd vits/monotonic_align && \
    mkdir -p monotonic_align && \
    python setup.py build_ext --inplace

# 5. Copy application files (separate from requirements for layer caching)
COPY app ./app

# 6. Set environment variables
ENV PYTHONPATH=/app/vits

# 7. Create models directory and pre-download models
RUN mkdir -p /app/models && \
    for lang in eng khm mya; do \
        wget -q https://dl.fbaipublicfiles.com/mms/tts/${lang}.tar.gz -O /app/models/${lang}.tar.gz && \
        tar zxf /app/models/${lang}.tar.gz -C /app/models && \
        rm /app/models/${lang}.tar.gz; \
    done

# 8. Clean up to reduce image size (optional)
RUN apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 9. Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]