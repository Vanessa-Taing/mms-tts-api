# Use updated base image
FROM python:3.10.13-slim-bookworm

WORKDIR /app

# Set UTF-8 locale by default
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/vits \
    UROMAN_DIR=/app/uroman

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    perl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install core Python dependencies first
RUN pip install --no-cache-dir numpy Cython==0.29.21

# 3. Clone and setup VITS
RUN git clone https://github.com/jaywalnut310/vits.git && \
    cd vits/monotonic_align && \
    mkdir -p monotonic_align && \
    python setup.py build_ext --inplace

# 4. Clone uroman
RUN git clone https://github.com/isi-nlp/uroman.git ${UROMAN_DIR}

# 5. Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application
COPY app ./app

# 7. Create models directory
RUN mkdir -p /app/models

# 8. Pre-download models (optional)
RUN for lang in eng khm mya; do \
    wget -q https://dl.fbaipublicfiles.com/mms/tts/${lang}.tar.gz -O /app/models/${lang}.tar.gz && \
    tar --no-same-owner -zxf /app/models/${lang}.tar.gz -C /app/models && \
    rm /app/models/${lang}.tar.gz; \
    done

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]