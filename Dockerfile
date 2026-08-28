FROM python:3.11-slim

WORKDIR /app

# libgl1 + libglib2.0-0 are needed by opencv-python-headless at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate the synthetic fallback dataset and train a small demo model at
# build time so the container works out of the box even without the real
# Kaggle dataset mounted in. Drop your own data/casting/ folder in before
# building to train on the real dataset instead.
RUN python -m src.data_prep && python -m src.train --backbone resnet50 --epochs 3

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
