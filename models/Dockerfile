FROM python:3.11-slim

WORKDIR /app

# System deps for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake && \
    rm -rf /var/lib/apt/lists/*

# Install Python ML deps
RUN pip install --no-cache-dir \
    "autogluon.tabular[all]>=1.0" \
    "yfinance>=0.2.31" \
    "pandas>=2.0" \
    "numpy>=1.24" \
    "scikit-learn>=1.3" \
    "matplotlib>=3.7"

# Code will be mounted at /app
CMD ["python", "train.py"]
