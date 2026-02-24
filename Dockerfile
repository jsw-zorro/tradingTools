FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

RUN pip install --no-cache-dir .

# Create output directory
RUN mkdir -p output/reports output/data_cache

ENTRYPOINT ["strategylab"]
CMD ["monitor", "--anytime"]
