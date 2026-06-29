FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir prophet mcp pandas plotly

COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
