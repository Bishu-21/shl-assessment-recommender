FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app/ ./app/
COPY data/shl_catalog.json ./data/shl_catalog.json

# Set default env vars
ENV PORT=8000
ENV CATALOG_PATH=data/shl_catalog.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
