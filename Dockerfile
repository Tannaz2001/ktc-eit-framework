FROM python:3.12-slim

WORKDIR /app

# Dependency layer first so code edits don't invalidate the pip cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code + install ktc_framework from src/
COPY . .
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
