# Docker Deployment Guide

This guide covers building and running the KTC EIT dashboard in Docker.

## Quick Start

### Build and run locally (development)

```bash
# Build the image
docker-compose build

# Run the dashboard
docker-compose up

# Visit http://localhost:8501
```

### Stop the container

```bash
docker-compose down
```

---

## Requirements

- **Docker** (19.03+) and **Docker Compose** (1.25+)
- **5 GB disk space** for results and datasets (mounted as volumes)
- **Port 8501** available on host

---

## Directory Structure

Before running, ensure you have:

```
ktc-eit-framework/
├── EvaluationData/           ← KTC 2023 evaluation dataset (required for benchmark)
├── Codes_Matlab/             ← FEM mesh and training data
├── outputs/                  ← Benchmark results (created by run)
├── docker-compose.yml
└── Dockerfile
```

**Important:** The dataset is mounted as a volume. Without `EvaluationData/`, the dashboard runs but shows no data. See [RUN_GUIDE.md](RUN_GUIDE.md#-data-setup) to download the dataset.

---

## Configuration

### Environment Variables

Set in `docker-compose.yml` or via `-e` flag:

```bash
docker run -e STREAMLIT_THEME_BASE=dark -p 8501:8501 ktc-dashboard
```

**Common Streamlit settings:**
- `STREAMLIT_THEME_BASE`: `light` or `dark`
- `STREAMLIT_LOGGER_LEVEL`: `info`, `debug`, `warning`, `error`

See [Streamlit configuration](https://docs.streamlit.io/library/advanced-features/configuration) for all options.

---

## Production Deployment

### Push to Docker Registry

```bash
# Tag the image
docker build -t sahil-exe/ktc-dashboard:v1.0.0 .

# Log in to Docker Hub (or your registry)
docker login

# Push the image
docker push sahil-exe/ktc-dashboard:v1.0.0
```

### Deploy on a Server

**Using Docker Compose:**

```bash
# On the server, create a directory
mkdir -p /opt/ktc-dashboard
cd /opt/ktc-dashboard

# Copy docker-compose.yml
curl -o docker-compose.yml https://raw.githubusercontent.com/Sahil-exe/ktc-eit-framework/main/docker-compose.yml

# Create data directories
mkdir -p EvaluationData outputs Codes_Matlab

# Download datasets (see RUN_GUIDE.md)
# ... place EvaluationData and Codes_Matlab files here ...

# Run the container
docker-compose up -d
```

**Using Kubernetes (advanced):**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ktc-dashboard
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ktc-dashboard
  template:
    metadata:
      labels:
        app: ktc-dashboard
    spec:
      containers:
      - name: dashboard
        image: sahil-exe/ktc-dashboard:v1.0.0
        ports:
        - containerPort: 8501
        volumeMounts:
        - name: outputs
          mountPath: /app/outputs
        - name: data
          mountPath: /app/EvaluationData
      volumes:
      - name: outputs
        emptyDir: {}
      - name: data
        persistentVolumeClaim:
          claimName: ktc-data-pvc
```

---

## Health Checks

The container includes a built-in health check:

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Manual health check
curl -f http://localhost:8501/_stcore/health || exit 1
```

---

## Troubleshooting

### Port 8501 already in use

```bash
# Map to a different port
docker-compose -f docker-compose.yml up -p 9000:8501
```

### Dataset not found

```bash
# Verify volume is mounted
docker exec ktc-eit-framework-dashboard-1 ls -la /app/EvaluationData

# If missing, copy data into running container
docker cp ./EvaluationData ktc-eit-framework-dashboard-1:/app/
```

### Out of memory

```bash
# Increase memory limit in docker-compose.yml
services:
  dashboard:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Slow startup

The first run downloads Streamlit plugins and builds caches. This takes 1-2 minutes. Subsequent restarts are faster.

---

## Image Size & Optimization

Current image: ~1.2 GB

**To reduce:**
- Use `python:3.12-alpine` base (saves ~200 MB)
- Remove test dependencies from requirements.txt
- Use multi-stage builds

**Current approach balances:**
- ✅ Fast pip installs (slim base + caching)
- ✅ All methods available (no dependency pruning)
- ✅ Simple debugging (slim = more tools)

---

## Continuous Integration

Every commit to `main` triggers CI that:
1. Builds the Docker image
2. Runs smoke tests
3. Pushes to registry (optional)

See `.github/workflows/ci.yml` for details.
