# Production Deployment Guide

## Pre-Deployment Checklist

### Configuration
- [ ] Copy `.env.example` to `.env` and customize
- [ ] Review `docker-compose.yml` settings
- [ ] Verify all environment variables are set
- [ ] Test in staging environment first

### Code Quality
- [ ] Run type checker: `mypy src/dashboard/`
- [ ] Run linter: `flake8 src/dashboard/`
- [ ] Run tests: `pytest tests/`
- [ ] Check security: `bandit src/dashboard/`

### Data & Volumes
- [ ] Verify `EvaluationData/` is present and valid
- [ ] Verify `Codes_Matlab/` is present
- [ ] Check `outputs/` has write permissions
- [ ] Backup existing `outputs/` directory

### Docker
- [ ] Build test image: `docker build -f Dockerfile.full -t ktc:test .`
- [ ] Test image: `docker run -p 8501:8501 ktc:test`
- [ ] Verify all volumes mount correctly
- [ ] Check healthcheck passes

---

## Deployment Steps

### 1. Prepare Environment

```bash
# Clone configuration
cp .env.example .env

# Edit .env with production values
vi .env

# Verify config
python3 -c "from src.dashboard.config import get_config; print(get_config().to_dict())"
```

### 2. Build Production Image

```bash
# Build final image
docker build -f Dockerfile.full -t ktc-dashboard:v1.0.0 .

# Tag for registry (if using)
docker tag ktc-dashboard:v1.0.0 registry.example.com/ktc-dashboard:v1.0.0
docker push registry.example.com/ktc-dashboard:v1.0.0
```

### 3. Deploy Container

```bash
# Start with compose
docker compose up -d --build

# Verify startup
docker compose logs -f dashboard

# Check health
docker compose ps
docker inspect <container-id> | grep -A 10 '"Health"'
```

### 4. Verify Deployment

```bash
# Test dashboard access
curl http://localhost:8501/_stcore/health

# Check system health via Python
python3 << 'EOF'
from src.dashboard.system_init import check_system_health
health = check_system_health()
print(health)
EOF

# View logs
docker compose logs dashboard --tail 100
```

---

## Production Configuration

### Environment Variables

```bash
# Safety settings
ENVIRONMENT=production
LOG_LEVEL=INFO

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true

# Run locking (2 hours)
RUN_LOCK_TIMEOUT=7200

# Cache
CACHE_VALIDATION=true

# Disk management (85% threshold)
MAX_DISK_USAGE_PCT=85
```

### Docker Compose Overrides

For production, create `docker-compose.prod.yml`:

```yaml
version: "3.9"

services:
  dashboard:
    restart: on-failure:3
    healthcheck:
      retries: 10
      interval: 60s
    environment:
      - LOG_LEVEL=WARNING  # Less verbose in prod
      - ENVIRONMENT=production
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G
```

Deploy with:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Monitoring & Maintenance

### Daily Checks

```bash
# Container health
docker compose ps

# Disk usage
docker exec ktc-dashboard python3 -c \
  "from src.dashboard.disk_manager import get_disk_report; \
   import json; print(json.dumps(get_disk_report(), indent=2))"

# Cache status
docker exec ktc-dashboard python3 -c \
  "from src.dashboard.cache_manager import get_cache_stats; \
   import json; print(json.dumps(get_cache_stats(), indent=2))"

# Run lock status
docker exec ktc-dashboard python3 -c \
  "from src.dashboard.run_lock import get_lock_status; \
   import json; print(json.dumps(get_lock_status(), indent=2))"
```

### Weekly Maintenance

```bash
# Clean cache (entries older than 30 days)
docker exec ktc-dashboard python3 << 'EOF'
from src.dashboard.cache_manager import clear_cache
deleted = clear_cache(older_than_days=30)
print(f"Cleaned {deleted} cache entries")
EOF

# Clean old runs (keep last 10, last 30 days)
docker exec ktc-dashboard python3 << 'EOF'
from src.dashboard.disk_manager import cleanup_old_runs
result = cleanup_old_runs(keep_days=30, keep_count=10)
print(f"Freed {result['freed_gb']} GB")
EOF

# Validate manifest
docker exec ktc-dashboard python3 << 'EOF'
from src.dashboard.run_manifest import validate_manifest
result = validate_manifest()
print(f"Fixed {result['fixed']} manifest issues")
EOF
```

### Log Monitoring

```bash
# Real-time logs
docker compose logs -f dashboard

# JSON logs (production)
docker compose logs dashboard --format json | jq '.log | fromjson'

# Search logs
docker compose logs dashboard | grep ERROR

# Archive logs
docker compose logs dashboard > logs/dashboard-$(date +%Y%m%d).log
```

---

## Rollback Procedure

If deployment fails:

```bash
# Stop current deployment
docker compose down

# Restore previous version
docker compose up -d  # Falls back to last working image

# Or explicitly use previous version
docker tag ktc-dashboard:v1.0.0-prev ktc-dashboard:latest
docker compose up -d --build
```

---

## Performance Optimization

### Resource Limits

Set in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: "2"
      memory: 4G
    reservations:
      cpus: "1"
      memory: 2G
```

### Caching Strategy

- Enable cache validation: `CACHE_VALIDATION=true`
- Run cleanup weekly to prevent growth
- Monitor cache size: keep under 5 GB

### Disk Management

- Set threshold: `MAX_DISK_USAGE_PCT=85`
- Auto-cleanup runs older than 30 days
- Keep last 10 runs minimum
- Archive old results to external storage

---

## Security Hardening

### File Permissions

```bash
# Ensure proper ownership
chown -R 1000:1000 EvaluationData/
chown -R 1000:1000 Codes_Matlab/
chown -R 1000:1000 outputs/

# Limit access
chmod 755 EvaluationData/
chmod 750 outputs/
```

### Container Security

- [x] Non-root user: `user: "1000:1000"`
- [x] Read-only filesystem: (consider adding)
- [x] No privilege escalation: (default)
- [x] Resource limits: (set in compose)

### Network Security

```yaml
# In docker-compose.yml
services:
  dashboard:
    networks:
      - internal
    # Don't expose unless necessary
    ports:
      - "127.0.0.1:8501:8501"  # Local only
```

---

## Disaster Recovery

### Backup Strategy

```bash
# Daily backup of outputs
0 2 * * * tar -czf /backups/ktc-outputs-$(date +\%Y\%m\%d).tar.gz outputs/

# Weekly backup of configs
0 3 * * 0 tar -czf /backups/ktc-config-$(date +\%Y\%m\%d).tar.gz .env .env.example docker-compose.yml
```

### Restore Procedure

```bash
# From backup
tar -xzf /backups/ktc-outputs-20260715.tar.gz

# Verify integrity
python3 << 'EOF'
from src.dashboard.run_manifest import validate_manifest
result = validate_manifest()
print(result)
EOF
```

---

## Support & Troubleshooting

### Common Issues

**Container won't start:**
```bash
docker compose logs dashboard | tail -50
# Check for config errors, missing volumes, port conflicts
```

**High disk usage:**
```bash
python3 -c "from src.dashboard.disk_manager import cleanup_old_runs; \
           cleanup_old_runs(keep_days=7, keep_count=5)"
```

**Cache not working:**
```bash
python3 -c "from src.dashboard.cache_manager import get_cache; \
           cache = get_cache(); cache.cleanup(older_than_days=0)"
```

**Run lock stuck:**
```bash
# Force release (admin only)
rm outputs/.locks/run.lock
```

---

## Version Tracking

Keep track of deployments:

```bash
# Tag each production release
git tag -a v1.0.0 -m "Production release"
git push origin v1.0.0

# Build with version
docker build --label version=1.0.0 --label date=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -f Dockerfile.full -t ktc-dashboard:v1.0.0 .
```

---

## Contact & Escalation

For production issues:
1. Check logs: `docker compose logs dashboard`
2. Run health check: `docker compose ps`
3. Contact: [Team contact info]
4. Escalation: [On-call contact]
