# Docker Commands for KTC Dashboard

## 🚀 Quick Start (First Time Setup)

```powershell
# 1. Clone configuration
Copy-Item .env.example .env

# 2. Build and start
docker compose up -d --build

# 3. Wait for startup
Start-Sleep -Seconds 5

# 4. Check status
docker compose ps

# 5. Test health
curl http://localhost:8501/_stcore/health
```

---

## 📅 Daily Usage

### **Start the Dashboard**
```powershell
docker compose up -d
```

### **Stop the Dashboard**
```powershell
docker compose down
```

### **Restart the Dashboard**
```powershell
docker compose restart
```

### **View Logs (Real-time)**
```powershell
docker compose logs -f dashboard
```

### **View Logs (Last 20 lines)**
```powershell
docker compose logs dashboard --tail 20
```

### **Check Status**
```powershell
docker compose ps
```

---

## 🏗️ Building & Deployment

### **Build New Image**
```powershell
docker compose build --no-cache
```

### **Full Rebuild & Start**
```powershell
docker compose up -d --build
```

### **Pull Latest & Restart**
```powershell
git pull
docker compose down
docker compose up -d --build
```

---

## 🔍 Monitoring & Diagnostics

### **Check Container Health**
```powershell
docker compose ps
```

### **View Resource Usage**
```powershell
docker stats
```

### **Test Health Endpoint**
```powershell
curl http://localhost:8501/_stcore/health
```

### **Check System Status**
```powershell
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.system_init import check_system_health; `
   import json; print(json.dumps(check_system_health(), indent=2))"
```

### **Check Disk Usage**
```powershell
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.disk_manager import get_disk_report; `
   import json; print(json.dumps(get_disk_report(), indent=2))"
```

### **Check Run Lock Status**
```powershell
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.run_lock import get_lock_status; `
   import json; print(json.dumps(get_lock_status(), indent=2))"
```

### **Check Cache Status**
```powershell
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.cache_manager import get_cache_stats; `
   import json; print(json.dumps(get_cache_stats(), indent=2))"
```

---

## 🧹 Maintenance

### **Clean Old Cache (Older than 30 days)**
```powershell
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.cache_manager import clear_cache; `
   deleted = clear_cache(older_than_days=30); `
   print(f'Cleaned {deleted} cache entries')"
```

### **Clean Old Runs (Keep Last 10, Last 30 days)**
```powershell
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.disk_manager import cleanup_old_runs; `
   result = cleanup_old_runs(keep_days=30, keep_count=10); `
   print(f'Freed {result[\"freed_gb\"]} GB')"
```

### **Validate Run Manifest**
```powershell
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.run_manifest import validate_manifest; `
   result = validate_manifest(); `
   print(f'Fixed {result[\"fixed\"]} issues')"
```

### **Force Release Stuck Lock**
```powershell
Remove-Item outputs\.locks\run.lock -ErrorAction SilentlyContinue
docker compose restart
```

---

## 📊 Common Workflows

### **Full Diagnostic Check**
```powershell
Write-Host "=== Docker Status ===" -ForegroundColor Green
docker compose ps

Write-Host "`n=== Disk Usage ===" -ForegroundColor Green
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.disk_manager import get_disk_usage; `
   stats = get_disk_usage(); `
   print(f'Disk: {stats[\"percent\"]}% ({stats[\"used_gb\"]}/{stats[\"total_gb\"]} GB)')"

Write-Host "`n=== Run Lock ===" -ForegroundColor Green
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.run_lock import get_lock_status; `
   status = get_lock_status(); `
   print(status)"

Write-Host "`n=== Cache ===" -ForegroundColor Green
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.cache_manager import get_cache_stats; `
   stats = get_cache_stats(); `
   print(f'Cache: {stats[\"entries\"]} entries, {stats[\"total_size_mb\"]} MB')"

Write-Host "`n=== Health ===" -ForegroundColor Green
$response = curl -s http://localhost:8501/_stcore/health
if ($response -eq "ok") {
    Write-Host "✓ Dashboard healthy" -ForegroundColor Green
} else {
    Write-Host "✗ Dashboard unhealthy" -ForegroundColor Red
}
```

### **Pre-Benchmark Checklist**
```powershell
# 1. Check if dashboard is running
docker compose ps

# 2. Check disk space
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.disk_manager import check_disk_threshold; `
   is_critical, stats = check_disk_threshold(85); `
   if is_critical: print(f'WARNING: {stats[\"percent\"]}% disk used')"

# 3. Check if run is already in progress
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.run_lock import get_lock_status; `
   status = get_lock_status(); `
   if status['locked']: print(f'Run in progress: {status[\"requester\"]}')"

# 4. Open dashboard
Start-Process http://localhost:8501
```

---

## 🆘 Troubleshooting

### **Container Won't Start**
```powershell
# Check logs
docker compose logs dashboard

# Rebuild
docker compose down
docker compose up -d --build

# Check if port 8501 is in use
netstat -ano | findstr :8501
```

### **Disk Full (>85%)**
```powershell
# Quick cleanup
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.disk_manager import cleanup_old_runs; `
   result = cleanup_old_runs(keep_days=7, keep_count=5); `
   print(f'Freed {result[\"freed_gb\"]} GB')"

# Restart
docker compose restart
```

### **Run Lock Stuck**
```powershell
# Check status
docker exec ktc_work_his-dashboard-1 python -c `
  "from src.dashboard.run_lock import get_lock_status; print(get_lock_status())"

# Force release (if sure no one is running)
Remove-Item outputs\.locks\run.lock
docker compose restart
```

### **High Memory Usage**
```powershell
# Check memory
docker stats

# Restart container
docker compose restart

# Check for memory leaks in logs
docker compose logs dashboard | findstr "memory\|error"
```

### **Clear Everything & Start Fresh**
```powershell
# Stop and remove
docker compose down -v

# Remove images
docker rmi ktc-dashboard:full

# Rebuild
docker compose up -d --build
```

---

## 📋 Container Name Reference

Your container name is: `ktc_work_his-dashboard-1`

Use this for exec commands:
```powershell
docker exec ktc_work_his-dashboard-1 [command]
```

---

## 🔐 Safety Features Enabled

✅ **Run Locking** - Prevents concurrent benchmarks  
✅ **Auto-Restart** - on-failure:5 (max 5 retries)  
✅ **Health Checks** - Every 30 seconds  
✅ **Disk Cleanup** - Auto-cleanup old runs  
✅ **Cache Validation** - Prevents stale results  
✅ **Non-Root User** - Security hardening  

---

## 📞 Quick Help

| Task | Command |
|------|---------|
| **Start** | `docker compose up -d` |
| **Stop** | `docker compose down` |
| **Status** | `docker compose ps` |
| **Logs** | `docker compose logs -f` |
| **Restart** | `docker compose restart` |
| **Clean** | `docker compose down -v` |
| **Rebuild** | `docker compose up -d --build` |

---

## Environment Variables

Located in `.env` file:

```env
ENVIRONMENT=production
LOG_LEVEL=INFO
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
RUN_LOCK_TIMEOUT=7200
CACHE_VALIDATION=true
MAX_DISK_USAGE_PCT=85
```

Edit `.env` to customize (then restart: `docker compose restart`)

---

**Questions?** Check logs: `docker compose logs dashboard`  
**Issues?** Run diagnostic: Use "Full Diagnostic Check" above
