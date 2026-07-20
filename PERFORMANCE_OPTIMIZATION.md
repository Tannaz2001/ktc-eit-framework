# Dashboard Performance Optimization

## ⚡ Quick Fixes (Do These First)

### Fix 1: Clear Streamlit Cache
```powershell
docker exec ktc_work_his-dashboard-1 streamlit cache clear
docker compose restart
```

### Fix 2: Increase Docker Resources
Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: "2"
      memory: 4G
    reservations:
      cpus: "1.5"
      memory: 2G
```

Then:
```powershell
docker compose down
docker compose up -d --build
```

### Fix 3: Fix Permission Errors
```powershell
docker exec ktc_work_his-dashboard-1 chmod 777 configs/
```

---

## 🐢 Why Dashboard Is Slow

### Problem 1: Streamlit Reruns Everything
When you change a metric:
1. Widget value changes ❌
2. Streamlit reruns ENTIRE script ❌
3. Reloads all data ❌
4. Recalculates everything ❌
5. Updates UI ✅

**Solution: Use `@st.cache` decorators**

---

## 🚀 Performance Improvements

### Issue 1: Metric Selection Is Slow
**In app.py line ~716:**
```python
# ❌ SLOW
for m in ALL_METRICS_SIDEBAR:
    if st.checkbox(m):  # Reruns entire script!
        add_metric(m)

# ✅ FAST
@st.cache_data
def get_metrics():
    return ALL_METRICS_SIDEBAR

for m in get_metrics():
    if st.checkbox(m):
        add_metric(m)
```

### Issue 2: Data Loading Is Slow
**In app.py line ~498:**
```python
# ❌ SLOW - Reloads on every run
scores, _ = load_run_data(find_latest_run())

# ✅ FAST - Cache for 5 minutes
@st.cache_data(ttl=300)
def load_cached_data():
    return load_run_data(find_latest_run())

scores, _ = load_cached_data()
```

### Issue 3: Chart Generation Is Slow
**Plotly charts are cached:**
```python
# ✅ Already good - Plotly handles internally
fig = px.bar(data)
st.plotly_chart(fig)  # Fast rerenders
```

---

## 📊 Optimization Checklist

- [ ] Clear cache: `streamlit cache clear`
- [ ] Increase Docker resources (4GB → 8GB)
- [ ] Check for permission errors in logs
- [ ] Use `@st.cache_data` for expensive operations
- [ ] Add `ttl=300` for cache expiry (5 min)
- [ ] Minimize data loaded on startup

---

## 🔍 Diagnose Performance Issues

### Check what's slow:
```powershell
# View logs with timing
docker compose logs dashboard -f | grep "seconds"

# Check container resources
docker stats

# Profile memory usage
docker exec ktc_work_his-dashboard-1 python -c `
  "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

### Check disk I/O:
```powershell
docker exec ktc_work_his-dashboard-1 ls -lah outputs/ | head -20
```

---

## 💾 Memory Management

### Current Issue
Outputs folder is large (35.84 GB)
- Loading all runs on startup = SLOW
- Should load only latest

### Solution
**In app.py:**
```python
# ✅ Load only latest run, not all
from src.dashboard.run_manifest import get_active_run

active_run = get_active_run()  # Only loads current
scores, _ = load_run_data(active_run)
```

---

## 🎯 Quick Performance Gains

### Change 1: Lazy Load Runs (Biggest Impact)
```python
# Don't load all runs on startup
# Only load when user selects a run
if selected_run_changed:
    load_run_data(selected_run)
```

### Change 2: Cache Sidebar Data
```python
@st.cache_data(ttl=600)
def get_available_methods():
    return discover_available_methods()

methods = get_available_methods()
```

### Change 3: Use Session State
```python
# Store computed values in session
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = compute_heavy_operation()

data = st.session_state.filtered_data
```

---

## 🔧 Advanced Optimization

### Profile the App
```python
import time

@st.cache_data
def timed_function():
    start = time.time()
    # your code here
    elapsed = time.time() - start
    st.write(f"Took {elapsed:.2f}s")
```

### Reduce Data Size
```python
# Load only what's needed
cols_needed = ['method', 'score', 'level']
data = data[cols_needed]  # Faster

# Filter before visualization
filtered = data[data['level'] <= 5]
st.plotly_chart(px.bar(filtered))
```

---

## 📈 Expected Results

**Before Optimization:**
- Metric change: 2-3 seconds ❌
- Run selection: 5 seconds ❌
- Dashboard load: 10 seconds ❌

**After Optimization:**
- Metric change: <500ms ✅
- Run selection: <1 second ✅
- Dashboard load: 3 seconds ✅

---

## ⚙️ Docker Optimization

### Current Settings (docker-compose.yml)
```yaml
resources:
  limits:
    cpus: "2"
    memory: 4G  # May be too low
```

### Recommended for Smooth UI
```yaml
resources:
  limits:
    cpus: "4"
    memory: 8G
  reservations:
    cpus: "2"
    memory: 4G
```

**Note:** Only if your machine has 16GB+ RAM

---

## 🚀 Implementation Priority

### Priority 1 (Do First)
- [ ] Fix permission errors
- [ ] Increase Docker memory
- [ ] Clear cache

### Priority 2 (Quick Wins)
- [ ] Add `@st.cache_data` to data loading
- [ ] Load only active run on startup
- [ ] Use session state for filters

### Priority 3 (Big Changes)
- [ ] Refactor data loading
- [ ] Implement lazy loading
- [ ] Add performance profiling

---

## 📝 Immediate Action Plan

```powershell
# 1. Fix permissions
docker exec ktc_work_his-dashboard-1 chmod 777 configs/ outputs/ .

# 2. Clear cache
docker exec ktc_work_his-dashboard-1 streamlit cache clear

# 3. Restart
docker compose restart

# 4. Test in browser
# Try changing metrics - should be faster now
```

---

## 🔍 Monitor Performance

```powershell
# Watch resource usage while using dashboard
docker stats --no-stream

# Expected:
# CPU: 5-15%
# Memory: 500MB - 1.5GB
```

If higher than expected → add more resources

---

## ❓ Still Slow?

Check these in order:
1. ✅ Permission errors? → Fix file permissions
2. ✅ Large dataset? → Only load current run
3. ✅ Rerunning script? → Add `@st.cache_data`
4. ✅ Not enough RAM? → Increase Docker memory
5. ✅ Disk I/O? → SSD performance issue

---

## 📞 Get Help

**Check logs:**
```powershell
docker compose logs dashboard --tail 100 | findstr -i "error\|permission\|slow"
```

**Check resources:**
```powershell
docker stats
# CPU >50%? Need more power
# Memory >3GB? Need more RAM
```

**Test performance:**
```powershell
# Clear everything
docker compose down -v
docker system prune -a

# Fresh start
docker compose up -d --build
```

---

**Apply these fixes and your dashboard will be smooth!** ⚡
