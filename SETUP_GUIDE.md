# Setup Guide: Docker vs Local Environment

## 📋 Quick Comparison

| Feature | Docker | Local (venv) |
|---------|--------|--------------|
| **Setup Time** | 5 minutes | 10-15 minutes |
| **Dependencies** | Docker only | Python 3.12 + packages |
| **Performance** | Good | Faster |
| **Isolation** | Excellent | None |
| **Team Sync** | Guaranteed | Version dependent |
| **Storage** | Inside container | On your machine |
| **Best For** | Production, CI/CD | Development |

---

# Option 1: 🐳 Docker (Recommended for Production)

## Prerequisites
- Docker Desktop installed
- 4 GB RAM available

## Quick Start

```powershell
# 1. Clone config
Copy-Item .env.example .env

# 2. Start dashboard
docker compose up -d

# 3. Wait for startup
Start-Sleep -Seconds 5

# 4. Open dashboard
Start-Process http://localhost:8501
```

## Daily Commands

```powershell
# Start
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f dashboard

# Restart
docker compose restart

# Status
docker compose ps
```

## Advantages
✅ Guaranteed consistency across team  
✅ Isolated environment (no conflicts)  
✅ Easy to update (just git pull + docker compose up -d --build)  
✅ Production-ready  
✅ No local dependencies  

## Disadvantages
❌ Requires Docker Desktop  
❌ Slightly slower startup  
❌ Less direct debugging  

---

# Option 2: 🐍 Local Python Environment (Development)

## Prerequisites

- Python 3.12 installed
- Git installed
- 3-5 GB disk space

## Step 1: Create Virtual Environment

```powershell
# Navigate to project directory
cd C:\Users\sahil\OneDrive\Desktop\KTC_WORK_HIS

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get permission error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

**Check activation:**
```powershell
# You should see (venv) at start of prompt
# Example: (venv) PS C:\Users\sahil\OneDrive\Desktop\KTC_WORK_HIS>
```

## Step 2: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
pip install -r requirements-full.txt
```

**If requirements don't exist:**
```powershell
# Install Streamlit and core dependencies
pip install streamlit streamlit-option-menu plotly pandas numpy scipy scikit-learn pyyaml pillow
```

## Step 3: Install Package in Development Mode

```powershell
# Install framework from source
pip install -e .
```

## Step 4: Run Dashboard

```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://YOUR_LOCAL_IP:8501
```

## Daily Workflow

```powershell
# 1. Activate venv
.\venv\Scripts\Activate.ps1

# 2. Pull latest changes
git pull

# 3. Install any new dependencies
pip install -r requirements.txt

# 4. Run dashboard
streamlit run app.py
```

## Advantages
✅ Direct Python environment  
✅ Faster iteration for development  
✅ Easy debugging  
✅ Full IDE integration  
✅ No Docker required  

## Disadvantages
❌ Must manage Python version  
❌ Dependency conflicts possible  
❌ Not guaranteed consistent across team  
❌ Requires manual updates  

---

# Option 3: 🔀 Hybrid (Docker for Production, Local for Dev)

**Recommended for teams:**

```powershell
# Development - Use local venv
.\venv\Scripts\Activate.ps1
streamlit run app.py

# Testing - Use Docker
docker compose up -d --build
curl http://localhost:8501/_stcore/health

# Production - Use Docker with CI/CD
git push
# (CI/CD automatically builds and deploys)
```

---

# Troubleshooting

## Docker Issues

### Port Already in Use
```powershell
# Find what's using port 8501
netstat -ano | findstr :8501

# Kill the process (replace PID with number from above)
taskkill /PID <PID> /F

# Or use different port in docker-compose.yml
# Change: ports: - "8502:8501"
```

### Out of Disk Space
```powershell
# Clean up Docker
docker system prune -a

# Or just remove old images
docker image prune
```

## Local (venv) Issues

### Python Version Wrong
```powershell
# Check Python version
python --version
# Should be 3.10 or higher

# If wrong, install Python 3.12:
# Download from python.org or use:
choco install python --version=3.12.0
```

### Permission Error on Activation
```powershell
# Fix execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Try again
.\venv\Scripts\Activate.ps1
```

### Module Not Found
```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or reinstall everything
pip install -e . --force-reinstall
```

### Streamlit Won't Start
```powershell
# Clear Streamlit cache
streamlit cache clear

# Run with verbose output
streamlit run app.py --logger.level=debug
```

---

# Environment Variables

## For Docker
Create `.env` from `.env.example`:
```env
ENVIRONMENT=production
LOG_LEVEL=INFO
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
RUN_LOCK_TIMEOUT=7200
CACHE_VALIDATION=true
MAX_DISK_USAGE_PCT=85
```

## For Local (venv)
Optional - set in PowerShell:
```powershell
$env:ENVIRONMENT = "development"
$env:LOG_LEVEL = "DEBUG"
$env:STREAMLIT_SERVER_PORT = "8501"
```

Or add to `~/.streamlit/config.toml`:
```toml
[client]
showErrorDetails = true

[logger]
level = "debug"
```

---

# Comparison Table

| Task | Docker | Local venv |
|------|--------|-----------|
| **Setup** | `docker compose up -d` | `.\venv\Scripts\Activate.ps1 && pip install -r requirements.txt` |
| **Start** | `docker compose up -d` | `.\venv\Scripts\Activate.ps1 && streamlit run app.py` |
| **Stop** | `docker compose down` | `Ctrl + C` |
| **Logs** | `docker compose logs -f` | Console output |
| **Update** | `git pull && docker compose up -d --build` | `git pull && pip install -r requirements.txt` |
| **Reset** | `docker compose down -v` | `rm -r venv && python -m venv venv && pip install -r requirements.txt` |

---

# Recommendations by Role

## 👨‍💼 **DevOps / Deployment**
→ Use **Docker** (Option 1)  
✅ Production-ready  
✅ Consistent  
✅ Easy CI/CD integration  

## 👨‍💻 **Developer**
→ Use **Local venv** (Option 2)  
✅ Fast iteration  
✅ Direct debugging  
✅ IDE integration  

## 🔄 **Data Scientist / Researcher**
→ Use **Hybrid** (Option 3)  
✅ Local development  
✅ Docker for validation  

## 📊 **Team Lead / Manager**
→ Recommend **Docker** for consistency  
✅ Everyone runs same version  
✅ No environment conflicts  

---

# Getting Help

## Check System Status

### Docker
```powershell
docker compose ps
docker compose logs dashboard --tail 20
curl http://localhost:8501/_stcore/health
```

### Local venv
```powershell
# Check Python
python --version

# Check Streamlit
streamlit --version

# Check installed packages
pip list | grep streamlit
```

## Common Issues

| Issue | Docker | Local |
|-------|--------|-------|
| Port in use | `docker compose down` | Change port in config |
| Memory high | `docker stats` | Check task manager |
| Missing package | Rebuild image | `pip install <package>` |
| Permission error | Docker user issue | Windows permission issue |

---

# Quick Setup Scripts

## Auto-Setup Docker
```powershell
# Copy and save as setup-docker.ps1
Copy-Item .env.example .env
docker compose build
docker compose up -d
Start-Sleep -Seconds 5
Start-Process http://localhost:8501
```

## Auto-Setup Local
```powershell
# Copy and save as setup-local.ps1
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-full.txt
pip install -e .
streamlit run app.py
```

---

# Decision Guide

**Choose Docker if:**
- ✅ Running on different machines
- ✅ Need guaranteed consistency
- ✅ Production deployment
- ✅ Team has mixed OS (Windows/Mac/Linux)

**Choose Local venv if:**
- ✅ Developing new features
- ✅ Need fast iteration
- ✅ Single developer machine
- ✅ Want direct IDE debugging

**Choose Hybrid if:**
- ✅ Development + testing + production
- ✅ Team with mixed roles
- ✅ Want best of both worlds

---

**Questions?**
- Docker: See `DOCKER_COMMANDS.md`
- Local: See Python documentation
- Both: Ask team lead
