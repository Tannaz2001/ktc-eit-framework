# Complete Setup From Scratch 🚀

**Start here if you have nothing installed!**

Everything you need to download and run the KTC Dashboard.

---

# Step 0: What Do You Need?

Choose ONE option:

**Option A: Docker** (Easiest, Recommended)
- Docker Desktop only
- Works on Windows/Mac/Linux
- 5 minutes to run

**Option B: Local Python** (For developers)
- Python 3.12
- Git
- Text editor/IDE
- 15 minutes to run

---

---

# ✅ OPTION A: DOCKER (Easiest)

## Step 1: Install Docker Desktop

**Windows/Mac:**
1. Go to: https://www.docker.com/products/docker-desktop
2. Click "Download"
3. Install it
4. Open Docker Desktop app
5. Wait for it to start (watch for whale icon)

**Verify Docker is working:**
```powershell
docker --version
```

---

## Step 2: Download the Repository

Open PowerShell and run:

```powershell
# Go to where you want to download the project
cd Desktop

# Download the project
git clone https://github.com/Tannaz2001/ktc-eit-framework.git

# Go into the project folder
cd ktc-eit-framework
```

**Or download manually:**
1. Go to https://github.com/Tannaz2001/ktc-eit-framework
2. Click green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file
5. Open PowerShell in that folder

---

## Step 3: Copy Configuration

In PowerShell (inside project folder):

```powershell
Copy-Item .env.example .env
```

---

## Step 4: Start Dashboard

```powershell
docker compose up -d
```

**Wait 10 seconds...**

You should see:
```
✓ Container started
```

---

## Step 5: Open in Browser

Go to:
```
http://localhost:8501
```

**🎉 Done!** Dashboard is running!

---

## Stop Dashboard

When done, run:
```powershell
docker compose down
```

---

---

# ✅ OPTION B: LOCAL PYTHON (For Developers)

## Step 1: Install Python 3.12

**Windows:**
1. Go to: https://www.python.org/downloads/
2. Download Python 3.12
3. Run installer
4. ✅ Check "Add Python to PATH"
5. Click Install

**Verify Python installed:**
```powershell
python --version
# Should show: Python 3.12.x
```

---

## Step 2: Install Git

1. Go to: https://git-scm.com/download/win
2. Download and install
3. Use default options

**Verify Git installed:**
```powershell
git --version
```

---

## Step 3: Download the Repository

Open PowerShell:

```powershell
# Go to where you want the project
cd Desktop

# Download the project
git clone https://github.com/Tannaz2001/ktc-eit-framework.git

# Go into the project folder
cd ktc-eit-framework
```

---

## Step 4: Create Python Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

**You should see `(venv)` at start of your PowerShell prompt**

Example:
```
(venv) PS C:\Users\sahil\Desktop\ktc-eit-framework>
```

---

## Step 5: Install Dependencies

```powershell
# Make sure venv is activated (see step above)
# You should see (venv) in the prompt

# Upgrade pip
python -m pip install --upgrade pip

# Install all packages
pip install -r requirements.txt
pip install -r requirements-full.txt

# Install the framework
pip install -e .
```

**This takes 2-5 minutes...**

---

## Step 6: Run Dashboard

```powershell
# Make sure venv is still activated
# (you should see (venv) in the prompt)

# Start the dashboard
streamlit run app.py
```

**You should see:**
```
Local URL: http://localhost:8501
Network URL: http://YOUR_IP:8501
```

---

## Step 7: Open in Browser

Click the link or go to:
```
http://localhost:8501
```

**🎉 Done!** Dashboard is running!

---

## Stop Dashboard

Press `Ctrl + C` in PowerShell

---

---

# 📊 Comparison

| | Docker | Local Python |
|---|--------|--------------|
| **Install Time** | 10 min | 15 min |
| **Run Time** | 1 command | 2 commands |
| **Easy?** | ✅ Yes | 🟡 Medium |
| **For Beginners** | ✅ Recommended | ⚠️ Advanced |

---

---

# ✅ Verify It's Working

### For Docker:
```powershell
docker compose ps
```
Should show container as "healthy"

### For Local:
Browser should show the dashboard at http://localhost:8501

---

---

# 🌐 Access From Another Computer

Find your computer's IP:
```powershell
ipconfig
```

Look for: `IPv4 Address` (example: 192.168.1.100)

Share this link with others:
```
http://192.168.1.100:8501
```

(Replace with your actual IP)

---

---

# ❓ Troubleshooting

### "Docker not found"
- Make sure Docker Desktop is installed AND running
- Look for whale icon in taskbar

### "Python not found"
- Restart PowerShell after installing Python
- Check: `python --version`

### "Port already in use"
```powershell
# For Docker:
docker compose down
docker compose up -d

# For Local:
# Kill whatever is using port 8501 (ask IT)
```

### "Can't connect to http://localhost:8501"
- Wait 10-15 seconds and try again
- Check if app is running: `docker compose ps` or check PowerShell

### "Permission denied"
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

---

# 📋 Quick Checklist

## For Docker:
- [ ] Docker Desktop installed
- [ ] Docker Desktop running (whale icon visible)
- [ ] Repository cloned
- [ ] Copied .env.example to .env
- [ ] Ran `docker compose up -d`
- [ ] Opened http://localhost:8501
- [ ] See dashboard ✅

## For Local Python:
- [ ] Python 3.12 installed
- [ ] Git installed
- [ ] Repository cloned
- [ ] Virtual environment created (`python -m venv venv`)
- [ ] Virtual environment activated (see `(venv)` in prompt)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Ran `streamlit run app.py`
- [ ] Opened http://localhost:8501
- [ ] See dashboard ✅

---

---

# 🎯 Next Steps After Setup

### View Logs (Troubleshoot)
```powershell
# Docker
docker compose logs -f dashboard

# Local (press Ctrl+C to stop)
# Already showing in PowerShell window
```

### Stop Dashboard
```powershell
# Docker
docker compose down

# Local
# Press Ctrl+C in PowerShell
```

### Update Code
```powershell
# Get latest code
git pull

# For Docker:
docker compose up -d --build

# For Local:
pip install -r requirements.txt
# Then run: streamlit run app.py again
```

---

---

# 🆘 Need Help?

| Issue | File to Read |
|-------|-------------|
| Docker command help | DOCKER_COMMANDS.md |
| First time with Docker? | DOCKER_BEGINNER_GUIDE.md |
| Just need to start? | QUICK_START.md |
| Production setup? | PRODUCTION_DEPLOYMENT.md |
| Local + Docker options? | SETUP_GUIDE.md |

---

**All set?** Go to http://localhost:8501 and start using! 🚀
