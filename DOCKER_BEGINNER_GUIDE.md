# Docker Setup - Beginner's Guide ✨

**For people using Docker for the first time!**

---

## 📋 What You Need

- ✅ Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop))
- ✅ This project folder
- ✅ Internet connection (first time only)

---

## 🚀 Step 1: Copy Configuration File

**What to do:**
1. Find `.env.example` file in the project folder
2. Make a copy and rename it to `.env`
3. Don't change anything inside, just leave it as is

**In PowerShell:**
```powershell
Copy-Item .env.example .env
```

---

## 🚀 Step 2: Start Docker

**What to do:**
1. Open PowerShell in the project folder
2. Type this command:

```powershell
docker compose up -d
```

3. Press Enter and wait 5-10 seconds
4. You should see: ✅ "Container started"

---

## 🚀 Step 3: Open Dashboard

**In your web browser, go to:**
```
http://localhost:8501
```

**That's it!** You should see the dashboard now. 🎉

---

## 📖 Common Commands (Copy & Paste)

### ✅ Check if it's running
```powershell
docker compose ps
```

### ▶️ Start dashboard
```powershell
docker compose up -d
```

### ⏹️ Stop dashboard
```powershell
docker compose down
```

### 🔄 Restart dashboard
```powershell
docker compose restart
```

### 📋 View what happened
```powershell
docker compose logs -f dashboard
```
Press `Ctrl + C` to stop viewing logs

---

## ❓ Troubleshooting (If Something Goes Wrong)

### **Problem: "Port already in use"**
```powershell
docker compose down
docker compose up -d
```

### **Problem: "Can't access http://localhost:8501"**
- Wait 10 seconds and try again
- Check if Docker is running (look for Docker icon)

### **Problem: "Disk full or out of space"**
```powershell
docker system prune -a
docker compose up -d
```

### **Problem: Nothing works**
```powershell
docker compose down -v
docker compose up -d --build
```

---

## 📞 Help

| What? | Command |
|-------|---------|
| Is it running? | `docker compose ps` |
| What happened? | `docker compose logs -f` |
| Where is it? | `http://localhost:8501` |
| Stop it | `docker compose down` |
| Start it | `docker compose up -d` |

---

## 📱 Access Dashboard From Another Device

1. Find your computer's IP:
```powershell
ipconfig
```
Look for: `IPv4 Address` (example: `192.168.1.100`)

2. Share this link with others:
```
http://192.168.1.100:8501
```

(Replace `192.168.1.100` with your actual IP)

---

## 🎯 Quick Checklist

- [ ] Docker Desktop installed
- [ ] Copied `.env.example` to `.env`
- [ ] Ran `docker compose up -d`
- [ ] Waited 10 seconds
- [ ] Opened `http://localhost:8501`
- [ ] See dashboard ✅

---

## 💡 Remember

- **`docker compose up -d`** = Start (the `-d` means "run in background")
- **`docker compose down`** = Stop
- **`docker compose logs -f`** = Watch what's happening
- **`docker compose ps`** = Check status

That's all you need to know! 🎊

---

## 🆘 Still Having Issues?

1. Restart Docker Desktop
2. Run: `docker compose down`
3. Run: `docker compose up -d --build`
4. Wait 15 seconds
5. Try again

**If nothing works:** Check if Docker Desktop is running (look for whale icon in taskbar)
