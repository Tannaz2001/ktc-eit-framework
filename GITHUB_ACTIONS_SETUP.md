# GitHub Actions Auto-Build Setup for Upstream Maintainers

After merging PR #79, follow these steps to enable automatic Docker builds.

## ⚡ Quick Setup (3 minutes)

### Step 1: Create Docker Hub Personal Access Token
1. Go to: https://hub.docker.com/settings/personal-access-tokens
2. Click **Generate New Token**
3. Name: `github-actions`
4. Permissions: **Read** + **Write**
5. Generate and **copy the token immediately** (won't show again)

### Step 2: Add GitHub Secret to Upstream Repo
1. Go to: https://github.com/Tannaz2001/ktc-eit-framework/settings/secrets/actions
2. Click **New repository secret**
3. Add **one secret**:
   - Name: `DOCKER_PASSWORD`
   - Value: Paste the token from Step 1

**That's it!** The username (sahil2705) is hardcoded in the workflow.

### Step 3: Test
Push any commit to main:
```bash
git push origin main
```

Monitor the build:
1. Go to **Actions** tab
2. Watch "Docker Build & Push" workflow run
3. Image appears on Docker Hub at: `docker.io/YOUR_USERNAME/ktc-dashboard`

---

## 📊 What Happens After Setup

Every push to `main` that modifies:
- `Dockerfile`
- `requirements.txt`
- `app.py`
- `src/` or `dashboard/`

Will automatically:
1. ✅ Run tests (Linux + Windows)
2. ✅ Build Docker image
3. ✅ Push to **your Docker Hub** account
4. ✅ Tag as: `latest`, commit hash, version tags
5. ✅ Update Docker Hub description

---

## 🔐 Security Notes

- ✓ Use **Personal Access Token** (safer than password)
- ✓ Token scoped to Docker Hub only
- ✓ GitHub masks secrets in logs (never visible)
- ✗ Never commit credentials to git
- ✗ If token leaks, revoke it immediately: https://hub.docker.com/settings/personal-access-tokens

---

## 🐳 After Auto-Builds Are Running

**Update README quick start:**
```bash
docker run -p 8501:8501 tannaz2001/ktc-dashboard:latest
```
(Replace `tannaz2001` with your Docker Hub username)

**Reference:** [docs/guides/CI-CD_SETUP.md](docs/guides/CI-CD_SETUP.md)

---

## ❓ Troubleshooting

**"Unauthorized: authentication required"**
- Check `DOCKER_PASSWORD` is a Personal Access Token, not your password
- Verify token has Read + Write permissions
- Verify secret names are exactly: `DOCKER_USERNAME`, `DOCKER_PASSWORD`

**"Workflow didn't run"**
- Confirm secrets are set in repo settings
- Confirm push was to `main` branch
- Confirm changed files match trigger conditions (see above)
- Use **Run workflow** manually in Actions tab to force build

**"Build failed but tests passed"**
- Check build logs in Actions tab
- Common: missing dependency, Python compatibility
- Fix locally: `docker build -t test .`
- Commit fix and push — workflow will retry

---

Done! Auto-builds are now live. 🚀
