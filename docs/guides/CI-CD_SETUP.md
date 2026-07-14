# CI/CD Setup & GitHub Actions

This guide walks through setting up automated Docker builds and pushes.

## Overview

Three GitHub Actions workflows are configured:

| Workflow | Trigger | Action |
|----------|---------|--------|
| **CI** (ci.yml) | Every push | Run tests (Linux + Windows) |
| **Tests** (test.yml) | PR to main | Run Windows tests |
| **Docker Build** (docker-build.yml) | Push to main | Build & push Docker image to Hub |

---

## Docker Auto-Build Setup

### Step 1: Add GitHub Secrets

GitHub needs your Docker Hub credentials to push images. Add them as **repository secrets**:

1. Go to your repo: `https://github.com/Sahil-exe/ktc-eit-framework`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add two secrets:

| Name | Value |
|------|-------|
| `DOCKER_USERNAME` | Your Docker Hub username (e.g., `sahil2705`) |
| `DOCKER_PASSWORD` | Docker Hub Personal Access Token (NOT password) |

### Step 2: Create Docker Hub Personal Access Token

Docker Hub doesn't allow pushing with your password directly. Create a token instead:

1. Go to https://hub.docker.com/settings/personal-access-tokens
2. Click **Generate New Token**
3. Name it: `github-actions` (or any name)
4. Set permissions: `Read, Write`
5. Click **Generate**
6. **Copy the token immediately** (you won't see it again)
7. Paste it as the `DOCKER_PASSWORD` secret in GitHub

### Step 3: Verify Secrets Are Set

```bash
# In GitHub repo settings, you should see:
✓ DOCKER_USERNAME
✓ DOCKER_PASSWORD
```

---

## How It Works

### Automatic Builds (every push to main)

When you push to `main`, GitHub Actions:

1. **Checks out code**
2. **Builds Docker image** (multi-stage, cached)
3. **Pushes to Docker Hub** with tags:
   - `sahil2705/ktc-dashboard:latest` (always points to newest)
   - `sahil2705/ktc-dashboard:v1.0.0` (git tag, if tagged release)
   - `sahil2705/ktc-dashboard:<short-sha>` (commit hash)
4. **Updates Docker Hub description** from this README

### Build Triggers

Image only builds when these files change:
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `requirements.txt`
- `app.py`
- `src/**` (framework code)
- `dashboard/**` (dashboard modules)
- `.github/workflows/docker-build.yml` (workflow itself)

This avoids unnecessary rebuilds when docs or tests change.

### Manual Trigger

To build without code changes:

1. Go to **Actions** tab
2. Click **Docker Build & Push**
3. Click **Run workflow** → **Run workflow**

---

## Monitoring Builds

### View Build Status

1. Go to **Actions** tab in GitHub
2. Click on a workflow run to see:
   - Build step-by-step logs
   - Success/failure status
   - Artifact details (image digest, size)

### Build Failures

Common issues:

**"Unauthorized: authentication required"**
- ✗ `DOCKER_PASSWORD` is wrong
- ✓ Use Personal Access Token, not password
- ✓ Token has `Read, Write` permissions

**"Layer caching failed"**
- This is a warning, not an error
- Next build will still succeed
- Caching is just an optimization

**"Build timed out"**
- Docker layer upload took >1 hour
- Rare, but happens on slow internet
- GitHub will retry automatically

---

## Deployment from CI

Once image is pushed to Docker Hub, you can deploy:

### Pull Latest Image

```bash
docker pull sahil2705/ktc-dashboard:latest
docker run -p 8501:8501 sahil2705/ktc-dashboard:latest
```

### Deploy to Server

```bash
# On your server:
docker run -d \
  -p 8501:8501 \
  -v /data/EvaluationData:/app/EvaluationData \
  -v /data/outputs:/app/outputs \
  sahil2705/ktc-dashboard:latest
```

### Auto-Deploy Pipeline (Advanced)

To automatically deploy on every push, add another GitHub Actions step that:
1. SSHes into your server
2. Pulls the new image
3. Restarts the container

Example (save as `.github/workflows/docker-deploy.yml`):

```yaml
name: Auto-Deploy to Server

on:
  workflow_run:
    workflows: ["Docker Build & Push"]
    types: [completed]

jobs:
  deploy:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_IP }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SERVER_SSH_KEY }}
          script: |
            docker pull sahil2705/ktc-dashboard:latest
            docker-compose -f /app/docker-compose.yml up -d
```

---

## Troubleshooting

### Image not updating on Docker Hub

1. Check **Actions** tab — did the workflow run?
2. If it didn't run, check the trigger conditions:
   - Push must be to `main` branch
   - Must have changed `Dockerfile`, `requirements.txt`, etc.
   - Use **Run workflow** manually to force a build

### Tests pass but Docker build fails

1. Check build logs in Actions tab
2. Common issues:
   - Missing dependency in `requirements.txt`
   - Python 3.12 incompatibility
   - File permissions in Dockerfile
3. Fix locally: `docker build -t test .`
4. Commit fix and push — CI will retry

### Secrets not working

1. Verify secrets are set in repo settings
2. Secrets are **case-sensitive**: `DOCKER_USERNAME` not `docker_username`
3. Secrets are **repo-level**, not org-level
4. After adding secrets, new workflows will see them (no cache delay)

---

## Security

### Personal Access Token (Not Password)

- ✓ Use PAT for CI/CD (safer, limited scope)
- ✗ Never commit credentials to git
- ✗ Never paste password into GitHub

### Secret Scope

- Secrets are visible to **all workflows in this repo**
- They're **not** visible to forked repos
- They're **not** visible in workflow logs (GitHub masks them)

### Revoking Access

If a token is compromised:

1. Revoke it: https://hub.docker.com/settings/personal-access-tokens
2. Create a new token
3. Update GitHub secret with new token

---

## Next Steps

- [ ] Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets to GitHub
- [ ] Push a commit to `main` to trigger first automated build
- [ ] Monitor Actions tab for success
- [ ] Verify image on Docker Hub
- [ ] (Optional) Set up auto-deploy pipeline

