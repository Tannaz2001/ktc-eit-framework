# syntax=docker/dockerfile:1
#
# KTC EIT Reconstruction Dashboard
# =============================================================================
# Two build targets, differing only in which Python dependencies they install:
#
#   full (default)  Every method in configs/ktc_all_methods.yaml. Adds
#                   TensorFlow (CompetitionCNN) and Torch
#                   (ktc2023_postprocessing_master).
#   slim            No TensorFlow/Torch. Runs the pure numpy/scipy methods:
#                   KTC2023_CUQI1, KTC2023_CUQI2_main,
#                   LinearDifferenceReconstruction and the built-in baselines.
#
#   docker build -t ktc-dashboard:full .
#   docker build -t ktc-dashboard:slim --target slim .
#
# EvaluationData/, Codes_Matlab/ and external_methods/ total ~42 MB and are
# baked in, so the image runs with no host setup. Only outputs/ (multi-GB and
# machine-specific) stays outside and is mounted — see docker-compose.yml.
# =============================================================================

ARG PYTHON_VERSION=3.12


# --- base: OS packages, runtime user, settings shared by every target --------
FROM python:${PYTHON_VERSION}-slim AS base

# STREAMLIT_SERVER_FILE_WATCHER_TYPE / RUN_ON_SAVE override .streamlit/config.toml,
# which sets fileWatcherType="poll" and runOnSave=true for local development.
# Both are harmful in a container: the watcher would continuously poll the
# mounted multi-GB outputs/ tree, and a rerun firing mid-upload raises
# RerunException (a BaseException), which slips past the upload handlers'
# `except Exception` cleanup — see _render_upload_new_plugin_widget in app.py.
# Setting them here leaves local development untouched.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_RUN_ON_SAVE=false

# libgomp1: OpenMP runtime the scipy / scikit-image / tensorflow wheels link
# against; not present in python:slim. The app writes to configs/ and
# external_methods/ at runtime, so it runs as a real user that owns /app.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app

WORKDIR /app

# WORKDIR creates /app as root. The app needs to create outputs/ here when the
# container runs without a bind mount, so hand the directory over.
RUN chown app:app /app

EXPOSE 8501

# Port is resolved at runtime so this still holds when STREAMLIT_SERVER_PORT is
# overridden. Streamlit's first paint is slow, hence the long start period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('STREAMLIT_SERVER_PORT','8501'))" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]


# --- dependency layers: separate stages so code edits never re-run pip -------
FROM base AS deps-slim
COPY requirements.txt ./
RUN pip install -r requirements.txt

FROM base AS deps-full
COPY requirements.txt requirements-full.txt ./
RUN pip install -r requirements-full.txt


# --- application layers -----------------------------------------------------
# Identical bodies; they differ only in the dependency stage they build on.
# ENV / EXPOSE / HEALTHCHECK / ENTRYPOINT are inherited from base.
# chmod is explicit because a Windows build context carries no exec bit.

FROM deps-slim AS slim
COPY --chown=app:app . .
# .dockerignore re-includes outputs/run_* via a negation rule; Docker creates
# the outputs/ parent synthetically as root before applying --chown, and may
# leave it with restrictive mode bits (e.g. 0555). Fix both owner and mode.
RUN pip install --no-deps . && chmod +x docker-entrypoint.sh \
    && mkdir -p /app/outputs && chown app:app /app/outputs && chmod 755 /app/outputs
USER app

FROM deps-full AS full
COPY --chown=app:app . .
RUN pip install --no-deps . && chmod +x docker-entrypoint.sh \
    && mkdir -p /app/outputs && chown app:app /app/outputs && chmod 755 /app/outputs
USER app
