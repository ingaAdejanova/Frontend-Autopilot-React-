# syntax=docker/dockerfile:1

########################
# Base (shared)
########################
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps shared by api + worker
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cached layer)
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# App code
COPY repoops /app/repoops
COPY apps /app/apps


########################
# API image
########################
FROM base AS api

# Optional: install ripgrep only if API needs it
RUN apt-get update && apt-get install -y --no-install-recommends \
      ripgrep \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


########################
# Worker image
########################
FROM base AS worker

# Tools required for patching + verification + keyword search
# Add build tools for node-gyp/native deps + make (since allowlisted)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ripgrep patch make \
      build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get update && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

# Enable Yarn/PNPM via Corepack
RUN corepack enable

# TypeScript language service for ts_graph
RUN npm install -g typescript
ENV NODE_PATH=/usr/local/lib/node_modules

CMD ["python", "-m", "apps.worker.worker"]
