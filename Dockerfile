# syntax=docker/dockerfile:1.6

##############################
# Stage 1: Builder
##############################
FROM python:3.9 as builder
WORKDIR /app

# 1) APT con caché (BuildKit)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git git-lfs openssh-client curl && \
    rm -rf /var/lib/apt/lists/* && git lfs install

# Copia mínimos para cachear pip en base a requirements (no TODO el repo)
COPY Equipo54_MLOps/requirements.txt .
COPY Equipo54_MLOps/setup.py .

# 2) PIP con caché (BuildKit) + preferir binarios
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=1
# Si quieres fallar si no hay wheel (evitar compilar):
# ENV PIP_ONLY_BINARY=:all:

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt || true && \
    python -m pip install "dvc[s3]==3.*" mlflow boto3 awscli

# Copia el resto del repo (esto no invalida la capa de pip)
COPY . .

##############################
# Stage 2: Runtime
##############################
FROM python:3.9-slim
WORKDIR /app

# APT mínimo con caché (BuildKit)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs openssh-client curl && \
    rm -rf /var/lib/apt/lists/* && git lfs install

# Copia paquetes instalados y binarios desde builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# (Opcional) instala tu paquete local SOLO si realmente lo importas como módulo
# Mejor evita reinstalarlo para no romper el caché:
# RUN cd /app/Equipo54_MLOps && python -m pip install .

RUN dvc --version && mlflow --version && aws --version

ENV PYTHONUNBUFFERED=1 DVC_NO_ANALYTICS=1
CMD ["/bin/bash"]
