# CI/CD Pipeline Guide

This document describes the Continuous Integration and Continuous Deployment setup for the MLOps pipeline.

## Overview

We have a main workflows:

- **Deploy Dev** (`deploy-dev.yml`) - Runs the full MLOps pipeline when merging to dev

## Workflows


### 1. Deploy to Dev (`deploy-dev.yml`)

**Triggers:** 
- Push to `dev` branch
- Manual workflow dispatch

**Purpose:** Run the complete MLOps pipeline with experiment tracking

**Steps:**
- ✅ Build Docker image with caching
- ✅ Configure AWS credentials
- ✅ Cache DVC data
- ✅ Run full DVC pipeline via `docker-compose.ci.yml`
- ✅ Upload pipeline artifacts (reports, models)

**Caching:**
- Docker build layers
- DVC cache for faster data pulls
- Cache key: `${{ runner.os }}-dvc-${{ hashFiles('**/*.dvc') }}`

## Required GitHub Secrets

Add these secrets to your repository settings:

| Secret | Description | Example |
|--------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key for S3/DVC | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `wJalrXUt...` |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://your-ec2-ip:5000` |
| `MLFLOW_EXPERIMENT` | Experiment name | `student-performance` |
| `ARTIFACT_ROOT` | S3 bucket for MLflow artifacts | `s3://your-bucket/mlflow` |

## Docker Compose Files

### `docker-compose.yml` (Local Development)
- Runs MLflow server locally
- Pipeline connects to `localhost:5000`
- Uses host networking

### `docker-compose.ci.yml` (CI/CD)
- Designed for GitHub Actions runners
- Uses environment variables from secrets
- Mounts `$GITHUB_WORKSPACE` for file access
- Includes DVC cache mounting

## Pipeline Stages

The DVC pipeline runs these stages in order:

1. **process_data** - Clean raw student data
2. **build_features** - Feature engineering and encoding
3. **train** - Train decision tree model with hyperparameter optimization
4. **evaluate** - Generate metrics and performance reports
5. **predict** - Create predictions on dataset

## Artifacts

### Uploaded to GitHub Actions
- Pipeline reports (`Equipo54_MLOps/reports/**`)
- Trained models (`*.pkl` files)
- Retention: 30 days

### Stored in S3 (via DVC)
- Raw and processed datasets
- Model files
- Feature encoders

### Tracked in MLflow
- Experiment parameters
- Model metrics
- Model artifacts (via S3)
