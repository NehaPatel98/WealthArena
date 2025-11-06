# RL Training Service Deployment Guide

## Overview

This guide covers deployment of the RL Training service for WealthArena, including local development with Docker Compose and production deployment to Azure Container Apps and Azure Container Registry (ACR).

## Prerequisites

### Required Software
- **Python 3.9+** (3.11 or 3.12 recommended)
- **Docker Desktop** (for local deployment)
- **Azure CLI** (for Azure deployment)
- **Azure Account** with active subscription
- **Virtual environment** (for local development)

### Required Accounts
- Azure Account with resource group access
- ngrok account (free tier works for public access)

### Required Files
- Trained model checkpoints in `checkpoints/` directory
- `requirements.txt` with all dependencies
- `.env` file with configuration (see `.env.example`)

## Local Deployment (Docker Compose)

### 1. Environment Setup

```bash
cd rl-training

# Create virtual environment (optional, for local development)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and configure:
- Database connection settings
- Model paths
- Training parameters
- Logging configuration
- API server settings

**Important:** Never commit `.env` files with secrets. Use `.env.example` as a template.

### 3. Run with Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

The API will be available at `http://localhost:8000`.

### 4. Test Local Deployment

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test models endpoint
curl http://localhost:8000/models

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "asx_stocks",
    "input_data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
    "symbol": "AAPL"
  }'
```

## Azure Deployment (Container Apps/ACR)

### Step 1: Install Required Software

#### Install Docker Desktop
1. Go to [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Download and install Docker Desktop
3. Start Docker Desktop
4. Verify installation:
   ```bash
   docker --version
   ```

#### Install Azure CLI
1. Go to [Azure CLI Installation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
2. Download and install Azure CLI
3. Verify installation:
   ```bash
   az --version
   ```

#### Install ngrok (for public access)
1. Go to [ngrok](https://ngrok.com/)
2. Sign up for a free account
3. Download ngrok
4. Extract and add to your PATH
5. Configure auth token:
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

### Step 2: Set Up Azure

#### Login to Azure
```bash
az login
```
This will open a browser window for authentication.

#### Set Your Resource Group
```bash
# List your resource groups
az group list --output table

# Set your resource group (replace with your actual name)
$env:RESOURCE_GROUP = "YOUR_RESOURCE_GROUP_NAME"  # Windows PowerShell
# or
export RESOURCE_GROUP="YOUR_RESOURCE_GROUP_NAME"   # macOS/Linux
```

#### Set Your Subscription
```bash
# List your subscriptions
az account list --output table

# Set the correct subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

### Step 3: Create Azure Container Registry (ACR)

```bash
# Create ACR (replace with unique name)
ACR_NAME="wealtharenaacr$(Get-Date -Format 'yyyyMMddHHmmss')"  # Windows PowerShell
# or
ACR_NAME="wealtharenaacr$(date +%s)"  # macOS/Linux

az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true
```

**Note:** Write down your ACR name! You'll need it later.

#### Login to ACR
```bash
az acr login --name $ACR_NAME
```

### Step 4: Build and Push Docker Image

#### Build Docker Image Locally
```bash
cd rl-training

# Build the Docker image
docker build -t wealtharena-api -f Dockerfile.api .

# Test the container locally
docker run -p 8000:8000 wealtharena-api
```

#### Tag and Push to ACR
```bash
# Tag the image for ACR
docker tag wealtharena-api $ACR_NAME.azurecr.io/wealtharena-api:latest

# Push to ACR
docker push $ACR_NAME.azurecr.io/wealtharena-api:latest
```

### Step 5: Deploy to Azure Container Instances

```bash
# Create container instance
az container create \
  --resource-group $RESOURCE_GROUP \
  --name wealtharena-ai-models \
  --image $ACR_NAME.azurecr.io/wealtharena-api:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server $ACR_NAME.azurecr.io \
  --registry-username $(az acr credential show --name $ACR_NAME --query username --output tsv) \
  --registry-password $(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv) \
  --ports 8000 \
  --environment-variables PYTHONPATH=/app
```

#### Get Container IP Address
```bash
az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models --query ipAddress.ip --output tsv
```

### Step 6: Set Up Public Access with ngrok

```bash
# Get your container IP
CONTAINER_IP=$(az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models --query ipAddress.ip --output tsv)

# Create ngrok tunnel
ngrok http $CONTAINER_IP:8000
```

**Note:** ngrok will give you a public URL like `https://abc123.ngrok.io`. Write this down!

### Step 7: Test Azure Deployment

```bash
# Test health endpoint
curl https://YOUR_NGROK_URL/health

# Test models endpoint
curl https://YOUR_NGROK_URL/models

# Test prediction endpoint
curl -X POST https://YOUR_NGROK_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "asx_stocks",
    "input_data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
    "symbol": "AAPL"
  }'
```

## Model Deployment

### Upload Models to Azure Blob Storage

Use the `upload_models.ps1` script to upload trained model checkpoints:

```powershell
# Navigate to infrastructure directory
cd infrastructure/azure_deployment

# Run upload script
.\upload_models.ps1 -StorageAccount "stwealtharenadev" -ResourceGroup "rg-wealtharena-northcentralus" -Container "rl-models"
```

The script will:
1. Verify Azure CLI connection
2. Verify storage account exists
3. Create blob container if needed
4. Upload model checkpoints for each asset class:
   - `asx_stocks`
   - `cryptocurrencies`
   - `currency_pairs`
   - `commodities`
   - `etf`
5. Verify uploads

**Model files uploaded:**
- `.pt` files (PyTorch model weights)
- `.pkl` files (training state, optimizer state)
- `.json` files (metadata, configuration)

**Blob storage structure:**
```
rl-models/
└── latest/
    ├── asx_stocks/
    ├── cryptocurrencies/
    ├── currency_pairs/
    ├── commodities/
    └── etf/
```

### Update RL Service Configuration

After uploading models, update the RL service environment variables:

```bash
# Set MODEL_PATH to blob storage mount point
MODEL_PATH=/home/site/models/latest

# Or use Azure Blob Storage connection string
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."
```

## Testing

### Local Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=xml --cov-report=term

# Run specific test file
pytest tests/test_environment.py

# Run with verbose output
pytest -v
```

### Integration Testing

```bash
# Test API endpoints
python test_api.py

# Test model loading
python test_models.py

# Test deployment
python test_deployment.py
```

### Health Checks

```bash
# Local health check
curl http://localhost:8000/health

# Azure health check
curl https://YOUR_NGROK_URL/health

# Container health check
az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models --query instanceView.state
```

## Troubleshooting

### Container Failed to Start

**Symptoms:** Container shows "Failed" or "Stopped" state

**Solutions:**
```bash
# Check container logs
az container logs --resource-group $RESOURCE_GROUP --name wealtharena-ai-models

# Common fixes:
# 1. Check if all dependencies are installed
# 2. Verify model files are present
# 3. Check memory requirements
# 4. Verify environment variables

# Restart container
az container restart --resource-group $RESOURCE_GROUP --name wealtharena-ai-models
```

### Cannot Connect to API

**Symptoms:** Connection timeout or connection refused

**Solutions:**
```bash
# Check if container is running
az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models --query instanceView.state

# Check container IP
az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models --query ipAddress.ip --output tsv

# Check if port is open
az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models --query ipAddress.ports

# Test connectivity
Test-NetConnection -ComputerName YOUR_CONTAINER_IP -Port 8000  # Windows
nc -zv YOUR_CONTAINER_IP 8000  # macOS/Linux
```

### ngrok Tunnel Not Working

**Symptoms:** ngrok shows "offline" or connection errors

**Solutions:**
```bash
# Check ngrok status
ngrok api tunnels list

# Verify auth token
ngrok config check

# Restart ngrok
ngrok http YOUR_CONTAINER_IP:8000

# Check ngrok web interface
# Open http://127.0.0.1:4040 in browser
```

### Models Not Loading

**Symptoms:** API returns 404 for model endpoints or "Model not found" errors

**Solutions:**
```bash
# Check if model files exist in container
az container exec --resource-group $RESOURCE_GROUP --name wealtharena-ai-models -- ls -la checkpoints/

# Verify model upload to blob storage
az storage blob list --container-name rl-models --account-name stwealtharenadev --prefix latest/ --auth-mode login --output table

# Rebuild and redeploy if needed
docker build -t wealtharena-api -f Dockerfile.api .
docker tag wealtharena-api $ACR_NAME.azurecr.io/wealtharena-api:latest
docker push $ACR_NAME.azurecr.io/wealtharena-api:latest
az container restart --resource-group $RESOURCE_GROUP --name wealtharena-ai-models
```

### Docker Build Fails

**Symptoms:** `docker build` command fails with errors

**Solutions:**
```bash
# Check Dockerfile syntax
docker build --no-cache -t wealtharena-api -f Dockerfile.api .

# Check for missing files
ls -la checkpoints/
ls -la requirements.txt

# Check Docker logs
docker logs <container_id>

# Try building with verbose output
docker build --progress=plain -t wealtharena-api -f Dockerfile.api .
```

### Azure CLI Authentication Issues

**Symptoms:** "Not logged in" or "Not authorized" errors

**Solutions:**
```bash
# Re-login to Azure
az login

# Check current account
az account show

# List available subscriptions
az account list --output table

# Set correct subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

## Cost Controls

### Azure Costs (Estimated Monthly)

- **Container Registry (ACR)**: $5/month (Basic tier)
- **Container Instance**: $10-15/month (2 CPU, 4GB RAM)
- **Blob Storage**: $1-2/month (for model storage)
- **Total**: ~$16-22/month

### ngrok Costs

- **Free Tier**: 1 tunnel, 40 connections/minute
- **Paid Tier**: $8/month for more features

### Cost Optimization Tips

1. **Stop container when not in use:**
   ```bash
   az container stop --resource-group $RESOURCE_GROUP --name wealtharena-ai-models
   ```

2. **Start container when needed:**
   ```bash
   az container start --resource-group $RESOURCE_GROUP --name wealtharena-ai-models
   ```

3. **Use smaller instance size for development:**
   ```bash
   az container create --resource-group $RESOURCE_GROUP --name wealtharena-ai-models-dev \
     --image $ACR_NAME.azurecr.io/wealtharena-api:latest \
     --cpu 1 --memory 2
   ```

4. **Monitor costs:**
   ```bash
   # View resource costs
   az consumption usage list --start-date 2024-01-01 --end-date 2024-01-31
   ```

5. **Set up cost alerts:**
   - Go to Azure Portal → Cost Management → Budgets
   - Create budget alerts for your resource group

## Monitoring and Maintenance

### Daily Tasks
- [ ] Check if container is running
- [ ] Monitor API response times
- [ ] Check ngrok tunnel status
- [ ] Review error logs

### Weekly Tasks
- [ ] Review Azure costs
- [ ] Test all API endpoints
- [ ] Check container logs
- [ ] Verify model availability

### Monthly Tasks
- [ ] Update models with new data
- [ ] Review and optimize costs
- [ ] Backup configuration
- [ ] Update dependencies

### Monitoring Commands

```bash
# Check container status
az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models

# View logs
az container logs --resource-group $RESOURCE_GROUP --name wealtharena-ai-models

# Check resource usage
az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models --query instanceView.currentState
```

## Quick Reference Commands

```bash
# Check container status
az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models

# View logs
az container logs --resource-group $RESOURCE_GROUP --name wealtharena-ai-models

# Restart container
az container restart --resource-group $RESOURCE_GROUP --name wealtharena-ai-models

# Stop container
az container stop --resource-group $RESOURCE_GROUP --name wealtharena-ai-models

# Start container
az container start --resource-group $RESOURCE_GROUP --name wealtharena-ai-models

# Get container IP
az container show --resource-group $RESOURCE_GROUP --name wealtharena-ai-models --query ipAddress.ip --output tsv

# Test health endpoint
curl http://YOUR_CONTAINER_IP:8000/health
```

## Additional Resources

- **API Documentation**: Available at `http://localhost:8000/docs` (local) or `https://YOUR_NGROK_URL/docs` (Azure)
- **Health Check**: `http://localhost:8000/health`
- **Models Info**: `http://localhost:8000/models`
- **Docker Compose**: See `docker-compose.yml` for local deployment configuration
- **Dockerfile**: See `Dockerfile.api` for container image configuration

## Support

For additional help:
1. Check application logs: `docker-compose logs` or `az container logs`
2. Review error messages in container output
3. Verify environment configuration in `.env` file
4. Check database connectivity (if applicable)
5. Review recent code changes

---

*Last Updated: 2025*
*Status: Production Ready* ✅
