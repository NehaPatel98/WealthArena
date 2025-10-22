#!/bin/bash
# WealthArena Backend Deployment Commands
# Run these commands step by step to deploy your model to Azure

echo "🚀 WealthArena Backend Deployment Script"
echo "========================================"

# Configuration
RESOURCE_GROUP="wealtharena-rg"
LOCATION="australiaeast"
ACR_NAME="wealtharenaacr"
CONTAINER_NAME="wealtharena-backend"
IMAGE_NAME="wealtharena-backend"
IMAGE_TAG="v1"

# Step 1: Login to Azure
echo ""
echo "Step 1: Logging in to Azure..."
az login

# Step 2: Create Resource Group
echo ""
echo "Step 2: Creating resource group..."
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Step 3: Create Azure Container Registry
echo ""
echo "Step 3: Creating Azure Container Registry..."
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic

# Step 4: Login to ACR
echo ""
echo "Step 4: Logging in to Container Registry..."
az acr login --name $ACR_NAME

# Step 5: Build and Push Image
echo ""
echo "Step 5: Building and pushing Docker image..."
echo "This may take 5-10 minutes..."
az acr build \
  --registry $ACR_NAME \
  --image $IMAGE_NAME:$IMAGE_TAG \
  --file Dockerfile \
  .

# Step 6: Get ACR credentials
echo ""
echo "Step 6: Getting ACR credentials..."
ACR_USERNAME=$ACR_NAME
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

echo "ACR Username: $ACR_USERNAME"
echo "ACR Password: $ACR_PASSWORD"

# Step 7: Deploy Container
echo ""
echo "Step 7: Deploying container to Azure..."
echo "IMPORTANT: You need to provide your Azure SQL connection string!"
echo "Replace <YOUR_CONNECTION_STRING> below with your actual connection string"

read -p "Enter your Azure SQL Connection String: " SQL_CONN_STRING

az container create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --image $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG \
  --cpu 2 \
  --memory 4 \
  --registry-login-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --dns-name-label wealtharena-api \
  --ports 8000 \
  --environment-variables \
    AZURE_SQL_CONNECTION_STRING="$SQL_CONN_STRING" \
    MODEL_PATH=/app/checkpoints \
    MODEL_VERSION=v2.3.1 \
    CACHE_TIMEOUT=300

# Step 8: Get Public URL
echo ""
echo "Step 8: Getting public URL..."
PUBLIC_URL=$(az container show \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --query ipAddress.fqdn \
  --output tsv)

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo "======================"
echo "Your API is live at: http://$PUBLIC_URL:8000"
echo ""
echo "Test it with:"
echo "  curl http://$PUBLIC_URL:8000/health"
echo ""
echo "API Documentation:"
echo "  http://$PUBLIC_URL:8000/docs"
echo ""
echo "To test predictions:"
echo "  curl -X POST http://$PUBLIC_URL:8000/api/predictions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"symbol\": \"CBA\", \"horizon\": 1}'"

