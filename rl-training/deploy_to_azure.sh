#!/bin/bash

# WealthArena Azure Deployment Script
# This script automates the deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP=""
ACR_NAME=""
LOCATION="eastus"
CONTAINER_NAME="wealtharena-ai-models"

echo -e "${BLUE}🚀 WealthArena Azure Deployment Script${NC}"
echo "================================================"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Get resource group name
if [ -z "$RESOURCE_GROUP" ]; then
    echo -e "${YELLOW}📝 Please enter your Azure resource group name:${NC}"
    read -p "Resource Group: " RESOURCE_GROUP
fi

# Get ACR name
if [ -z "$ACR_NAME" ]; then
    echo -e "${YELLOW}📝 Please enter a unique name for your Azure Container Registry:${NC}"
    read -p "ACR Name (must be unique): " ACR_NAME
fi

echo -e "${BLUE}📋 Configuration:${NC}"
echo "Resource Group: $RESOURCE_GROUP"
echo "ACR Name: $ACR_NAME"
echo "Location: $LOCATION"
echo "Container Name: $CONTAINER_NAME"
echo ""

# Step 1: Login to Azure
echo -e "${BLUE}🔐 Step 1: Logging into Azure...${NC}"
az login

# Step 2: Set subscription
echo -e "${BLUE}📋 Step 2: Setting subscription...${NC}"
az account set --subscription $(az account show --query id --output tsv)

# Step 3: Create ACR
echo -e "${BLUE}🏗️ Step 3: Creating Azure Container Registry...${NC}"
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Step 4: Login to ACR
echo -e "${BLUE}🔑 Step 4: Logging into ACR...${NC}"
az acr login --name $ACR_NAME

# Step 5: Build Docker image
echo -e "${BLUE}🐳 Step 5: Building Docker image...${NC}"
docker build -t wealtharena-api .

# Step 6: Tag image
echo -e "${BLUE}🏷️ Step 6: Tagging image for ACR...${NC}"
docker tag wealtharena-api $ACR_NAME.azurecr.io/wealtharena-api:latest

# Step 7: Push to ACR
echo -e "${BLUE}⬆️ Step 7: Pushing image to ACR...${NC}"
docker push $ACR_NAME.azurecr.io/wealtharena-api:latest

# Step 8: Create container instance
echo -e "${BLUE}☁️ Step 8: Creating Azure Container Instance...${NC}"
az container create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --image $ACR_NAME.azurecr.io/wealtharena-api:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server $ACR_NAME.azurecr.io \
  --registry-username $(az acr credential show --name $ACR_NAME --query username --output tsv) \
  --registry-password $(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv) \
  --ports 8000 \
  --environment-variables PYTHONPATH=/app

# Step 9: Get container IP
echo -e "${BLUE}🌐 Step 9: Getting container IP address...${NC}"
CONTAINER_IP=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.ip --output tsv)

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}📊 Deployment Information:${NC}"
echo "Resource Group: $RESOURCE_GROUP"
echo "Container Name: $CONTAINER_NAME"
echo "Container IP: $CONTAINER_IP"
echo "API URL: http://$CONTAINER_IP:8000"
echo "Health Check: http://$CONTAINER_IP:8000/health"
echo "API Docs: http://$CONTAINER_IP:8000/docs"
echo ""

echo -e "${YELLOW}🔧 Next Steps:${NC}"
echo "1. Test the API: curl http://$CONTAINER_IP:8000/health"
echo "2. Set up ngrok: ngrok http $CONTAINER_IP:8000"
echo "3. Update your backend with the ngrok URL"
echo ""

echo -e "${GREEN}🎉 Your WealthArena AI Models are now running on Azure!${NC}"
