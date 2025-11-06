# WealthArena Azure Deployment Script (PowerShell)
# This script automates the deployment process for Windows users

param(
    [string]$ResourceGroup = "",
    [string]$ACRName = "",
    [string]$Location = "eastus",
    [string]$ContainerName = "wealtharena-ai-models"
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

Write-Host "🚀 WealthArena Azure Deployment Script" -ForegroundColor $Blue
Write-Host "================================================" -ForegroundColor $Blue

# Check if Azure CLI is installed
try {
    $azVersion = az --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI not found"
    }
} catch {
    Write-Host "❌ Azure CLI is not installed. Please install it first." -ForegroundColor $Red
    Write-Host "Download from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli" -ForegroundColor $Yellow
    exit 1
}

# Check if Docker is installed
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not found"
    }
} catch {
    Write-Host "❌ Docker is not installed. Please install it first." -ForegroundColor $Red
    Write-Host "Download from: https://www.docker.com/products/docker-desktop/" -ForegroundColor $Yellow
    exit 1
}

# Get resource group name if not provided
if ([string]::IsNullOrEmpty($ResourceGroup)) {
    $ResourceGroup = Read-Host "📝 Please enter your Azure resource group name"
}

# Get ACR name if not provided
if ([string]::IsNullOrEmpty($ACRName)) {
    $ACRName = Read-Host "📝 Please enter a unique name for your Azure Container Registry"
}

Write-Host "📋 Configuration:" -ForegroundColor $Blue
Write-Host "Resource Group: $ResourceGroup"
Write-Host "ACR Name: $ACRName"
Write-Host "Location: $Location"
Write-Host "Container Name: $ContainerName"
Write-Host ""

# Step 1: Login to Azure
Write-Host "🔐 Step 1: Logging into Azure..." -ForegroundColor $Blue
az login

# Step 2: Set subscription
Write-Host "📋 Step 2: Setting subscription..." -ForegroundColor $Blue
$subscriptionId = az account show --query id --output tsv
az account set --subscription $subscriptionId

# Step 3: Create ACR
Write-Host "🏗️ Step 3: Creating Azure Container Registry..." -ForegroundColor $Blue
az acr create --resource-group $ResourceGroup --name $ACRName --sku Basic --admin-enabled true

# Step 4: Login to ACR
Write-Host "🔑 Step 4: Logging into ACR..." -ForegroundColor $Blue
az acr login --name $ACRName

# Step 5: Build Docker image
Write-Host "🐳 Step 5: Building Docker image..." -ForegroundColor $Blue
docker build -t wealtharena-api .

# Step 6: Tag image
Write-Host "🏷️ Step 6: Tagging image for ACR..." -ForegroundColor $Blue
docker tag wealtharena-api "$ACRName.azurecr.io/wealtharena-api:latest"

# Step 7: Push to ACR
Write-Host "⬆️ Step 7: Pushing image to ACR..." -ForegroundColor $Blue
docker push "$ACRName.azurecr.io/wealtharena-api:latest"

# Step 8: Create container instance
Write-Host "☁️ Step 8: Creating Azure Container Instance..." -ForegroundColor $Blue
$acrUsername = az acr credential show --name $ACRName --query username --output tsv
$acrPassword = az acr credential show --name $ACRName --query passwords[0].value --output tsv

az container create `
  --resource-group $ResourceGroup `
  --name $ContainerName `
  --image "$ACRName.azurecr.io/wealtharena-api:latest" `
  --cpu 2 `
  --memory 4 `
  --registry-login-server "$ACRName.azurecr.io" `
  --registry-username $acrUsername `
  --registry-password $acrPassword `
  --ports 8000 `
  --environment-variables PYTHONPATH=/app

# Step 9: Get container IP
Write-Host "🌐 Step 9: Getting container IP address..." -ForegroundColor $Blue
$containerIP = az container show --resource-group $ResourceGroup --name $ContainerName --query ipAddress.ip --output tsv

Write-Host "✅ Deployment completed successfully!" -ForegroundColor $Green
Write-Host ""
Write-Host "📊 Deployment Information:" -ForegroundColor $Blue
Write-Host "Resource Group: $ResourceGroup"
Write-Host "Container Name: $ContainerName"
Write-Host "Container IP: $containerIP"
Write-Host "API URL: http://$containerIP:8000"
Write-Host "Health Check: http://$containerIP:8000/health"
Write-Host "API Docs: http://$containerIP:8000/docs"
Write-Host ""

Write-Host "🔧 Next Steps:" -ForegroundColor $Yellow
Write-Host "1. Test the API: Invoke-WebRequest -Uri 'http://$containerIP:8000/health'"
Write-Host "2. Set up ngrok: ngrok http $containerIP:8000"
Write-Host "3. Update your backend with the ngrok URL"
Write-Host ""

Write-Host "🎉 Your WealthArena AI Models are now running on Azure!" -ForegroundColor $Green

# Test the API
Write-Host ""
Write-Host "🧪 Testing API connection..." -ForegroundColor $Blue
try {
    $response = Invoke-WebRequest -Uri "http://$containerIP:8000/health" -TimeoutSec 30
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ API is responding successfully!" -ForegroundColor $Green
        Write-Host "Response: $($response.Content)" -ForegroundColor $Green
    } else {
        Write-Host "⚠️ API responded with status code: $($response.StatusCode)" -ForegroundColor $Yellow
    }
} catch {
    Write-Host "❌ API test failed: $($_.Exception.Message)" -ForegroundColor $Red
    Write-Host "The container might still be starting up. Please wait a few minutes and try again." -ForegroundColor $Yellow
}
