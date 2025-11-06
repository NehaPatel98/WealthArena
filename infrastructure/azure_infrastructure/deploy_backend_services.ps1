# WealthArena Backend Services Deployment Script
# This script deploys the RAG chatbot and backend API to Azure Container Apps

param(
    [string]$ResourceGroupName = "rg-wealtharena-northcentralus",
    [string]$Environment = "dev"
)

# Set error action preference
$ErrorActionPreference = "Continue"

# Colors for output
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Blue = "Blue"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Build-AndPush-Container {
    param([string]$ServiceName, [string]$ServicePath, [string]$RegistryName)
    
    Write-ColorOutput "Building and pushing $ServiceName container..." $Blue
    
    try {
        # Change to service directory
        Push-Location $ServicePath
        
        # Build Docker image
        Write-ColorOutput "   Building Docker image for $ServiceName..." $Blue
        docker build -t $ServiceName .
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "   Failed to build Docker image for $ServiceName" $Red
            return $false
        }
        
        # Tag image for Azure Container Registry
        $imageTag = "$RegistryName.azurecr.io/$ServiceName" + ":latest"
        docker tag $ServiceName $imageTag
        
        # Push to Azure Container Registry
        Write-ColorOutput "   Pushing image to Azure Container Registry..." $Blue
        docker push $imageTag
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "   Failed to push image to registry" $Red
            return $false
        }
        
        Write-ColorOutput "   $ServiceName container built and pushed successfully" $Green
        return $true
    }
    catch {
        Write-ColorOutput "   Error building/pushing $ServiceName container: $($_.Exception.Message)" $Red
        return $false
    }
    finally {
        Pop-Location
    }
}

function Deploy-ContainerApp {
    param([string]$AppName, [string]$ImageName, [string]$RegistryName, [string]$ResourceGroup, [string]$EnvironmentName)
    
    Write-ColorOutput "Deploying $AppName to Azure Container Apps..." $Blue
    
    try {
        $imageUrl = "$RegistryName.azurecr.io/$ImageName" + ":latest"
        
        # Create container app
        az containerapp create `
            --name $AppName `
            --resource-group $ResourceGroup `
            --environment $EnvironmentName `
            --image $imageUrl `
            --target-port 8000 `
            --ingress external `
            --registry-server $RegistryName.azurecr.io `
            --output none
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "   $AppName deployed successfully" $Green
            
            # Get the app URL
            $appUrl = az containerapp show --name $AppName --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn --output tsv
            Write-ColorOutput "   App URL: https://$appUrl" $Blue
            
            return $true
        }
        else {
            Write-ColorOutput "   Failed to deploy $AppName" $Red
            return $false
        }
    }
    catch {
        Write-ColorOutput "   Error deploying ${AppName}: $($_.Exception.Message)" $Red
        return $false
    }
}

function Test-ServiceEndpoint {
    param([string]$ServiceUrl, [string]$ServiceName)
    
    Write-ColorOutput "Testing $ServiceName endpoint..." $Blue
    
    try {
        $response = Invoke-WebRequest -Uri "$ServiceUrl/healthz" -Method GET -TimeoutSec 30
        
        if ($response.StatusCode -eq 200) {
            Write-ColorOutput "   $ServiceName health check passed" $Green
            return $true
        }
        else {
            Write-ColorOutput "   $ServiceName health check failed with status: $($response.StatusCode)" $Red
            return $false
        }
    }
    catch {
        Write-ColorOutput "   $ServiceName health check failed: $($_.Exception.Message)" $Red
        return $false
    }
}

function Setup-EnvironmentVariables {
    param([string]$AppName, [string]$ResourceGroup)
    
    Write-ColorOutput "Setting up environment variables for $AppName..." $Blue
    
    try {
        # Set environment variables for the container app
        $envVars = @(
            "GROQ_API_KEY=YOUR_GROQ_API_KEY",
            "AZURE_STORAGE_ACCOUNT=stwealtharena$Environment",
            "AZURE_SQL_SERVER=sql-wealtharena-$Environment.database.windows.net",
            "AZURE_SQL_DATABASE=wealtharena_db",
            "AZURE_SQL_USER=wealtharena_admin",
            "AZURE_SQL_PASSWORD=Secure@Db2024!$%",
            "AZURE_COSMOS_ACCOUNT=cosmos-wealtharena-$Environment"
        )
        
        foreach ($envVar in $envVars) {
            $key, $value = $envVar -split "=", 2
            az containerapp update `
                --name $AppName `
                --resource-group $ResourceGroup `
                --set-env-vars "$key=$value" `
                --output none
        }
        
        Write-ColorOutput "   Environment variables set for $AppName" $Green
        return $true
    }
    catch {
        Write-ColorOutput "   Error setting environment variables for ${AppName}: $($_.Exception.Message)" $Red
        return $false
    }
}

# Main execution
Write-ColorOutput "WealthArena Backend Services Deployment" $Blue
Write-ColorOutput "=========================================" $Blue

# Check if Docker is available
try {
    $dockerVersion = docker --version
    Write-ColorOutput "Docker found: $dockerVersion" $Green
}
catch {
    Write-ColorOutput "Docker not found. Please install Docker Desktop first." $Red
    exit 1
}

# Check if Azure CLI is logged in
try {
    $account = az account show --output json | ConvertFrom-Json
    Write-ColorOutput "Azure CLI connected as: $($account.user.name)" $Green
}
catch {
    Write-ColorOutput "Not logged in to Azure. Please run 'az login' first." $Red
    exit 1
}

# Get Azure Container Registry name
$registryName = "acrwealtharena$Environment"
Write-ColorOutput "Using Azure Container Registry: $registryName" $Blue

# Login to Azure Container Registry
Write-ColorOutput "Logging in to Azure Container Registry..." $Blue
az acr login --name $registryName

# Build and push RAG Chatbot service
Write-ColorOutput "Deploying RAG Chatbot Service..." $Blue
$chatbotPath = "azure_services/rag_chatbot_service"
if (Test-Path $chatbotPath) {
    $chatbotBuilt = Build-AndPush-Container -ServiceName "rag-chatbot" -ServicePath $chatbotPath -RegistryName $registryName
}
else {
    Write-ColorOutput "RAG Chatbot service directory not found: $chatbotPath" $Yellow
    $chatbotBuilt = $false
}

# Build and push Backend API service
Write-ColorOutput "Deploying Backend API Service..." $Blue
$backendPath = "azure_services/wealtharena_backend_api"
if (Test-Path $backendPath) {
    $backendBuilt = Build-AndPush-Container -ServiceName "wealtharena-backend" -ServicePath $backendPath -RegistryName $registryName
}
else {
    Write-ColorOutput "Backend API service directory not found: $backendPath" $Yellow
    $backendBuilt = $false
}

# Deploy to Azure Container Apps (if Container Apps environment exists)
$containerAppsEnv = "cae-wealtharena-$Environment"
Write-ColorOutput "Checking Container Apps environment..." $Blue

try {
    $envExists = az containerapp env show --name $containerAppsEnv --resource-group $ResourceGroupName --output json | ConvertFrom-Json
    if ($envExists) {
        Write-ColorOutput "Container Apps environment found: $containerAppsEnv" $Green
        
        # Deploy RAG Chatbot
        if ($chatbotBuilt) {
            $chatbotDeployed = Deploy-ContainerApp -AppName "rag-chatbot" -ImageName "rag-chatbot" -RegistryName $registryName -ResourceGroup $ResourceGroupName -EnvironmentName $containerAppsEnv
            if ($chatbotDeployed) {
                Setup-EnvironmentVariables -AppName "rag-chatbot" -ResourceGroup $ResourceGroupName
            }
        }
        
        # Deploy Backend API
        if ($backendBuilt) {
            $backendDeployed = Deploy-ContainerApp -AppName "wealtharena-backend" -ImageName "wealtharena-backend" -RegistryName $registryName -ResourceGroup $ResourceGroupName -EnvironmentName $containerAppsEnv
            if ($backendDeployed) {
                Setup-EnvironmentVariables -AppName "wealtharena-backend" -ResourceGroup $ResourceGroupName
            }
        }
    }
    else {
        Write-ColorOutput "Container Apps environment not found. Services will be deployed manually." $Yellow
    }
}
catch {
    Write-ColorOutput "Container Apps environment not available. Services will be deployed manually." $Yellow
}

# Summary
Write-ColorOutput "" $Blue
Write-ColorOutput "Backend Services Deployment Summary:" $Blue
Write-ColorOutput "====================================" $Blue

if ($chatbotBuilt) {
    Write-ColorOutput "RAG Chatbot: BUILT AND PUSHED" $Green
}
else {
    Write-ColorOutput "RAG Chatbot: BUILD FAILED" $Red
}

if ($backendBuilt) {
    Write-ColorOutput "Backend API: BUILT AND PUSHED" $Green
}
else {
    Write-ColorOutput "Backend API: BUILD FAILED" $Red
}

Write-ColorOutput "" $Blue
Write-ColorOutput "Next steps:" $Blue
Write-ColorOutput "1. Update frontend environment variables" $Blue
Write-ColorOutput "2. Run integration tests" $Blue
Write-ColorOutput "3. Test mobile app connectivity" $Blue

Write-ColorOutput "" $Blue
Write-ColorOutput "Backend services deployment complete!" $Blue
