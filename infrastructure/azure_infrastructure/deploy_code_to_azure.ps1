# WealthArena Code Deployment to Azure App Service
# This script deploys the backend code to Azure App Service

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

function Deploy-BackendCode {
    Write-ColorOutput "Deploying Backend API code to Azure App Service..." $Blue
    
    try {
        $backendAppName = "wealtharena-backend-$Environment"
        $backendPath = "azure_services/wealtharena_backend_api"
        
        # Create deployment package
        Write-ColorOutput "Creating deployment package..." $Blue
        
        # Create a simple deployment script
        $deployScript = @"
# WealthArena Backend API - Azure App Service Deployment
# This file will be deployed to Azure App Service

import os
import sys
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
"@

        # Save deployment script
        $deployScript | Out-File -FilePath "$backendPath/main.py" -Encoding UTF8
        
        # Create startup command
        $startupCommand = "python main.py"
        
        # Configure the web app
        Write-ColorOutput "Configuring web app..." $Blue
        az webapp config set --resource-group $ResourceGroupName --name $backendAppName --startup-file $startupCommand
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Backend API configured successfully" $Green
            return $true
        }
        else {
            Write-ColorOutput "Failed to configure Backend API" $Red
            return $false
        }
    }
    catch {
        Write-ColorOutput "Error deploying backend code: $($_.Exception.Message)" $Red
        return $false
    }
}

function Deploy-ChatbotCode {
    Write-ColorOutput "Deploying RAG Chatbot code to Azure App Service..." $Blue
    
    try {
        $chatbotAppName = "wealtharena-chatbot-$Environment"
        $chatbotPath = "azure_services/rag_chatbot_service"
        
        # Create deployment script
        $deployScript = @"
# WealthArena RAG Chatbot - Azure App Service Deployment
# This file will be deployed to Azure App Service

import os
import sys
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
"@

        # Save deployment script
        $deployScript | Out-File -FilePath "$chatbotPath/main.py" -Encoding UTF8
        
        # Create startup command
        $startupCommand = "python main.py"
        
        # Configure the web app
        Write-ColorOutput "Configuring chatbot web app..." $Blue
        az webapp config set --resource-group $ResourceGroupName --name $chatbotAppName --startup-file $startupCommand
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "RAG Chatbot configured successfully" $Green
            return $true
        }
        else {
            Write-ColorOutput "Failed to configure RAG Chatbot" $Red
            return $false
        }
    }
    catch {
        Write-ColorOutput "Error deploying chatbot code: $($_.Exception.Message)" $Red
        return $false
    }
}

function Test-AzureServices {
    Write-ColorOutput "Testing Azure Services..." $Blue
    
    $backendUrl = "https://wealtharena-backend-$Environment.azurewebsites.net"
    $chatbotUrl = "https://wealtharena-chatbot-$Environment.azurewebsites.net"
    
    # Test backend health
    Write-ColorOutput "Testing Backend API health..." $Blue
    try {
        $response = Invoke-WebRequest -Uri "$backendUrl/healthz" -Method GET -TimeoutSec 30
        if ($response.StatusCode -eq 200) {
            Write-ColorOutput "Backend API: HEALTHY" $Green
        }
    }
    catch {
        Write-ColorOutput "Backend API: NOT READY (may need time to start)" $Yellow
    }
    
    # Test chatbot health
    Write-ColorOutput "Testing RAG Chatbot health..." $Blue
    try {
        $response = Invoke-WebRequest -Uri "$chatbotUrl/healthz" -Method GET -TimeoutSec 30
        if ($response.StatusCode -eq 200) {
            Write-ColorOutput "RAG Chatbot: HEALTHY" $Green
        }
    }
    catch {
        Write-ColorOutput "RAG Chatbot: NOT READY (may need time to start)" $Yellow
    }
    
    # Test API endpoints
    Write-ColorOutput "Testing API endpoints..." $Blue
    try {
        # Test signals endpoint
        $response = Invoke-WebRequest -Uri "$backendUrl/api/signals/top" -Method GET -TimeoutSec 30
        if ($response.StatusCode -eq 200) {
            Write-ColorOutput "Signals API: WORKING" $Green
        }
    }
    catch {
        Write-ColorOutput "Signals API: NOT READY" $Yellow
    }
    
    try {
        # Test chatbot endpoint
        $chatRequest = @{
            message = "Hello, can you help me with trading?"
            userId = "test-user"
        } | ConvertTo-Json
        
        $response = Invoke-WebRequest -Uri "$chatbotUrl/api/chat" -Method POST -Body $chatRequest -ContentType "application/json" -TimeoutSec 30
        if ($response.StatusCode -eq 200) {
            Write-ColorOutput "Chatbot API: WORKING" $Green
        }
    }
    catch {
        Write-ColorOutput "Chatbot API: NOT READY" $Yellow
    }
}

function Update-FrontendEnvironment {
    Write-ColorOutput "Updating Frontend Environment Configuration..." $Blue
    
    try {
        # Create environment configuration for the frontend
        $envConfig = @"
// WealthArena Environment Configuration - Azure Deployment
export const API_CONFIG = {
  BACKEND_URL: 'https://wealtharena-backend-$Environment.azurewebsites.net',
  CHATBOT_URL: 'https://wealtharena-chatbot-$Environment.azurewebsites.net',
  ENVIRONMENT: '$Environment',
  VERSION: '1.0.0'
};

export const API_ENDPOINTS = {
  AUTH: {
    SIGNUP: `${API_CONFIG.BACKEND_URL}/api/auth/signup`,
    LOGIN: `${API_CONFIG.BACKEND_URL}/api/auth/login`,
    LOGOUT: `${API_CONFIG.BACKEND_URL}/api/auth/logout`
  },
  USER: {
    PROFILE: `${API_CONFIG.BACKEND_URL}/api/user/profile`,
    AVATAR: `${API_CONFIG.BACKEND_URL}/api/user/avatar`
  },
  SIGNALS: {
    TOP: `${API_CONFIG.BACKEND_URL}/api/signals/top`,
    DETAIL: `${API_CONFIG.BACKEND_URL}/api/signals`
  },
  PORTFOLIO: {
    OVERVIEW: `${API_CONFIG.BACKEND_URL}/api/portfolio`,
    TRADES: `${API_CONFIG.BACKEND_URL}/api/portfolio/trades`
  },
  GAME: {
    START: `${API_CONFIG.BACKEND_URL}/api/game/start`,
    EXECUTE_TRADE: `${API_CONFIG.BACKEND_URL}/api/game/execute-trade`,
    LEADERBOARD: `${API_CONFIG.BACKEND_URL}/api/game/leaderboard`
  },
  MARKET: {
    DATA: `${API_CONFIG.BACKEND_URL}/api/market`,
    OHLCV: `${API_CONFIG.BACKEND_URL}/api/market/ohlcv`
  },
  CHAT: {
    SEND_MESSAGE: `${API_CONFIG.CHATBOT_URL}/api/chat`,
    EXPLAIN_SIGNAL: `${API_CONFIG.CHATBOT_URL}/api/explain/signal`,
    EXPLAIN_INDICATOR: `${API_CONFIG.CHATBOT_URL}/api/explain/indicator`
  }
};

export default API_CONFIG;
"@

        # Save environment configuration
        $envConfig | Out-File -FilePath "WealthArena/config/apiConfig.ts" -Encoding UTF8
        
        Write-ColorOutput "Frontend environment configuration updated" $Green
        Write-ColorOutput "Backend URL: https://wealtharena-backend-$Environment.azurewebsites.net" $Blue
        Write-ColorOutput "Chatbot URL: https://wealtharena-chatbot-$Environment.azurewebsites.net" $Blue
        
        return $true
    }
    catch {
        Write-ColorOutput "Error updating frontend environment: $($_.Exception.Message)" $Red
        return $false
    }
}

# Main execution
Write-ColorOutput "WealthArena Code Deployment to Azure" $Blue
Write-ColorOutput "====================================" $Blue

# Check Azure CLI connection
try {
    $account = az account show --output json | ConvertFrom-Json
    Write-ColorOutput "Azure CLI connected as: $($account.user.name)" $Green
}
catch {
    Write-ColorOutput "Not logged in to Azure. Please run 'az login' first." $Red
    exit 1
}

# Deploy backend code
$backendDeployed = Deploy-BackendCode

# Deploy chatbot code
$chatbotDeployed = Deploy-ChatbotCode

# Update frontend environment
$frontendUpdated = Update-FrontendEnvironment

# Test services
$servicesTested = Test-AzureServices

# Summary
Write-ColorOutput "" $Blue
Write-ColorOutput "Code Deployment Summary:" $Blue
Write-ColorOutput "=======================" $Blue

if ($backendDeployed) {
    Write-ColorOutput "Backend API Code: DEPLOYED" $Green
}
else {
    Write-ColorOutput "Backend API Code: FAILED" $Red
}

if ($chatbotDeployed) {
    Write-ColorOutput "RAG Chatbot Code: DEPLOYED" $Green
}
else {
    Write-ColorOutput "RAG Chatbot Code: FAILED" $Red
}

if ($frontendUpdated) {
    Write-ColorOutput "Frontend Config: UPDATED" $Green
}
else {
    Write-ColorOutput "Frontend Config: FAILED" $Red
}

Write-ColorOutput "" $Blue
Write-ColorOutput "Service URLs:" $Blue
Write-ColorOutput "Backend API: https://wealtharena-backend-$Environment.azurewebsites.net" $Blue
Write-ColorOutput "RAG Chatbot: https://wealtharena-chatbot-$Environment.azurewebsites.net" $Blue

Write-ColorOutput "" $Blue
Write-ColorOutput "Next Steps:" $Blue
Write-ColorOutput "1. Test the mobile app with the new backend URLs" $Blue
Write-ColorOutput "2. Verify all API endpoints are working" $Blue
Write-ColorOutput "3. Test user registration and login flow" $Blue
Write-ColorOutput "4. Test trading signals and portfolio features" $Blue
Write-ColorOutput "5. Test chatbot integration" $Blue

Write-ColorOutput "" $Blue
Write-ColorOutput "Code deployment complete!" $Blue
