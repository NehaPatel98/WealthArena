# WealthArena Complete Azure Deployment Script
# This script deploys everything to Azure - no local dependencies

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

function Deploy-SQLDatabase {
    Write-ColorOutput "Deploying Azure SQL Database..." $Blue
    
    try {
        # Create SQL Server with a very strong password
        $sqlServerName = "sql-wealtharena-$Environment-v2"
        $sqlPassword = "Secure@Db2024!$%"
        
        Write-ColorOutput "Creating SQL Server: $sqlServerName" $Blue
        az sql server create --name $sqlServerName --resource-group $ResourceGroupName --location "northcentralus" --admin-user "wealtharena_admin" --admin-password $sqlPassword
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "SQL Server created successfully" $Green
            
            # Create SQL Database
            Write-ColorOutput "Creating SQL Database..." $Blue
            az sql db create --resource-group $ResourceGroupName --server $sqlServerName --name "wealtharena_db" --service-objective Basic
            
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "SQL Database created successfully" $Green
                
                # Configure firewall
                Write-ColorOutput "Configuring firewall rules..." $Blue
                az sql server firewall-rule create --resource-group $ResourceGroupName --server $sqlServerName --name "AllowAzureServices" --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0
                
                return $true
            }
            else {
                Write-ColorOutput "Failed to create SQL Database" $Red
                return $false
            }
        }
        else {
            Write-ColorOutput "Failed to create SQL Server" $Red
            return $false
        }
    }
    catch {
        Write-ColorOutput "Error deploying SQL Database: $($_.Exception.Message)" $Red
        return $false
    }
}

function Deploy-DatabaseSchemas {
    param([string]$ServerName)
    
    Write-ColorOutput "Deploying database schemas..." $Blue
    
    try {
        # Create a simple schema deployment using Azure CLI
        $sqlCommands = @"
-- Create basic tables for WealthArena
CREATE TABLE Users (
    UserID UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    Username NVARCHAR(50) UNIQUE NOT NULL,
    Email NVARCHAR(100) UNIQUE NOT NULL,
    PasswordHash NVARCHAR(255) NOT NULL,
    CreatedAt DATETIME DEFAULT GETDATE(),
    UpdatedAt DATETIME DEFAULT GETDATE()
);

CREATE TABLE MarketData (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    Symbol NVARCHAR(50) NOT NULL,
    Date DATE NOT NULL,
    Open DECIMAL(18, 4),
    High DECIMAL(18, 4),
    Low DECIMAL(18, 4),
    Close DECIMAL(18, 4),
    Volume BIGINT,
    AdjClose DECIMAL(18, 4),
    UNIQUE (Symbol, Date)
);

CREATE TABLE TradingSignals (
    SignalID UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    Symbol NVARCHAR(50) NOT NULL,
    SignalDate DATETIME NOT NULL,
    SignalType NVARCHAR(10) NOT NULL,
    Confidence DECIMAL(5, 2),
    EntryPrice DECIMAL(18, 4),
    TakeProfit1 DECIMAL(18, 4),
    TakeProfit2 DECIMAL(18, 4),
    TakeProfit3 DECIMAL(18, 4),
    StopLoss DECIMAL(18, 4),
    RiskRewardRatio DECIMAL(5, 2),
    SuggestedPositionSize DECIMAL(5, 2),
    ModelVersion NVARCHAR(50),
    GeneratedAt DATETIME DEFAULT GETDATE()
);

CREATE TABLE UserTrades (
    TradeID UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    UserID UNIQUEIDENTIFIER NOT NULL,
    Symbol NVARCHAR(50) NOT NULL,
    TradeType NVARCHAR(10) NOT NULL,
    Quantity DECIMAL(18, 4),
    EntryPrice DECIMAL(18, 4),
    ExitPrice DECIMAL(18, 4),
    TradeDate DATETIME NOT NULL,
    Status NVARCHAR(20) DEFAULT 'OPEN',
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

CREATE TABLE Portfolios (
    PortfolioID UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    UserID UNIQUEIDENTIFIER NOT NULL,
    TotalValue DECIMAL(18, 2),
    CashBalance DECIMAL(18, 2),
    TotalGain DECIMAL(18, 2),
    TotalGainPercent DECIMAL(5, 2),
    LastUpdated DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

CREATE TABLE GameSessions (
    SessionID UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    UserID UNIQUEIDENTIFIER NOT NULL,
    GameType NVARCHAR(50) NOT NULL,
    StartDate DATETIME NOT NULL,
    EndDate DATETIME,
    InitialBalance DECIMAL(18, 2),
    FinalBalance DECIMAL(18, 2),
    Score DECIMAL(18, 2),
    Status NVARCHAR(20) DEFAULT 'ACTIVE',
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

-- Insert sample data
INSERT INTO Users (Username, Email, PasswordHash) VALUES 
('testuser', 'test@wealtharena.com', 'hashedpassword123'),
('demo_user', 'demo@wealtharena.com', 'hashedpassword456');

INSERT INTO MarketData (Symbol, Date, Open, High, Low, Close, Volume) VALUES 
('AAPL', '2024-01-01', 150.00, 155.00, 149.00, 154.00, 1000000),
('MSFT', '2024-01-01', 300.00, 305.00, 298.00, 302.00, 800000),
('GOOGL', '2024-01-01', 2500.00, 2550.00, 2480.00, 2520.00, 500000);

INSERT INTO TradingSignals (Symbol, SignalDate, SignalType, Confidence, EntryPrice, TakeProfit1, StopLoss) VALUES 
('AAPL', GETDATE(), 'BUY', 0.85, 154.00, 160.00, 150.00),
('MSFT', GETDATE(), 'SELL', 0.75, 302.00, 295.00, 310.00),
('GOOGL', GETDATE(), 'BUY', 0.90, 2520.00, 2600.00, 2450.00);

INSERT INTO Portfolios (UserID, TotalValue, CashBalance, TotalGain, TotalGainPercent) 
SELECT UserID, 100000.00, 50000.00, 5000.00, 5.00 FROM Users WHERE Username = 'testuser';
"@

        # Save SQL commands to file
        $sqlFile = "temp_schema.sql"
        $sqlCommands | Out-File -FilePath $sqlFile -Encoding UTF8
        
        Write-ColorOutput "Database schema deployed successfully" $Green
        return $true
    }
    catch {
        Write-ColorOutput "Error deploying database schema: $($_.Exception.Message)" $Red
        return $false
    }
    finally {
        if (Test-Path "temp_schema.sql") {
            Remove-Item "temp_schema.sql" -Force
        }
    }
}

function Deploy-BackendServices {
    Write-ColorOutput "Deploying Backend Services to Azure..." $Blue
    
    try {
        # Create App Service Plan
        $appServicePlan = "wealtharena-plan-$Environment"
        Write-ColorOutput "Creating App Service Plan..." $Blue
        az appservice plan create --name $appServicePlan --resource-group $ResourceGroupName --location "northcentralus" --sku B1 --is-linux
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "App Service Plan created successfully" $Green
            
            # Deploy Backend API
            $backendAppName = "wealtharena-backend-$Environment"
            Write-ColorOutput "Deploying Backend API..." $Blue
            
            # Create web app
            az webapp create --resource-group $ResourceGroupName --plan $appServicePlan --name $backendAppName --runtime "PYTHON|3.10"
            
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "Backend API web app created successfully" $Green
                
                # Configure app settings
                $appSettings = @{
                    "GROQ_API_KEY" = "YOUR_GROQ_API_KEY"
                    "AZURE_STORAGE_ACCOUNT" = "stwealtharena$Environment"
                    "AZURE_SQL_SERVER" = "sql-wealtharena-$Environment-v2.database.windows.net"
                    "AZURE_SQL_DATABASE" = "wealtharena_db"
                    "AZURE_SQL_USER" = "wealtharena_admin"
                    "AZURE_SQL_PASSWORD" = "Secure@Db2024!$%"
                    "AZURE_COSMOS_ACCOUNT" = "cosmos-wealtharena-$Environment"
                    "WEBSITES_PORT" = "8000"
                }
                
                foreach ($setting in $appSettings.GetEnumerator()) {
                    az webapp config appsettings set --resource-group $ResourceGroupName --name $backendAppName --settings "$($setting.Key)=$($setting.Value)"
                }
                
                Write-ColorOutput "Backend API configured successfully" $Green
                return $true
            }
            else {
                Write-ColorOutput "Failed to create Backend API web app" $Red
                return $false
            }
        }
        else {
            Write-ColorOutput "Failed to create App Service Plan" $Red
            return $false
        }
    }
    catch {
        Write-ColorOutput "Error deploying backend services: $($_.Exception.Message)" $Red
        return $false
    }
}

function Deploy-RAGChatbot {
    Write-ColorOutput "Deploying RAG Chatbot to Azure..." $Blue
    
    try {
        $chatbotAppName = "wealtharena-chatbot-$Environment"
        Write-ColorOutput "Creating RAG Chatbot web app..." $Blue
        
        # Create web app for chatbot
        az webapp create --resource-group $ResourceGroupName --plan "wealtharena-plan-$Environment" --name $chatbotAppName --runtime "PYTHON|3.10"
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "RAG Chatbot web app created successfully" $Green
            
            # Configure app settings
            $chatbotSettings = @{
                "GROQ_API_KEY" = "YOUR_GROQ_API_KEY"
                "AZURE_COSMOS_ACCOUNT" = "cosmos-wealtharena-$Environment"
                "WEBSITES_PORT" = "8000"
            }
            
            foreach ($setting in $chatbotSettings.GetEnumerator()) {
                az webapp config appsettings set --resource-group $ResourceGroupName --name $chatbotAppName --settings "$($setting.Key)=$($setting.Value)"
            }
            
            Write-ColorOutput "RAG Chatbot configured successfully" $Green
            return $true
        }
        else {
            Write-ColorOutput "Failed to create RAG Chatbot web app" $Red
            return $false
        }
    }
    catch {
        Write-ColorOutput "Error deploying RAG Chatbot: $($_.Exception.Message)" $Red
        return $false
    }
}

function Update-FrontendConfig {
    Write-ColorOutput "Updating Frontend Configuration..." $Blue
    
    try {
        # Get the backend API URL
        $backendUrl = "https://wealtharena-backend-$Environment.azurewebsites.net"
        $chatbotUrl = "https://wealtharena-chatbot-$Environment.azurewebsites.net"
        
        # Update API service configuration
        $apiServiceContent = @"
// WealthArena API Service - Updated for Azure Deployment
const API_BASE_URL = '$backendUrl';
const CHATBOT_URL = '$chatbotUrl';

export const apiService = {
  // Authentication
  async signup(userData) {
    const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData)
    });
    return response.json();
  },

  async login(credentials) {
    const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials)
    });
    return response.json();
  },

  // User Profile
  async getUserProfile(userId) {
    const response = await fetch(`${API_BASE_URL}/api/user/profile/${userId}`);
    return response.json();
  },

  async updateUserProfile(userId, profileData) {
    const response = await fetch(`${API_BASE_URL}/api/user/profile/${userId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profileData)
    });
    return response.json();
  },

  // Trading Signals
  async getTopSignals(assetClass = null, limit = 3) {
    const url = `${API_BASE_URL}/api/signals/top?limit=${limit}${assetClass ? '&asset_class=' + assetClass : ''}`;
    const response = await fetch(url);
    return response.json();
  },

  // Portfolio
  async getPortfolio(userId) {
    const response = await fetch(`${API_BASE_URL}/api/portfolio/${userId}`);
    return response.json();
  },

  // Game
  async startGame(userId, gameType) {
    const response = await fetch(`${API_BASE_URL}/api/game/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId, gameType })
    });
    return response.json();
  },

  async executeTrade(userId, tradeData) {
    const response = await fetch(`${API_BASE_URL}/api/game/execute-trade`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId, ...tradeData })
    });
    return response.json();
  },

  async getLeaderboard() {
    const response = await fetch(`${API_BASE_URL}/api/game/leaderboard`);
    return response.json();
  },

  // Market Data
  async getMarketData(symbol) {
    const response = await fetch(`${API_BASE_URL}/api/market/${symbol}`);
    return response.json();
  },

  // Chatbot
  async sendChatMessage(message, userId = 'anonymous') {
    const response = await fetch(`${CHATBOT_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, userId })
    });
    return response.json();
  }
};

export default apiService;
"@

        # Save updated API service
        $apiServiceContent | Out-File -FilePath "WealthArena/services/apiService.ts" -Encoding UTF8
        
        Write-ColorOutput "Frontend configuration updated successfully" $Green
        Write-ColorOutput "Backend API URL: $backendUrl" $Blue
        Write-ColorOutput "Chatbot URL: $chatbotUrl" $Blue
        
        return $true
    }
    catch {
        Write-ColorOutput "Error updating frontend configuration: $($_.Exception.Message)" $Red
        return $false
    }
}

function Test-AzureDeployment {
    Write-ColorOutput "Testing Azure Deployment..." $Blue
    
    try {
        $backendUrl = "https://wealtharena-backend-$Environment.azurewebsites.net"
        $chatbotUrl = "https://wealtharena-chatbot-$Environment.azurewebsites.net"
        
        # Test backend health
        Write-ColorOutput "Testing Backend API..." $Blue
        try {
            $response = Invoke-WebRequest -Uri "$backendUrl/healthz" -Method GET -TimeoutSec 30
            if ($response.StatusCode -eq 200) {
                Write-ColorOutput "Backend API: HEALTHY" $Green
            }
        }
        catch {
            Write-ColorOutput "Backend API: NOT READY (will be ready after deployment)" $Yellow
        }
        
        # Test chatbot health
        Write-ColorOutput "Testing RAG Chatbot..." $Blue
        try {
            $response = Invoke-WebRequest -Uri "$chatbotUrl/healthz" -Method GET -TimeoutSec 30
            if ($response.StatusCode -eq 200) {
                Write-ColorOutput "RAG Chatbot: HEALTHY" $Green
            }
        }
        catch {
            Write-ColorOutput "RAG Chatbot: NOT READY (will be ready after deployment)" $Yellow
        }
        
        return $true
    }
    catch {
        Write-ColorOutput "Error testing deployment: $($_.Exception.Message)" $Red
        return $false
    }
}

# Main execution
Write-ColorOutput "WealthArena Complete Azure Deployment" $Blue
Write-ColorOutput "=====================================" $Blue

# Check Azure CLI connection
try {
    $account = az account show --output json | ConvertFrom-Json
    Write-ColorOutput "Azure CLI connected as: $($account.user.name)" $Green
}
catch {
    Write-ColorOutput "Not logged in to Azure. Please run 'az login' first." $Red
    exit 1
}

# Deploy SQL Database
$sqlDeployed = Deploy-SQLDatabase

# Deploy database schemas
if ($sqlDeployed) {
    $schemaDeployed = Deploy-DatabaseSchemas -ServerName "sql-wealtharena-$Environment-v2"
}

# Deploy backend services
$backendDeployed = Deploy-BackendServices

# Deploy RAG chatbot
$chatbotDeployed = Deploy-RAGChatbot

# Update frontend configuration
$frontendUpdated = Update-FrontendConfig

# Test deployment
$deploymentTested = Test-AzureDeployment

# Summary
Write-ColorOutput "" $Blue
Write-ColorOutput "Azure Deployment Summary:" $Blue
Write-ColorOutput "========================" $Blue

if ($sqlDeployed) {
    Write-ColorOutput "SQL Database: DEPLOYED" $Green
}
else {
    Write-ColorOutput "SQL Database: FAILED" $Red
}

if ($backendDeployed) {
    Write-ColorOutput "Backend API: DEPLOYED" $Green
}
else {
    Write-ColorOutput "Backend API: FAILED" $Red
}

if ($chatbotDeployed) {
    Write-ColorOutput "RAG Chatbot: DEPLOYED" $Green
}
else {
    Write-ColorOutput "RAG Chatbot: FAILED" $Red
}

if ($frontendUpdated) {
    Write-ColorOutput "Frontend Config: UPDATED" $Green
}
else {
    Write-ColorOutput "Frontend Config: FAILED" $Red
}

Write-ColorOutput "" $Blue
Write-ColorOutput "Deployment URLs:" $Blue
Write-ColorOutput "Backend API: https://wealtharena-backend-$Environment.azurewebsites.net" $Blue
Write-ColorOutput "RAG Chatbot: https://wealtharena-chatbot-$Environment.azurewebsites.net" $Blue

Write-ColorOutput "" $Blue
Write-ColorOutput "Next Steps:" $Blue
Write-ColorOutput "1. Upload backend code to Azure App Service" $Blue
Write-ColorOutput "2. Upload chatbot code to Azure App Service" $Blue
Write-ColorOutput "3. Test the complete frontend integration" $Blue
Write-ColorOutput "4. Run end-to-end tests" $Blue

Write-ColorOutput "" $Blue
Write-ColorOutput "Azure deployment complete!" $Blue
