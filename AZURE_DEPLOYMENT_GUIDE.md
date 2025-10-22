# WealthArena Azure Deployment Guide
**Complete Step-by-Step Cloud Deployment**

---

## 📊 Current System Inventory

### Data Symbols Currently Tracked

#### ✅ **Currently Processed (32 symbols)**
- **US Stocks**: 2 symbols
  - AAPL (Apple)
  - GOOGL (Google)

- **ASX Stocks**: 30 symbols
  - Major Banks: ANZ, CBA, NAB, WBC
  - Mining: BHP, RIO, FMG, S32
  - Retail: WOW, WES, COL, COH, HVN, JBH
  - Tech: XRO, REA
  - Energy: WDS, STO, ORG
  - Telecom: TLS, TCL
  - Healthcare: CSL
  - Financials: MQG
  - Others: AGL, VCX, NXT, BXB, SCG

#### 📈 **Available for Expansion (500+ symbols)**

**ASX Stocks**: 300+ symbols available
- Large Cap: Top 50 ASX 200
- Mid Cap: ASX 200 (51-200)
- Small Cap: 100+ additional symbols
- Sectors: Banks, Mining, Energy, Healthcare, Tech, REITs, Industrials, Consumer, Telecom, Financials, Utilities

**Cryptocurrencies**: 20 major coins
- Major: BTC, ETH, BNB, XRP, ADA, SOL
- DeFi: UNI, AAVE, COMP, MKR
- Layer 1: DOT, AVAX, ATOM, ALGO
- Meme: DOGE, SHIB

**Currency Pairs**: 28 pairs
- Major: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- Minor: EURGBP, EURJPY, GBPJPY, AUDJ PY, etc.
- Exotic: USDTRY, USDZAR, USDMXN, etc.

**Commodities**: 30+ commodities
- Precious Metals: Gold, Silver, Platinum, Palladium
- Energy: Crude Oil, Brent Oil, Natural Gas
- Agricultural: Wheat, Corn, Soybeans, Coffee, Sugar, Cotton
- Industrial Metals: Copper, Aluminum, Zinc, Nickel

---

## 🏗️ Azure Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AZURE CLOUD INFRASTRUCTURE                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐        ┌──────────────────────┐
│   Azure Container    │        │   Azure SQL Database │
│   (Backend FastAPI)  │◄──────►│   (Market Data +     │
│   Port: 8000         │        │    Predictions)      │
└──────────────────────┘        └──────────────────────┘
           ▲                              ▲
           │                              │
           ▼                              │
┌──────────────────────┐                  │
│   Azure Container    │                  │
│   (Airflow DAGs)     │──────────────────┘
│   Data Pipeline      │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│   Azure Blob Storage │
│   (Model Checkpoints,│
│    CSV Files)        │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│   Azure Static       │
│   Web Apps           │
│   (Frontend - React) │
└──────────────────────┘
```

---

## 📋 Step-by-Step Deployment Process

### Phase 1: Azure Setup & Resource Creation

#### Step 1.1: Create Azure Account & Resource Group
```bash
# Login to Azure
az login

# Create resource group
az group create \
  --name wealtharena-rg \
  --location australiaeast

# Expected output: Resource group created successfully
```

#### Step 1.2: Create Azure SQL Database
```bash
# Create SQL Server
az sql server create \
  --name wealtharena-sql-server \
  --resource-group wealtharena-rg \
  --location australiaeast \
  --admin-user sqladmin \
  --admin-password <YourStrongPassword123!>

# Create SQL Database
az sql db create \
  --resource-group wealtharena-rg \
  --server wealtharena-sql-server \
  --name wealtharena_db \
  --service-objective S1 \
  --max-size 10GB

# Configure firewall (allow Azure services)
az sql server firewall-rule create \
  --resource-group wealtharena-rg \
  --server wealtharena-sql-server \
  --name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

# Allow your IP for management
az sql server firewall-rule create \
  --resource-group wealtharena-rg \
  --server wealtharena-sql-server \
  --name AllowMyIP \
  --start-ip-address <YOUR_PUBLIC_IP> \
  --end-ip-address <YOUR_PUBLIC_IP>
```

**📝 Save This Connection String:**
```
Server=tcp:wealtharena-sql-server.database.windows.net,1433;
Initial Catalog=wealtharena_db;
Persist Security Info=False;
User ID=sqladmin;
Password=<YourStrongPassword123!>;
MultipleActiveResultSets=False;
Encrypt=True;
TrustServerCertificate=False;
Connection Timeout=30;
```

#### Step 1.3: Setup Database Schema
```bash
# Connect to Azure SQL and run schema
# Option 1: Using Azure Data Studio or SSMS
# - Open wealtharena_rl/backend/azure_sql_schema.sql
# - Connect to: wealtharena-sql-server.database.windows.net
# - Run the entire script

# Option 2: Using sqlcmd
sqlcmd -S wealtharena-sql-server.database.windows.net \
  -d wealtharena_db \
  -U sqladmin \
  -P <YourStrongPassword123!> \
  -i wealtharena_rl/backend/azure_sql_schema.sql
```

**✅ Expected Database Tables Created:**
1. `raw_market_data` - OHLCV data (Open, High, Low, Close, Volume)
2. `processed_features` - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. `model_predictions` - Predictions with TP/SL
4. `portfolio_state` - User positions
5. `game_leaderboard` - Rankings
6. `game_sessions` - Game data
7. `model_registry` - ML models metadata
8. `audit_log` - System logs

---

### Phase 2: Azure Blob Storage for Model Checkpoints

#### Step 2.1: Create Storage Account
```bash
# Create storage account
az storage account create \
  --name wealtharenastorage \
  --resource-group wealtharena-rg \
  --location australiaeast \
  --sku Standard_LRS

# Create blob container for models
az storage container create \
  --account-name wealtharenastorage \
  --name model-checkpoints \
  --auth-mode login

# Create blob container for data
az storage container create \
  --account-name wealtharenastorage \
  --name market-data \
  --auth-mode login

# Get storage connection string
az storage account show-connection-string \
  --name wealtharenastorage \
  --resource-group wealtharena-rg \
  --output tsv
```

**📝 Save Storage Connection String:**
```
DefaultEndpointsProtocol=https;
AccountName=wealtharenastorage;
AccountKey=<YOUR_KEY>;
EndpointSuffix=core.windows.net
```

#### Step 2.2: Upload Model Checkpoints
```bash
# Upload model checkpoints
az storage blob upload-batch \
  --account-name wealtharenastorage \
  --destination model-checkpoints \
  --source wealtharena_rl/checkpoints/

# Expected upload:
# - asx_stocks/ (13 files: 6 .pkl, 4 .json, 3 .pt)
# - commodities/ (13 files)
# - cryptocurrencies/ (13 files)
# - currency_pairs/ (21 files)
# - etf/ (13 files)
```

---

### Phase 3: Deploy Backend API (FastAPI)

#### Step 3.1: Create Dockerfile for Backend
```dockerfile
# Create: wealtharena_rl/backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc-dev \
    curl

# Install ODBC Driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 3.2: Create requirements.txt for Backend
```txt
# Create: wealtharena_rl/backend/requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pandas==2.1.3
numpy==1.24.3
pyodbc==5.0.1
sqlalchemy==2.0.23
python-dotenv==1.0.0
httpx==0.25.1
```

#### Step 3.3: Create Azure Container Registry (ACR)
```bash
# Create ACR
az acr create \
  --resource-group wealtharena-rg \
  --name wealtharenaacr \
  --sku Basic

# Login to ACR
az acr login --name wealtharenaacr

# Build and push backend image
cd wealtharena_rl/backend
az acr build \
  --registry wealtharenaacr \
  --image wealtharena-backend:v1 \
  .
```

#### Step 3.4: Deploy to Azure Container Instances (ACI)
```bash
# Create container instance
az container create \
  --resource-group wealtharena-rg \
  --name wealtharena-backend \
  --image wealtharenaacr.azurecr.io/wealtharena-backend:v1 \
  --cpu 2 \
  --memory 4 \
  --registry-login-server wealtharenaacr.azurecr.io \
  --registry-username <ACR_USERNAME> \
  --registry-password <ACR_PASSWORD> \
  --dns-name-label wealtharena-api \
  --ports 8000 \
  --environment-variables \
    AZURE_SQL_CONNECTION_STRING='<YOUR_SQL_CONNECTION_STRING>' \
    AZURE_STORAGE_CONNECTION_STRING='<YOUR_STORAGE_CONNECTION_STRING>'

# Get public IP
az container show \
  --resource-group wealtharena-rg \
  --name wealtharena-backend \
  --query ipAddress.fqdn \
  --output tsv

# Expected output: wealtharena-api.australiaeast.azurecontainer.io
```

**✅ Backend API Now Available At:**
```
http://wealtharena-api.australiaeast.azurecontainer.io:8000
```

**Test It:**
```bash
# Health check
curl http://wealtharena-api.australiaeast.azurecontainer.io:8000/health

# Get prediction
curl -X POST http://wealtharena-api.australiaeast.azurecontainer.io:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizon": 1}'
```

---

### Phase 4: Deploy Airflow Data Pipeline

#### Step 4.1: Create Dockerfile for Airflow
```dockerfile
# Create: wealtharena_rl/airflow/Dockerfile
FROM apache/airflow:2.7.3-python3.10

USER root
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc-dev

USER airflow

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy DAGs
COPY dags/ ${AIRFLOW_HOME}/dags/
```

#### Step 4.2: Deploy Airflow to Azure Container Instances
```bash
# Build and push Airflow image
cd wealtharena_rl/airflow
az acr build \
  --registry wealtharenaacr \
  --image wealtharena-airflow:v1 \
  .

# Create Airflow container
az container create \
  --resource-group wealtharena-rg \
  --name wealtharena-airflow \
  --image wealtharenaacr.azurecr.io/wealtharena-airflow:v1 \
  --cpu 2 \
  --memory 4 \
  --registry-login-server wealtharenaacr.azurecr.io \
  --registry-username <ACR_USERNAME> \
  --registry-password <ACR_PASSWORD> \
  --dns-name-label wealtharena-airflow \
  --ports 8080 \
  --environment-variables \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__CORE__SQL_ALCHEMY_CONN='<AIRFLOW_DB_CONNECTION>' \
    AZURE_SQL_CONNECTION_STRING='<YOUR_SQL_CONNECTION_STRING>' \
    AZURE_STORAGE_CONNECTION_STRING='<YOUR_STORAGE_CONNECTION_STRING>'
```

**✅ Airflow UI Available At:**
```
http://wealtharena-airflow.australiaeast.azurecontainer.io:8080
Username: admin
Password: admin (change this!)
```

---

### Phase 5: Deploy Frontend (React)

#### Step 5.1: Build Frontend
```bash
# Navigate to frontend directory
cd wealtharena_ui

# Install dependencies
npm install

# Update API endpoint in .env
echo "REACT_APP_API_URL=http://wealtharena-api.australiaeast.azurecontainer.io:8000" > .env

# Build for production
npm run build
```

#### Step 5.2: Deploy to Azure Static Web Apps
```bash
# Create Static Web App
az staticwebapp create \
  --name wealtharena-frontend \
  --resource-group wealtharena-rg \
  --location australiaeast

# Deploy build
az staticwebapp deploy \
  --name wealtharena-frontend \
  --resource-group wealtharena-rg \
  --app-location ./build \
  --output-location ./build

# Get URL
az staticwebapp show \
  --name wealtharena-frontend \
  --resource-group wealtharena-rg \
  --query defaultHostname \
  --output tsv

# Expected output: wealtharena-frontend.azurestaticapps.net
```

**✅ Frontend Now Available At:**
```
https://wealtharena-frontend.azurestaticapps.net
```

---

## 📊 Database Schema & Data Expectations

### What Data the System Expects

#### 1. **Market Data Ingestion** (`raw_market_data` table)

**Required Fields:**
```sql
INSERT INTO raw_market_data (
    symbol,
    asset_type,
    date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    created_at,
    updated_at
) VALUES (
    'AAPL',                    -- Symbol name
    'stock',                   -- Asset type: 'stock', 'crypto', 'currency_pair', 'commodity'
    '2025-10-08',              -- Trading date
    178.20,                    -- Open price
    179.50,                    -- High price
    177.80,                    -- Low price
    178.45,                    -- Close price
    52847392,                  -- Volume
    GETDATE(),                 -- Created timestamp
    GETDATE()                  -- Updated timestamp
);
```

**Daily Ingestion Volume:**
- **32 symbols** (current) = 32 rows/day
- **500 symbols** (expanded) = 500 rows/day
- **Annual**: ~125,000 rows (500 symbols × 250 trading days)

**Storage Requirements:**
- Per row: ~100 bytes
- Annual (500 symbols): ~12.5 MB raw data
- With indexes: ~50 MB

---

#### 2. **Feature Engineering** (`processed_features` table)

**What Gets Calculated:**
```sql
INSERT INTO processed_features (
    symbol, date,
    
    -- Price Features (4 columns)
    returns, log_returns,
    volatility_5, volatility_20,
    
    -- Moving Averages (8 columns)
    sma_5, sma_10, sma_20, sma_50, sma_200,
    ema_12, ema_26, ema_50,
    
    -- Momentum Indicators (9 columns)
    rsi, rsi_6, rsi_21,
    macd, macd_signal, macd_hist,
    momentum_5, momentum_10, momentum_20,
    
    -- Bollinger Bands (5 columns)
    bb_upper, bb_middle, bb_lower,
    bb_width, bb_position,
    
    -- Volume Indicators (3 columns)
    volume_sma_20, volume_ratio, obv,
    
    -- Support/Resistance (4 columns)
    atr, support_20, resistance_20, price_position,
    
    processed_at
) VALUES (...);
```

**Total Features per Symbol per Day: 33 columns**

**Calculation Requirements:**
- **Minimum History**: 200 days (for SMA-200)
- **Processing Time**: ~100ms per symbol
- **Total Processing**: 32 symbols × 100ms = 3.2 seconds/day

**Storage Requirements:**
- Per row: ~300 bytes (33 columns)
- Annual (500 symbols): ~37.5 MB
- With indexes: ~150 MB

---

#### 3. **Model Predictions** (`model_predictions` table)

**What RL Models Generate:**
```sql
INSERT INTO model_predictions (
    symbol, asset_type, prediction_date,
    
    -- Signal & Confidence
    signal,                              -- 'BUY', 'SELL', 'HOLD'
    confidence,                          -- 0.87
    
    -- Entry Strategy (from Trading Agent)
    entry_price,                         -- 178.45
    entry_price_min, entry_price_max,    -- 177.91, 178.99
    entry_timing,                        -- 'immediate'
    
    -- Take Profit Levels (from Risk Management Agent)
    tp1_price, tp1_percent, tp1_close_percent, tp1_probability,   -- TP1
    tp2_price, tp2_percent, tp2_close_percent, tp2_probability,   -- TP2
    tp3_price, tp3_percent, tp3_close_percent, tp3_probability,   -- TP3
    
    -- Stop Loss (from Risk Management Agent)
    sl_price, sl_percent, sl_type, sl_trail_amount,
    
    -- Risk Metrics
    risk_reward_ratio, max_risk_per_share, max_reward_per_share,
    win_probability, expected_value,
    
    -- Position Sizing (from Portfolio Manager Agent)
    recommended_position_percent, recommended_dollar_amount, max_risk_percent,
    
    -- Metadata
    model_version, model_type, reasoning, ranking_score
) VALUES (...);
```

**Generation Frequency:**
- **On-Demand**: When user requests prediction
- **Batch**: Daily for all symbols (optional)
- **Cache**: 5 minutes

**Storage Requirements:**
- Per prediction: ~500 bytes (40+ columns)
- Daily batch (500 symbols): ~250 KB
- Annual: ~62.5 MB
- With indexes: ~250 MB

---

### Data Flow Timeline (Daily)

```
┌─────────────────────────────────────────────────────────────┐
│                    DAILY DATA CYCLE                         │
└─────────────────────────────────────────────────────────────┘

04:00 PM (16:00) ─────▶ Market Close (US/ASX)

06:00 PM (18:00) ─────▶ Airflow DAG: Fetch Market Data
                        ├── Download OHLCV from Yahoo Finance
                        ├── Insert into raw_market_data table
                        ├── 32 symbols × 5 seconds = ~3 minutes
                        └── INSERT: 32 rows into raw_market_data

07:00 PM (19:00) ─────▶ Airflow DAG: Process Features
                        ├── Read from raw_market_data
                        ├── Calculate 33 features per symbol
                        ├── Technical indicators (RSI, MACD, BB, etc.)
                        ├── 32 symbols × 100ms = ~3 seconds
                        └── INSERT: 32 rows into processed_features

07:30 PM (19:30) ─────▶ Data Ready for Predictions
                        └── Backend API can now generate predictions

08:00 PM - Next Day ──▶ Real-time Predictions
                        ├── User requests prediction
                        ├── Model loads from checkpoints
                        ├── Generates TP/SL levels
                        ├── INSERT into model_predictions
                        └── Returns to frontend
```

---

### Database Growth Projections

#### **Current System (32 symbols)**
| Table | Rows/Day | Rows/Year | Size/Year |
|-------|----------|-----------|-----------|
| raw_market_data | 32 | 8,000 | 800 KB |
| processed_features | 32 | 8,000 | 2.4 MB |
| model_predictions | 32 | 8,000 | 4 MB |
| **Total** | **96** | **24,000** | **~7 MB** |

#### **Expanded System (500 symbols)**
| Table | Rows/Day | Rows/Year | Size/Year |
|-------|----------|-----------|-----------|
| raw_market_data | 500 | 125,000 | 12.5 MB |
| processed_features | 500 | 125,000 | 37.5 MB |
| model_predictions | 500 | 125,000 | 62.5 MB |
| **Total** | **1,500** | **375,000** | **~112 MB** |

#### **5-Year Projection (500 symbols)**
- Raw Data: 62.5 MB
- Features: 187.5 MB
- Predictions: 312.5 MB
- Indexes: ~200 MB
- **Total: ~762 MB** (well within Azure SQL 10GB limit)

---

## 🔧 Environment Variables Configuration

### Backend (.env)
```bash
# Azure SQL Database
AZURE_SQL_SERVER=wealtharena-sql-server.database.windows.net
AZURE_SQL_DATABASE=wealtharena_db
AZURE_SQL_USERNAME=sqladmin
AZURE_SQL_PASSWORD=<YourStrongPassword123!>
AZURE_SQL_CONNECTION_STRING=<FULL_CONNECTION_STRING>

# Azure Blob Storage
AZURE_STORAGE_ACCOUNT=wealtharenastorage
AZURE_STORAGE_KEY=<YOUR_STORAGE_KEY>
AZURE_STORAGE_CONNECTION_STRING=<FULL_STORAGE_STRING>
AZURE_MODEL_CONTAINER=model-checkpoints
AZURE_DATA_CONTAINER=market-data

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
API_WORKERS=4
CACHE_TIMEOUT=300  # 5 minutes

# Model Configuration
MODEL_VERSION=v2.3.1
MODEL_PATH=/app/checkpoints
```

### Airflow (.env)
```bash
# Airflow Configuration
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__SQL_ALCHEMY_CONN=<AIRFLOW_DB_CONNECTION>
AIRFLOW__CORE__LOAD_EXAMPLES=False

# Azure connections (same as backend)
AZURE_SQL_CONNECTION_STRING=<FULL_CONNECTION_STRING>
AZURE_STORAGE_CONNECTION_STRING=<FULL_STORAGE_STRING>

# Data Pipeline Settings
FETCH_SCHEDULE=0 18 * * 1-5  # 6 PM weekdays
PROCESS_SCHEDULE=0 19 * * 1-5  # 7 PM weekdays
DATA_RETENTION_DAYS=365
```

### Frontend (.env)
```bash
# API Endpoint
REACT_APP_API_URL=http://wealtharena-api.australiaeast.azurecontainer.io:8000

# Feature Flags
REACT_APP_ENABLE_CHAT=true
REACT_APP_ENABLE_GAME=true
REACT_APP_REFRESH_INTERVAL=30000  # 30 seconds
```

---

## ✅ Deployment Verification Checklist

### Database Verification
```sql
-- 1. Check tables exist
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES;
-- Expected: 8 tables

-- 2. Check raw_market_data
SELECT COUNT(*), MIN(date), MAX(date) FROM raw_market_data;
-- Expected: 32+ rows

-- 3. Check processed_features
SELECT COUNT(*), AVG(rsi), AVG(macd) FROM processed_features;
-- Expected: 32+ rows, RSI 30-70, MACD varies

-- 4. Check model_predictions
SELECT symbol, signal, confidence, risk_reward_ratio 
FROM model_predictions 
ORDER BY ranking_score DESC 
LIMIT 10;
-- Expected: Top predictions with TP/SL
```

### API Verification
```bash
# 1. Health check
curl http://wealtharena-api.australiaeast.azurecontainer.io:8000/health
# Expected: {"status": "healthy"}

# 2. Get market data
curl -X POST http://wealtharena-api.australiaeast.azurecontainer.io:8000/api/market-data \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "days": 30}'
# Expected: OHLCV data

# 3. Get prediction
curl -X POST http://wealtharena-api.australiaeast.azurecontainer.io:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizon": 1}'
# Expected: Prediction with TP/SL

# 4. Get top setups
curl -X POST http://wealtharena-api.australiaeast.azurecontainer.io:8000/api/top-setups \
  -H "Content-Type: application/json" \
  -d '{"asset_type": "stocks", "count": 3}'
# Expected: Top 3 ranked setups
```

### Airflow Verification
```bash
# 1. Access Airflow UI
open http://wealtharena-airflow.australiaeast.azurecontainer.io:8080

# 2. Check DAGs are loaded
# Expected: fetch_market_data, preprocess_market_data

# 3. Trigger manual run
# In UI: Click "Trigger DAG" on fetch_market_data
# Expected: Success status after ~5 minutes
```

### Frontend Verification
```bash
# 1. Access frontend
open https://wealtharena-frontend.azurestaticapps.net

# 2. Check pages load
# - Dashboard: Top 3 setups visible
# - Trading Signals: Chart with TP/SL lines
# - Portfolio: Holdings displayed
# - Leaderboard: Rankings shown
```

---

## 🔐 Security Configuration

### Azure SQL Firewall Rules
```bash
# 1. Restrict to Azure services only
az sql server firewall-rule update \
  --resource-group wealtharena-rg \
  --server wealtharena-sql-server \
  --name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

# 2. Add specific IP for management
az sql server firewall-rule create \
  --resource-group wealtharena-rg \
  --server wealtharena-sql-server \
  --name MyOfficeIP \
  --start-ip-address <YOUR_IP> \
  --end-ip-address <YOUR_IP>
```

### API Authentication (Future)
```python
# Add to backend/main.py
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/api/predictions")
async def get_predictions(
    request: PredictionRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    # Verify JWT token
    token = credentials.credentials
    # ... authentication logic
```

---

## 📈 Monitoring & Logging

### Azure Application Insights
```bash
# Create Application Insights
az monitor app-insights component create \
  --app wealtharena-insights \
  --location australiaeast \
  --resource-group wealtharena-rg

# Get instrumentation key
az monitor app-insights component show \
  --app wealtharena-insights \
  --resource-group wealtharena-rg \
  --query instrumentationKey \
  --output tsv

# Add to backend container
az container create ... \
  --environment-variables \
    APPINSIGHTS_INSTRUMENTATIONKEY='<YOUR_KEY>'
```

### Key Metrics to Monitor
1. **API Performance**
   - Request latency (target: <500ms)
   - Error rate (target: <1%)
   - Requests per second

2. **Database Performance**
   - Query execution time (target: <100ms)
   - Connection pool usage
   - Storage growth rate

3. **Model Performance**
   - Inference time (target: <300ms)
   - Prediction accuracy
   - Cache hit rate

---

## 💰 Cost Estimation (Monthly)

| Service | Tier | Cost (AUD) |
|---------|------|------------|
| Azure SQL Database (S1) | 10GB | $30 |
| Container Instances (Backend) | 2 vCPU, 4GB | $80 |
| Container Instances (Airflow) | 2 vCPU, 4GB | $80 |
| Blob Storage | 50GB | $2 |
| Static Web App | Free tier | $0 |
| Container Registry | Basic | $7 |
| Application Insights | 5GB/month | $15 |
| **Total** | | **~$214/month** |

**Cost Optimization:**
- Use Azure Reserved Instances: Save 30%
- Schedule containers (shut down off-hours): Save 50%
- Optimized tier: **~$107/month**

---

## 🚀 Going Live Checklist

### Pre-Launch
- [ ] Database schema created (8 tables)
- [ ] Model checkpoints uploaded to Blob Storage
- [ ] Backend API deployed and tested
- [ ] Airflow DAGs scheduled and running
- [ ] Frontend deployed and connected to API
- [ ] Environment variables configured
- [ ] Security rules in place
- [ ] Monitoring enabled
- [ ] Data pipeline tested end-to-end

### Launch Day
- [ ] Run manual data fetch
- [ ] Verify 32 symbols processed
- [ ] Check predictions generated
- [ ] Test all API endpoints
- [ ] Verify frontend displays data correctly
- [ ] Monitor logs for errors
- [ ] Backup database

### Post-Launch
- [ ] Monitor performance metrics
- [ ] Review error logs daily
- [ ] Plan symbol expansion (32 → 500)
- [ ] Optimize slow queries
- [ ] Set up automated alerts
- [ ] Schedule regular backups

---

## 📞 Troubleshooting

### Common Issues

**Issue 1: Database Connection Failed**
```bash
# Check firewall rules
az sql server firewall-rule list \
  --resource-group wealtharena-rg \
  --server wealtharena-sql-server

# Test connection
sqlcmd -S wealtharena-sql-server.database.windows.net \
  -d wealtharena_db -U sqladmin -P <password> -Q "SELECT 1"
```

**Issue 2: Container Won't Start**
```bash
# Check container logs
az container logs \
  --resource-group wealtharena-rg \
  --name wealtharena-backend

# Check container status
az container show \
  --resource-group wealtharena-rg \
  --name wealtharena-backend \
  --query instanceView.state
```

**Issue 3: API Returns 500 Error**
```bash
# Check backend logs
az container logs \
  --resource-group wealtharena-rg \
  --name wealtharena-backend \
  --tail 100

# Common fixes:
# - Verify environment variables
# - Check database connectivity
# - Verify model checkpoints accessible
```

---

## 📚 Next Steps

1. **Complete Deployment** (Week 1)
   - Deploy all Azure resources
   - Upload data and models
   - Configure pipelines

2. **Expand Symbol Coverage** (Week 2-3)
   - Add 100 ASX stocks
   - Add major cryptocurrencies
   - Add currency pairs

3. **Optimize Performance** (Week 4)
   - Implement caching
   - Optimize database queries
   - Load test API

4. **Add Features** (Week 5-6)
   - User authentication
   - Portfolio tracking
   - Game mode

5. **Production Hardening** (Week 7-8)
   - Security audit
   - Backup/recovery plan
   - Monitoring alerts
   - Documentation

---

**Deployment Guide Version**: 1.0.0  
**Last Updated**: October 8, 2025  
**Next Review**: November 8, 2025

