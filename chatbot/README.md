# WealthArena - AI Trading Education Platform

A comprehensive trading education platform with AI-powered chat, sentiment analysis, and financial data integration.

## ⚠️ First Time Setup

**IMPORTANT**: You must start the server before running any tests. The `dev_up.ps1` script handles everything automatically.

**Expected time**: 2-3 minutes for first run, 30 seconds for subsequent runs.

---

## 🚀 Quick Start

### Step 1: Start the Development Server

```powershell
# Copy environment file (if not already done)
copy .env.example .env

# Edit .env and add your GROQ_API_KEY

# Start the server (first time setup)
powershell -ExecutionPolicy Bypass -File scripts/dev_up.ps1
```

The script will:
- ✅ Create a virtual environment (`.venv`)
- ✅ Install minimal runtime dependencies
- ✅ Ingest Knowledge Base into vector database
- ✅ Start the API server on port 8000

**Wait for the server to start** - you'll see "Server started successfully!" message.

### Step 2: Verify Server is Running (in a NEW terminal)

```powershell
python scripts/check_server.py
```

Expected output: `✅ All prerequisites met! Server is ready.`

### Step 3: Run the Metrics Test

```powershell
python scripts/print_metrics.py --url http://127.0.0.1:8000 --runs 5
```

**API will be available at:**
- Health Check: http://127.0.0.1:8000/healthz
- API Docs: http://127.0.0.1:8000/docs
- Interactive docs: http://127.0.0.1:8000/docs

---

## 🚨 Common Issues

If you encounter problems, see the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for detailed solutions.

**Quick fixes:**
- **Server not running** → See [TROUBLESHOOTING.md Section 2](docs/TROUBLESHOOTING.md#section-2-server-not-running-issues)
- **Connection refused** → See [TROUBLESHOOTING.md Section 2](docs/TROUBLESHOOTING.md#section-2-server-not-running-issues)
- **Port already in use** → See [TROUBLESHOOTING.md Section 3](docs/TROUBLESHOOTING.md#section-3-port-conflicts)
- **Import errors** → See [TROUBLESHOOTING.md Section 4](docs/TROUBLESHOOTING.md#section-4-dependency-issues)

### PDF Ingestion Issues

**Problem**: PDF ingestion gets stuck or takes hours

**Solutions**:

1. **Use smaller batch sizes** for better progress feedback:

   ```bash
   python scripts/pdf_ingest.py --batch-size 20 --duplicate-check-batch-size 200
   ```

2. **Enable parallel processing** (process 2-3 PDFs at once):

   ```bash
   python scripts/pdf_ingest.py --parallel --max-workers 2
   ```

3. **Resume interrupted uploads**:

   ```bash
   # Resume (partials are saved by default)
   python scripts/pdf_ingest.py --resume
   
   # Resume but retry partial ingestions
   python scripts/pdf_ingest.py --resume --no-resume-partial
   ```

4. **Skip problematic PDFs** and continue:

   ```bash
   python scripts/pdf_ingest.py --skip-on-error
   ```

5. **Check progress**: Look for log messages showing batch progress (e.g., "Processing batch 3/10")

6. **First-time setup**: The first PDF takes longer (downloads embedding model ~80MB)

**Performance Tips**:

- Typical processing time: 2-5 minutes per PDF (depends on size and CPU)
- First batch is slowest (model download + initialization)
- Subsequent batches are faster (model cached)
- Use `--parallel` for 20+ PDFs to save time

## 🚀 Alternative Setup (Manual)

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

**Get Your Groq API Key (REQUIRED):**
1. Visit https://console.groq.com/
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key (starts with `gsk_`)

**IMPORTANT:** The application requires a valid Groq API key to function. Without it, all LLM-powered features will be unavailable.

**Create and Configure .env File:**
```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your API key (REQUIRED):
```env
GROQ_API_KEY=gsk_your_actual_key_here  # REQUIRED: Replace with your actual Groq API key from https://console.groq.com/
GROQ_MODEL=llama3-8b-8192
LLM_PROVIDER=groq
CHROMA_PERSIST_DIR=data/vectorstore  # Use absolute path for reliability
APP_HOST=0.0.0.0
APP_PORT=8000

# PDF Ingestion Configuration (optional)
PDF_INGEST_INTERVAL_HOURS=24  # Background job interval for PDF ingestion
ENABLE_PDF_INGESTION=true     # Enable automatic PDF ingestion
```

### 3. Run the API Server
```bash
# Development mode
python -m uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 4. Docker Setup (Alternative)
```bash
# Build and run with Docker
docker-compose up --build

# Run in background
docker-compose up -d
```

## 📡 API Endpoints (Handoff Quickstart)

### Page 4 – Chat & Knowledge
- **POST** `/v1/chat` → `{"message":"..."}`  # uses Groq; "analyze: ..." hits local sentiment
- **GET/POST** `/v1/chat/stream` → Server-Sent Events (SSE) with `Content-Type: text/event-stream`. Format: `data: {"chunk": "...", "index": 0}\n\n`. Supports query params (GET) or JSON body (POST).
- **GET** `/v1/search?q=&k=5`
- **POST** `/v1/explain` → `{"question":"...", "k":3}`
- **POST** `/v1/chat/history`
- **GET** `/v1/chat/history?user_id=`
- **POST** `/v1/chat/feedback` → `{"message_id":"...","vote":"up"|"down"}`
- **GET** `/v1/chat/export?user_id=`

### Page 3 – Game Mode
- **GET** `/v1/game/episodes`
- **POST** `/v1/game/start` → `{"user_id":"u1","episode_id":"covid_crash_2020","difficulty":"medium"}`
- **POST** `/v1/game/tick` → `{"game_id":"...","speed":1}`
- **POST** `/v1/game/trade` → `{"game_id":"...","symbol":"AAPL","side":"buy","qty":5,"type":"market"}`
- **GET** `/v1/game/portfolio?game_id=...`
- **GET** `/v1/game/summary?game_id=...`
- **WS** `/v1/game/stream?game_id=...`

### Metrics & Documentation
- **GET** `/metrics`            # Prometheus metrics
- **GET** `/metrics/basic`      # JSON summary
- **GET** `/openapi.json`       # OpenAPI specification
- **GET** `/docs`               # Interactive API documentation
- **GET** `/healthz`            # Health check

### Background Jobs & Data Pipeline
- **GET** `/v1/background/status` → Job health and last run times

## 📊 RAG Data Pipeline

### Overview

WealthArena uses a RAG (Retrieval Augmented Generation) system that combines:

- **Knowledge Base**: Curated educational content (markdown files from `docs/kb/`)
- **PDF Documents**: User-uploaded PDF files from the `docs/` directory

### PDF Ingestion

Process PDF files from the `docs/` directory:

```bash
# Basic usage
python scripts/pdf_ingest.py

# With parallel processing (recommended for 20+ PDFs)
python scripts/pdf_ingest.py --parallel --max-workers 2

# Optimized defaults for parallel processing (recommended for 20+ PDFs)
python scripts/pdf_ingest.py --parallel --max-workers 2 --batch-size 30 --duplicate-check-batch-size 200 --skip-on-error --resume

# Resume interrupted uploads (partials are saved by default)
python scripts/pdf_ingest.py --resume

# Resume but retry partial ingestions on next run
python scripts/pdf_ingest.py --resume --no-resume-partial

# Optimized batch sizes for better progress feedback
python scripts/pdf_ingest.py --batch-size 30 --duplicate-check-batch-size 200

# Skip problematic PDFs and continue
python scripts/pdf_ingest.py --skip-on-error
```

See [PDF Ingestion Issues](#pdf-ingestion-issues) section for troubleshooting.

### Initial Setup

```bash
# Load initial data (run once)
python scripts/initial_data_load.py

# Verify data pipeline
python scripts/verify_data_pipeline.py
```

### Background Jobs

The API automatically runs background jobs for:

- PDF ingestion: Periodically processes PDF files from the `docs/` directory (configurable via `PDF_INGEST_INTERVAL_HOURS`, default: 24 hours)

Check job status: `GET /v1/background/status`

### Configuration

Required environment variables:

- `GROQ_API_KEY`: Must be a valid Groq API key from https://console.groq.com/ (starts with `gsk_`)
- `CHROMA_PERSIST_DIR`: Use absolute path for reliability (code resolves relative paths programmatically)

Optional environment variables:

- `PDF_INGEST_INTERVAL_HOURS`: Background job interval for PDF ingestion in hours (default: 24)
- `ENABLE_PDF_INGESTION`: Enable/disable automatic PDF ingestion (default: true)

See `.env.example` for full configuration options.

For detailed documentation, see [RAG Pipeline Documentation](docs/RAG_PIPELINE.md).

### Data Pipeline Orchestration

Run the complete data pipeline using the orchestrator script:

```powershell
# Run full pipeline (all phases)
powershell -ExecutionPolicy Bypass -File deploy-master.ps1

# Skip PDF ingestion
powershell -ExecutionPolicy Bypass -File deploy-master.ps1 --skip-pdf-ingest

# Full refresh (clear collections and reload)
powershell -ExecutionPolicy Bypass -File deploy-master.ps1 --full-refresh
```

The pipeline orchestrator runs these phases:
1. **PHASE 0**: Environment setup (Python, packages, directories)
2. **PHASE 1**: PDF ingestion from `docs/` directory
3. **PHASE 2**: API verification (test endpoints)
4. **PHASE 3**: Summary and next steps

## 🧪 Testing & Development

### Server Health Check

Before running tests, verify the server is running:

```powershell
python scripts/check_server.py
```

This checks:
- Server is responding at http://127.0.0.1:8000
- Virtual environment exists
- Knowledge base is ingested
- Required packages are installed
- Environment variables are configured

### Performance Metrics

**⚠️ IMPORTANT: Server Must Be Running Before Testing**

Before running any metrics or tests:

1. **Start the server:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/dev_up.ps1
   ```

2. **Verify it's running:**
   ```powershell
   python scripts/check_server.py
   ```
   Wait for: "✅ Server is running at http://127.0.0.1:8000"

3. **Then run tests:**
   ```powershell
   python scripts/print_metrics.py
   ```

**Common Mistake:** Running `print_metrics.py` before starting the server results in 0% success rate in `metrics/runtime_http.json`. Always start the server first!

**Note:** Metrics should be generated dynamically using `scripts/print_metrics.py` rather than viewing static snapshots. Static metric files become outdated and can be misleading.

---

Run performance tests (requires server running):

```powershell
python scripts/print_metrics.py --url http://127.0.0.1:8000 --runs 5
```

Arguments:
- `--url`: Base URL of the API server (default: `http://127.0.0.1:8000`)
- `--runs`: Number of test iterations per endpoint (default: `5`)

> **Note:** If metrics show 0% success rate, the server was not running during the test. Follow the steps above to regenerate valid metrics.

### Basic Endpoint Tests

Run basic endpoint tests (uses port 8000 by default):

```powershell
python scripts/smoke_local.py
```

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report html

# Run specific test file
pytest tests/test_chat.py

# Run API sanity check
python scripts/sanity_check.py
```

### API Sanity Check
The sanity check script tests all major API endpoints to ensure the system is working correctly:

```bash
# Run comprehensive API tests
python scripts/sanity_check.py

# Test against different URL
python scripts/sanity_check.py --url http://localhost:8000
```

The sanity check will test:
- Health check endpoint
- Game episodes listing
- Starting a new game
- Game tick (advance time)
- Stock trading (buy AAPL)
- Portfolio retrieval
- Game summary
- Search functionality
- Explain functionality
- Metrics endpoints

**Example Output:**
```
🧪 WealthArena API Sanity Check
==================================================
PASS: Health Check - Status: healthy
PASS: Game Episodes - Found 4 episodes
PASS: Start Game - Game ID: abc123-def456
PASS: Game Tick - Advanced to: 2020-02-20
PASS: Game Trade - Trade ID: trade_789
PASS: Game Portfolio - Cash: $49,950.00, Holdings: 1
PASS: Game Summary - Total Value: $50,100.00, P&L: $100.00
PASS: Search - Found 5 results
PASS: Explain - Answer length: 245 chars, Sources: 3
PASS: Metrics Basic - Metrics available: 8 keys

==================================================
📊 Test Summary: 10/10 tests passed
✅ All tests passed!
```

### Code Quality
```bash
# Format code
black .

# Lint code
ruff check . --fix

# Type checking
mypy app/
```

### Export API Documentation
```bash
# Export OpenAPI spec
python scripts/export_openapi.py
```

## 🤖 Machine Learning Models

### Train Sentiment Analysis Model
1. Open Jupyter notebook:
   ```bash
   jupyter notebook ml/notebooks/02_finetune_sentiment.ipynb
   ```

2. Run all cells to train the DistilBERT sentiment model

3. The model automatically saves to `models/sentiment-finetuned/`

### Train Intent Classification Model
1. Open Jupyter notebook:
   ```bash
   jupyter notebook ml/notebooks/03_finetune_intent.ipynb
   ```

2. Run all cells to train the intent classification model

3. Model saves to `models/intent-finetuned/`

### ML Pipeline Scripts
The ML directory contains additional scripts for data processing and model training:

```bash
# Export financial phrasebank data
python ml/scripts/export_finphrasebank.py

# Run complete ML pipeline
python ml/scripts/pipeline_prepare_and_train.py

# Run pipeline with PowerShell (Windows)
ml/scripts/run_pipeline.ps1
```

## 📊 Monitoring & Metrics

### API Performance Metrics
- **Response Time**: Average API response time (ms)
- **Error Rate**: Percentage of failed requests
- **Throughput**: Requests per minute
- **Uptime**: Service availability percentage

### Machine Learning Metrics
- **Accuracy**: Model prediction accuracy (%)
- **F1-Score**: Macro-averaged F1 score
- **Inference Time**: Model prediction speed (ms)
- **Training Loss**: Model training convergence

### RSS Scraping Metrics
- **Success Rate**: Percentage of successful RSS fetches
- **Pages per Minute**: RSS feed processing throughput
- **Error Rate**: Failed RSS requests percentage
- **Response Time**: Average RSS fetch time

## 🏗️ Project Structure

```
WealthArena/
├── app/                    # FastAPI application
│   ├── background/        # Background job scheduler
│   │   └── scheduler.py   # Periodic PDF ingestion
│   ├── api/               # API endpoints
│   │   ├── chat.py        # Chat endpoints
│   │   ├── chat_stream.py # WebSocket chat streaming
│   │   ├── game.py        # Game mode endpoints
│   │   ├── game_stream.py # WebSocket game streaming
│   │   ├── search.py      # Vector search functionality
│   │   ├── explain.py     # AI explanation with KB
│   │   ├── market.py      # Market data endpoints
│   │   ├── context.py     # Context & knowledge
│   │   ├── history.py     # Chat history
│   │   ├── feedback.py    # User feedback
│   │   ├── export.py      # Data export
│   │   ├── metrics.py     # System metrics
│   │   └── background.py # Background job status
│   ├── llm/               # LLM client integration
│   │   ├── client.py      # Groq LLM client
│   │   └── guard_prompt.txt # Educational guardrails
│   ├── models/            # ML model wrappers
│   │   └── sentiment.py  # Sentiment analysis
│   ├── tools/             # Utility tools
│   │   ├── prices.py      # Price data tools
│   │   ├── document_processor.py # Document chunking
│   │   ├── pdf_processor.py # PDF processing
│   │   ├── vector_ingest.py    # Vector store ingestion
│   │   └── retrieval.py   # KB vector search
│   ├── metrics/           # Prometheus metrics
│   ├── middleware/        # FastAPI middleware
│   └── main.py           # Application entry point
├── docs/                 # Documentation
│   ├── kb/               # Knowledge Base (NEW)
│   │   ├── intro.md      # Platform introduction
│   │   ├── indicators_rsi.md # RSI guide
│   │   └── risk_management.md # Risk management
│   ├── CHAT_HISTORY_API.md
│   ├── CHAT_STREAMING.md
│   ├── CONTEXT_KNOWLEDGE_API.md
│   ├── INTEGRATION_ANDROID.md
│   ├── INTEGRATION_IOS.md
│   └── INTEGRATION_RN.md
├── packages/             # Mobile SDKs
│   ├── mobile-sdk-android/ # Android SDK
│   ├── mobile-sdk-ios/    # iOS SDK
│   ├── mobile-sdk-rn/     # React Native SDK
│   └── wealtharena-rn/    # RN Components
├── examples/              # Demo applications
│   ├── android-demo/      # Android demo
│   ├── ios-demo/          # iOS demo
│   └── rn-demo/           # React Native demo
├── ml/                    # Machine Learning
│   ├── notebooks/         # Jupyter notebooks
│   │   ├── 01_prepare_data.ipynb
│   │   ├── 02_finetune_sentiment.ipynb
│   │   └── 03_finetune_intent.ipynb
│   └── scripts/           # ML pipeline scripts
├── scripts/               # Development scripts
│   ├── dev_up.ps1         # Windows one-command setup
│   ├── dev_up.sh          # Unix one-command setup
│   ├── kb_ingest.py       # Knowledge Base ingestion
│   ├── initial_data_load.py    # One-time data loading
│   ├── verify_data_pipeline.py # Pre-deployment checks
│   ├── run_pipeline.py         # Data pipeline orchestrator
│   ├── smoke_local.py     # API testing
│   └── export_openapi.py  # API documentation
├── tests/                 # Test files
├── data/                  # Runtime data (gitignored)
│   ├── chat_history.db    # SQLite chat history
│   ├── vectorstore/       # ChromaDB vector store
│   └── game_state/        # Game state storage
├── models/                # Trained ML models (gitignored)
├── requirements.txt       # Dependencies
└── .env.example          # Environment template
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM (REQUIRED - application will not function without it) | Required |
| `GROQ_MODEL` | Groq model to use | `llama3-8b-8192` |
| `LLM_PROVIDER` | LLM provider | `groq` |
| `SENTIMENT_MODEL_DIR` | Path to sentiment model | `models/sentiment-finetuned` |
| `SENTRY_DSN` | Sentry DSN for error tracking | Optional |
| `CHROMA_PERSIST_DIR` | Vector database directory (use absolute path) | `data/vectorstore` (resolved to absolute) |
| `PDF_INGEST_INTERVAL_HOURS` | Background job interval for PDF ingestion | `24` |
| `ENABLE_PDF_INGESTION` | Enable/disable automatic PDF ingestion | `true` |
| `ENABLE_TOOLS` | Enable/disable non-LLM features (sentiment analysis, price queries) | `false` |
| `ENABLE_SENTIMENT_ANALYSIS` | Enable/disable sentiment analysis (requires ENABLE_TOOLS=true) | `false` |
| `APP_HOST` | Server host | `0.0.0.0` |
| `APP_PORT` | Server port | `8000` |

## 🚨 Troubleshooting

For detailed troubleshooting steps, see the [Troubleshooting Guide](docs/TROUBLESHOOTING.md).

### Data Pipeline Issues

- **No documents in vector store**: Run `python scripts/kb_ingest.py` to ingest knowledge base, and `python scripts/pdf_ingest.py` to ingest PDF documents
- **PDF ingestion not running**: Check `/v1/background/status` and logs. Verify `ENABLE_PDF_INGESTION=true` in `.env`
- **PDF ingestion slow**: Use parallel processing with `python scripts/pdf_ingest.py --parallel --max-workers 2`

### Quick Diagnostics

```powershell
# Check server health and prerequisites
python scripts/check_server.py

# Check if API is running
curl http://localhost:8000/healthz

# Check metrics
curl http://localhost:8000/metrics
```

### Common Issues

1. **GROQ_API_KEY not set or invalid**: 
   - Error: "GROQ_API_KEY is required" or "LLM service unavailable"
   - Solution: Ensure `.env` file exists with a valid `GROQ_API_KEY` starting with `gsk_`
   - Get your key from: https://console.groq.com/
   - Verify the key is correctly formatted (no extra spaces, correct prefix)
   - Restart the server after adding/updating the API key

2. **Server not running**: Run `powershell -ExecutionPolicy Bypass -File scripts/dev_up.ps1` first
3. **Port 8000 busy**: See [TROUBLESHOOTING.md Section 3](docs/TROUBLESHOOTING.md#section-3-port-conflicts)
4. **Import Errors**: See [TROUBLESHOOTING.md Section 4](docs/TROUBLESHOOTING.md#section-4-dependency-issues)
5. **Vector store issues**: See [TROUBLESHOOTING.md Section 5](docs/TROUBLESHOOTING.md#section-5-chromadb--vector-store-issues)
6. **Testing issues**: See [TROUBLESHOOTING.md Section 6](docs/TROUBLESHOOTING.md#section-6-testing-issues)

## 📚 Additional Resources

- **API Documentation**: Visit `http://localhost:8000/docs` when server is running
- **OpenAPI Specification**: Run `python scripts/export_openapi.py` to generate `docs/openapi.json`
- **Jupyter Notebooks**: Detailed ML training examples in `notebooks/`
- **Model Metrics**: Check `metrics_*.json` files for training results
- **Docker Documentation**: See `Dockerfile` and `docker-compose.yml` for container setup

## 🚀 Deployment

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

### Quick Deploy Options

#### Docker (Recommended for Local/Production)

**Using docker-compose:**
```bash
docker-compose up -d
```

**Using master deployment script:**
```powershell
# Windows PowerShell
.\deploy-master.ps1 --deploy docker

# Or with options
.\deploy-master.ps1 --deploy docker -Build -Run
.\deploy-master.ps1 --deploy docker -Stop
.\deploy-master.ps1 --deploy docker -Logs
```

**Manual Docker commands:**
```bash
# Build image
docker build -t wealtharena-api .

# Run container
docker run -d --name wealtharena-api -p 8000:8000 --env-file .env wealtharena-api
```

#### Azure App Service

**Automated deployment:**
```powershell
.\deploy-master.ps1 --deploy azure `
  -ResourceGroup "rg-wealtharena" `
  -AppName "wealtharena-api" `
  -Location "eastus"
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment guide including Azure, Docker, and production configuration.

---

## 🔧 Azure Deployment Troubleshooting

### Common Issue: Module Import Errors (HTTP 503)

If your Azure deployment shows **"Application container failed to start"** with module import errors:

**Symptoms:**
- HTTP 503 errors when accessing the app
- Azure portal error: "interpreter is unable to locate a module or package"
- Logs show: `ModuleNotFoundError` or `No module named 'uvicorn'`

**Quick Fix (Recommended):**

Use the automated fix script:
```powershell
.\scripts\azure_fix_deployment.ps1 -AppName "wealtharena-api" -ResourceGroup "rg-wealtharena"
```

This script will:
- ✅ Check your current configuration
- ✅ Fix all critical settings automatically
- ✅ Remove problematic configurations
- ✅ Restart the app
- ✅ Wait for health check to pass

**Diagnostic Check:**

Verify your configuration before applying fixes:
```powershell
.\scripts\azure_verify_config.ps1 -AppName "wealtharena-api" -ResourceGroup "rg-wealtharena"
```

**Manual Quick Fix:**

If you prefer manual fixes:
```bash
# 1. Enable Oryx build
az webapp config appsettings set --name wealtharena-api --resource-group rg-wealtharena --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true

# 2. Remove blocking setting
az webapp config appsettings delete --name wealtharena-api --resource-group rg-wealtharena --setting-names WEBSITE_RUN_FROM_PACKAGE

# 3. Set PYTHONPATH
az webapp config appsettings set --name wealtharena-api --resource-group rg-wealtharena --settings PYTHONPATH=/home/site/wwwroot

# 4. Restart
az webapp restart --name wealtharena-api --resource-group rg-wealtharena

# 5. Check logs
az webapp log tail --name wealtharena-api --resource-group rg-wealtharena
```

**For detailed troubleshooting**, see:
- [DEPLOYMENT.md - Module Import Errors](DEPLOYMENT.md#module-import-errors-application-container-failed-to-start)
- [TROUBLESHOOTING.md - Azure App Service](docs/TROUBLESHOOTING.md#section-9-azure-app-service-module-import-errors)

---

**Happy Trading! 📈🤖**

## Verified Metrics (local)

| Endpoint | Success % | Avg (ms) | P50 | P90 | P95 | P99 |
|----------|-----------|----------|-----|-----|-----|-----|
| episodes | 0.0% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| healthz | 0.0% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| explain | 0.0% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

*Overall: 0.0% success, 0.0ms avg latency*
*Last verified: 2025-11-06T12:47:27.485701*

## Verified Metrics (latest run)

| Component | Metric | Value |
|-----------|--------|-------|
| Chatbot | Inference(ms) avg | 1931.4 |
| Chatbot | ROUGE-L | n/a |
| Chatbot | BERT-F1 | n/a |
| Retrieval | Latency(ms) avg | 550.7 |
| Retrieval | MAP@5 | n/a |
| Retrieval | MRR@5 | n/a |
| Classification | Inference(ms) avg | n/a |
| Classification | F1-macro | n/a |
| Classification | AUC | n/a |
| Overall | Success Rate % | 90.0 |
| Overall | P50(ms) | 1312.0 |
| Overall | P95(ms) | 3329.0 |

