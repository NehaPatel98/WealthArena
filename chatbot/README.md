# WealthArena - AI Trading Education Platform

A comprehensive trading education platform with AI-powered chat, sentiment analysis, and financial data integration.

## 🚀 Quick Setup

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
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
LLM_PROVIDER=groq
SENTIMENT_MODEL_DIR=models/sentiment-finetuned
SENTRY_DSN=
CHROMA_PERSIST_DIR=data/vectorstore
APP_HOST=0.0.0.0
APP_PORT=8000
```

**Important**: Never commit your actual API keys to the repository. Always use `.env.example` as a template and keep your `.env` file in `.gitignore`.

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
- **WS** `/v1/chat/stream?user_id=...`
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

## 🧪 Testing & Development

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
│   ├── api/               # API endpoints
│   │   ├── chat.py        # Chat endpoints
│   │   ├── game.py        # Game mode endpoints
│   │   ├── search.py      # Search functionality
│   │   └── ...            # Other endpoints
│   ├── llm/               # LLM client integration
│   ├── models/            # ML model wrappers
│   ├── tools/             # Utility tools (prices, news)
│   └── main.py           # Application entry point
├── ml/                    # Machine Learning components
│   ├── notebooks/         # Jupyter notebooks for ML training
│   │   ├── 01_prepare_data.ipynb
│   │   ├── 02_finetune_sentiment.ipynb
│   │   └── 03_finetune_intent.ipynb
│   └── scripts/          # ML pipeline scripts
│       ├── export_finphrasebank.py
│       ├── pipeline_prepare_and_train.py
│       └── run_pipeline.ps1
├── models/               # Trained ML models (gitignored)
├── data/                 # Training data and vectorstore (gitignored)
├── scripts/              # Utility scripts
├── docs/                 # API documentation
├── tests/                # Test files
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
└── .env.example        # Environment template
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM | Required |
| `GROQ_MODEL` | Groq model to use | `llama3-8b-8192` |
| `LLM_PROVIDER` | LLM provider | `groq` |
| `SENTIMENT_MODEL_DIR` | Path to sentiment model | `models/sentiment-finetuned` |
| `SENTRY_DSN` | Sentry DSN for error tracking | Optional |
| `CHROMA_PERSIST_DIR` | Vector database directory | `data/vectorstore` |
| `APP_HOST` | Server host | `0.0.0.0` |
| `APP_PORT` | Server port | `8000` |

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure virtual environment is activated and dependencies installed
2. **Model Not Found**: Train the sentiment model using the notebook first
3. **API Connection**: Ensure server is running on correct port (8000)
4. **RSS Errors**: Some feeds may be blocked - this is normal and tracked in metrics
5. **Docker Issues**: Check if ports are available and Docker is running

### Health Check
```bash
# Check if API is running
curl http://localhost:8000/healthz

# Check metrics
curl http://localhost:8000/metrics
```

### Logs
```bash
# View Docker logs
docker-compose logs -f api

# View specific service logs
docker-compose logs api
```

## 📚 Additional Resources

- **API Documentation**: Visit `http://localhost:8000/docs` when server is running
- **OpenAPI Specification**: Run `python scripts/export_openapi.py` to generate `docs/openapi.json`
- **Jupyter Notebooks**: Detailed ML training examples in `notebooks/`
- **Model Metrics**: Check `metrics_*.json` files for training results
- **Docker Documentation**: See `Dockerfile` and `docker-compose.yml` for container setup

## 🚀 Deployment

### Production Deployment
1. Set up environment variables
2. Train your ML models
3. Use Docker for containerized deployment
4. Configure monitoring and logging
5. Set up health checks and metrics

### Docker Commands
```bash
# Build image
docker build -t wealtharena-api .

# Run container
docker run -p 8000:8000 --env-file .env wealtharena-api

# Use docker-compose
docker-compose up -d
```

---

**Happy Trading! 📈🤖**