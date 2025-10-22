# WealthArena Data Flow Architecture
**Visual Guide to Data Pipeline and API Architecture**

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          WEALTHARENA SYSTEM                              │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   AIRFLOW    │──────│ PREPROCESSING│──────│  RL MODELS   │──────│   BACKEND    │
│  DATA FETCH  │      │   PIPELINE   │      │   INFERENCE  │      │     API      │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
       │                      │                      │                     │
       │                      │                      │                     │
       ▼                      ▼                      ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Raw Market  │      │  Processed   │      │    Model     │      │   Frontend   │
│     Data     │      │   Features   │      │ Predictions  │      │   Display    │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
```

---

## Data Pipeline Flow

### 1. Data Acquisition (Airflow DAG)
```
┌─────────────────────────────────────────────────────────────┐
│               FETCH MARKET DATA DAG                         │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │  Fetch   │──▶│ Validate │──▶│  Store   │──▶│ Notify  │ │
│  │  Yahoo   │   │   Data   │   │  to CSV  │   │Complete │ │
│  │ Finance  │   │          │   │          │   │         │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘ │
│                                                             │
│  Schedule: Daily at 6:00 PM (after market close)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Raw CSV Files   │
                    │  data/processed/  │
                    │ AAPL_processed.csv│
                    │GOOGL_processed.csv│
                    └──────────────────┘
```

### 2. Feature Engineering (Preprocessing DAG)
```
┌──────────────────────────────────────────────────────────────┐
│            PREPROCESS MARKET DATA DAG                        │
│                                                              │
│  ┌────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐  │
│  │  Load  │──▶│Calculate │──▶│ Validate │──▶│   Save    │  │
│  │  Data  │   │ Features │   │ Features │   │  Features │  │
│  └────────┘   └──────────┘   └──────────┘   └───────────┘  │
│                                                              │
│  Features Calculated:                                       │
│  • Price: Returns, Volatility, Momentum                     │
│  • Technical: RSI, MACD, Bollinger Bands                    │
│  • Volume: Volume Ratio, OBV, VWAP                          │
│  • Support/Resistance levels                                │
│                                                              │
│  Schedule: Daily at 7:00 PM (after fetch DAG)              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Feature Files   │
                    │  data/features/  │
                    │AAPL_features.csv │
                    │feature_stats.json│
                    └──────────────────┘
```

### 3. Model Inference (Backend Service)
```
┌─────────────────────────────────────────────────────────────────┐
│                    RL MODEL INFERENCE                           │
│                                                                 │
│  ┌────────────────┐      ┌─────────────────┐                   │
│  │ Trading Agent  │      │  Risk Mgmt      │                   │
│  │                │      │  Agent          │                   │
│  │ • Signal       │      │ • TP Levels     │                   │
│  │ • Entry Price  │      │ • SL Level      │                   │
│  │ • Confidence   │      │ • Risk/Reward   │                   │
│  └────────────────┘      └─────────────────┘                   │
│                                                                 │
│  ┌────────────────┐                                            │
│  │ Portfolio Mgr  │                                            │
│  │                │                                            │
│  │ • Position %   │                                            │
│  │ • Dollar Amt   │                                            │
│  │ • Max Risk     │                                            │
│  └────────────────┘                                            │
│                                                                 │
│  Models: v2.3.1                                                │
│  Types: asx_stocks, currency_pairs, commodities, crypto, etf   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Predictions    │
                    │ (with TP/SL)     │
                    └──────────────────┘
```

---

## API Endpoint Architecture

### Backend API Structure
```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GET  /                      → API Info                         │
│  GET  /health                → Health Check                     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  MARKET DATA ENDPOINTS                    │  │
│  │                                                           │  │
│  │  POST /api/market-data   → Get OHLCV data               │  │
│  │      Request: {symbols, days}                            │  │
│  │      Response: {data: {symbol: {OHLCV array}}}          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                PREDICTION/SIGNAL ENDPOINTS                │  │
│  │                                                           │  │
│  │  POST /api/predictions   → Get signal with TP/SL        │  │
│  │      Request: {symbol, horizon}                          │  │
│  │      Response: {                                         │  │
│  │         signal, confidence, entry,                       │  │
│  │         take_profit: [TP1, TP2, TP3],                   │  │
│  │         stop_loss: {price, type},                        │  │
│  │         risk_metrics, position_sizing,                   │  │
│  │         indicators, chart_data                           │  │
│  │      }                                                    │  │
│  │                                                           │  │
│  │  POST /api/top-setups    → Get top 3 setups             │  │
│  │      Request: {asset_type, count, risk_tolerance}       │  │
│  │      Response: {                                         │  │
│  │         setups: [ranked predictions with TP/SL],        │  │
│  │         metadata                                         │  │
│  │      }                                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 PORTFOLIO ENDPOINTS                       │  │
│  │                                                           │  │
│  │  POST /api/portfolio     → Analyze portfolio             │  │
│  │      Request: {symbols, weights}                         │  │
│  │      Response: {                                         │  │
│  │         portfolio, metrics, recommendations              │  │
│  │      }                                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   GAME ENDPOINTS                          │  │
│  │                                                           │  │
│  │  GET /api/game/leaderboard → Get leaderboard            │  │
│  │      Response: {leaderboard: [users with scores]}       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   OTHER ENDPOINTS                         │  │
│  │                                                           │  │
│  │  POST /api/chat          → RAG chatbot                   │  │
│  │  GET  /api/metrics/summary → System metrics              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Architecture (Azure SQL)

```
┌─────────────────────────────────────────────────────────────────┐
│                     AZURE SQL DATABASE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  raw_market_data                                         │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │  id, symbol, asset_type, date                           │  │
│  │  open_price, high_price, low_price, close_price, volume │  │
│  │  created_at, updated_at                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ▲                                    │
│                            │                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  processed_features                                      │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │  id, symbol, date                                        │  │
│  │  returns, volatility_5, volatility_20                    │  │
│  │  sma_5, sma_20, sma_50, sma_200                         │  │
│  │  rsi, macd, macd_signal, macd_hist                      │  │
│  │  bb_upper, bb_lower, atr, obv                           │  │
│  │  momentum_5, momentum_10, volume_ratio                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  model_predictions (TP/SL Storage)                       │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │  id, symbol, asset_type, prediction_date                │  │
│  │  signal, confidence                                      │  │
│  │  entry_price, entry_price_min, entry_price_max          │  │
│  │  tp1_price, tp1_percent, tp1_close_percent              │  │
│  │  tp2_price, tp2_percent, tp2_close_percent              │  │
│  │  tp3_price, tp3_percent, tp3_close_percent              │  │
│  │  sl_price, sl_percent, sl_type                          │  │
│  │  risk_reward_ratio, win_probability                     │  │
│  │  recommended_position_percent, ranking_score            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  portfolio_state                                         │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │  id, user_id, symbol                                     │  │
│  │  quantity, entry_price, current_value                    │  │
│  │  pnl, pnl_percent                                        │  │
│  │  tp1_price, tp2_price, tp3_price, sl_price              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  game_leaderboard & game_sessions                        │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │  User stats, game results, rankings                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Frontend Data Flow

### Component-Level Data Usage

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND COMPONENTS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  DASHBOARD VIEW                                          │  │
│  │  ──────────────────────────────────────────────────────  │  │
│  │  GET /api/metrics/summary                                │  │
│  │                                                           │  │
│  │  Displays:                                               │  │
│  │  • System status & uptime                                │  │
│  │  • Symbols tracked                                       │  │
│  │  • Active models                                         │  │
│  │  • Active users & games                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TOP SETUPS VIEW                                         │  │
│  │  ──────────────────────────────────────────────────────  │  │
│  │  POST /api/top-setups                                    │  │
│  │                                                           │  │
│  │  Displays:                                               │  │
│  │  ┌─────────────────────────────────────────────────────┐│  │
│  │  │ Rank 1: NVDA - BUY (92% conf) | R/R: 3.58          ││  │
│  │  │ Entry: $458.75                                      ││  │
│  │  │ TP1: $472.50 (3.0%) - Close 50%                    ││  │
│  │  │ TP2: $486.25 (6.0%) - Close 30%                    ││  │
│  │  │ TP3: $500.00 (9.0%) - Close 20%                    ││  │
│  │  │ SL:  $450.00 (-1.91%) - Trailing                   ││  │
│  │  │ Position: 6.8% ($6,800)                             ││  │
│  │  │ [View Chart] [Trade Setup]                          ││  │
│  │  └─────────────────────────────────────────────────────┘│  │
│  │                                                           │  │
│  │  (Repeat for Rank 2, 3...)                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TRADING SIGNAL VIEW                                     │  │
│  │  ──────────────────────────────────────────────────────  │  │
│  │  POST /api/predictions {symbol: "AAPL"}                  │  │
│  │                                                           │  │
│  │  Displays:                                               │  │
│  │  ┌─────────────────────────────────────────────────────┐│  │
│  │  │ [CANDLESTICK CHART - 30 days]                       ││  │
│  │  │                                                      ││  │
│  │  │  ─── TP3: $190.00 (6.47%)                          ││  │
│  │  │  ─── TP2: $186.15 (4.31%)                          ││  │
│  │  │  ─── TP1: $182.30 (2.16%)                          ││  │
│  │  │  ● Entry: $178.45                                   ││  │
│  │  │  ─── SL: $175.89 (-1.43%)                          ││  │
│  │  └─────────────────────────────────────────────────────┘│  │
│  │                                                           │  │
│  │  Technical Indicators:                                   │  │
│  │  • RSI: 58.42 (Neutral-Bearish)                         │  │
│  │  • MACD: 0.87 (Bullish)                                 │  │
│  │  • Volatility: 0.023 (Medium)                           │  │
│  │  • Volume: High                                          │  │
│  │  • Trend: Up (Strong)                                    │  │
│  │                                                           │  │
│  │  Risk Metrics:                                           │  │
│  │  • R/R Ratio: 3.02                                       │  │
│  │  • Win Prob: 74%                                         │  │
│  │  • Expected Value: $5.05/share                           │  │
│  │                                                           │  │
│  │  Position Sizing:                                        │  │
│  │  • Recommended: 5.2% ($5,200)                            │  │
│  │  • Max Risk: 2.6%                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PORTFOLIO ANALYTICS                                     │  │
│  │  ──────────────────────────────────────────────────────  │  │
│  │  POST /api/portfolio {symbols, weights}                  │  │
│  │                                                           │  │
│  │  Displays:                                               │  │
│  │  • Portfolio Value: $125,450 (+12.45%)                   │  │
│  │  • Sharpe Ratio: 1.30                                    │  │
│  │  • Max Drawdown: -8.42%                                  │  │
│  │  • Volatility: 18.75%                                    │  │
│  │                                                           │  │
│  │  Holdings:                                               │  │
│  │  • AAPL:  30% | $37,635 | +5.2%                         │  │
│  │  • GOOGL: 25% | $31,362 | +8.7%                         │  │
│  │  • NVDA:  25% | $31,362 | +18.3%                        │  │
│  │  • TSLA:  20% | $25,090 | +2.1%                         │  │
│  │                                                           │  │
│  │  [Pie Chart] [Performance Chart] [Rebalance]             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GAME/LEADERBOARD                                        │  │
│  │  ──────────────────────────────────────────────────────  │  │
│  │  GET /api/game/leaderboard                               │  │
│  │                                                           │  │
│  │  Rank | User        | Score | Sharpe | Games             │  │
│  │  ───────────────────────────────────────────────         │  │
│  │   1   | TradingPro  | 25.4  |  2.1   |  15               │  │
│  │   2   | RL_Master   | 22.8  |  1.9   |  12               │  │
│  │   3   | QuanTitan   | 21.5  |  1.8   |  20               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  CHATBOT/RAG                                             │  │
│  │  ──────────────────────────────────────────────────────  │  │
│  │  POST /api/chat {message}                                │  │
│  │                                                           │  │
│  │  User: "Should I buy AAPL now?"                          │  │
│  │                                                           │  │
│  │  Bot: Based on current market analysis, our RL models    │  │
│  │       suggest considering stocks with strong momentum    │  │
│  │       and positive technical indicators...               │  │
│  │                                                           │  │
│  │  Confidence: 85%                                          │  │
│  │  Sources: Technical Analysis, Model Predictions          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Update Schedule

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAILY DATA CYCLE                             │
└─────────────────────────────────────────────────────────────────┘

16:00 (4:00 PM) ──────────▶ Market Close (US Stocks)

18:00 (6:00 PM) ──────────▶ Airflow: Fetch Market Data DAG
                            • Download OHLCV from Yahoo Finance
                            • Validate data quality
                            • Save to data/processed/*.csv
                            Duration: ~15 minutes

19:00 (7:00 PM) ──────────▶ Airflow: Preprocess Data DAG
                            • Load processed data
                            • Calculate features (RSI, MACD, etc.)
                            • Validate features
                            • Save to data/features/*.csv
                            Duration: ~20 minutes

19:30 (7:30 PM) ──────────▶ Data Ready for Model Inference

00:00 - 23:59  ──────────▶ Backend API Serving
                            • Models generate predictions on-demand
                            • Fresh data available for all symbols
                            • Cache predictions for 5 minutes

Real-time Updates:
• Portfolio values: Updated on every API call
• Leaderboard: Updated after each game completion
• System metrics: Cached for 30 seconds
```

---

## Key Data Structures Summary

### 1. **Market Data**
```json
{
  "OHLCV": ["Date", "Open", "High", "Low", "Close", "Volume"],
  "frequency": "daily",
  "source": "Yahoo Finance"
}
```

### 2. **Prediction with TP/SL** (Most Important!)
```json
{
  "signal": "BUY/SELL/HOLD",
  "confidence": 0.87,
  "entry": {"price": 178.45, "range": [177.91, 178.99]},
  "take_profit": [
    {"level": 1, "price": 182.30, "close_percent": 50},
    {"level": 2, "price": 186.15, "close_percent": 30},
    {"level": 3, "price": 190.00, "close_percent": 20}
  ],
  "stop_loss": {"price": 175.89, "type": "trailing"},
  "risk_metrics": {"risk_reward_ratio": 3.02, "win_probability": 0.74},
  "position_sizing": {"recommended_percent": 5.2, "dollar_amount": 5200}
}
```

### 3. **Technical Indicators**
```json
{
  "rsi": {"value": 58.42, "status": "neutral-bearish"},
  "macd": {"value": 0.87, "status": "bullish"},
  "volatility": {"value": 0.023, "status": "medium"},
  "trend": {"direction": "up", "strength": "strong"}
}
```

### 4. **Portfolio Metrics**
```json
{
  "total_return": 12.45,
  "annualized_return": 24.32,
  "volatility": 18.75,
  "sharpe_ratio": 1.30,
  "max_drawdown": -8.42
}
```

---

## Integration Points

### Backend ↔ Database
- **Write**: Airflow DAGs write processed data to CSV/AzureSQL
- **Read**: Backend API reads from CSV/AzureSQL for predictions
- **Cache**: 5-minute cache for frequently accessed data

### Backend ↔ Frontend
- **Protocol**: REST API over HTTP
- **Format**: JSON
- **Auth**: JWT tokens (future)
- **CORS**: Enabled for development

### RL Models ↔ Backend
- **Load**: Models loaded on-demand (lazy loading)
- **Inference**: Real-time prediction generation
- **Checkpoints**: Stored in `checkpoints/` directory
- **Version**: v2.3.1 (all agents)

---

## Performance Considerations

### API Response Times (Target)
- `/api/market-data`: < 200ms
- `/api/predictions`: < 500ms (includes model inference)
- `/api/top-setups`: < 1000ms (analyzes multiple symbols)
- `/api/portfolio`: < 300ms
- `/api/chat`: < 800ms (RAG inference)

### Optimization Strategies
1. **Caching**: 5-minute cache for market data and predictions
2. **Lazy Loading**: Models loaded only when needed
3. **Database Indexing**: Indexed on (symbol, date)
4. **Batch Processing**: Process multiple symbols in parallel
5. **CDN**: Static assets served from CDN (future)

---

## Error Handling

### Data Pipeline Errors
```
Missing Data → Retry (3 times) → Alert → Skip symbol
Invalid Data → Validation Error → Log → Skip record
Model Error  → Fallback to HOLD → Log → Continue
```

### API Errors
```
404: Symbol not found → Return error message
500: Model error → Return generic error
503: Model loading → Return "service unavailable"
```

---

## Security Notes (Future Implementation)

1. **Authentication**: JWT tokens for user authentication
2. **API Keys**: Rate-limited API keys for external access
3. **Data Encryption**: TLS 1.3 for data in transit
4. **SQL Injection**: Parameterized queries only
5. **Rate Limiting**: 100 requests/minute per user

---

## Monitoring & Logging

### Metrics to Track
- API request/response times
- Model inference times
- Data pipeline success rates
- Database query performance
- User activity (games, trades)

### Logging Strategy
```
INFO  → Normal operations
WARN  → Data quality issues, retries
ERROR → Pipeline failures, model errors
DEBUG → Detailed trace for development
```

---

## Quick Reference: Common Queries

### Get Latest Price
```
POST /api/market-data
{"symbols": ["AAPL"], "days": 1}
→ response.data.AAPL.latest_price
```

### Get Trading Signal
```
POST /api/predictions
{"symbol": "AAPL", "horizon": 1}
→ response.signal, response.take_profit, response.stop_loss
```

### Get Top 3 Stocks
```
POST /api/top-setups
{"asset_type": "stocks", "count": 3}
→ response.setups[0..2]
```

### Check System Health
```
GET /health
→ response.status
```

---

**Last Updated**: October 8, 2025  
**Version**: 1.0.0  
**Team**: WealthArena RL Engineering

