# WealthArena Backend-Frontend Data Schema
**Complete API Data Structure Reference**  
Version: 1.0.0  
Last Updated: October 8, 2025

---

## Table of Contents
1. [Market Data](#1-market-data)
2. [Model Predictions & Trading Signals](#2-model-predictions--trading-signals)
3. [Top Trading Setups](#3-top-trading-setups)
4. [Portfolio Analytics](#4-portfolio-analytics)
5. [Game & Leaderboard](#5-game--leaderboard)
6. [Chat/RAG Responses](#6-chatrag-responses)
7. [System Metrics](#7-system-metrics)

---

## 1. Market Data

### Endpoint: `POST /api/market-data`

### Request
```json
{
  "symbols": ["AAPL", "GOOGL", "BTC-USD"],
  "days": 30
}
```

### Response
```json
{
  "symbols": ["AAPL", "GOOGL", "BTC-USD"],
  "days": 30,
  "timestamp": "2025-10-08T10:30:00.123456",
  "data": {
    "AAPL": {
      "latest_price": 178.45,
      "records": 30,
      "columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
      "data": [
        {
          "Date": "2025-10-01",
          "Open": 175.20,
          "High": 178.90,
          "Low": 174.85,
          "Close": 178.45,
          "Volume": 52847392
        },
        {
          "Date": "2025-10-02",
          "Open": 178.50,
          "High": 180.20,
          "Low": 177.30,
          "Close": 179.85,
          "Volume": 48392847
        }
        // ... more records
      ]
    },
    "GOOGL": {
      "latest_price": 142.35,
      "records": 30,
      "columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
      "data": [ /* similar structure */ ]
    }
  }
}
```

### Database Schema
```sql
-- Raw market data from database
raw_market_data (
  id, symbol, asset_type, date,
  open_price, high_price, low_price, close_price, volume,
  created_at, updated_at
)
```

---

## 2. Model Predictions & Trading Signals

### Endpoint: `POST /api/predictions`

### Request
```json
{
  "symbol": "AAPL",
  "horizon": 1
}
```

### Response (Complete Trading Setup)
```json
{
  "symbol": "AAPL",
  "asset_type": "stock",
  "signal": "BUY",
  "confidence": 0.87,
  "timestamp": "2025-10-08T10:30:00.123456",
  
  "entry": {
    "price": 178.45,
    "range": [177.91, 178.99],
    "timing": "immediate"
  },
  
  "take_profit": [
    {
      "level": 1,
      "price": 182.30,
      "percent": 2.16,
      "close_percent": 50,
      "probability": 0.75
    },
    {
      "level": 2,
      "price": 186.15,
      "percent": 4.31,
      "close_percent": 30,
      "probability": 0.55
    },
    {
      "level": 3,
      "price": 190.00,
      "percent": 6.47,
      "close_percent": 20,
      "probability": 0.35
    }
  ],
  
  "stop_loss": {
    "price": 175.89,
    "percent": -1.43,
    "type": "trailing",
    "trail_amount": 1.28
  },
  
  "risk_metrics": {
    "risk_reward_ratio": 3.02,
    "max_risk_per_share": 2.56,
    "max_reward_per_share": 7.73,
    "win_probability": 0.74,
    "expected_value": 5.05
  },
  
  "position_sizing": {
    "recommended_percent": 5.2,
    "dollar_amount": 5200.00,
    "method": "Kelly Criterion + Volatility Adjusted",
    "max_risk_percent": 2.6,
    "confidence_factor": 0.87,
    "volatility_factor": 0.023
  },
  
  "indicators": {
    "rsi": {
      "value": 58.42,
      "status": "neutral-bearish"
    },
    "macd": {
      "value": 0.87,
      "status": "bullish"
    },
    "volatility": {
      "value": 0.023,
      "status": "medium"
    },
    "volume": {
      "status": "high"
    },
    "trend": {
      "direction": "up",
      "strength": "strong"
    }
  },
  
  "chart_data": [
    {
      "date": "2025-09-08",
      "open": 172.50,
      "high": 174.20,
      "low": 171.80,
      "close": 173.90,
      "volume": 48392847
    },
    {
      "date": "2025-09-09",
      "open": 174.00,
      "high": 175.60,
      "low": 173.50,
      "close": 175.30,
      "volume": 52847392
    }
    // ... last 30 days
  ],
  
  "model_version": "v2.3.1",
  "model_type": "stock",
  "reasoning": "Model confidence: 87%. Bullish signals detected with favorable risk/reward setup."
}
```

### Database Schema
```sql
model_predictions (
  id, symbol, asset_type, prediction_date,
  signal, confidence,
  entry_price, entry_price_min, entry_price_max, entry_timing,
  tp1_price, tp1_percent, tp1_close_percent, tp1_probability,
  tp2_price, tp2_percent, tp2_close_percent, tp2_probability,
  tp3_price, tp3_percent, tp3_close_percent, tp3_probability,
  sl_price, sl_percent, sl_type, sl_trail_amount,
  risk_reward_ratio, max_risk_per_share, max_reward_per_share,
  win_probability, expected_value,
  recommended_position_percent, recommended_dollar_amount, max_risk_percent,
  model_version, model_type, reasoning, ranking_score
)
```

---

## 3. Top Trading Setups

### Endpoint: `POST /api/top-setups`

### Request
```json
{
  "asset_type": "stocks",
  "count": 3,
  "risk_tolerance": "medium"
}
```

### Response
```json
{
  "asset_type": "stocks",
  "timestamp": "2025-10-08T10:30:00.123456",
  "count": 3,
  "setups": [
    {
      "rank": 1,
      "symbol": "NVDA",
      "asset_type": "stocks",
      "signal": "BUY",
      "confidence": 0.92,
      "ranking_score": 0.8745,
      
      "entry": {
        "price": 458.75,
        "range": [457.37, 460.13],
        "timing": "immediate"
      },
      
      "take_profit": [
        {
          "level": 1,
          "price": 472.50,
          "percent": 3.00,
          "close_percent": 50,
          "probability": 0.80
        },
        {
          "level": 2,
          "price": 486.25,
          "percent": 6.00,
          "close_percent": 30,
          "probability": 0.62
        },
        {
          "level": 3,
          "price": 500.00,
          "percent": 9.00,
          "close_percent": 20,
          "probability": 0.42
        }
      ],
      
      "stop_loss": {
        "price": 450.00,
        "percent": -1.91,
        "type": "trailing",
        "trail_amount": 4.37
      },
      
      "risk_metrics": {
        "risk_reward_ratio": 3.58,
        "max_risk_per_share": 8.75,
        "max_reward_per_share": 31.31,
        "win_probability": 0.78,
        "expected_value": 22.51
      },
      
      "position_sizing": {
        "recommended_percent": 6.8,
        "dollar_amount": 6800.00,
        "method": "Kelly Criterion + Volatility Adjusted",
        "max_risk_percent": 3.4,
        "confidence_factor": 0.92,
        "volatility_factor": 0.028
      },
      
      "indicators": {
        "rsi": {"value": 62.15, "status": "neutral-bearish"},
        "macd": {"value": 1.23, "status": "bullish"},
        "volatility": {"value": 0.028, "status": "medium"},
        "volume": {"status": "high"},
        "trend": {"direction": "up", "strength": "strong"}
      },
      
      "chart_data": [ /* 30 days of OHLCV data */ ],
      "model_version": "v2.3.1",
      "reasoning": "Strong momentum with favorable R/R. High model confidence.",
      "risk_tolerance_match": true
    },
    {
      "rank": 2,
      "symbol": "TSLA",
      "signal": "BUY",
      "confidence": 0.84,
      "ranking_score": 0.8231,
      // ... similar structure
    },
    {
      "rank": 3,
      "symbol": "AMD",
      "signal": "BUY",
      "confidence": 0.79,
      "ranking_score": 0.7892,
      // ... similar structure
    }
  ],
  "metadata": {
    "symbols_analyzed": 32,
    "model_version": "v2.3.1",
    "risk_tolerance": "medium"
  }
}
```

### Ranking Algorithm
- **Confidence**: 35% weight
- **Risk/Reward Ratio**: 30% weight (capped at 5:1)
- **Win Probability**: 20% weight
- **Expected Value**: 15% weight
- **Bonus**: +10% for high confidence (>85%) AND high R/R (>3.0)

---

## 4. Portfolio Analytics

### Endpoint: `POST /api/portfolio`

### Request
```json
{
  "symbols": ["AAPL", "GOOGL", "NVDA", "TSLA"],
  "weights": [0.3, 0.25, 0.25, 0.2]
}
```

### Response
```json
{
  "portfolio": {
    "symbols": ["AAPL", "GOOGL", "NVDA", "TSLA"],
    "weights": [0.3, 0.25, 0.25, 0.2],
    "holdings": [
      {
        "symbol": "AAPL",
        "weight": 30.0,
        "current_price": 178.45,
        "allocation": "30.0%"
      },
      {
        "symbol": "GOOGL",
        "weight": 25.0,
        "current_price": 142.35,
        "allocation": "25.0%"
      },
      {
        "symbol": "NVDA",
        "weight": 25.0,
        "current_price": 458.75,
        "allocation": "25.0%"
      },
      {
        "symbol": "TSLA",
        "weight": 20.0,
        "current_price": 242.18,
        "allocation": "20.0%"
      }
    ]
  },
  
  "metrics": {
    "total_return": 12.45,
    "annualized_return": 24.32,
    "volatility": 18.75,
    "sharpe_ratio": 1.30,
    "max_drawdown": -8.42,
    "period_days": 30,
    "timestamp": "2025-10-08T10:30:00.123456"
  },
  
  "recommendations": {
    "rebalance_needed": false,
    "risk_level": "Moderate",
    "diversification_score": 40
  },
  
  "timestamp": "2025-10-08T10:30:00.123456"
}
```

### Database Schema
```sql
portfolio_state (
  id, user_id, symbol,
  quantity, entry_price, entry_date,
  current_price, current_value, pnl, pnl_percent,
  tp1_price, tp2_price, tp3_price, sl_price,
  updated_at
)
```

---

## 5. Game & Leaderboard

### Endpoint: `GET /api/game/leaderboard`

### Response
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "username": "TradingPro",
      "score": 25.4,
      "sharpe": 2.1,
      "games": 15
    },
    {
      "rank": 2,
      "username": "RL_Master",
      "score": 22.8,
      "sharpe": 1.9,
      "games": 12
    },
    {
      "rank": 3,
      "username": "QuanTitan",
      "score": 21.5,
      "sharpe": 1.8,
      "games": 20
    },
    {
      "rank": 4,
      "username": "MarketGuru",
      "score": 19.2,
      "sharpe": 1.6,
      "games": 8
    },
    {
      "rank": 5,
      "username": "AITrader",
      "score": 18.7,
      "sharpe": 1.5,
      "games": 10
    }
  ],
  "period": "all_time",
  "last_updated": "2025-10-08T10:30:00.123456"
}
```

### Database Schema
```sql
game_leaderboard (
  id, user_id, username,
  total_score, games_played, games_won,
  avg_return, avg_sharpe, avg_max_drawdown,
  best_return, worst_return,
  last_played, created_at, updated_at
)

game_sessions (
  id, game_id, user_id,
  period_start, period_end, initial_capital, available_symbols,
  status, current_day,
  final_value, total_return, sharpe_ratio, max_drawdown, num_trades,
  agent_return, agent_sharpe, user_vs_agent_score,
  started_at, completed_at
)
```

---

## 6. Chat/RAG Responses

### Endpoint: `POST /api/chat`

### Request
```json
{
  "message": "Should I buy AAPL now?",
  "context": {
    "portfolio": ["GOOGL", "MSFT"],
    "risk_tolerance": "medium"
  }
}
```

### Response
```json
{
  "query": "Should I buy AAPL now?",
  "response": {
    "answer": "Based on current market analysis, our RL models suggest considering stocks with strong momentum and positive technical indicators. Always ensure proper diversification and risk management.",
    "confidence": 0.85,
    "sources": ["Technical Analysis", "Model Predictions", "Risk Assessment"]
  },
  "timestamp": "2025-10-08T10:30:00.123456",
  "model": "WealthArena RAG v1.0"
}
```

---

## 7. System Metrics

### Endpoint: `GET /api/metrics/summary`

### Response
```json
{
  "system": {
    "status": "operational",
    "uptime": "99.9%",
    "last_data_update": "2025-10-08T10:30:00.123456",
    "database_type": "AzureSQL"
  },
  
  "data": {
    "symbols_tracked": 125,
    "symbols": ["AAPL", "GOOGL", "NVDA", "TSLA", "AMD", "..."],
    "last_fetch": "2025-10-08T18:00:00.000000"
  },
  
  "models": {
    "active_models": 5,
    "loaded_models": 5,
    "model_types": ["asx_stocks", "currency_pairs", "commodities", "cryptocurrencies", "etf"]
  },
  
  "users": {
    "active_users": 42,
    "games_in_progress": 8
  }
}
```

### Health Check: `GET /health`
```json
{
  "status": "healthy",
  "timestamp": "2025-10-08T10:30:00.123456",
  "database": "connected",
  "model_service": "ready"
}
```

---

## 8. Processed Features (Internal/Database)

### Feature Set for Each Symbol
These features are calculated during preprocessing and stored in the database. They are used by RL models but typically not directly exposed to frontend.

```sql
processed_features (
  -- Price Features
  returns, log_returns,
  volatility_5, volatility_20,
  
  -- Moving Averages
  sma_5, sma_10, sma_20, sma_50, sma_200,
  ema_12, ema_26, ema_50,
  
  -- Momentum Indicators
  rsi, rsi_6, rsi_21,
  macd, macd_signal, macd_hist,
  momentum_5, momentum_10, momentum_20,
  
  -- Bollinger Bands
  bb_upper, bb_middle, bb_lower,
  bb_width, bb_position,
  
  -- Volume Indicators
  volume_sma_20, volume_ratio,
  obv (On-Balance Volume),
  
  -- Support/Resistance
  atr (Average True Range),
  support_20, resistance_20,
  price_position,
  
  -- Metadata
  processed_at
)
```

### Feature Engineering Pipeline
```python
# Price-based features
- Returns (simple and log returns)
- Volatility (5-day and 20-day)
- Momentum (5, 10, 20 periods)
- Price ratios (open/close, high/close, low/close)

# Volume-based features
- Volume moving average (20-day)
- Volume ratio (current vs. average)
- Volume-weighted average price (VWAP)
- On-Balance Volume (OBV)

# Technical indicators
- RSI (6, 14, 21 periods)
- MACD (12, 26, 9)
- Bollinger Bands (20-day, 2 std dev)
- Moving Averages (SMA: 5,10,20,50,200; EMA: 12,26,50)
- ATR (Average True Range)

# Support/Resistance levels
- 20-day support (rolling min)
- 20-day resistance (rolling max)
- Price position relative to range

# Time-based features
- Hour of day
- Day of week
- Month
- Quarter
```

---

## Asset Type Categories

### Supported Asset Types
```json
{
  "stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"],
  "asx_stocks": ["CBA", "BHP", "CSL", "ANZ", "NAB", "WBC", "RIO", "FMG", "WOW", "WES"],
  "currency_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
  "commodities": ["GC", "CL", "SI"],
  "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
  "etf": ["SPY", "QQQ", "IWM", "DIA"]
}
```

---

## API Authentication (Future)

### Headers
```
Authorization: Bearer <jwt_token>
X-API-Key: <api_key>
Content-Type: application/json
```

---

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "SYMBOL_NOT_FOUND",
    "message": "Data not available for symbol: XYZ",
    "timestamp": "2025-10-08T10:30:00.123456",
    "path": "/api/predictions"
  }
}
```

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request (validation error)
- `404` - Not Found (symbol/data not available)
- `500` - Internal Server Error
- `503` - Service Unavailable (model loading)

---

## Key Metrics Glossary

### Trading Metrics
- **Sharpe Ratio**: Risk-adjusted return (>1.0 is good, >2.0 is excellent)
- **Max Drawdown**: Largest peak-to-trough decline (%)
- **Risk/Reward Ratio**: Expected reward divided by risk (>2.0 is good)
- **Win Probability**: Estimated probability of profitable trade (0-1)
- **Expected Value**: (Win Prob × Reward) - (Loss Prob × Risk)

### Technical Indicators
- **RSI**: Relative Strength Index (0-100, <30 oversold, >70 overbought)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands (2 std dev from 20-day SMA)
- **ATR**: Average True Range (volatility measure)
- **Volume Ratio**: Current volume / 20-day average

---

## Notes for Frontend/Backend Design

### Frontend Considerations
1. **Charts**: Use `chart_data` array for candlestick/line charts (30 days)
2. **TP/SL Visualization**: Display 3 take-profit levels and stop-loss on chart
3. **Real-time Updates**: Poll `/api/metrics/summary` every 30 seconds
4. **Position Sizing**: Display recommended % and $ amounts clearly
5. **Risk Indicators**: Color-code by risk level (green/yellow/red)

### Backend Considerations
1. **Caching**: Cache market data for 5 minutes
2. **Rate Limiting**: 100 requests/minute per user
3. **Model Loading**: Lazy load models on first request
4. **Database**: Use AzureSQL for production, local files for dev
5. **Async Processing**: Use background tasks for heavy computations

### Data Flow
```
Market Data → Airflow DAG → Preprocessing → Features →
→ RL Models → Predictions → Database → API → Frontend
```

---

## Example Frontend UI Components

### 1. Trading Signal Card
```
┌─────────────────────────────────────┐
│ AAPL - BUY Signal (87% confidence)  │
├─────────────────────────────────────┤
│ Entry: $178.45                      │
│ TP1: $182.30 (2.16%) - 50% close   │
│ TP2: $186.15 (4.31%) - 30% close   │
│ TP3: $190.00 (6.47%) - 20% close   │
│ SL: $175.89 (-1.43%) - Trailing    │
├─────────────────────────────────────┤
│ R/R: 3.02 | Win Prob: 74%          │
│ Position: 5.2% ($5,200)             │
└─────────────────────────────────────┘
```

### 2. Top Setups List
```
Rank  Symbol  Signal  Confidence  R/R   Score
1     NVDA    BUY     92%        3.58   0.87
2     TSLA    BUY     84%        3.12   0.82
3     AMD     BUY     79%        2.89   0.79
```

### 3. Portfolio Dashboard
```
Portfolio Value: $125,450 (+12.45%)
Sharpe: 1.30 | Max DD: -8.42%

Holdings:
- AAPL: 30% | $37,635 | +5.2%
- GOOGL: 25% | $31,362 | +8.7%
- NVDA: 25% | $31,362 | +18.3%
- TSLA: 20% | $25,090 | +2.1%
```

---

## Version History

- **v1.0.0** (2025-10-08): Initial schema definition
  - Market data endpoints
  - Prediction/signal structure with TP/SL
  - Top setups ranking
  - Portfolio analytics
  - Game & leaderboard
  - Chat/RAG interface

---

## Contact & Support

For questions about this API schema:
- Documentation: See `/docs` folder
- API Spec: `docs/rl_api_specification.md`
- Architecture: `docs/technical_architecture.md`

