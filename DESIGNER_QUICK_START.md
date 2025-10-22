# WealthArena Designer Quick Start Guide
**Everything You Need to Design the Frontend & Backend**

---

## 🎯 Overview

WealthArena is an RL-powered trading platform that provides:
- **Real-time trading signals** with Take Profit (TP) and Stop Loss (SL) levels
- **Top trading setups** ranked by AI confidence
- **Portfolio analytics** with risk metrics
- **Game/Leaderboard** for competitive trading
- **RAG chatbot** for trading advice

---

## 📊 Core Data You'll Be Displaying

### 1. Trading Signal Card (Most Important!)

**What the user sees:**
```
┌──────────────────────────────────────────┐
│ 🟢 AAPL - BUY Signal                     │
│ Confidence: 87% | Model: v2.3.1          │
├──────────────────────────────────────────┤
│ Entry Price: $178.45                     │
│ Range: $177.91 - $178.99                 │
│ Timing: Immediate                        │
├──────────────────────────────────────────┤
│ 🎯 Take Profit Levels:                   │
│   TP1: $182.30 (+2.16%) → Close 50%     │
│   TP2: $186.15 (+4.31%) → Close 30%     │
│   TP3: $190.00 (+6.47%) → Close 20%     │
│                                          │
│ 🛑 Stop Loss:                            │
│   SL: $175.89 (-1.43%) | Trailing       │
├──────────────────────────────────────────┤
│ 📈 Risk Metrics:                         │
│   Risk/Reward: 3.02:1                    │
│   Win Probability: 74%                   │
│   Expected Value: $5.05/share            │
├──────────────────────────────────────────┤
│ 💰 Position Sizing:                      │
│   Recommended: 5.2% ($5,200)             │
│   Max Risk: 2.6%                         │
├──────────────────────────────────────────┤
│ 📊 Technical Indicators:                 │
│   RSI: 58.42 (Neutral-Bearish) ⚪       │
│   MACD: 0.87 (Bullish) 🟢               │
│   Volatility: 0.023 (Medium) ⚪          │
│   Volume: High 🟢                        │
│   Trend: Up (Strong) 🟢                  │
├──────────────────────────────────────────┤
│ [View Chart] [Execute Trade] [Details]   │
└──────────────────────────────────────────┘
```

**Data structure:**
```json
{
  "symbol": "AAPL",
  "signal": "BUY",
  "confidence": 0.87,
  "entry": {
    "price": 178.45,
    "range": [177.91, 178.99],
    "timing": "immediate"
  },
  "take_profit": [
    {"level": 1, "price": 182.30, "percent": 2.16, "close_percent": 50},
    {"level": 2, "price": 186.15, "percent": 4.31, "close_percent": 30},
    {"level": 3, "price": 190.00, "percent": 6.47, "close_percent": 20}
  ],
  "stop_loss": {
    "price": 175.89,
    "percent": -1.43,
    "type": "trailing"
  },
  "risk_metrics": {
    "risk_reward_ratio": 3.02,
    "win_probability": 0.74,
    "expected_value": 5.05
  },
  "position_sizing": {
    "recommended_percent": 5.2,
    "dollar_amount": 5200
  }
}
```

---

### 2. Chart with TP/SL Visualization

**What the user sees:**
```
Price Chart (Last 30 Days)
┌──────────────────────────────────────────────────────┐
│                                                      │
│  $190 ━━━━━━━━━━━━━━━━━━━━━━━━━ TP3 (Green line)  │
│                                                      │
│  $186 ━━━━━━━━━━━━━━━━━━━━━━━━━ TP2 (Green line)  │
│                                                      │
│  $182 ━━━━━━━━━━━━━━━━━━━━━━━━━ TP1 (Green line)  │
│                    📈                                │
│  $178 ●━━━━━━━━━━━━━━━━━━━━━━━━ Entry (Blue dot)  │
│     ╱ ╲    ╱╲  ╱╲  ╱╲                              │
│    ╱   ╲  ╱  ╲╱  ╲╱  ╲  📊 Candlesticks           │
│   ╱     ╲╱                                          │
│  $175 ━━━━━━━━━━━━━━━━━━━━━━━━━ SL (Red line)    │
│                                                      │
│  ───────────────────────────────────────────────    │
│  Sep 8    Sep 15    Sep 22    Sep 29    Oct 6      │
└──────────────────────────────────────────────────────┘

Volume
┌──────────────────────────────────────────────────────┐
│  ▂▅▃▇▅▂▃▅▇▂▅▃▇▅▂▃▅▇▂▅▃▇▅▂▃▅▇▂▅▃                    │
└──────────────────────────────────────────────────────┘
```

**Data structure:**
```json
{
  "chart_data": [
    {
      "date": "2025-09-08",
      "open": 172.50,
      "high": 174.20,
      "low": 171.80,
      "close": 173.90,
      "volume": 48392847
    }
  ],
  "tp_levels": [182.30, 186.15, 190.00],
  "sl_level": 175.89,
  "entry_level": 178.45
}
```

---

### 3. Top 3 Trading Setups (Homepage)

**What the user sees:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ 🏆 Top Trading Setups - Stocks                                      │
│ Updated: Oct 8, 2025 10:30 AM                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ #1  NVDA - BUY (92% confidence) ⭐⭐⭐                              │
│     Entry: $458.75 → TP3: $500.00 (+9.0%)                          │
│     R/R: 3.58 | Position: 6.8% ($6,800)                            │
│     [View Details] [Trade Now]                                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ #2  TSLA - BUY (84% confidence) ⭐⭐                                │
│     Entry: $242.18 → TP3: $265.00 (+9.4%)                          │
│     R/R: 3.12 | Position: 5.4% ($5,400)                            │
│     [View Details] [Trade Now]                                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ #3  AMD - BUY (79% confidence) ⭐⭐                                 │
│     Entry: $142.50 → TP3: $158.00 (+10.9%)                         │
│     R/R: 2.89 | Position: 4.8% ($4,800)                            │
│     [View Details] [Trade Now]                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Filter by: [Stocks] [Crypto] [Currency Pairs] [Commodities]
Risk Tolerance: [Low] [Medium] [High]
```

**Data structure:**
```json
{
  "setups": [
    {
      "rank": 1,
      "symbol": "NVDA",
      "signal": "BUY",
      "confidence": 0.92,
      "entry": {"price": 458.75},
      "take_profit": [
        {"level": 3, "price": 500.00, "percent": 9.0}
      ],
      "risk_metrics": {"risk_reward_ratio": 3.58},
      "position_sizing": {"recommended_percent": 6.8}
    }
  ]
}
```

---

### 4. Portfolio Dashboard

**What the user sees:**
```
┌──────────────────────────────────────────────────────────┐
│ 💼 My Portfolio                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Portfolio Value: $125,450                              │
│  Total Return: +$12,450 (+12.45%) 🟢                    │
│                                                          │
│  📊 Performance Metrics:                                 │
│  • Sharpe Ratio: 1.30                                   │
│  • Volatility: 18.75%                                   │
│  • Max Drawdown: -8.42%                                 │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Holdings:                                              │
│                                                          │
│  AAPL  ████████████ 30.0% | $37,635 | +5.2% 🟢         │
│  GOOGL ██████████   25.0% | $31,362 | +8.7% 🟢         │
│  NVDA  ██████████   25.0% | $31,362 | +18.3% 🟢        │
│  TSLA  ████████     20.0% | $25,090 | +2.1% 🟢         │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  [Rebalance] [Add Position] [View History]              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Data structure:**
```json
{
  "portfolio": {
    "holdings": [
      {"symbol": "AAPL", "weight": 30.0, "current_price": 178.45, "allocation": "30.0%"}
    ]
  },
  "metrics": {
    "total_return": 12.45,
    "sharpe_ratio": 1.30,
    "volatility": 18.75,
    "max_drawdown": -8.42
  }
}
```

---

### 5. Leaderboard

**What the user sees:**
```
┌──────────────────────────────────────────────────────────┐
│ 🏆 Top Traders - All Time                               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Rank  Player       Score   Sharpe   Games   Streak     │
│  ──────────────────────────────────────────────────     │
│  🥇 1  TradingPro   25.4    2.1      15      🔥🔥🔥     │
│  🥈 2  RL_Master    22.8    1.9      12      🔥🔥       │
│  🥉 3  QuanTitan    21.5    1.8      20      🔥         │
│  ─  4  MarketGuru   19.2    1.6       8      ─         │
│  ─  5  AITrader     18.7    1.5      10      ─         │
│  ─  6  You          15.3    1.2       5      ─         │
│                                                          │
└──────────────────────────────────────────────────────────┘

Period: [All Time] [This Month] [This Week]
```

**Data structure:**
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "username": "TradingPro",
      "score": 25.4,
      "sharpe": 2.1,
      "games": 15
    }
  ]
}
```

---

### 6. Chatbot

**What the user sees:**
```
┌──────────────────────────────────────────────────────────┐
│ 💬 WealthArena Assistant                                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  You: Should I buy AAPL now?                            │
│                                                          │
│  🤖 Bot: Based on current market analysis, our RL       │
│         models suggest AAPL shows strong momentum       │
│         with favorable risk/reward (3.02:1). Current    │
│         signal: BUY with 87% confidence.                │
│                                                          │
│         Entry: $178.45                                  │
│         TP1: $182.30 (+2.16%)                          │
│         SL: $175.89 (-1.43%)                           │
│                                                          │
│         Recommendation: Consider a 5.2% position        │
│         ($5,200) based on Kelly Criterion.              │
│                                                          │
│  Confidence: 85% ⭐⭐⭐⭐                                 │
│  Sources: Technical Analysis, Model Predictions         │
│                                                          │
│  [View Full Analysis] [Execute Trade]                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Data structure:**
```json
{
  "query": "Should I buy AAPL now?",
  "response": {
    "answer": "Based on current market analysis...",
    "confidence": 0.85,
    "sources": ["Technical Analysis", "Model Predictions"]
  }
}
```

---

## 🎨 Design Guidelines

### Color Scheme

**Signal Colors:**
- 🟢 **BUY Signal**: Green (#10B981)
- 🔴 **SELL Signal**: Red (#EF4444)
- ⚪ **HOLD Signal**: Gray (#6B7280)

**UI Elements:**
- **Take Profit**: Green gradients (#10B981 → #34D399)
- **Stop Loss**: Red (#EF4444)
- **Entry Point**: Blue (#3B82F6)
- **Background**: Dark mode (#1F2937) or Light mode (#F9FAFB)

**Status Indicators:**
- **High Confidence (>85%)**: ⭐⭐⭐ (3 stars) - Green
- **Medium Confidence (70-85%)**: ⭐⭐ (2 stars) - Yellow
- **Low Confidence (<70%)**: ⭐ (1 star) - Orange

**Technical Indicator Status:**
- 🟢 **Bullish**: Green
- 🔴 **Bearish**: Red
- ⚪ **Neutral**: Gray

### Typography

**Headings:**
- H1: 32px, Bold (Portfolio Value, Page Titles)
- H2: 24px, Semibold (Section Headers)
- H3: 18px, Medium (Card Titles)

**Body:**
- Regular: 16px, Normal (General text)
- Small: 14px, Normal (Metadata, timestamps)
- Tiny: 12px, Normal (Labels, captions)

**Numbers:**
- Prices: 18px, Bold, Monospace
- Percentages: 16px, Semibold
- Metrics: 14px, Medium

### Layout Components

**Card:**
```
┌─────────────────────────┐
│ Header                  │
├─────────────────────────┤
│ Content                 │
│ • Key metrics           │
│ • Visual elements       │
├─────────────────────────┤
│ [Actions]               │
└─────────────────────────┘
```

**Data Grid:**
```
┌──────────┬──────────┬──────────┐
│ Column 1 │ Column 2 │ Column 3 │
├──────────┼──────────┼──────────┤
│ Data     │ Data     │ Data     │
│ Data     │ Data     │ Data     │
└──────────┴──────────┴──────────┘
```

---

## 📱 Responsive Breakpoints

- **Mobile**: < 640px (Single column, stack cards)
- **Tablet**: 640px - 1024px (2 columns)
- **Desktop**: > 1024px (3 columns, sidebar)

---

## 🔄 Real-time Updates

**Polling Intervals:**
- Market prices: Every 30 seconds
- Portfolio value: Every 60 seconds
- Leaderboard: Every 5 minutes
- System status: Every 30 seconds

**WebSocket (Future):**
- Live price updates
- Trade notifications
- Game events

---

## 🧩 Component Hierarchy

```
App
├── Dashboard
│   ├── SystemStatus
│   ├── TopSetups (3 cards)
│   └── QuickStats
│
├── TradingSignals
│   ├── SymbolSearch
│   ├── SignalCard (with TP/SL)
│   ├── PriceChart
│   └── TechnicalIndicators
│
├── Portfolio
│   ├── PortfolioSummary
│   ├── HoldingsTable
│   ├── PerformanceChart
│   └── Recommendations
│
├── Game
│   ├── Leaderboard
│   ├── GameSession
│   └── PlayerStats
│
└── Chatbot
    ├── ChatHistory
    ├── MessageInput
    └── SuggestedActions
```

---

## 🔗 API Endpoints Reference

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|---------------|
| `/api/market-data` | POST | Get OHLCV data | < 200ms |
| `/api/predictions` | POST | Get signal with TP/SL | < 500ms |
| `/api/top-setups` | POST | Get top 3 setups | < 1000ms |
| `/api/portfolio` | POST | Analyze portfolio | < 300ms |
| `/api/game/leaderboard` | GET | Get leaderboard | < 200ms |
| `/api/chat` | POST | Ask chatbot | < 800ms |
| `/health` | GET | Health check | < 50ms |

---

## 📋 Implementation Checklist

### Frontend (React/Vue/Angular)

- [ ] Dashboard with top 3 setups
- [ ] Trading signal page with TP/SL visualization
- [ ] Interactive price chart with TP/SL lines
- [ ] Portfolio analytics page
- [ ] Leaderboard with rankings
- [ ] Chatbot interface
- [ ] Mobile responsive design
- [ ] Dark/Light mode toggle
- [ ] Loading states & error handling
- [ ] Real-time data polling

### Backend (FastAPI)

- [ ] `/api/market-data` endpoint
- [ ] `/api/predictions` endpoint with TP/SL
- [ ] `/api/top-setups` endpoint with ranking
- [ ] `/api/portfolio` endpoint
- [ ] `/api/game/leaderboard` endpoint
- [ ] `/api/chat` endpoint (RAG)
- [ ] Database integration (AzureSQL or local CSV)
- [ ] Model loading service
- [ ] Caching layer (5 min cache)
- [ ] CORS configuration
- [ ] Error handling & logging

---

## 🚀 Quick Start Commands

### Run Backend
```bash
cd wealtharena_rl/backend
python main.py
# Server starts at http://localhost:8000
```

### Test API
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizon": 1}'
```

### View Documentation
```bash
# Open browser to http://localhost:8000/docs
# Swagger UI with interactive API testing
```

---

## 📚 Additional Resources

- **Full API Schema**: `API_DATA_SCHEMA.md`
- **JSON Examples**: `API_EXAMPLES.json`
- **Architecture**: `DATA_FLOW_ARCHITECTURE.md`
- **Technical Docs**: `docs/rl_api_specification.md`

---

## 💡 Design Tips

1. **Highlight TP/SL Clearly**: This is the most important feature!
2. **Use Color Coding**: Green = profit, Red = loss, Blue = entry
3. **Show Confidence Visually**: Stars, progress bars, or badges
4. **Make Charts Interactive**: Hover to see exact values
5. **Mobile-First**: Ensure all features work on mobile
6. **Loading States**: Show skeleton screens while loading
7. **Error Messages**: Be clear about what went wrong
8. **Tooltips**: Explain technical terms (RSI, MACD, etc.)
9. **Call-to-Actions**: Make "Trade Now" buttons prominent
10. **Real-time Feel**: Update prices smoothly

---

## ❓ Common Questions

**Q: Where does the TP/SL data come from?**  
A: The RL Risk Management Agent calculates optimal TP/SL levels based on volatility, confidence, and learned risk/reward patterns.

**Q: How often is data updated?**  
A: Market data is fetched daily at 6 PM. API serves fresh predictions on-demand with 5-minute caching.

**Q: What does "Close 50%" mean for TP1?**  
A: It means close 50% of your position when price hits TP1, keep the rest for TP2/TP3.

**Q: What is "Trailing" stop loss?**  
A: The stop loss moves up as price increases, locking in profits.

**Q: How is the ranking score calculated?**  
A: Weighted combination of confidence (35%), R/R ratio (30%), win probability (20%), and expected value (15%).

---

**Happy Designing! 🎨**

For questions, refer to the full documentation or contact the RL team.

