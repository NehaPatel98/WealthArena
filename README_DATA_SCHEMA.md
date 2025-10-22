# WealthArena Data Schema Documentation

## 📚 Complete Guide to Backend & Frontend Data Structures

I've created comprehensive documentation showing all the data that will be shared between your backend and frontend. This will help you design both systems effectively.

---

## 🎯 What You Asked For

> "I want you to show me the data you will be sharing with the backend to be displayed on the frontend so that I can design the back and frontends"

✅ **Delivered:** Complete data schema documentation with:
- All API endpoint structures
- Database schemas (AzureSQL)
- JSON examples for every endpoint
- Visual mockups for frontend design
- Data flow architecture diagrams

---

## 📁 Files Created (5 Documents)

### 1. **📘 API_DATA_SCHEMA.md** - Complete API Reference
**What's inside:**
- All 8 API endpoints with request/response formats
- Complete database schemas (AzureSQL tables)
- Model predictions with TP/SL structure ⭐
- Feature engineering details
- Error handling formats
- Metrics glossary

**Size:** ~50 pages  
**Best for:** Backend developers, Database designers

---

### 2. **📗 API_EXAMPLES.json** - Ready-to-Use Examples
**What's inside:**
- Real JSON examples for all endpoints
- Sample database records
- Feature sets and asset types
- Complete prediction structure with TP/SL

**Size:** ~300 lines of JSON  
**Best for:** API testing, Frontend mocking

---

### 3. **📕 DATA_FLOW_ARCHITECTURE.md** - Visual Architecture
**What's inside:**
- System architecture diagrams (ASCII art)
- Data pipeline flow (Airflow → Database → API)
- Integration points
- Performance targets
- Update schedules

**Size:** ~40 pages  
**Best for:** Understanding the complete system flow

---

### 4. **📙 DESIGNER_QUICK_START.md** - Designer's Guide
**What's inside:**
- Visual UI mockups for all components
- Trading signal card with TP/SL visualization ⭐
- Color schemes and typography
- Responsive design guidelines
- Component hierarchy
- Implementation checklist

**Size:** ~30 pages  
**Best for:** Frontend developers, UI/UX designers

---

### 5. **📔 DATA_SCHEMA_INDEX.md** - Navigation Guide
**What's inside:**
- Index of all documentation
- Quick navigation by role
- Search guide
- Implementation checklists

**Size:** This navigation file  
**Best for:** Finding what you need quickly

---

## 🌟 Key Feature: Trading Signals with TP/SL

The most important data structure you'll be working with:

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
    "max_risk_percent": 2.6
  }
}
```

---

## 🎨 Visual Preview: What Users Will See

### Trading Signal Card
```
┌──────────────────────────────────────────┐
│ 🟢 AAPL - BUY Signal                     │
│ Confidence: 87% | Model: v2.3.1          │
├──────────────────────────────────────────┤
│ Entry Price: $178.45                     │
├──────────────────────────────────────────┤
│ 🎯 Take Profit Levels:                   │
│   TP1: $182.30 (+2.16%) → Close 50%     │
│   TP2: $186.15 (+4.31%) → Close 30%     │
│   TP3: $190.00 (+6.47%) → Close 20%     │
│                                          │
│ 🛑 Stop Loss:                            │
│   SL: $175.89 (-1.43%) | Trailing       │
├──────────────────────────────────────────┤
│ 📈 Risk/Reward: 3.02:1                   │
│ 💰 Position: 5.2% ($5,200)               │
└──────────────────────────────────────────┘
```

### Price Chart with TP/SL
```
  $190 ━━━━━━━━━━━━━━━━━━━━━━━━━ TP3 (Green)
  $186 ━━━━━━━━━━━━━━━━━━━━━━━━━ TP2 (Green)
  $182 ━━━━━━━━━━━━━━━━━━━━━━━━━ TP1 (Green)
  $178 ●━━━━━━━━━━━━━━━━━━━━━━━━ Entry (Blue)
  $175 ━━━━━━━━━━━━━━━━━━━━━━━━━ SL (Red)
```

---

## 🔗 API Endpoints Overview

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|---------------|
| `/api/predictions` | POST | Get signal with TP/SL | < 500ms |
| `/api/top-setups` | POST | Get top 3 setups | < 1000ms |
| `/api/market-data` | POST | Get OHLCV data | < 200ms |
| `/api/portfolio` | POST | Analyze portfolio | < 300ms |
| `/api/game/leaderboard` | GET | Get leaderboard | < 200ms |
| `/api/chat` | POST | RAG chatbot | < 800ms |
| `/api/metrics/summary` | GET | System metrics | < 200ms |
| `/health` | GET | Health check | < 50ms |

---

## 🗄️ Database Tables

### Core Tables (AzureSQL)

1. **`raw_market_data`** - OHLCV data from Yahoo Finance
2. **`processed_features`** - Technical indicators (RSI, MACD, etc.)
3. **`model_predictions`** - Predictions with TP/SL ⭐
4. **`portfolio_state`** - User positions & P&L
5. **`game_leaderboard`** - Player rankings
6. **`game_sessions`** - Active/completed games
7. **`model_registry`** - ML model versions
8. **`audit_log`** - System audit trail

Full SQL schema: `backend/azure_sql_schema.sql`

---

## 📊 Data Flow Architecture

```
┌─────────────────────────────────────────────────┐
│             COMPLETE DATA PIPELINE              │
└─────────────────────────────────────────────────┘

Yahoo Finance (Market Data)
        │
        ▼
Airflow DAG (Daily 6 PM)
    ├── Fetch Market Data
    └── Calculate Features (RSI, MACD, etc.)
        │
        ▼
Data Storage
    ├── CSV Files (data/features/)
    └── AzureSQL Database
        │
        ▼
RL Models (3 Agents)
    ├── Trading Agent → Signal, Entry
    ├── Risk Mgmt Agent → TP/SL Levels
    └── Portfolio Mgr → Position Sizing
        │
        ▼
FastAPI Backend (Port 8000)
    ├── /api/predictions
    ├── /api/top-setups
    ├── /api/portfolio
    └── ... 5 more endpoints
        │
        ▼
Frontend (React/Vue/Angular)
    ├── Trading Signal Cards
    ├── Price Charts with TP/SL
    ├── Portfolio Dashboard
    └── Leaderboard
```

---

## 🚀 Quick Start Guide

### Step 1: Read the Documentation (30 min)

**For Frontend Design:**
1. Start with `DESIGNER_QUICK_START.md` (UI mockups)
2. Review `API_EXAMPLES.json` (data structures)
3. Reference `API_DATA_SCHEMA.md` (detailed specs)

**For Backend Development:**
1. Start with `API_DATA_SCHEMA.md` (complete API spec)
2. Review `DATA_FLOW_ARCHITECTURE.md` (system flow)
3. Test with `API_EXAMPLES.json` (JSON examples)

**For Project Management:**
1. Start with `DATA_SCHEMA_INDEX.md` (navigation)
2. Review `DESIGNER_QUICK_START.md` (visual overview)
3. Check `DATA_FLOW_ARCHITECTURE.md` (architecture)

---

### Step 2: Implement Your Part

**Frontend Developer:**
```javascript
// Example: Fetch trading signal
const response = await fetch('http://localhost:8000/api/predictions', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({symbol: 'AAPL', horizon: 1})
});

const signal = await response.json();
// Now display: signal.entry, signal.take_profit, signal.stop_loss
```

**Backend Developer:**
```python
# Example: Return prediction with TP/SL
@app.post("/api/predictions")
async def get_predictions(request: PredictionRequest):
    prediction = model_service.generate_prediction(
        request.symbol, df, asset_type='stock'
    )
    return prediction  # Includes TP/SL from Risk Mgmt agent
```

---

### Step 3: Test Everything

Use examples from `API_EXAMPLES.json` to:
- ✅ Test API endpoints
- ✅ Mock frontend data
- ✅ Validate data structures
- ✅ Check integration

---

## 🎨 Design System Preview

### Color Palette
- **BUY Signal**: Green `#10B981`
- **SELL Signal**: Red `#EF4444`
- **HOLD Signal**: Gray `#6B7280`
- **Entry Point**: Blue `#3B82F6`
- **TP Levels**: Green gradients
- **SL Level**: Red

### Typography
- **Headings**: 32px/24px/18px (Bold/Semibold/Medium)
- **Body**: 16px Regular
- **Prices**: 18px Bold Monospace
- **Metrics**: 14px Medium

### Components
1. Trading Signal Card (with TP/SL)
2. Price Chart (with TP/SL lines)
3. Top Setups List
4. Portfolio Dashboard
5. Leaderboard Table
6. Chatbot Interface

---

## 📋 Implementation Checklist

### Backend Tasks
- [ ] Set up FastAPI server (`backend/main.py`)
- [ ] Implement 8 API endpoints
- [ ] Configure database (AzureSQL or CSV)
- [ ] Load RL models (from `checkpoints/`)
- [ ] Add caching layer (5 min cache)
- [ ] Error handling & logging
- [ ] CORS configuration

### Frontend Tasks
- [ ] Create Trading Signal component
- [ ] Implement price chart with TP/SL visualization
- [ ] Build Top Setups page
- [ ] Design Portfolio Dashboard
- [ ] Add Leaderboard view
- [ ] Integrate Chatbot
- [ ] Responsive design (mobile/tablet/desktop)
- [ ] Real-time data polling

### Database Tasks
- [ ] Run `azure_sql_schema.sql`
- [ ] Set up tables and indexes
- [ ] Configure stored procedures
- [ ] Test data insertion
- [ ] Set up backups

---

## 📈 Performance Targets

- **API Response**: < 500ms for predictions
- **Page Load**: < 2 seconds
- **Real-time Updates**: Every 30 seconds
- **Database Queries**: < 100ms
- **Model Inference**: < 300ms

---

## 🔍 Finding What You Need

**Looking for API structure?** → `API_DATA_SCHEMA.md`  
**Need JSON examples?** → `API_EXAMPLES.json`  
**Want UI mockups?** → `DESIGNER_QUICK_START.md`  
**System architecture?** → `DATA_FLOW_ARCHITECTURE.md`  
**Navigation help?** → `DATA_SCHEMA_INDEX.md`  

---

## 💡 Key Insights

### What Makes WealthArena Unique
1. **RL-Powered Predictions**: All signals from trained RL agents
2. **Complete TP/SL Strategy**: 3 take-profit levels + trailing stop loss
3. **Risk-Aware Position Sizing**: Kelly Criterion + volatility adjustment
4. **Multi-Agent System**: Trading, Risk Mgmt, Portfolio Manager agents
5. **Real-Time Analytics**: Live portfolio tracking and leaderboard

### The Data Journey
```
Market Data → Features → RL Agents → Predictions (TP/SL) → API → UI
```

### The User Experience
```
1. User opens app
2. Sees Top 3 trading setups
3. Clicks on a signal
4. Views TP/SL levels on chart
5. Executes trade with recommended position size
6. Tracks P&L in portfolio
7. Competes on leaderboard
```

---

## 📞 Support & Resources

### Documentation Files (All in this directory)
- ✅ `API_DATA_SCHEMA.md` - Complete API reference
- ✅ `API_EXAMPLES.json` - JSON examples
- ✅ `DATA_FLOW_ARCHITECTURE.md` - System architecture
- ✅ `DESIGNER_QUICK_START.md` - Design guide
- ✅ `DATA_SCHEMA_INDEX.md` - Navigation index

### Code Files (In repository)
- `backend/main.py` - FastAPI server
- `backend/model_service.py` - RL model inference
- `backend/azure_sql_schema.sql` - Database schema
- `airflow/dags/preprocess_data_dag.py` - Data pipeline

### Additional Docs
- `docs/rl_api_specification.md` - Detailed API spec
- `docs/technical_architecture.md` - System design
- `README.md` - Project overview

---

## ✨ Summary

You now have **complete data schema documentation** covering:

✅ **8 API Endpoints** with full request/response specs  
✅ **8 Database Tables** with SQL schemas  
✅ **300+ Lines** of JSON examples  
✅ **Visual UI Mockups** for all components  
✅ **System Architecture** diagrams  
✅ **Design Guidelines** (colors, typography, responsive)  
✅ **Implementation Checklists** for frontend & backend  

**Total Documentation**: ~150 pages across 5 files!

---

## 🎯 Next Steps

1. ✅ **Review** the documentation files
2. ✅ **Design** your frontend using the mockups
3. ✅ **Implement** your backend using the API specs
4. ✅ **Test** with the JSON examples
5. ✅ **Deploy** following the architecture guide

---

## 🚀 Ready to Build!

You have everything you need to design and build both the backend and frontend for WealthArena. The data structures are clear, the API is well-defined, and the UI mockups are ready.

**Happy coding! 🎉**

---

**Documentation Version**: 1.0.0  
**Created**: October 8, 2025  
**Team**: WealthArena RL Engineering

---

**Questions?** Refer to `DATA_SCHEMA_INDEX.md` for quick navigation or review the specific documentation file for detailed information.

