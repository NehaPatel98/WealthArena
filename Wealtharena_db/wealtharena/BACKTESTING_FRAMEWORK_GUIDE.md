# Backtesting Framework Guide

## 📊 **Overview**

The WealthArena backtesting framework allows you to test trading strategies against historical data to evaluate their performance before risking real money. This comprehensive framework supports multiple strategy types and provides detailed performance analytics.

## 🎯 **Available Strategies**

### **1. Buy and Hold Strategy**
- **Description:** Purchase assets and hold them for the entire period
- **Best for:** Long-term investors, benchmark comparison
- **Risk Level:** Low to Medium
- **Use Case:** Baseline performance measurement

### **2. Rebalancing Strategy**
- **Description:** Periodically rebalance portfolio to maintain target weights
- **Best for:** Risk management, maintaining diversification
- **Risk Level:** Low to Medium
- **Use Case:** Systematic portfolio management

### **3. Momentum Strategy**
- **Description:** Buy assets showing positive momentum, sell those with negative momentum
- **Best for:** Trend-following investors
- **Risk Level:** Medium to High
- **Use Case:** Capturing market trends

### **4. Mean Reversion Strategy**
- **Description:** Buy assets that have fallen below their average, sell those above
- **Best for:** Contrarian investors
- **Risk Level:** Medium to High
- **Use Case:** Exploiting market inefficiencies

## 🚀 **API Endpoints**

### **Individual Strategy Backtests**

#### **Buy and Hold Strategy**
```http
POST /api/backtesting/buy-and-hold
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "startDate": "2023-01-01",
  "endDate": "2024-01-01",
  "initialCapital": 100000
}
```

#### **Rebalancing Strategy**
```http
POST /api/backtesting/rebalancing
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "startDate": "2023-01-01",
  "endDate": "2024-01-01",
  "initialCapital": 100000,
  "rebalanceFrequencyDays": 30
}
```

#### **Momentum Strategy**
```http
POST /api/backtesting/momentum
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "startDate": "2023-01-01",
  "endDate": "2024-01-01",
  "initialCapital": 100000,
  "lookbackDays": 20,
  "holdingDays": 10
}
```

#### **Mean Reversion Strategy**
```http
POST /api/backtesting/mean-reversion
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "startDate": "2023-01-01",
  "endDate": "2024-01-01",
  "initialCapital": 100000,
  "lookbackDays": 20,
  "threshold": 0.02
}
```

### **Strategy Comparison**
```http
POST /api/backtesting/compare
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "startDate": "2023-01-01",
  "endDate": "2024-01-01",
  "initialCapital": 100000
}
```

### **Performance Summary**
```http
GET /api/backtesting/summary?symbols=AAPL,MSFT,GOOGL&startDate=2023-01-01&endDate=2024-01-01&initialCapital=100000
```

### **Performance Metrics**
```http
GET /api/backtesting/metrics?symbols=AAPL,MSFT,GOOGL&startDate=2023-01-01&endDate=2024-01-01&initialCapital=100000
```

## 📈 **Response Format**

### **Backtest Result**
```json
{
  "strategyName": "Buy and Hold",
  "startDate": "2023-01-01",
  "endDate": "2024-01-01",
  "initialCapital": 100000.00,
  "finalCapital": 115000.00,
  "totalReturn": 0.15,
  "annualizedReturn": 0.15,
  "maxDrawdown": 0.08,
  "sharpeRatio": 1.25,
  "volatility": 0.18,
  "winRate": 0.65,
  "totalTrades": 4,
  "winningTrades": 3,
  "losingTrades": 1,
  "averageWin": 5000.00,
  "averageLoss": 2000.00,
  "profitFactor": 2.5,
  "trades": [
    {
      "date": "2023-01-01",
      "symbol": "AAPL",
      "action": "BUY",
      "quantity": 100.00,
      "price": 150.00,
      "value": 15000.00,
      "commission": 15.00,
      "pnl": 0.00,
      "cumulativePnl": 0.00
    }
  ],
  "portfolioSnapshots": [
    {
      "date": "2023-01-01",
      "totalValue": 100000.00,
      "cash": 0.00,
      "positions": {
        "AAPL": 100.00,
        "MSFT": 50.00
      },
      "prices": {
        "AAPL": 150.00,
        "MSFT": 300.00
      },
      "dailyReturn": 0.00,
      "cumulativeReturn": 0.00
    }
  ],
  "performanceMetrics": {
    "riskFreeRate": 0.02,
    "beta": 1.1,
    "alpha": 0.03
  },
  "calculatedAt": "2024-01-01"
}
```

## 🎯 **Performance Metrics Explained**

### **Return Metrics**
- **Total Return:** Overall percentage gain/loss
- **Annualized Return:** Return adjusted for time period
- **Daily Return:** Day-to-day percentage change

### **Risk Metrics**
- **Volatility:** Standard deviation of returns (risk measure)
- **Max Drawdown:** Largest peak-to-trough decline
- **Sharpe Ratio:** Risk-adjusted return (higher is better)

### **Trading Metrics**
- **Win Rate:** Percentage of profitable trades
- **Profit Factor:** Gross profit / Gross loss
- **Average Win/Loss:** Average profit/loss per trade

## 🔧 **Strategy Parameters**

### **Buy and Hold**
- **Symbols:** List of assets to invest in
- **Start/End Date:** Backtesting period
- **Initial Capital:** Starting investment amount

### **Rebalancing**
- **Rebalance Frequency:** Days between rebalancing (e.g., 30 for monthly)
- **Target Weights:** Desired allocation percentages

### **Momentum**
- **Lookback Days:** Period to calculate momentum (e.g., 20 days)
- **Holding Days:** How long to hold positions
- **Momentum Threshold:** Minimum momentum to trigger trade

### **Mean Reversion**
- **Lookback Days:** Period to calculate moving average
- **Threshold:** Deviation from mean to trigger trade
- **Reversion Period:** Time to hold mean-reverting positions

## 📊 **How to Use**

### **Step 1: Choose Your Strategy**
Select the strategy that matches your investment style:
- **Conservative:** Buy and Hold, Rebalancing
- **Moderate:** Mean Reversion
- **Aggressive:** Momentum

### **Step 2: Set Parameters**
- **Symbols:** Choose 3-10 stocks
- **Time Period:** At least 1 year of data
- **Initial Capital:** Realistic amount for testing

### **Step 3: Run Backtest**
Use the appropriate API endpoint for your strategy.

### **Step 4: Analyze Results**
- **Compare returns** across strategies
- **Check risk metrics** (volatility, drawdown)
- **Review trade details** for insights

## 🎯 **Best Practices**

### **1. Data Quality**
- Use at least 1 year of historical data
- Ensure data includes all trading days
- Check for missing or erroneous prices

### **2. Realistic Parameters**
- Include transaction costs (commissions)
- Use realistic initial capital amounts
- Consider market liquidity constraints

### **3. Multiple Time Periods**
- Test across different market conditions
- Include both bull and bear markets
- Validate strategy robustness

### **4. Risk Management**
- Monitor maximum drawdown
- Set position size limits
- Implement stop-loss rules

## 🔍 **Common Use Cases**

### **Portfolio Optimization**
- Test different asset allocations
- Compare rebalancing frequencies
- Evaluate risk-return tradeoffs

### **Strategy Development**
- Develop new trading algorithms
- Test parameter sensitivity
- Validate strategy logic

### **Risk Assessment**
- Measure portfolio volatility
- Calculate maximum drawdown
- Assess tail risk

### **Performance Attribution**
- Identify best/worst performing assets
- Analyze timing of trades
- Understand strategy behavior

## 🚀 **Advanced Features**

### **Strategy Comparison**
- Side-by-side performance comparison
- Risk-adjusted return analysis
- Drawdown comparison

### **Performance Attribution**
- Asset-level performance breakdown
- Timing analysis
- Risk factor decomposition

### **Monte Carlo Simulation**
- Random scenario testing
- Stress testing
- Confidence interval analysis

## 📚 **Mathematical Background**

### **Return Calculations**
- **Simple Return:** (End Value - Start Value) / Start Value
- **Log Return:** ln(End Value / Start Value)
- **Annualized Return:** (1 + Total Return)^(365/Days) - 1

### **Risk Calculations**
- **Volatility:** Standard deviation of returns
- **Sharpe Ratio:** (Return - Risk-Free Rate) / Volatility
- **Max Drawdown:** Maximum peak-to-trough decline

### **Trading Metrics**
- **Win Rate:** Winning Trades / Total Trades
- **Profit Factor:** Gross Profit / Gross Loss
- **Average Win:** Total Profit / Winning Trades

## 🎯 **Getting Started**

1. **Start your WealthArena application**
2. **Ensure you have historical data** for your chosen symbols
3. **Test with a simple buy-and-hold strategy** first
4. **Compare different strategies** to see which fits your style
5. **Analyze the results** and adjust parameters

## 📞 **Support**

For questions about backtesting:
- Check the API documentation
- Test with sample data first
- Monitor your strategy performance
- Adjust parameters based on results

---

**Happy Backtesting! 📈🚀**
