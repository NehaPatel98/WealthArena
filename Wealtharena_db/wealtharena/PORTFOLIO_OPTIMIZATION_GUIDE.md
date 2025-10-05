# Portfolio Optimization Algorithms Guide

## 📊 **Overview**

This guide explains the portfolio optimization algorithms implemented in WealthArena. These algorithms help investors find the optimal mix of assets to maximize returns while minimizing risk.

## 🎯 **Available Algorithms**

### 1. **Modern Portfolio Theory (MPT)**
- **Purpose:** Maximizes Sharpe ratio (return per unit of risk)
- **Best for:** Balanced investors seeking optimal risk-return tradeoff
- **Method:** Markowitz optimization

### 2. **Risk Parity**
- **Purpose:** Equalizes risk contribution from each asset
- **Best for:** Conservative investors who want balanced risk exposure
- **Method:** Equal risk contribution optimization

### 3. **Black-Litterman Model**
- **Purpose:** Incorporates investor views into optimization
- **Best for:** Experienced investors with specific market views
- **Method:** Bayesian approach combining market equilibrium with investor opinions

### 4. **Minimum Variance**
- **Purpose:** Finds portfolio with lowest possible risk
- **Best for:** Very conservative investors prioritizing capital preservation
- **Method:** Risk minimization optimization

## 🚀 **API Endpoints**

### **Modern Portfolio Theory**
```http
POST /api/portfolio/optimization/mpt
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "riskFreeRate": 0.02
}
```

### **Risk Parity**
```http
POST /api/portfolio/optimization/risk-parity
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"]
}
```

### **Black-Litterman**
```http
POST /api/portfolio/optimization/black-litterman
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "investorViews": {
    "AAPL": 0.15,
    "MSFT": 0.12
  },
  "riskFreeRate": 0.02,
  "confidenceLevel": 0.5
}
```

### **Minimum Variance**
```http
POST /api/portfolio/optimization/minimum-variance
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"]
}
```

### **Compare All Algorithms**
```http
GET /api/portfolio/optimization/compare?symbols=AAPL,MSFT,GOOGL,AMZN&riskFreeRate=0.02
```

### **Get Recommendations**
```http
GET /api/portfolio/optimization/recommendations?symbols=AAPL,MSFT,GOOGL,AMZN&riskTolerance=moderate
```

## 📈 **Response Format**

```json
{
  "optimizationType": "Modern Portfolio Theory",
  "expectedReturn": 0.125,
  "expectedRisk": 0.18,
  "sharpeRatio": 0.58,
  "allocations": [
    {
      "symbol": "AAPL",
      "weight": 0.35,
      "expectedReturn": 0.15,
      "risk": 0.25,
      "currentPrice": 150.00,
      "shares": 233.33
    }
  ],
  "metadata": {
    "riskFreeRate": 0.02,
    "optimizationMethod": "Markowitz",
    "dataPoints": 252
  },
  "calculatedAt": "2025-09-26T17:30:00"
}
```

## 🎯 **How to Use**

### **Step 1: Choose Your Risk Tolerance**
- **Conservative:** Use Minimum Variance
- **Moderate:** Use Risk Parity
- **Aggressive:** Use Modern Portfolio Theory
- **Custom Views:** Use Black-Litterman

### **Step 2: Select Your Assets**
Choose 3-10 stocks from your portfolio or market favorites:
- `["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]`

### **Step 3: Run Optimization**
Use the appropriate endpoint based on your risk tolerance.

### **Step 4: Analyze Results**
- **Expected Return:** Annualized return expectation
- **Expected Risk:** Portfolio volatility (standard deviation)
- **Sharpe Ratio:** Risk-adjusted return (higher is better)
- **Allocations:** How much to invest in each asset

## 🔧 **Advanced Features**

### **Risk Tolerance Levels**
- **Conservative:** Prioritizes capital preservation
- **Moderate:** Balanced risk-return approach
- **Aggressive:** Maximizes potential returns

### **Investor Views (Black-Litterman)**
Express your market opinions:
```json
{
  "AAPL": 0.15,    // 15% expected return for Apple
  "MSFT": 0.12,    // 12% expected return for Microsoft
  "GOOGL": 0.10    // 10% expected return for Google
}
```

### **Confidence Levels**
- **0.1:** Low confidence in views
- **0.5:** Medium confidence
- **0.9:** High confidence

## 📊 **Example Scenarios**

### **Scenario 1: Conservative Investor**
```http
GET /api/portfolio/optimization/recommendations?symbols=AAPL,MSFT,GOOGL&riskTolerance=conservative
```
**Result:** Minimum Variance portfolio with low risk

### **Scenario 2: Balanced Investor**
```http
GET /api/portfolio/optimization/recommendations?symbols=AAPL,MSFT,GOOGL&riskTolerance=moderate
```
**Result:** Risk Parity portfolio with balanced risk exposure

### **Scenario 3: Aggressive Investor**
```http
GET /api/portfolio/optimization/recommendations?symbols=AAPL,MSFT,GOOGL&riskTolerance=aggressive
```
**Result:** MPT portfolio maximizing Sharpe ratio

## 🎯 **Best Practices**

### **1. Diversification**
- Use 5-10 different assets
- Include different sectors
- Mix stocks and bonds

### **2. Regular Rebalancing**
- Re-run optimization monthly
- Adjust allocations as markets change
- Monitor performance metrics

### **3. Risk Management**
- Set maximum position sizes
- Monitor correlation between assets
- Consider transaction costs

### **4. Backtesting**
- Test strategies on historical data
- Compare different algorithms
- Validate assumptions

## 🔍 **Troubleshooting**

### **Common Issues**
1. **Insufficient Data:** Need at least 1 year of price data
2. **High Correlation:** Assets too similar reduce diversification benefits
3. **Extreme Weights:** Some algorithms may suggest 100% in one asset

### **Solutions**
1. **Add More Assets:** Increase diversification
2. **Adjust Timeframe:** Use longer historical periods
3. **Set Constraints:** Limit maximum position sizes

## 📚 **Mathematical Background**

### **Modern Portfolio Theory**
- **Objective:** Maximize (Expected Return - Risk-Free Rate) / Risk
- **Constraints:** Weights sum to 1, non-negative weights
- **Method:** Quadratic programming

### **Risk Parity**
- **Objective:** Equal risk contribution from each asset
- **Formula:** Risk Contribution = Weight × Asset Risk
- **Method:** Iterative optimization

### **Black-Litterman**
- **Formula:** E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹Π + P'Ω⁻¹Q]
- **Where:** τ = scaling factor, Σ = covariance matrix, P = pick matrix, Ω = uncertainty matrix, Q = investor views

### **Minimum Variance**
- **Objective:** Minimize portfolio variance
- **Formula:** σ²p = w'Σw
- **Method:** Quadratic programming with equality constraint

## 🚀 **Getting Started**

1. **Start your WealthArena application**
2. **Ensure you have market data** for your chosen symbols
3. **Test with a simple 3-asset portfolio** first
4. **Compare different algorithms** to see which fits your style
5. **Implement the recommended allocations** in your actual portfolio

## 📞 **Support**

For questions about portfolio optimization:
- Check the API documentation
- Test with sample data first
- Monitor your portfolio performance
- Adjust strategies based on results

---

**Happy Optimizing! 📈🚀**
