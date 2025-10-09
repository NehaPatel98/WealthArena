# Real-Time Portfolio Tracking Guide

## 📊 **Overview**

The WealthArena real-time portfolio tracking system provides live updates of portfolio performance, market data, and alerts. This comprehensive system includes WebSocket connections, real-time calculations, and automated monitoring.

## 🚀 **Key Features**

### **1. Real-Time Portfolio Snapshots**
- **Live portfolio values** updated every 30 seconds
- **Position-level tracking** with current prices
- **P&L calculations** in real-time
- **Performance metrics** and analytics

### **2. WebSocket Streaming**
- **Server-Sent Events (SSE)** for live updates
- **Automatic reconnection** on connection loss
- **Efficient data streaming** with minimal bandwidth

### **3. Market Data Integration**
- **Live price feeds** from Yahoo Finance
- **Cached fallback** for reliability
- **Multi-symbol** price updates

### **4. Portfolio Analytics**
- **Performance tracking** over time
- **Risk metrics** and alerts
- **Diversification analysis**
- **Position concentration** warnings

## 🎯 **API Endpoints**

### **Portfolio Snapshots**

#### **Get Single Portfolio Snapshot**
```http
GET /api/realtime/portfolio/{portfolioId}
```

#### **Get User Portfolio Snapshots**
```http
GET /api/realtime/portfolios/{userId}
```

#### **Subscribe to Portfolio Updates (SSE)**
```http
GET /api/realtime/portfolio/{portfolioId}/stream
```

### **Market Data**

#### **Get Real-Time Prices**
```http
POST /api/realtime/prices
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"]
}
```

#### **Get Market Overview**
```http
GET /api/realtime/market/overview
```

### **Portfolio Analytics**

#### **Get Portfolio History**
```http
GET /api/realtime/portfolio/{portfolioId}/history?days=30
```

#### **Get Portfolio Alerts**
```http
GET /api/realtime/portfolio/{portfolioId}/alerts
```

#### **Get Portfolio Analytics**
```http
GET /api/realtime/portfolio/{portfolioId}/analytics
```

### **Dashboard & Comparison**

#### **Get Dashboard Data**
```http
GET /api/realtime/dashboard/{userId}
```

#### **Compare Portfolios**
```http
GET /api/realtime/compare/{userId}
```

## 📈 **Response Format**

### **Portfolio Snapshot**
```json
{
  "portfolioId": 1,
  "portfolioName": "My Portfolio",
  "timestamp": "2024-01-01T10:30:00",
  "totalValue": 105000.00,
  "cash": 5000.00,
  "totalCost": 100000.00,
  "totalPnl": 5000.00,
  "totalPnlPercent": 0.05,
  "dayChange": 250.00,
  "dayChangePercent": 0.0024,
  "positions": {
    "AAPL": {
      "symbol": "AAPL",
      "quantity": 100.00,
      "currentPrice": 150.00,
      "marketValue": 15000.00,
      "costBasis": 14000.00,
      "pnl": 1000.00,
      "pnlPercent": 0.0714,
      "dayChange": 50.00,
      "dayChangePercent": 0.0033,
      "weight": 0.1429,
      "lastUpdated": "2024-01-01T10:30:00"
    }
  },
  "performanceMetrics": {
    "totalValue": 105000.00,
    "totalCost": 100000.00,
    "totalPnl": 5000.00,
    "totalPnlPercent": 0.05,
    "cash": 5000.00,
    "positionCount": 5,
    "totalWeight": 1.0000,
    "largestPosition": "AAPL",
    "largestPositionWeight": 0.1429,
    "positionPerformance": {
      "AAPL": {
        "marketValue": 15000.00,
        "pnl": 1000.00,
        "pnlPercent": 0.0714,
        "weight": 0.1429
      }
    }
  }
}
```

### **Portfolio Alerts**
```json
[
  {
    "type": "P&L_ALERT",
    "message": "Portfolio P&L: 5.00%",
    "severity": "POSITIVE",
    "timestamp": "2024-01-01T10:30:00"
  },
  {
    "type": "CONCENTRATION_ALERT",
    "message": "High concentration in AAPL: 35.00%",
    "severity": "WARNING",
    "timestamp": "2024-01-01T10:30:00"
  }
]
```

### **Dashboard Data**
```json
{
  "portfolios": [...],
  "totalValue": 250000.00,
  "totalPnl": 15000.00,
  "portfolioCount": 3,
  "lastUpdated": "2024-01-01T10:30:00"
}
```

## 🔧 **How to Use**

### **Step 1: Get Portfolio Snapshot**
```javascript
// Get current portfolio state
fetch('/api/realtime/portfolio/1')
  .then(response => response.json())
  .then(data => console.log(data));
```

### **Step 2: Subscribe to Live Updates**
```javascript
// Subscribe to real-time updates
const eventSource = new EventSource('/api/realtime/portfolio/1/stream');

eventSource.onmessage = function(event) {
  const snapshot = JSON.parse(event.data);
  updatePortfolioDisplay(snapshot);
};

eventSource.onerror = function(event) {
  console.error('SSE connection error:', event);
};
```

### **Step 3: Get Market Prices**
```javascript
// Get real-time prices
fetch('/api/realtime/prices', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ symbols: ['AAPL', 'MSFT', 'GOOGL'] })
})
.then(response => response.json())
.then(prices => console.log(prices));
```

### **Step 4: Monitor Portfolio Alerts**
```javascript
// Get portfolio alerts
fetch('/api/realtime/portfolio/1/alerts')
  .then(response => response.json())
  .then(alerts => {
    alerts.forEach(alert => {
      showNotification(alert.message, alert.severity);
    });
  });
});
```

## 📊 **Real-Time Features**

### **1. Live Price Updates**
- **30-second intervals** for portfolio updates
- **Cached fallback** for reliability
- **Multi-source** price feeds

### **2. Performance Tracking**
- **Real-time P&L** calculations
- **Position-level** performance metrics
- **Portfolio-level** analytics

### **3. Risk Monitoring**
- **Concentration alerts** for over-weighted positions
- **P&L alerts** for significant changes
- **Performance warnings** for underperforming assets

### **4. Dashboard Integration**
- **Multi-portfolio** overview
- **Comparative analysis** across portfolios
- **Market overview** with key metrics

## 🎯 **WebSocket Implementation**

### **Server-Sent Events (SSE)**
```javascript
// Connect to portfolio stream
const eventSource = new EventSource('/api/realtime/portfolio/1/stream');

// Handle updates
eventSource.onmessage = function(event) {
  const portfolioSnapshot = JSON.parse(event.data);
  
  // Update UI with new data
  updatePortfolioValue(portfolioSnapshot.totalValue);
  updatePositions(portfolioSnapshot.positions);
  updateAlerts(portfolioSnapshot.alerts);
};

// Handle connection events
eventSource.onopen = function(event) {
  console.log('Connected to portfolio stream');
};

eventSource.onerror = function(event) {
  console.error('Connection error:', event);
  // Implement reconnection logic
};
```

### **Automatic Reconnection**
```javascript
function connectToPortfolioStream(portfolioId) {
  const eventSource = new EventSource(`/api/realtime/portfolio/${portfolioId}/stream`);
  
  eventSource.onerror = function(event) {
    eventSource.close();
    // Reconnect after 5 seconds
    setTimeout(() => connectToPortfolioStream(portfolioId), 5000);
  };
  
  return eventSource;
}
```

## 📱 **Frontend Integration**

### **React Component Example**
```jsx
import React, { useState, useEffect } from 'react';

function PortfolioTracker({ portfolioId }) {
  const [snapshot, setSnapshot] = useState(null);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    // Get initial snapshot
    fetch(`/api/realtime/portfolio/${portfolioId}`)
      .then(response => response.json())
      .then(data => setSnapshot(data));

    // Subscribe to updates
    const eventSource = new EventSource(`/api/realtime/portfolio/${portfolioId}/stream`);
    
    eventSource.onmessage = function(event) {
      const data = JSON.parse(event.data);
      setSnapshot(data);
    };

    return () => eventSource.close();
  }, [portfolioId]);

  return (
    <div>
      <h2>Portfolio: {snapshot?.portfolioName}</h2>
      <p>Total Value: ${snapshot?.totalValue}</p>
      <p>P&L: ${snapshot?.totalPnl} ({snapshot?.totalPnlPercent}%)</p>
      {/* Render positions and alerts */}
    </div>
  );
}
```

### **Vue.js Component Example**
```vue
<template>
  <div class="portfolio-tracker">
    <h2>{{ snapshot?.portfolioName }}</h2>
    <div class="portfolio-value">
      <span class="value">${{ snapshot?.totalValue }}</span>
      <span class="pnl" :class="pnlClass">
        ${{ snapshot?.totalPnl }} ({{ snapshot?.totalPnlPercent }}%)
      </span>
    </div>
    <div class="positions">
      <div v-for="(position, symbol) in snapshot?.positions" :key="symbol">
        <h3>{{ symbol }}</h3>
        <p>Value: ${{ position.marketValue }}</p>
        <p>P&L: ${{ position.pnl }}</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      snapshot: null,
      eventSource: null
    };
  },
  computed: {
    pnlClass() {
      if (!this.snapshot) return '';
      return this.snapshot.totalPnl >= 0 ? 'positive' : 'negative';
    }
  },
  mounted() {
    this.connectToStream();
  },
  beforeUnmount() {
    if (this.eventSource) {
      this.eventSource.close();
    }
  },
  methods: {
    connectToStream() {
      this.eventSource = new EventSource(`/api/realtime/portfolio/${this.portfolioId}/stream`);
      
      this.eventSource.onmessage = (event) => {
        this.snapshot = JSON.parse(event.data);
      };
    }
  }
};
</script>
```

## 🔍 **Performance Optimization**

### **1. Caching Strategy**
- **Portfolio snapshots** cached for 30 seconds
- **Price data** cached with fallback to database
- **Efficient updates** only when values change

### **2. Connection Management**
- **Automatic cleanup** of inactive connections
- **Connection pooling** for multiple subscribers
- **Graceful degradation** on connection loss

### **3. Data Efficiency**
- **Delta updates** for changed values only
- **Compressed data** transmission
- **Batch updates** for multiple portfolios

## 🚀 **Getting Started**

### **Step 1: Test Basic Endpoints**
```bash
# Get portfolio snapshot
curl http://localhost:8081/api/realtime/portfolio/1

# Get market prices
curl -X POST http://localhost:8081/api/realtime/prices \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"]}'
```

### **Step 2: Test WebSocket Connection**
```javascript
// Open browser console and test SSE connection
const eventSource = new EventSource('http://localhost:8081/api/realtime/portfolio/1/stream');
eventSource.onmessage = console.log;
```

### **Step 3: Monitor Portfolio Alerts**
```bash
# Get portfolio alerts
curl http://localhost:8081/api/realtime/portfolio/1/alerts
```

## 📚 **Advanced Features**

### **1. Custom Alerts**
- **P&L thresholds** for automatic notifications
- **Position concentration** warnings
- **Performance benchmarks** comparison

### **2. Historical Analysis**
- **Performance tracking** over time
- **Trend analysis** and patterns
- **Comparative studies** across portfolios

### **3. Risk Management**
- **Real-time risk metrics** calculation
- **Portfolio stress testing** scenarios
- **Diversification analysis** and recommendations

## 🎯 **Best Practices**

### **1. Connection Management**
- **Implement reconnection** logic
- **Handle connection errors** gracefully
- **Clean up connections** on component unmount

### **2. Data Updates**
- **Debounce rapid updates** to prevent UI flicker
- **Show loading states** during data fetch
- **Handle stale data** appropriately

### **3. Error Handling**
- **Implement fallback** mechanisms
- **Show user-friendly** error messages
- **Log errors** for debugging

---

**Your WealthArena application now has professional-grade real-time portfolio tracking! 📈🚀**
