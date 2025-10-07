# WealthArena - Complete Portfolio Management System Demo

## 🚀 System Overview
**WealthArena** is a comprehensive financial portfolio management system built with Spring Boot, featuring:

- **Market Data Ingestion** from Yahoo Finance & Alpha Vantage
- **User Authentication** with JWT tokens
- **Portfolio Management** with buy/sell functionality
- **Real-time Value Calculation** using current market prices
- **PostgreSQL/TimescaleDB** for data persistence

## 📊 Available Endpoints

### Health Check
- `GET /api/hello` - API status
- `GET /api/ping` - Quick health check

### Market Data
- `GET /api/market/data/{symbol}?limit=N` - Get recent price bars
- `GET /api/market/price/{symbol}` - Get current price
- `POST /api/marketdata/ingest/yahoo/{symbol}` - Fetch new data
- `POST /api/marketdata/ingest/alphaVantage/{symbol}` - Fetch Alpha Vantage data

### User Management
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `GET /api/users/profile` - Get user profile
- `PUT /api/users/profile` - Update user profile

### Portfolio Management
- `GET /api/portfolios?userId={id}` - Get user's portfolios
- `POST /api/portfolios` - Create new portfolio
- `GET /api/portfolios/{id}?userId={id}` - Get specific portfolio
- `POST /api/portfolios/{id}/buy` - Buy stocks
- `POST /api/portfolios/{id}/sell` - Sell stocks
- `GET /api/portfolios/{id}/value?userId={id}` - Get portfolio value
- `GET /api/portfolios/{id}/holdings?userId={id}` - Get holdings

## 🧪 Testing Instructions

### 1. Test Health Endpoints
```bash
curl http://localhost:8081/api/hello
curl http://localhost:8081/api/ping
```

### 2. Test Market Data
```bash
# Get AAPL data
curl http://localhost:8081/api/market/data/AAPL?limit=5

# Get current price
curl http://localhost:8081/api/market/price/AAPL

# Fetch new data (POST request)
curl -X POST http://localhost:8081/api/marketdata/ingest/yahoo/AAPL
```

### 3. Test User Registration & Login
```bash
# Register a new user
curl -X POST http://localhost:8081/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@example.com",
    "password": "password123",
    "fullName": "Demo User"
  }'

# Login (returns user ID)
curl -X POST http://localhost:8081/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@example.com",
    "password": "password123"
  }'
```

### 4. Test Portfolio Management
```bash
# Create a portfolio (use userId from registration)
curl -X POST http://localhost:8081/api/portfolios \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 1,
    "name": "My Tech Portfolio",
    "description": "Technology stocks"
  }'

# Get user's portfolios
curl http://localhost:8081/api/portfolios?userId=1

# Buy stocks
curl -X POST http://localhost:8081/api/portfolios/1/buy \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 1,
    "symbol": "AAPL",
    "quantity": 10,
    "price": 150.00
  }'

# Get portfolio value
curl http://localhost:8081/api/portfolios/1/value?userId=1

# Get holdings
curl http://localhost:8081/api/portfolios/1/holdings?userId=1
```

## 🏗️ Architecture

### Backend Stack
- **Spring Boot 3.5.6** - Main framework
- **Spring Data JPA** - Data persistence
- **PostgreSQL/TimescaleDB** - Database
- **Spring Security** - Authentication
- **JWT** - Token-based auth
- **WebClient** - HTTP client for external APIs

### Key Features
- **Real-time Market Data** - Yahoo Finance & Alpha Vantage integration
- **Portfolio Tracking** - Buy/sell with average cost calculation
- **Value Calculation** - Real-time portfolio valuation
- **User Security** - JWT-based authentication
- **Database Optimization** - TimescaleDB for time-series data

## 📈 Business Logic

### Portfolio Management
- Create multiple portfolios per user
- Track stock positions with average cost
- Calculate real-time portfolio value
- Support partial buy/sell operations
- Automatic P&L calculation

### Market Data
- Fetch historical price data
- Real-time price updates
- Multiple data sources (Yahoo Finance, Alpha Vantage)
- Data normalization and persistence

### Security
- Password encryption with bcrypt
- JWT token generation and validation
- User-specific data access control
- Secure API endpoints

## 🎯 Demo Flow for Professor

1. **Start Application** - Show successful startup
2. **Health Check** - Verify API is running
3. **Market Data** - Fetch real stock data
4. **User Registration** - Create demo user
5. **Portfolio Creation** - Create investment portfolio
6. **Stock Trading** - Buy/sell stocks
7. **Value Tracking** - Show real-time portfolio value
8. **Database Verification** - Show data persistence

## 🔧 Technical Highlights

- **Clean Architecture** - Separation of concerns
- **RESTful APIs** - Standard HTTP methods
- **Error Handling** - Comprehensive exception management
- **Data Validation** - Input validation and constraints
- **Performance** - Optimized database queries
- **Scalability** - Microservice-ready architecture

---

**Ready for Professor Demo! 🎓**
