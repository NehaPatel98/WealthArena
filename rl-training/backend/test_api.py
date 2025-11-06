#!/usr/bin/env python3
"""
API Endpoint Tests
Tests all backend API endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and info endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "WealthArena API"
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestMarketDataEndpoints:
    """Test market data endpoints"""
    
    def test_get_market_data(self):
        """Test market data endpoint"""
        response = client.post(
            "/api/market-data",
            json={"symbols": ["AAPL"], "days": 30}
        )
        assert response.status_code in [200, 404]  # 404 if no data yet
        
        if response.status_code == 200:
            data = response.json()
            assert "symbols" in data
            assert "data" in data
    
    def test_get_market_data_multiple_symbols(self):
        """Test with multiple symbols"""
        response = client.post(
            "/api/market-data",
            json={"symbols": ["AAPL", "GOOGL", "MSFT"], "days": 7}
        )
        assert response.status_code in [200, 404]


class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_get_prediction(self):
        """Test single symbol prediction"""
        response = client.post(
            "/api/predictions",
            json={"symbol": "AAPL", "horizon": 1}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "signal" in data
            assert "confidence" in data
            assert data["signal"] in ["BUY", "SELL", "HOLD"]
            assert 0 <= data["confidence"] <= 1
            
            # Check TP/SL from RL models
            if data["signal"] != "HOLD":
                assert "take_profit" in data
                assert "stop_loss" in data
                assert "position_sizing" in data
    
    def test_top_setups_stocks(self):
        """Test top setups for stocks"""
        response = client.post(
            "/api/top-setups",
            json={"asset_type": "stocks", "count": 3, "risk_tolerance": "medium"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "asset_type" in data
            assert data["asset_type"] == "stocks"
            assert "setups" in data
            assert len(data["setups"]) <= 3
            
            # Verify each setup has required fields
            for setup in data["setups"]:
                assert "symbol" in setup
                assert "signal" in setup
                assert "confidence" in setup
                assert "entry" in setup
                assert "take_profit" in setup  # From RL model
                assert "stop_loss" in setup     # From RL model
                assert "risk_metrics" in setup
                assert "position_sizing" in setup
                assert "rank" in setup
    
    def test_top_setups_currency_pairs(self):
        """Test top setups for currency pairs"""
        response = client.post(
            "/api/top-setups",
            json={"asset_type": "currency_pairs", "count": 3}
        )
        assert response.status_code in [200, 404]


class TestPortfolioEndpoints:
    """Test portfolio endpoints"""
    
    def test_analyze_portfolio(self):
        """Test portfolio analysis"""
        response = client.post(
            "/api/portfolio",
            json={
                "symbols": ["AAPL", "GOOGL"],
                "weights": [0.6, 0.4]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "portfolio" in data
            assert "metrics" in data
            assert "recommendations" in data
    
    def test_portfolio_invalid_weights(self):
        """Test portfolio with invalid weights"""
        response = client.post(
            "/api/portfolio",
            json={
                "symbols": ["AAPL", "GOOGL"],
                "weights": [0.6, 0.6]  # Sum > 1.0
            }
        )
        assert response.status_code == 400


class TestChatbotEndpoints:
    """Test chatbot endpoints"""
    
    def test_chat_basic_question(self):
        """Test basic chat query"""
        response = client.post(
            "/api/chat",
            json={"message": "Why should I buy AAPL?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "response" in data
        assert "answer" in data["response"]
    
    def test_chat_with_context(self):
        """Test chat with context"""
        response = client.post(
            "/api/chat",
            json={
                "message": "Explain this setup",
                "context": {"symbol": "AAPL", "signal": "BUY"}
            }
        )
        assert response.status_code == 200


class TestGameEndpoints:
    """Test game endpoints"""
    
    def test_get_leaderboard(self):
        """Test leaderboard endpoint"""
        response = client.get("/api/game/leaderboard")
        assert response.status_code == 200
        data = response.json()
        assert "leaderboard" in data
        assert len(data["leaderboard"]) > 0
    
    def test_get_metrics_summary(self):
        """Test metrics summary"""
        response = client.get("/api/metrics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "data" in data
        assert "models" in data


class TestResponseFormats:
    """Test response formats and data structures"""
    
    def test_prediction_response_structure(self):
        """Test prediction response has all required fields"""
        response = client.post(
            "/api/predictions",
            json={"symbol": "AAPL", "horizon": 1}
        )
        
        if response.status_code == 200:
            prediction = response.json()
            
            # Required fields
            required = ['symbol', 'signal', 'confidence', 'entry', 
                       'risk_metrics', 'position_sizing', 'model_version']
            
            for field in required:
                assert field in prediction, f"Missing field: {field}"
            
            # Entry structure
            assert 'price' in prediction['entry']
            assert 'range' in prediction['entry']
            
            # Take profit structure (if not HOLD)
            if prediction['signal'] != 'HOLD':
                assert len(prediction['take_profit']) == 3
                for tp in prediction['take_profit']:
                    assert 'level' in tp
                    assert 'price' in tp
                    assert 'percent' in tp
                    assert 'close_percent' in tp
            
            # Stop loss structure
            if prediction['signal'] != 'HOLD':
                assert 'price' in prediction['stop_loss']
                assert 'percent' in prediction['stop_loss']
                assert 'type' in prediction['stop_loss']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

