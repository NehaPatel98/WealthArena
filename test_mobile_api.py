#!/usr/bin/env python3
"""
WealthArena Mobile API Test Script
Tests the v1 mobile API endpoints
"""

import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing /v1/healthz...")
    try:
        response = requests.get(f"{BASE_URL}/v1/healthz", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_chat():
    """Test chat endpoint"""
    print("🔍 Testing /v1/chat...")
    try:
        payload = {
            "message": "Explain RSI to me",
            "symbol": "AAPL",
            "mode": "teach"
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        print(f"✅ Chat test passed: {data['reply'][:50]}...")
        return True
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

def test_analyze():
    """Test analyze endpoint"""
    print("🔍 Testing /v1/analyze...")
    try:
        payload = {"symbol": "AAPL"}
        response = requests.post(
            f"{BASE_URL}/v1/analyze",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        print(f"✅ Analysis test passed: {data['symbol']} - ${data['current_price']}")
        return True
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False

def test_state():
    """Test state endpoint"""
    print("🔍 Testing /v1/state...")
    try:
        response = requests.get(f"{BASE_URL}/v1/state", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ State test passed: Balance ${data['balance']}")
        return True
    except Exception as e:
        print(f"❌ State test failed: {e}")
        return False

def test_paper_trade():
    """Test paper trade endpoint"""
    print("🔍 Testing /v1/papertrade...")
    try:
        payload = {
            "action": "buy",
            "symbol": "AAPL",
            "quantity": 1.0
        }
        response = requests.post(
            f"{BASE_URL}/v1/papertrade",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        print(f"✅ Paper trade test passed: {data['message']}")
        return True
    except Exception as e:
        print(f"❌ Paper trade test failed: {e}")
        return False

def test_learn():
    """Test learn endpoint"""
    print("🔍 Testing /v1/learn...")
    try:
        response = requests.get(f"{BASE_URL}/v1/learn", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Learn test passed: {len(data['lessons'])} lessons, {len(data['quizzes'])} quizzes")
        return True
    except Exception as e:
        print(f"❌ Learn test failed: {e}")
        return False

def test_cors():
    """Test CORS headers"""
    print("🔍 Testing CORS headers...")
    try:
        response = requests.options(f"{BASE_URL}/v1/healthz", timeout=5)
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
        }
        print(f"✅ CORS headers: {cors_headers}")
        return True
    except Exception as e:
        print(f"❌ CORS test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 WealthArena Mobile API Test Suite")
    print("=" * 50)
    
    tests = [
        test_health,
        test_chat,
        test_analyze,
        test_state,
        test_paper_trade,
        test_learn,
        test_cors
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Mobile API is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
