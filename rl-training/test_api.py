#!/usr/bin/env python3
"""
Test script for WealthArena AI Models API
This script tests all API endpoints to ensure they're working correctly
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def test_health(self) -> bool:
        """Test the health endpoint"""
        try:
            print("🔍 Testing health endpoint...")
            response = self.session.get(f"{self.base_url}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   Available agents: {data['available_agents']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_models(self) -> bool:
        """Test the models endpoint"""
        try:
            print("🔍 Testing models endpoint...")
            response = self.session.get(f"{self.base_url}/models", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Models endpoint working: {len(data)} models found")
                for model in data:
                    print(f"   - {model['agent_name']}: {model['status']}")
                return True
            else:
                print(f"❌ Models endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Models endpoint error: {e}")
            return False
    
    def test_prediction(self, agent_name: str) -> bool:
        """Test prediction endpoint for a specific agent"""
        try:
            print(f"🔍 Testing prediction for {agent_name}...")
            
            # Create test data (140 features as expected by the models)
            test_data = [[0.1] * 140]  # Simple test data
            
            payload = {
                "agent_name": agent_name,
                "input_data": test_data,
                "symbol": "TEST"
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Prediction successful for {agent_name}")
                print(f"   Prediction shape: {len(data['prediction'])}")
                print(f"   Confidence: {data['confidence']:.4f}")
                return True
            else:
                print(f"❌ Prediction failed for {agent_name}: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Prediction error for {agent_name}: {e}")
            return False
    
    def test_predict_all(self) -> bool:
        """Test the predict-all endpoint"""
        try:
            print("🔍 Testing predict-all endpoint...")
            
            # Create test data
            test_data = [[0.1] * 140]
            
            payload = {
                "agent_name": "dummy",  # Required but not used
                "input_data": test_data,
                "symbol": "TEST"
            }
            
            response = self.session.post(
                f"{self.base_url}/predict-all",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Predict-all successful: {len(data)} predictions")
                for prediction in data:
                    print(f"   - {prediction['agent_name']}: {len(prediction['prediction'])} outputs")
                return True
            else:
                print(f"❌ Predict-all failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Predict-all error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("🚀 Starting WealthArena API Tests")
        print("=" * 50)
        
        results = {}
        
        # Test health
        results['health'] = self.test_health()
        print()
        
        # Test models
        results['models'] = self.test_models()
        print()
        
        # Test individual predictions
        agents = ['asx_stocks', 'currency_pairs', 'cryptocurrencies', 'etf', 'commodities']
        for agent in agents:
            results[f'prediction_{agent}'] = self.test_prediction(agent)
            print()
        
        # Test predict-all
        results['predict_all'] = self.test_predict_all()
        print()
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Print test summary"""
        print("📊 Test Summary")
        print("=" * 30)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print()
        
        print("Detailed Results:")
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        if passed == total:
            print("\n🎉 All tests passed! Your API is working perfectly!")
        else:
            print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python test_api.py <API_URL>")
        print("Example: python test_api.py http://localhost:8000")
        print("Example: python test_api.py https://abc123.ngrok.io")
        sys.exit(1)
    
    api_url = sys.argv[1]
    
    print(f"Testing API at: {api_url}")
    print()
    
    # Wait a moment for the API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(5)
    
    # Create tester and run tests
    tester = APITester(api_url)
    results = tester.run_all_tests()
    
    # Print summary
    tester.print_summary(results)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()
