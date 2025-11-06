"""
WealthArena - Groq Integration Test Script
Tests the Groq API integration to ensure it's configured correctly
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_env_configuration():
    """Test environment configuration"""
    print("=" * 60)
    print("Testing Groq Environment Configuration")
    print("=" * 60)
    print()
    
    # Check required environment variables
    provider = os.getenv("LLM_PROVIDER", "groq")
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    print(f"✓ LLM_PROVIDER: {provider}")
    print(f"✓ GROQ_MODEL: {model}")
    
    if api_key:
        # Mask the key for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✓ GROQ_API_KEY: {masked_key} (configured)")
        
        # Check key format
        if api_key.startswith("gsk_"):
            print(f"  ✓ Key format is correct (starts with 'gsk_')")
        else:
            print(f"  ⚠ Warning: Key should start with 'gsk_'")
    else:
        print(f"✗ GROQ_API_KEY: NOT SET")
        print(f"  ⚠ Please set GROQ_API_KEY in your .env file")
        return False
    
    print()
    return True

async def test_groq_api_call():
    """Test actual Groq API call"""
    print("=" * 60)
    print("Testing Groq API Call")
    print("=" * 60)
    print()
    
    try:
        from app.llm.client import LLMClient
        
        print("Initializing LLM client...")
        llm_client = LLMClient()
        
        # Check if Groq is configured
        if not llm_client.groq_api_key:
            print("✗ GROQ_API_KEY not found in LLM client")
            print("  Make sure .env file is loaded correctly")
            return False
        
        if llm_client.provider != "groq":
            print(f"✗ LLM_PROVIDER is '{llm_client.provider}', expected 'groq'")
            return False
        
        print("✓ LLM client initialized successfully")
        print(f"  Provider: {llm_client.provider}")
        print(f"  Model: {llm_client.groq_model}")
        print()
        
        # Test API call with a simple message
        print("Testing Groq API with a simple message...")
        test_messages = [
            {"role": "system", "content": "You are a helpful educational trading assistant. Always emphasize that your advice is for educational purposes only and that users should practice with paper trading first."},
            {"role": "user", "content": "What is RSI in trading? Keep the explanation brief (2-3 sentences)."}
        ]
        
        response = await llm_client.chat(test_messages)
        
        if response:
            print("✓ Groq API call successful!")
            print()
            print("Response:")
            print("-" * 60)
            print(response)
            print("-" * 60)
            print()
            return True
        else:
            print("✗ Groq API returned empty response")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Make sure you're running this from the wealtharena_chatbot directory")
        return False
    except Exception as e:
        print(f"✗ Groq API call failed: {e}")
        print()
        print("Common issues:")
        print("  1. Invalid API key - check your GROQ_API_KEY in .env")
        print("  2. Network connectivity - check your internet connection")
        print("  3. Rate limiting - wait a few seconds and try again")
        return False

async def main():
    """Main test function"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "WealthArena Groq Integration Test" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Test 1: Environment configuration
    env_ok = test_env_configuration()
    
    if not env_ok:
        print()
        print("=" * 60)
        print("Configuration test failed. Please fix the issues above.")
        print("=" * 60)
        sys.exit(1)
    
    print()
    
    # Test 2: Groq API call
    api_ok = await test_groq_api_call()
    
    print()
    print("=" * 60)
    if api_ok:
        print("✅ All tests passed! Groq integration is working correctly.")
        print("=" * 60)
        print()
        print("Your chatbot is ready to use Groq for AI-powered responses.")
    else:
        print("❌ Groq API test failed. Please check the error messages above.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)

