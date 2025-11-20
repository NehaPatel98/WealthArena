# Chatbot Groq API Troubleshooting Guide

## Issue: AI is not picking feeds from Groq API

This guide will help you diagnose and fix the issue where the chatbot is not connecting to the Groq API.

## Quick Diagnosis

When you send a message to the chatbot, check the backend logs. The improved error logging will show:

1. **Connection Refused (`ECONNREFUSED`)**: Chatbot service is not running
2. **Timeout (`ETIMEDOUT`)**: Chatbot service is running but not responding
3. **500 Error**: Chatbot service is running but Groq API key is missing or invalid

## Step 1: Check if Chatbot Service is Running

The chatbot service is a Python FastAPI application that needs to be running separately.

### Check if it's running:
```bash
# Test if the service is responding
curl http://localhost:8000/healthz
# or visit in browser: http://localhost:8000/healthz
```

### If not running, start it:
```bash
cd chatbot
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or if using a virtual environment:
```bash
cd chatbot
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Step 2: Configure Groq API Key

The chatbot service needs a Groq API key to connect to Groq API.

### 1. Get a Groq API Key (Free)
- Visit: https://console.groq.com/keys
- Sign up for a free account
- Create an API key
- Copy the key

### 2. Configure the Chatbot Service

Create or update `chatbot/.env` file:

```bash
cd chatbot
cp env.example .env
```

Edit `chatbot/.env` and set:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
LLM_PROVIDER=groq
APP_PORT=8000
```

### 3. Verify Configuration

Test if the Groq API key works:
```bash
cd chatbot
python -c "from app.llm.client import LLMClient; import asyncio; async def test(): client = LLMClient(); print('Groq API Key:', 'SET' if client.groq_api_key else 'NOT SET'); result = await client.chat([{'role': 'user', 'content': 'Hello'}]); print('Response:', result); asyncio.run(test())"
```

Or create a simple test file `test_groq.py`:
```python
import asyncio
from app.llm.client import LLMClient

async def test():
    client = LLMClient()
    print('Groq API Key:', 'SET' if client.groq_api_key else 'NOT SET')
    if client.groq_api_key:
        result = await client.chat([{'role': 'user', 'content': 'Say "Hello from Groq" if you are Groq'}])
        print('Response:', result)
    else:
        print('ERROR: GROQ_API_KEY not set in environment')

asyncio.run(test())
```

Run it:
```bash
python test_groq.py
```

## Step 3: Configure Backend Chatbot URL

Make sure the backend knows where to find the chatbot service.

Check `backend/.env` or `backend/.env.local`:
```env
CHATBOT_API_URL=http://localhost:8000
```

## Step 4: Test the Full Flow

### 1. Start Chatbot Service
```bash
cd chatbot
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Test Chatbot Service Directly
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is forex trading?"}'
```

You should get a JSON response with `reply` field containing Groq's answer.

### 3. Test via Backend Proxy
```bash
curl -X POST http://localhost:3000/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is forex trading?"}'
```

## Common Issues and Solutions

### Issue 1: "Chatbot service is not running"
**Solution**: Start the chatbot service (see Step 1)

### Issue 2: "GROQ_API_KEY not configured"
**Solution**: 
1. Check `chatbot/.env` file exists and has `GROQ_API_KEY=your_key`
2. Restart the chatbot service after updating `.env`

### Issue 3: "Connection timeout"
**Solution**:
- Check if chatbot service is on port 8000: `netstat -an | findstr 8000` (Windows) or `lsof -i :8000` (Mac/Linux)
- Verify `CHATBOT_API_URL` in backend `.env` matches the chatbot service URL

### Issue 4: "Groq API error: 401 Unauthorized"
**Solution**: 
- Your Groq API key is invalid or expired
- Get a new key from https://console.groq.com/keys
- Update `chatbot/.env` with the new key
- Restart chatbot service

### Issue 5: Backend shows "ECONNREFUSED"
**Solution**:
- Chatbot service is not running
- Check if it's running on the correct port (default: 8000)
- Verify firewall is not blocking the connection

## Verification Checklist

- [ ] Chatbot service is running (`curl http://localhost:8000/healthz` returns OK)
- [ ] `chatbot/.env` file exists with `GROQ_API_KEY` set
- [ ] Groq API key is valid (tested with test script)
- [ ] `backend/.env` has `CHATBOT_API_URL=http://localhost:8000`
- [ ] Backend can reach chatbot service (check logs for connection errors)
- [ ] Chatbot service can reach Groq API (check chatbot service logs)

## Getting Help

If you're still having issues:

1. Check backend logs for detailed error messages (now with improved logging)
2. Check chatbot service logs for Groq API errors
3. Verify all environment variables are set correctly
4. Test each component individually (chatbot service → Groq API, backend → chatbot service)

## Expected Log Output

When everything is working, you should see in backend logs:
```
[CHATBOT] Forwarding chat request to chatbot service: http://localhost:8000/v1/chat
[CHATBOT] API response received successfully
```

When there's an issue, you'll see detailed error information:
```
[CHATBOT ERROR] Error calling chatbot API:
   URL: http://localhost:8000/v1/chat
   Error Code: ECONNREFUSED
   HTTP Status: 500
   Error Message: connect ECONNREFUSED 127.0.0.1:8000
   Tip: Make sure the chatbot service is running. Run: cd chatbot && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

