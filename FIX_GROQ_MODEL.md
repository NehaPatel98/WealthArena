# Fix: Groq Model Decommissioned

## Issue Found

The chatbot is not using Groq API because the model name `llama3-8b-8192` has been **decommissioned** by Groq.

**Error Message:**
```
"The model `llama3-8b-8192` has been decommissioned and is no longer supported."
```

## Solution

Update the `GROQ_MODEL` in your `chatbot/.env` file to use a supported model.

### Step 1: Update chatbot/.env

Edit `chatbot/.env` and change:
```env
GROQ_MODEL=llama3-8b-8192
```

To:
```env
GROQ_MODEL=llama-3.1-8b-instant
```

### Step 2: Restart Chatbot Service

After updating the `.env` file, you **must restart the chatbot service** for the change to take effect.

1. Stop the current chatbot service (Ctrl+C if running in terminal)
2. Start it again:
   ```bash
   cd chatbot
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Available Groq Models

- `llama-3.1-8b-instant` - Fast, efficient model (recommended)
- `llama-3.1-70b-versatile` - More powerful, slower model
- `mixtral-8x7b-32768` - Large context window model

### Verify Fix

After restarting, test the chatbot:
```powershell
$body = @{message = "Say hello from Groq"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/v1/chat" -Method Post -Body $body -ContentType "application/json"
```

You should now get responses from Groq API instead of fallback responses!

