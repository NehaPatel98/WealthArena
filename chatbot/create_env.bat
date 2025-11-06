@echo off
REM Create .env file for chatbot service

if exist .env (
    echo .env file already exists. Skipping creation.
    exit /b 0
)

(
echo # WealthArena Mobile Integration Environment Variables
echo.
echo # API Configuration
echo BASE_URL=http://127.0.0.1:5001
echo AUTH_REQUIRED=false
echo API_TOKEN=wealtharena-mobile-token
echo.
echo # CORS Origins ^(comma-separated^)
echo CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://10.0.2.2:8000,http://localhost:5001
echo.
echo # LLM Configuration ^(Groq is completely free^)
echo LLM_PROVIDER=groq
echo GROQ_API_KEY=your_groq_api_key_here
echo GROQ_MODEL=llama3-8b-8192
echo.
echo # Model Directories
echo SENTIMENT_MODEL_DIR=models/sentiment-finetuned
echo INTENT_MODEL_DIR=models/intent-finetuned
echo.
echo # Vector Database ^(Chroma^)
echo CHROMA_PERSIST_DIR=data/chroma_db
echo.
echo # Application Configuration
echo APP_HOST=0.0.0.0
echo PORT=5001
echo.
echo # Development
echo DEBUG=true
echo LOG_LEVEL=info
echo.
echo # Monitoring ^(optional^)
echo SENTRY_DSN=
) > .env

echo Created .env file successfully

