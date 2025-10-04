"""
WealthArena Mobile Integration Main Application
FastAPI application with mobile SDK endpoints
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .api.chat import router as chat_router
from .api.metrics import router as metrics_router
from .middleware.metrics import MetricsMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="WealthArena Mobile API",
    description="Mobile SDK backend for WealthArena trading education platform",
    version="1.0.0"
)

# Add CORS middleware for mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://10.0.2.2:8000",  # Android emulator
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://localhost:19006",  # Expo
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

# Include API routers
app.include_router(chat_router, prefix="/v1", tags=["chat"])
app.include_router(metrics_router, prefix="/v1/metrics", tags=["metrics"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "WealthArena Mobile API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "wealtharena-mobile-api",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

