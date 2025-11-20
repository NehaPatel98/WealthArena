# Simple chatbot test script for PowerShell
# Usage: .\test_chatbot_simple.ps1 "Your message here"

param(
    [string]$Message = "What is the best forex strategy on the market?"
)

Write-Host "Testing Chatbot API..." -ForegroundColor Cyan
Write-Host "Message: $Message" -ForegroundColor Yellow
Write-Host ""

try {
    $body = @{
        message = $Message
    } | ConvertTo-Json

    $response = Invoke-RestMethod -Uri "http://localhost:8000/v1/chat" -Method Post -Body $body -ContentType "application/json"
    
    Write-Host "Response:" -ForegroundColor Green
    Write-Host $response.reply -ForegroundColor White
    Write-Host ""
    Write-Host "Tools Used: $($response.tools_used -join ', ')" -ForegroundColor Gray
    Write-Host "Trace ID: $($response.trace_id)" -ForegroundColor Gray
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure:" -ForegroundColor Yellow
    Write-Host "1. Chatbot service is running on port 8000" -ForegroundColor White
    Write-Host "2. Run: cd chatbot; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000" -ForegroundColor White
}

