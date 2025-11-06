# Wrapper script for verify_infrastructure.ps1
# This script calls the actual script in infrastructure/azure_deployment/

$scriptPath = Join-Path $PSScriptRoot "infrastructure\azure_deployment\verify_infrastructure.ps1"

if (-not (Test-Path $scriptPath)) {
    Write-Host "Error: Script not found at $scriptPath" -ForegroundColor Red
    exit 1
}

& $scriptPath @args

