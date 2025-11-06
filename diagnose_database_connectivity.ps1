# Wrapper script for diagnose_database_connectivity.ps1
# This script calls the actual script in infrastructure/azure_deployment/

$scriptPath = Join-Path $PSScriptRoot "infrastructure\azure_deployment\diagnose_database_connectivity.ps1"

if (-not (Test-Path $scriptPath)) {
    Write-Host "Error: Script not found at $scriptPath" -ForegroundColor Red
    exit 1
}

& $scriptPath @args

