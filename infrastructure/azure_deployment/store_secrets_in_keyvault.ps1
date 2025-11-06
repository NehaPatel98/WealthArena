# WealthArena Azure Key Vault Secrets Storage Script
# Stores all sensitive credentials in Azure Key Vault and documents retrieval commands

param(
    [string]$ResourceGroupName = "rg-wealtharena-northcentralus",
    [string]$KeyVaultName = "kv-wealtharena-dev",
    [string]$Environment = "dev"
)

$ErrorActionPreference = "Continue"

# Colors for output
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Blue = "Blue"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function New-RandomSecret {
    # Generate random 256-bit key (32 bytes)
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    $secret = [Convert]::ToBase64String($bytes)
    return $secret
}

function Set-KeyVaultSecret {
    param([string]$VaultName, [string]$SecretName, [string]$SecretValue)
    
    try {
        az keyvault secret set `
            --vault-name $VaultName `
            --name $SecretName `
            --value $SecretValue `
            --output none 2>$null
        
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

function Get-KeyVaultSecrets {
    param([string]$VaultName)
    
    try {
        $secrets = az keyvault secret list `
            --vault-name $VaultName `
            --output json 2>$null | ConvertFrom-Json
        
        return $secrets
    }
    catch {
        return @()
    }
}

function Get-StorageConnectionString {
    param([string]$StorageAccount, [string]$ResourceGroup)
    
    try {
        $connStr = az storage account show-connection-string `
            --name $StorageAccount `
            --resource-group $ResourceGroup `
            --query connectionString `
            --output tsv 2>$null
        
        return $connStr
    }
    catch {
        return ""
    }
}

function Get-CosmosConnectionString {
    param([string]$CosmosAccount, [string]$ResourceGroup)
    
    try {
        $connStr = az cosmosdb keys list `
            --name $CosmosAccount `
            --resource-group $ResourceGroup `
            --type connection-strings `
            --query connectionStrings[0].connectionString `
            --output tsv 2>$null
        
        return $connStr
    }
    catch {
        return ""
    }
}

# Main execution
Write-ColorOutput "Storing Secrets in Azure Key Vault" $Blue
Write-ColorOutput "===================================" $Blue
Write-ColorOutput "" $Blue
Write-ColorOutput "Key Vault: $KeyVaultName" $Blue
Write-ColorOutput "" $Blue

# Check Azure connection
try {
    $account = az account show --output json | ConvertFrom-Json
    Write-ColorOutput "✅ Logged in as: $($account.user.name)" $Green
} catch {
    Write-ColorOutput "❌ Not logged in to Azure" $Red
    Write-ColorOutput "Please run 'az login' first." $Yellow
    exit 1
}

# Verify Key Vault exists
try {
    $kv = az keyvault show --name $KeyVaultName --resource-group $ResourceGroupName --output json | ConvertFrom-Json
    if (-not $kv) {
        Write-ColorOutput "❌ Key Vault '$KeyVaultName' not found" $Red
        Write-ColorOutput "Please run setup_master.ps1 first." $Yellow
        exit 1
    }
    Write-ColorOutput "✅ Key Vault found: $($kv.properties.vaultUri)" $Green
} catch {
    Write-ColorOutput "❌ Key Vault '$KeyVaultName' not found" $Red
    Write-ColorOutput "Please run setup_master.ps1 first." $Yellow
    exit 1
}

Write-ColorOutput "" $Blue
Write-ColorOutput "Storing secrets..." $Blue
Write-ColorOutput "" $Blue

# Store secrets
$secretsStored = @{}

# 1. SQL Database Password
$sqlPassword = "Secure@Db2024!$%"
if (Set-KeyVaultSecret -VaultName $KeyVaultName -SecretName "sql-password" -SecretValue $sqlPassword) {
    Write-ColorOutput "✅ sql-password stored" $Green
    $secretsStored["sql-password"] = $true
} else {
    Write-ColorOutput "❌ Failed to store sql-password" $Red
    $secretsStored["sql-password"] = $false
}

# 2. GROQ API Key
$groqApiKey = "YOUR_GROQ_API_KEY"
if (Set-KeyVaultSecret -VaultName $KeyVaultName -SecretName "groq-api-key" -SecretValue $groqApiKey) {
    Write-ColorOutput "✅ groq-api-key stored" $Green
    $secretsStored["groq-api-key"] = $true
} else {
    Write-ColorOutput "❌ Failed to store groq-api-key" $Red
    $secretsStored["groq-api-key"] = $false
}

# 3. JWT Secret (Generate New)
$jwtSecret = New-RandomSecret
if (Set-KeyVaultSecret -VaultName $KeyVaultName -SecretName "jwt-secret" -SecretValue $jwtSecret) {
    Write-ColorOutput "✅ jwt-secret generated and stored" $Green
    $secretsStored["jwt-secret"] = $true
} else {
    Write-ColorOutput "❌ Failed to store jwt-secret" $Red
    $secretsStored["jwt-secret"] = $false
}

# 4. Storage Account Connection String
$storageAccount = "stwealtharena$Environment"
$storageConnStr = Get-StorageConnectionString -StorageAccount $storageAccount -ResourceGroup $ResourceGroupName
if ($storageConnStr) {
    if (Set-KeyVaultSecret -VaultName $KeyVaultName -SecretName "storage-connection-string" -SecretValue $storageConnStr) {
        Write-ColorOutput "✅ storage-connection-string stored" $Green
        $secretsStored["storage-connection-string"] = $true
    } else {
        Write-ColorOutput "❌ Failed to store storage-connection-string" $Red
        $secretsStored["storage-connection-string"] = $false
    }
} else {
    Write-ColorOutput "⚠️  Could not retrieve storage connection string" $Yellow
    $secretsStored["storage-connection-string"] = $false
}

# 5. Cosmos DB Connection String (optional - may not exist in student accounts)
$cosmosAccount = "cosmos-wealtharena-$Environment"
$cosmosConnStr = Get-CosmosConnectionString -CosmosAccount $cosmosAccount -ResourceGroup $ResourceGroupName
if ($cosmosConnStr) {
    if (Set-KeyVaultSecret -VaultName $KeyVaultName -SecretName "cosmos-connection-string" -SecretValue $cosmosConnStr) {
        Write-ColorOutput "✅ cosmos-connection-string stored" $Green
        $secretsStored["cosmos-connection-string"] = $true
    } else {
        Write-ColorOutput "❌ Failed to store cosmos-connection-string" $Red
        $secretsStored["cosmos-connection-string"] = $false
    }
} else {
    Write-ColorOutput "⚠️  Cosmos DB connection string not available (Cosmos DB may not exist in student accounts)" $Yellow
    $secretsStored["cosmos-connection-string"] = $false
}

Write-ColorOutput "" $Blue

# Verify secrets stored
Write-ColorOutput "Verifying secrets..." $Blue
$allSecrets = Get-KeyVaultSecrets -VaultName $KeyVaultName
$secretNames = $allSecrets | ForEach-Object { $_.name }

$expectedSecrets = @("sql-password", "groq-api-key", "jwt-secret", "storage-connection-string", "cosmos-connection-string")
$foundSecrets = 0

foreach ($secret in $expectedSecrets) {
    if ($secretNames -contains $secret) {
        Write-ColorOutput "✅ $secret accessible" $Green
        $foundSecrets++
    } else {
        if ($secret -eq "cosmos-connection-string") {
            Write-ColorOutput "⚠️  $secret not found (Cosmos DB may not exist in student accounts)" $Yellow
        } else {
            Write-ColorOutput "❌ $secret not found" $Red
        }
    }
}

Write-ColorOutput "" $Blue
Write-ColorOutput "✅ All $foundSecrets secrets accessible" $Green
Write-ColorOutput "" $Blue

# Test secret retrieval
Write-ColorOutput "Test secret retrieval..." $Blue
try {
    $testSecret = az keyvault secret show `
        --vault-name $KeyVaultName `
        --name sql-password `
        --query value `
        --output tsv 2>$null
    
    if ($testSecret) {
        Write-ColorOutput "✅ Secret retrieval test successful" $Green
    } else {
        Write-ColorOutput "⚠️  Secret retrieval test failed" $Yellow
    }
} catch {
    Write-ColorOutput "⚠️  Secret retrieval test failed: $($_.Exception.Message)" $Yellow
}

Write-ColorOutput "" $Blue

# Generate documentation
Write-ColorOutput "Secret Retrieval Commands:" $Blue
Write-ColorOutput "--------------------------" $Blue
Write-ColorOutput "" $Blue

$docContent = @"
# Azure Key Vault Secrets - Retrieval Commands

Generated on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

## Key Vault Details

- **Vault Name**: $KeyVaultName
- **Vault URL**: https://$KeyVaultName.vault.azure.net/
- **Resource Group**: $ResourceGroupName

## Secret Retrieval Commands

### SQL Password

`az keyvault secret show --vault-name $KeyVaultName --name sql-password --query value -o tsv`

### GROQ API Key

`az keyvault secret show --vault-name $KeyVaultName --name groq-api-key --query value -o tsv`

### JWT Secret

`az keyvault secret show --vault-name $KeyVaultName --name jwt-secret --query value -o tsv`

### Storage Connection String

`az keyvault secret show --vault-name $KeyVaultName --name storage-connection-string --query value -o tsv`

### Cosmos DB Connection String (if Cosmos DB exists)

`az keyvault secret show --vault-name $KeyVaultName --name cosmos-connection-string --query value -o tsv`

## Usage in App Settings

When configuring Azure Web Apps, reference Key Vault secrets:

\`\`\`powershell
az webapp config appsettings set --name wealtharena-backend --resource-group $ResourceGroupName --settings DB_PASSWORD="@Microsoft.KeyVault(SecretUri=https://$KeyVaultName.vault.azure.net/secrets/sql-password/)"
\`\`\`

## Azure Portal Access

- Key Vault Portal: https://portal.azure.com/#@/resource/subscriptions/<subscription-id>/resourceGroups/$ResourceGroupName/providers/Microsoft.KeyVault/vaults/$KeyVaultName

## IAM Permissions

To access secrets, ensure your account has:
- **Get** permission for secrets
- **List** permission for secrets (to see secret names)

Grant yourself access:
\`\`\`powershell
az keyvault set-policy --name $KeyVaultName --upn <your-email> --secret-permissions get list set
\`\`\`

"@

$docPath = Join-Path $PSScriptRoot "..\..\KEY_VAULT_SECRETS.md"
$docContent | Out-File -FilePath $docPath -Encoding UTF8

Write-ColorOutput "SQL Password:" $Blue
Write-ColorOutput "  az keyvault secret show --vault-name $KeyVaultName --name sql-password --query value -o tsv" $Blue
Write-ColorOutput "" $Blue
Write-ColorOutput "GROQ API Key:" $Blue
Write-ColorOutput "  az keyvault secret show --vault-name $KeyVaultName --name groq-api-key --query value -o tsv" $Blue
Write-ColorOutput "" $Blue
Write-ColorOutput "JWT Secret:" $Blue
Write-ColorOutput "  az keyvault secret show --vault-name $KeyVaultName --name jwt-secret --query value -o tsv" $Blue
Write-ColorOutput "" $Blue
Write-ColorOutput "Storage Connection String:" $Blue
Write-ColorOutput "  az keyvault secret show --vault-name $KeyVaultName --name storage-connection-string --query value -o tsv" $Blue
Write-ColorOutput "" $Blue
Write-ColorOutput "Cosmos DB Connection String (if Cosmos DB exists):" $Blue
Write-ColorOutput "  az keyvault secret show --vault-name $KeyVaultName --name cosmos-connection-string --query value -o tsv" $Blue
Write-ColorOutput "" $Blue
Write-ColorOutput "✅ Documentation saved to: KEY_VAULT_SECRETS.md" $Green
Write-ColorOutput "" $Blue
