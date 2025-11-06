<#
.SYNOPSIS
    Fix Key Vault permissions and store required secrets

.DESCRIPTION
    This script grants secret permissions to the current user/principal and stores all required secrets.
    Supports both Access Policy and RBAC-based Key Vaults.
    Grants permissions to:
    - GROQ API key
    - SQL password
    - Storage connection string

.PARAMETER ResourceGroup
    Azure resource group name (default: rg-wealtharena-northcentralus)

.PARAMETER KeyVault
    Key Vault name (optional, will auto-derive from Resource Group suffix if not provided)

.PARAMETER UserEmail
    User email for permissions (optional, will auto-detect current user if not provided)

.PARAMETER GroqApiKey
    GROQ API key (optional, will read from environment variable GROQ_API_KEY if not provided)

.PARAMETER SqlPassword
    SQL password (optional, will read from environment variable SQL_PASSWORD if not provided)

.NOTES
    Author: WealthArena DevOps Team
    Version: 2.0
    Created: 2024-12-22
    Updated: 2025-01-XX - Added RBAC fallback and improved vault discovery
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "rg-wealtharena-northcentralus",
    
    [Parameter(Mandatory=$false)]
    [string]$KeyVault = "",
    
    [Parameter(Mandatory=$false)]
    [string]$UserEmail = "",
    
    [Parameter(Mandatory=$false)]
    [string]$GroqApiKey = "",
    
    [Parameter(Mandatory=$false)]
    [string]$SqlPassword = ""
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# Colors for output
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Blue = "Blue"
$Cyan = "Cyan"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

Write-ColorOutput "========================================" $Cyan
Write-ColorOutput "Key Vault Permissions Fix" $Cyan
Write-ColorOutput "========================================" $Cyan
Write-ColorOutput ""

# Derive Key Vault name if not provided
if ([string]::IsNullOrWhiteSpace($KeyVault)) {
    if ($ResourceGroup -match 'rg-wealtharena-(\w+)') {
        $suffix = $matches[1]
        $KeyVault = "kv-wealtharena-$suffix"
        Write-ColorOutput "Derived Key Vault name from Resource Group: $KeyVault" $Cyan
    } else {
        Write-ColorOutput "Could not derive Key Vault name from Resource Group" $Yellow
        Write-ColorOutput "Please provide -KeyVault parameter" $Red
        exit 1
    }
}

$vaultName = $KeyVault
Write-ColorOutput "Resource Group: $ResourceGroup" $Cyan
Write-ColorOutput "Key Vault: $vaultName" $Cyan
Write-ColorOutput ""

# Step 1: Get current user/principal
Write-ColorOutput "Step 1: Detecting current Azure principal" $Blue
Write-ColorOutput "---------------------------------------------------" $Blue

$currentUser = $null
$currentObjectId = $null
$isServicePrincipal = $false

$accountOutput = az account show --output json 2>&1
$account = $null

if ($LASTEXITCODE -eq 0) {
    try {
        $account = $accountOutput | ConvertFrom-Json
        if ($account.user) {
            $currentUser = $account.user.name
            Write-ColorOutput "   Current user: $currentUser" $Green
        } elseif ($account.name) {
            $currentUser = $account.name
            $isServicePrincipal = $true
            Write-ColorOutput "   Current principal: $currentUser (Service Principal)" $Green
        }
    } catch {
        Write-ColorOutput "   Could not parse account information" $Yellow
    }
} else {
    Write-ColorOutput "   FAIL: Not logged in to Azure" $Red
    Write-ColorOutput "   Run 'az login' to authenticate" $Yellow
    if ($accountOutput) {
        Write-ColorOutput "   Error details: $accountOutput" $Yellow
    }
    exit 1
}

# Try to get signed-in user details and object ID
if (-not $isServicePrincipal) {
    $userOutput = az ad signed-in-user show --query "{name:userPrincipalName, id:id}" -o json 2>&1
    if ($LASTEXITCODE -eq 0) {
        try {
            $userInfo = $userOutput | ConvertFrom-Json
            if ($userInfo.name) {
                $currentUser = $userInfo.name
            }
            if ($userInfo.id) {
                $currentObjectId = $userInfo.id
                Write-ColorOutput "   Object ID: $currentObjectId" $Green
            }
        } catch {
            # Ignore parsing errors
        }
    }
} else {
    # For service principal, get object ID
    $spOutput = az ad sp show --id $currentUser --query id -o tsv 2>&1
    if ($LASTEXITCODE -eq 0 -and $spOutput) {
        $currentObjectId = $spOutput.Trim()
    }
}

if (-not $currentUser) {
    Write-ColorOutput "   FAIL: Could not determine current user from Azure CLI" $Red
    exit 1
}

Write-ColorOutput ""

# Use provided email or current user
if ([string]::IsNullOrWhiteSpace($UserEmail)) {
    $UserEmail = $currentUser
}

# Step 2: Verify Key Vault exists (with discovery fallback)
Write-ColorOutput "Step 2: Verifying Key Vault exists" $Blue
Write-ColorOutput "---------------------------------------------------" $Blue

$kvOutput = az keyvault show --name $vaultName --resource-group $ResourceGroup --output json 2>&1
$kvExists = $null

if ($LASTEXITCODE -eq 0) {
    try {
        $kvExists = $kvOutput | ConvertFrom-Json
        if ($kvExists -and $kvExists.name) {
            Write-ColorOutput "   Key Vault exists: $vaultName" $Green
        } else {
            $kvExists = $null
        }
    } catch {
        Write-ColorOutput "   Could not parse Key Vault output" $Yellow
        $kvExists = $null
    }
}

# If vault not found by name, try to discover by prefix
if (-not $kvExists) {
    Write-ColorOutput "   Key Vault not found by name: $vaultName" $Yellow
    Write-ColorOutput "   Searching for Key Vaults in resource group..." $Blue
    
    $kvListOutput = az keyvault list --resource-group $ResourceGroup --query "[?starts_with(name, 'kv-wealtharena-')].{name:name, id:id}" -o json 2>&1
    if ($LASTEXITCODE -eq 0) {
        try {
            $kvList = $kvListOutput | ConvertFrom-Json
            if ($kvList -and $kvList.Count -gt 0) {
                $vaultName = $kvList[0].name
                Write-ColorOutput "   Found Key Vault by prefix: $vaultName" $Green
                
                # Get full vault details
                $kvOutput = az keyvault show --name $vaultName --resource-group $ResourceGroup --output json 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $kvExists = $kvOutput | ConvertFrom-Json
                }
            }
        } catch {
            Write-ColorOutput "   Could not parse Key Vault list" $Yellow
        }
    }
}

if (-not $kvExists) {
    Write-ColorOutput "   FAIL: Key Vault not found: $vaultName" $Red
    Write-ColorOutput "   Create Key Vault with: az keyvault create --name $vaultName --resource-group $ResourceGroup --location <location>" $Blue
    Write-ColorOutput "   Or verify the Resource Group and Key Vault name are correct." $Yellow
    exit 1
}

Write-ColorOutput ""

# Step 3: Check current permissions
Write-ColorOutput "Step 3: Checking current permissions" $Blue
Write-ColorOutput "---------------------------------------------------" $Blue

$hasAccess = $false
$secretListOutput = az keyvault secret list --vault-name $vaultName --query "[].name" -o tsv 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput "   Permissions are already sufficient" $Green
    $hasAccess = $true
} else {
    Write-ColorOutput "   Current permissions insufficient" $Yellow
    if ($secretListOutput) {
        Write-ColorOutput "   Error details: $secretListOutput" $Yellow
    }
}

Write-ColorOutput ""

# Step 4: Grant permissions if needed
if (-not $hasAccess) {
    Write-ColorOutput "Step 4: Granting secret permissions" $Blue
    Write-ColorOutput "---------------------------------------------------" $Blue
    Write-ColorOutput "   Granting permissions to: $UserEmail" $Cyan
    
    # Check if Key Vault uses RBAC (enableRbacAuthorization)
    $kvUsesRbac = $false
    if ($kvExists.properties.enableRbacAuthorization -eq $true) {
        $kvUsesRbac = $true
        Write-ColorOutput "   Key Vault uses RBAC authorization (Access Policies disabled)" $Cyan
    }
    
    $permissionsGranted = $false
    
    # Strategy 1: Try Access Policy approach first (only if RBAC is not enabled)
    if (-not $kvUsesRbac) {
        Write-ColorOutput "   Attempting Access Policy approach..." $Blue
        $policyOutput = az keyvault set-policy `
            --name $vaultName `
            --upn $UserEmail `
            --secret-permissions get list `
            --output none 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "   PASS: Granted access policy permissions" $Green
            $permissionsGranted = $true
        } else {
            Write-ColorOutput "   Access policy approach failed (exit code: $LASTEXITCODE)" $Yellow
            Write-ColorOutput "   This may indicate the Key Vault uses RBAC instead of Access Policies" $Blue
            if ($policyOutput) {
                Write-ColorOutput "   Error details: $policyOutput" $Yellow
            }
        }
    } else {
        Write-ColorOutput "   Skipping Access Policy (Key Vault is RBAC-only)" $Cyan
    }
    
    # Strategy 2: If access policy failed, try RBAC fallback
    if (-not $permissionsGranted) {
        Write-ColorOutput "   Trying RBAC-based permissions..." $Blue
        # Get Key Vault resource ID
        $kvId = $kvExists.id
        if (-not $kvId) {
            Write-ColorOutput "   Could not get Key Vault resource ID" $Yellow
        } else {
            # Get object ID if not already retrieved
            if (-not $currentObjectId) {
                # First try using the signed-in user's object ID (most reliable)
                if ($account -and $account.user) {
                    # Try to get from signed-in user
                    $signedInUserOutput = az ad signed-in-user show --query id -o tsv 2>&1
                    if ($LASTEXITCODE -eq 0 -and $signedInUserOutput) {
                        $currentObjectId = $signedInUserOutput.Trim()
                        Write-ColorOutput "   Using signed-in user object ID: $currentObjectId" $Cyan
                    }
                }
                
                # If still not found, try looking up by email/UPN
                if (-not $currentObjectId) {
                    # Try user first
                    $objectIdOutput = az ad user show --id $UserEmail --query id -o tsv 2>&1
                    if ($LASTEXITCODE -eq 0 -and $objectIdOutput) {
                        $currentObjectId = $objectIdOutput.Trim()
                    } else {
                        # Try service principal
                        $spOutput = az ad sp show --id $UserEmail --query id -o tsv 2>&1
                        if ($LASTEXITCODE -eq 0 -and $spOutput) {
                            $currentObjectId = $spOutput.Trim()
                        }
                    }
                }
            }
            
            if ($currentObjectId) {
                Write-ColorOutput "   Assigning RBAC role 'Key Vault Secrets User'..." $Blue
                $rbacOutput = az role assignment create `
                    --role "Key Vault Secrets User" `
                    --assignee $currentObjectId `
                    --scope $kvId `
                    --output none 2>&1
                
                if ($LASTEXITCODE -eq 0) {
                    Write-ColorOutput "   PASS: Granted RBAC permissions (Key Vault Secrets User)" $Green
                    $permissionsGranted = $true
                } else {
                    # Check if assignment already exists
                    $checkOutput = az role assignment list --assignee $currentObjectId --scope $kvId --role "Key Vault Secrets User" --query "[].id" -o tsv 2>&1
                    if ($LASTEXITCODE -eq 0 -and $checkOutput) {
                        Write-ColorOutput "   PASS: RBAC role assignment already exists" $Green
                        $permissionsGranted = $true
                    } else {
                        Write-ColorOutput "   RBAC role assignment failed" $Yellow
                        if ($rbacOutput) {
                            Write-ColorOutput "   Error details: $rbacOutput" $Yellow
                        }
                    }
                }
            } else {
                # Try assigning by UPN/email directly
                Write-ColorOutput "   Attempting RBAC assignment by UPN..." $Blue
                $rbacOutput = az role assignment create `
                    --role "Key Vault Secrets User" `
                    --assignee $UserEmail `
                    --scope $kvId `
                    --output none 2>&1
                
                if ($LASTEXITCODE -eq 0) {
                    Write-ColorOutput "   PASS: Granted RBAC permissions by UPN" $Green
                    $permissionsGranted = $true
                } else {
                    $checkOutput = az role assignment list --assignee $UserEmail --scope $kvId --role "Key Vault Secrets User" --query "[].id" -o tsv 2>&1
                    if ($LASTEXITCODE -eq 0 -and $checkOutput) {
                        Write-ColorOutput "   PASS: RBAC role assignment already exists" $Green
                        $permissionsGranted = $true
                    } else {
                        Write-ColorOutput "   RBAC assignment by UPN failed" $Yellow
                        if ($rbacOutput) {
                            Write-ColorOutput "   Error details: $rbacOutput" $Yellow
                        }
                    }
                }
            }
        }
    }
    
    if (-not $permissionsGranted) {
        Write-ColorOutput "   FAIL: Failed to grant permissions automatically" $Red
        Write-ColorOutput "" $Red
        Write-ColorOutput "MANUAL ACTION REQUIRED:" $Yellow
        Write-ColorOutput "Option 1 (Access Policy):" $Cyan
        Write-ColorOutput "  az keyvault set-policy --name $vaultName --upn $UserEmail --secret-permissions get list" $Blue
        Write-ColorOutput "Option 2 (RBAC):" $Cyan
        Write-ColorOutput "  az role assignment create --role 'Key Vault Secrets User' --assignee $UserEmail --scope $($kvExists.id)" $Blue
        exit 1
    }
    
    Write-ColorOutput ""
}

# Step 5: Wait for propagation
Write-ColorOutput "Step 5: Waiting for permission propagation" $Blue
Write-ColorOutput "---------------------------------------------------" $Blue
Write-ColorOutput "   Waiting 10 seconds for permissions to propagate..." $Cyan
Start-Sleep -Seconds 10
Write-ColorOutput ""

# Step 6: Verify Permissions
Write-ColorOutput "Step 6: Verifying permissions" $Blue
Write-ColorOutput "---------------------------------------------------" $Blue

$verifySuccess = $false
$secretListOutput = az keyvault secret list --vault-name $vaultName --query "[].name" -o tsv 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput "   PASS: Permissions verified" $Green
    Write-ColorOutput "   Successfully listed secrets from Key Vault" $Green
    $verifySuccess = $true
} else {
    Write-ColorOutput "   FAIL: Permission verification failed (exit code: $LASTEXITCODE)" $Red
    if ($secretListOutput) {
        Write-ColorOutput "   Error details: $secretListOutput" $Yellow
    }
    Write-ColorOutput "   User may need to sign out and sign back in to Azure CLI" $Yellow
    Write-ColorOutput "   Or wait a few minutes for permissions to propagate" $Yellow
}

Write-ColorOutput ""

if ($verifySuccess) {
    Write-ColorOutput "========================================" $Cyan
    Write-ColorOutput "Verification Summary" $Cyan
    Write-ColorOutput "========================================" $Cyan
    Write-ColorOutput ""
    Write-ColorOutput "Key Vault permissions fixed successfully" $Green
    Write-ColorOutput "Key Vault: $vaultName" $Cyan
    Write-ColorOutput "User: $UserEmail" $Cyan
    Write-ColorOutput ""
    Write-ColorOutput "Next steps:" $Yellow
    Write-ColorOutput "  1. Verify access: az keyvault secret list --vault-name $vaultName" $Blue
    Write-ColorOutput "  2. Retrieve secrets as needed for deployment" $Blue
    Write-ColorOutput ""
    exit 0
} else {
    Write-ColorOutput "========================================" $Cyan
    Write-ColorOutput "Verification Summary" $Cyan
    Write-ColorOutput "========================================" $Cyan
    Write-ColorOutput ""
    Write-ColorOutput "Permission verification failed" $Red
    Write-ColorOutput ""
    Write-ColorOutput "Troubleshooting:" $Yellow
    Write-ColorOutput "  1. Wait a few minutes and try again" $Blue
    Write-ColorOutput "  2. Sign out and sign back in to Azure CLI: az logout && az login" $Blue
    Write-ColorOutput "  3. Verify manually: az keyvault secret list --vault-name $vaultName" $Blue
    Write-ColorOutput ""
    exit 1
}

# Note: Secret storage functionality removed per requirements
# This script focuses on permissions only

