# WealthArena Services Deployment Script
# This script deploys RAG chatbot and backend API services to Azure Container Apps

param(
    [string]$ResourceGroupName = "rg-wealtharena-northcentralus",
    [string]$Environment = "dev",
    [string]$ContainerRegistryName = "acrwealtharenadev",
    [string]$ContainerAppsEnvironment = "cae-wealtharena-dev"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$Green = "`e[32m"
$Red = "`e[31m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Reset = "`e[0m"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = $Reset)
    Write-Host "$Color$Message$Reset"
}

function Test-AzureCLI {
    try {
        $azVersion = az version --output json | ConvertFrom-Json
        Write-ColorOutput "✅ Azure CLI found: $($azVersion.'azure-cli')" $Green
        return $true
    }
    catch {
        Write-ColorOutput "❌ Azure CLI not found. Please install Azure CLI first." $Red
        return $false
    }
}

function Test-Docker {
    try {
        $dockerVersion = docker --version
        Write-ColorOutput "✅ Docker found: $dockerVersion" $Green
        return $true
    }
    catch {
        Write-ColorOutput "❌ Docker not found. Please install Docker Desktop first." $Red
        return $false
    }
}

function Build-DockerImage {
    param([string]$ServiceName, [string]$DockerfilePath, [string]$ImageTag)
    
    Write-ColorOutput "🐳 Building Docker image for $ServiceName..." $Blue
    
    try {
        # Build Docker image
        docker build -t $ImageTag -f $DockerfilePath .
        
        Write-ColorOutput "✅ Docker image built successfully: $ImageTag" $Green
        return $true
    }
    catch {
        Write-ColorOutput "❌ Failed to build Docker image for $ServiceName: $($_.Exception.Message)" $Red
        return $false
    }
}

function Push-DockerImage {
    param([string]$ImageTag, [string]$RegistryName)
    
    Write-ColorOutput "📤 Pushing Docker image to Azure Container Registry..." $Blue
    
    try {
        # Login to Azure Container Registry
        az acr login --name $RegistryName
        
        # Tag image for registry
        $registryImageTag = "$RegistryName.azurecr.io/$ImageTag"
        docker tag $ImageTag $registryImageTag
        
        # Push image
        docker push $registryImageTag
        
        Write-ColorOutput "✅ Docker image pushed successfully: $registryImageTag" $Green
        return $registryImageTag
    }
    catch {
        Write-ColorOutput "❌ Failed to push Docker image: $($_.Exception.Message)" $Red
        return $null
    }
}

function Deploy-ContainerApp {
    param([string]$AppName, [string]$ImageTag, [string]$ResourceGroup, [string]$Environment, [string]$Port, [hashtable]$EnvironmentVariables)
    
    Write-ColorOutput "🚀 Deploying Container App: $AppName..." $Blue
    
    try {
        # Build environment variables string
        $envVars = @()
        foreach ($key in $EnvironmentVariables.Keys) {
            $envVars += "$key=$($EnvironmentVariables[$key])"
        }
        $envVarsString = $envVars -join " "
        
        # Deploy Container App
        az containerapp create `
            --name $AppName `
            --resource-group $ResourceGroup `
            --environment $Environment `
            --image $ImageTag `
            --target-port $Port `
            --ingress external `
            --cpu 1.0 `
            --memory 2.0Gi `
            --env-vars $envVarsString `
            --registry-server "$ContainerRegistryName.azurecr.io" `
            --query-strings "registry=$ContainerRegistryName.azurecr.io"
        
        Write-ColorOutput "✅ Container App deployed successfully: $AppName" $Green
        return $true
    }
    catch {
        Write-ColorOutput "❌ Failed to deploy Container App $AppName: $($_.Exception.Message)" $Red
        return $false
    }
}

function Deploy-RAGChatbot {
    Write-ColorOutput "🤖 Deploying RAG Chatbot Service..." $Blue
    
    # Build Docker image
    $imageTag = "rag-chatbot:latest"
    $dockerfilePath = "azure_services/rag_chatbot_service/Dockerfile"
    
    if (-not (Build-DockerImage -ServiceName "RAG Chatbot" -DockerfilePath $dockerfilePath -ImageTag $imageTag)) {
        return $false
    }
    
    # Push to registry
    $registryImageTag = Push-DockerImage -ImageTag $imageTag -RegistryName $ContainerRegistryName
    if (-not $registryImageTag) {
        return $false
    }
    
    # Environment variables for RAG chatbot
    $chatbotEnvVars = @{
        "GROQ_API_KEY" = "YOUR_GROQ_API_KEY"
        "AZURE_COSMOS_ENDPOINT" = "https://cosmos-wealtharena-$Environment.documents.azure.com:443/"
        "AZURE_COSMOS_KEY" = "your_cosmos_key"
        "AZURE_COSMOS_DATABASE" = "wealtharena_cosmos"
        "AZURE_COSMOS_CONTAINER" = "knowledge_vectors"
        "VECTOR_DB_PATH" = "./chroma_db"
    }
    
    # Deploy Container App
    if (Deploy-ContainerApp -AppName "rag-chatbot" -ImageTag $registryImageTag -ResourceGroup $ResourceGroupName -Environment $ContainerAppsEnvironment -Port 8000 -EnvironmentVariables $chatbotEnvVars) {
        Write-ColorOutput "✅ RAG Chatbot deployed successfully!" $Green
        return $true
    }
    else {
        Write-ColorOutput "❌ Failed to deploy RAG Chatbot" $Red
        return $false
    }
}

function Deploy-BackendAPI {
    Write-ColorOutput "🔧 Deploying Backend API Service..." $Blue
    
    # Build Docker image
    $imageTag = "wealtharena-backend:latest"
    $dockerfilePath = "azure_services/wealtharena_backend_api/Dockerfile"
    
    if (-not (Build-DockerImage -ServiceName "Backend API" -DockerfilePath $dockerfilePath -ImageTag $imageTag)) {
        return $false
    }
    
    # Push to registry
    $registryImageTag = Push-DockerImage -ImageTag $imageTag -RegistryName $ContainerRegistryName
    if (-not $registryImageTag) {
        return $false
    }
    
    # Environment variables for backend API
    $backendEnvVars = @{
        "AZURE_SQL_SERVER" = "sql-wealtharena-$Environment.database.windows.net"
        "AZURE_SQL_DATABASE" = "wealtharena_db"
        "AZURE_SQL_USERNAME" = "wealtharena_admin"
        "AZURE_SQL_PASSWORD" = "Secure@Db2024!$%"
        "AZURE_STORAGE_ACCOUNT" = "stwealtharena$Environment"
        "AZURE_STORAGE_KEY" = "your_storage_key"
        "AZURE_COSMOS_ENDPOINT" = "https://cosmos-wealtharena-$Environment.documents.azure.com:443/"
        "AZURE_COSMOS_KEY" = "your_cosmos_key"
        "AZURE_COSMOS_DATABASE" = "wealtharena_cosmos"
        "CHATBOT_URL" = "https://rag-chatbot-dev.azurecontainerapps.io"
        "JWT_SECRET" = "wealtharena-secret-key-2024"
    }
    
    # Deploy Container App
    if (Deploy-ContainerApp -AppName "wealtharena-backend" -ImageTag $registryImageTag -ResourceGroup $ResourceGroupName -Environment $ContainerAppsEnvironment -Port 8000 -EnvironmentVariables $backendEnvVars) {
        Write-ColorOutput "✅ Backend API deployed successfully!" $Green
        return $true
    }
    else {
        Write-ColorOutput "❌ Failed to deploy Backend API" $Red
        return $false
    }
}

function Test-Services {
    Write-ColorOutput "🧪 Testing deployed services..." $Blue
    
    $services = @(
        @{
            Name = "RAG Chatbot"
            Url = "https://rag-chatbot-dev.azurecontainerapps.io/healthz"
        },
        @{
            Name = "Backend API"
            Url = "https://wealtharena-backend-dev.azurecontainerapps.io/healthz"
        }
    )
    
    $testResults = @{}
    
    foreach ($service in $services) {
        try {
            $response = Invoke-RestMethod -Uri $service.Url -Method GET -TimeoutSec 30
            
            if ($response.status -eq "healthy") {
                $testResults[$service.Name] = $true
                Write-ColorOutput "✅ $($service.Name): Healthy" $Green
            }
            else {
                $testResults[$service.Name] = $false
                Write-ColorOutput "❌ $($service.Name): Unhealthy" $Red
            }
        }
        catch {
            $testResults[$service.Name] = $false
            Write-ColorOutput "❌ $($service.Name): $($_.Exception.Message)" $Red
        }
    }
    
    return $testResults
}

function Show-DeploymentSummary {
    param([hashtable]$TestResults)
    
    Write-ColorOutput "📊 WealthArena Services Deployment Summary" $Blue
    Write-ColorOutput "===========================================" $Blue
    
    $totalServices = $TestResults.Count
    $healthyServices = ($TestResults.Values | Where-Object { $_ -eq $true }).Count
    $unhealthyServices = $totalServices - $healthyServices
    
    Write-ColorOutput "Total Services: $totalServices" $Blue
    Write-ColorOutput "Healthy: $healthyServices" $Green
    Write-ColorOutput "Unhealthy: $unhealthyServices" $(if ($unhealthyServices -gt 0) { $Red } else { $Green })
    
    Write-ColorOutput "`nService Details:" $Blue
    foreach ($service in $TestResults.GetEnumerator()) {
        $status = if ($service.Value) { "✅" } else { "❌" }
        Write-ColorOutput "  $status $($service.Key)" $Blue
    }
    
    if ($unhealthyServices -eq 0) {
        Write-ColorOutput "`n🎉 All services deployed successfully!" $Green
        Write-ColorOutput "Service URLs:" $Blue
        Write-ColorOutput "  RAG Chatbot: https://rag-chatbot-dev.azurecontainerapps.io" $Blue
        Write-ColorOutput "  Backend API: https://wealtharena-backend-dev.azurecontainerapps.io" $Blue
        Write-ColorOutput "`nNext steps:" $Blue
        Write-ColorOutput "  1. Update frontend with service URLs" $Blue
        Write-ColorOutput "  2. Run integration tests" $Blue
        Write-ColorOutput "  3. Deploy frontend to Azure Static Web Apps" $Blue
    }
    else {
        Write-ColorOutput "`n⚠️ Some services failed to deploy. Please check the errors above." $Yellow
    }
}

# Main execution
Write-ColorOutput "🚀 WealthArena Services Deployment" $Blue
Write-ColorOutput "===================================" $Blue

# Check prerequisites
if (-not (Test-AzureCLI)) {
    exit 1
}

if (-not (Test-Docker)) {
    exit 1
}

# Deploy services
$deploymentResults = @{}

# Deploy RAG Chatbot
if (Deploy-RAGChatbot) {
    $deploymentResults["RAG Chatbot"] = $true
} else {
    $deploymentResults["RAG Chatbot"] = $false
}

# Deploy Backend API
if (Deploy-BackendAPI) {
    $deploymentResults["Backend API"] = $true
} else {
    $deploymentResults["Backend API"] = $false
}

# Test services
$testResults = Test-Services

# Show summary
Show-DeploymentSummary -TestResults $testResults

# Generate deployment report
$deploymentReport = @{
    timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    deployment_results = $deploymentResults
    test_results = $testResults
    service_urls = @{
        "rag_chatbot" = "https://rag-chatbot-dev.azurecontainerapps.io"
        "backend_api" = "https://wealtharena-backend-dev.azurecontainerapps.io"
    }
}

$deploymentReportJson = $deploymentReport | ConvertTo-Json -Depth 3
$deploymentReportJson | Out-File -FilePath "services_deployment_report.json" -Encoding UTF8

Write-ColorOutput "`n📄 Deployment report saved to: services_deployment_report.json" $Blue

if ($testResults.Values -contains $false) {
    exit 1
} else {
    exit 0
}
