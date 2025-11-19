# Azure Fixes Applied

## Issues Found and Fixed

### 1. ✅ Malformed App Settings
**Problem:** The `SCM_DO_BUILD_DURING_DEPLOYMENT` setting had all other settings concatenated into it as a single value, causing configuration issues.

**Fix:** Reset all app settings individually with proper values:
- `SCM_DO_BUILD_DURING_DEPLOYMENT=true`
- `PYTHONPATH=/home/site/wwwroot`
- `PYTHONUNBUFFERED=1`
- `PORT=8000`
- `WEBSITES_CONTAINER_START_TIME_LIMIT=600` (10 minutes)
- `SCM_COMMAND_IDLE_TIMEOUT=1800` (30 minutes)
- `SCM_BUILD_TIMEOUT=1800` (30 minutes)
- `CHROMA_PERSIST_DIR=/home/data/vectorstore`
- `GROQ_API_KEY` (set correctly)
- All other required settings

### 2. ✅ Container Timeout
**Problem:** Container was timing out after 230 seconds because dependencies couldn't install due to configuration issues.

**Fix:** 
- Increased `WEBSITES_CONTAINER_START_TIME_LIMIT` to 600 seconds (10 minutes)
- Fixed app settings so dependencies can install properly
- Restarted the app

### 3. ✅ Dependency Resolution Issues
**Problem:** Logs showed `ERROR: ResolutionImpossible` - pip couldn't resolve dependencies.

**Status:** This should be resolved now that app settings are fixed. If it persists, we may need to:
- Use `--use-deprecated=legacy-resolver` flag (but Azure Oryx doesn't support this directly)
- Review and update conflicting package versions in `requirements.txt`

## Next Steps

1. **Monitor the app** - Check if it starts successfully now
2. **Check logs** - Verify dependencies are installing correctly
3. **Test health endpoint** - Ensure `/healthz` responds
4. **Test API endpoints** - Verify chat/search endpoints work

## Commands to Check Status

```powershell
# Check app state
az webapp show --name "wealtharena-api-kanika" --resource-group "rg-wealtharena-kanika" --query "state"

# Check logs
az webapp log tail --name "wealtharena-api-kanika" --resource-group "rg-wealtharena-kanika"

# Test health endpoint
Invoke-WebRequest -Uri "https://wealtharena-api-kanika.azurewebsites.net/healthz" -UseBasicParsing

# Check app settings
az webapp config appsettings list --name "wealtharena-api-kanika" --resource-group "rg-wealtharena-kanika"
```

## If Issues Persist

If the container still times out or dependencies fail to install:

1. **Check dependency conflicts:**
   ```powershell
   # Review requirements.txt for version conflicts
   # Common issues: pandas/numpy version mismatches, chromadb dependencies
   ```

2. **Try redeploying:**
   ```powershell
   .\deploy-master.ps1 --deploy azure -ResourceGroup "rg-wealtharena-kanika" -AppName "wealtharena-api-kanika"
   ```

3. **Check Oryx build logs:**
   - Go to Azure Portal → App Service → Deployment Center → Logs
   - Look for Oryx build output and dependency installation errors

4. **Consider using Docker deployment instead:**
   - Docker builds locally where you have more control
   - Then push to Azure Container Registry
