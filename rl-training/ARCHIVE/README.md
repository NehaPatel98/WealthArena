# RL Training Archive

This directory contains archived documentation files that have been consolidated into other locations.

## Archived Files

Files in this directory have been merged into consolidated documentation:

| Archived File | Replacement | Section |
|--------------|-------------|---------|
| `STEP_BY_STEP_AZURE_DEPLOYMENT.md` | `rl-training/DEPLOYMENT.md` | Azure Deployment |
| `AZURE_DEPLOYMENT_COMPLETE_GUIDE.md` | `rl-training/DEPLOYMENT.md` | Azure Deployment |
| `COMPLETE_LAYMAN_DEPLOYMENT_GUIDE.md` | `rl-training/DEPLOYMENT.md` | Deployment Overview |
| `SETUP_SUMMARY.md` | `rl-training/TESTING_AND_COVERAGE.md` | Pytest Setup |
| `RUN_COVERAGE.md` | `rl-training/TESTING_AND_COVERAGE.md` | Coverage Reports |
| `SONARQUBE_COVERAGE_SETUP.md` | `rl-training/TESTING_AND_COVERAGE.md` | SonarQube Integration |
| `SONARQUBE_FIX.md` | `rl-training/TESTING_AND_COVERAGE.md` | SonarQube Troubleshooting |

## Consolidation Targets

- **`rl-training/DEPLOYMENT.md`** - Consolidated deployment documentation including:
  - Prerequisites
  - Local Deployment (Docker Compose)
  - Azure Deployment (Container Apps/ACR)
  - Model Deployment (`upload_models.ps1`)
  - Testing
  - Troubleshooting
  - Cost Controls

- **`rl-training/TESTING_AND_COVERAGE.md`** - Consolidated testing and coverage documentation including:
  - Pytest Setup
  - Running Tests
  - Generating `coverage.xml`
  - SonarQube Integration
  - Improving Coverage
  - CI Instructions
  - References to `.coveragerc` and `pytest.ini`

## Note

Do not delete archived files immediately. Keep them for reference during the consolidation period, then remove after confirming all content has been properly merged.
