# RL Training Testing and Coverage Guide

## Overview

The RL Training service uses pytest for testing with coverage reporting. This guide covers test setup, running tests, generating coverage reports, and integrating with SonarQube for code quality analysis.

## Prerequisites

### Required Packages

```bash
pip install pytest pytest-cov coverage
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include:
- `pytest>=7.0.0`
- `pytest-cov>=4.0.0`
- `coverage>=7.0.0`

## Pytest Setup

### Configuration Files

#### `pytest.ini`

Located in `rl-training/pytest.ini`, this file configures pytest behavior:

```ini
[pytest]
# Test discovery patterns
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Test paths
testpaths = .

# Output options
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings

# Asyncio configuration
asyncio_mode = auto

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    gpu: marks tests that require GPU

# Ignore paths
norecursedirs = 
    .git
    .tox
    dist
    build
    *.egg
    __pycache__
    checkpoints
    logs
    results
    models
    data
    notebooks
```

#### `.coveragerc`

Located in `rl-training/.coveragerc`, this file configures coverage reporting:

```ini
[run]
source = src
omit =
    */tests/*
    */test_*.py
    */notebooks/*
    */data/*
    */models/*
    */checkpoints/*
    */logs/*
    */__pycache__/*
    */venv/*
    */env/*
    */ENV/*
    */.venv/*

[report]
precision = 2
show_missing = True
skip_covered = False

[xml]
output = coverage.xml
```

## Running Tests

### Basic Test Execution

```bash
cd rl-training

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_environment.py

# Run specific test function
pytest tests/test_environment.py::test_environment_initialization

# Run tests matching pattern
pytest -k "test_environment"

# Run tests with specific marker
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

### Test Structure

Tests are located in `tests/` directory:
- `tests/test_environments.py` - Environment tests
- `tests/test_models.py` - Model tests
- `tests/test_training.py` - Training pipeline tests
- `tests/test_metrics.py` - Metrics collection tests

### Example Test

```python
import pytest
from src.environments.trading_env import TradingEnvironment

def test_environment_initialization():
    env = TradingEnvironment()
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None

def test_environment_reset():
    env = TradingEnvironment()
    obs = env.reset()
    assert obs is not None
    assert len(obs) > 0

@pytest.mark.slow
def test_training_episode():
    env = TradingEnvironment()
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        steps += 1
    assert steps > 0
```

## Generating Coverage Reports

### Quick Command

```bash
# Generate coverage.xml for SonarQube
pytest --cov=src --cov-report=xml --cov-report=term
```

This will:
- Run all tests in the project
- Generate a `coverage.xml` file in the root directory (for SonarQube)
- Display coverage summary in terminal

### Coverage Options

```bash
# Generate XML report only
pytest --cov=src --cov-report=xml

# Generate HTML report (for local viewing)
pytest --cov=src --cov-report=html

# Generate both XML and HTML
pytest --cov=src --cov-report=xml --cov-report=html

# Generate terminal output only
pytest --cov=src --cov-report=term

# Generate all report types
pytest --cov=src --cov-report=xml --cov-report=html --cov-report=term
```

### Coverage for Specific Modules

```bash
# Coverage for src directory only
pytest --cov=src --cov-report=xml

# Coverage for specific module
pytest --cov=src/environments --cov-report=xml

# Coverage for specific file
pytest --cov=src/environments/trading_env --cov-report=xml
```

### Using Helper Scripts

#### Windows Batch Script

```cmd
cd rl-training
run_coverage.bat
```

#### Python Script

```bash
cd rl-training
python run_coverage.py
```

Both scripts will:
1. Run tests with coverage
2. Generate `coverage.xml` for SonarQube
3. Display coverage summary

## Coverage Reports

### Terminal Output

When running with `--cov-report=term`, you'll see:

```
---------- coverage: platform win32, python 3.11 -----------
Name                                 Stmts   Miss  Cover
--------------------------------------------------------
src/environments/trading_env.py       180     50    72%
src/models/ppo_agent.py              200     60    70%
src/training/trainer.py               150     45    70%
--------------------------------------------------------
TOTAL                                2000    600    70%
```

### XML Report (`coverage.xml`)

The XML report is generated for SonarQube integration. It contains:
- File paths and line numbers
- Coverage percentages per file
- Branch coverage information
- Source code locations

**Location:** `rl-training/coverage.xml`

### HTML Report (`htmlcov/index.html`)

The HTML report provides interactive coverage visualization:

```bash
# Generate HTML report
pytest --cov=src --cov-report=html

# Open in browser
# Windows:
start htmlcov/index.html

# macOS:
open htmlcov/index.html

# Linux:
xdg-open htmlcov/index.html
```

## SonarQube Integration

### Setup Summary

The project is configured for SonarQube code coverage analysis with:

1. **`pytest.ini`** - Configures pytest test discovery and options
2. **`.coveragerc`** - Configures coverage reporting and exclusions
3. **`requirements.txt`** - Includes `pytest-cov>=4.0.0`
4. **`coverage.xml`** - Generated coverage report for SonarQube

### Generating Coverage for SonarQube

```bash
cd rl-training
pytest --cov=src --cov-report=xml
```

This generates `coverage.xml` which SonarQube needs for coverage analysis.

### SonarQube Configuration

Ensure your `sonar-project.properties` includes:

```properties
# Source code location
sonar.sources=rl-training/src

# Test patterns
sonar.test.inclusions=**/test_*.py

# Coverage
sonar.python.coverage.reportPaths=rl-training/coverage.xml

# Exclusions
sonar.exclusions=**/data/**,**/checkpoints/**,**/logs/**,**/results/**,**/notebooks/**,**/__pycache__/**
```

### Fixing Zero Coverage Issues

If SonarQube shows 0% coverage, ensure:

1. **Coverage XML has correct paths:**
   ```bash
   # Check coverage.xml source path
   grep "<source>" rl-training/coverage.xml
   ```
   Should show absolute path to `rl-training` directory.

2. **File paths in coverage.xml are relative:**
   ```bash
   # Check file paths
   grep 'filename="' rl-training/coverage.xml | head -3
   ```
   Should show filenames relative to source directory.

3. **Regenerate coverage after config changes:**
   ```bash
   # Delete old coverage.xml
   rm rl-training/coverage.xml
   
   # Regenerate
   cd rl-training
   pytest --cov=src --cov-report=xml
   ```

### Viewing SonarQube Results

After running SonarQube scan:

1. Go to your SonarQube project dashboard
2. Check "Measures" → "Coverage"
3. Review coverage by file, module, or directory
4. Check for uncovered lines and branches

## Coverage Goals

### Target Coverage Levels

- **Overall coverage**: >80%
- **Critical paths**: >90%
- **Utility functions**: >70%
- **Test files**: Excluded from coverage

### Improving Coverage

1. **Identify uncovered code:**
   ```bash
   # Generate HTML report to see uncovered lines
   pytest --cov=src --cov-report=html
   open htmlcov/index.html
   ```

2. **Write tests for uncovered code:**
   - Focus on critical paths first
   - Add unit tests for utility functions
   - Add integration tests for complex workflows

3. **Run coverage regularly:**
   ```bash
   # Before committing
   pytest --cov=src --cov-report=term
   ```

## CI Integration

### GitHub Actions

```yaml
name: Tests and Coverage

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd rl-training
          pip install -r requirements.txt
      
      - name: Run tests with coverage
        run: |
          cd rl-training
          pytest --cov=src --cov-report=xml --cov-report=term
      
      - name: SonarQube Scan
        uses: SonarSource/sonarcloud-scan-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

### GitLab CI

```yaml
test:
  stage: test
  script:
    - cd rl-training
    - pip install -r requirements.txt
    - pytest --cov=src --cov-report=xml --cov-report=term
  coverage: '/TOTAL.+ ([0-9]{1,3}%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: rl-training/coverage.xml

sonarqube:
  stage: quality
  dependencies:
    - test
  script:
    - sonar-scanner
```

## Writing Tests

### Best Practices

1. **Test user interactions, not implementation details**
2. **Use descriptive test names:**
   ```python
   def test_environment_reset_returns_valid_observation():
       # Good
   
   def test_reset():
       # Less descriptive
   ```

3. **Mock external dependencies:**
   ```python
   from unittest.mock import Mock, patch
   
   @patch('src.data.market_data.fetch_data')
   def test_training_with_mock_data(mock_fetch):
       mock_fetch.return_value = sample_data
       # Test with mocked data
   ```

4. **Use fixtures for common setup:**
   ```python
   @pytest.fixture
   def trading_env():
       return TradingEnvironment()
   
   def test_environment_reset(trading_env):
       obs = trading_env.reset()
       assert obs is not None
   ```

5. **Aim for >80% code coverage**
6. **Write tests before fixing bugs (TDD when possible)**

### Test Markers

Use markers to organize tests:

```python
@pytest.mark.unit
def test_unit_function():
    # Fast unit test
    pass

@pytest.mark.integration
def test_integration_workflow():
    # Slower integration test
    pass

@pytest.mark.slow
def test_long_running_test():
    # Very slow test
    pass

@pytest.mark.gpu
def test_gpu_required():
    # Requires GPU
    pass
```

Run tests by marker:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"
```

## Troubleshooting

### Issue: "No module named pytest"

```bash
pip install pytest pytest-cov
```

### Issue: "No data to report"

Make sure you have test files:
- Files should be named `test_*.py` or `*_test.py`
- Test functions should start with `test_`
- Test classes should start with `Test`

Check what pytest finds:
```bash
pytest --collect-only
```

### Issue: Coverage is 0% or very low

1. **Make sure tests are actually running:**
   ```bash
   pytest -v
   ```

2. **Check that your source code is being imported in tests:**
   ```python
   # In test file
   from src.environments.trading_env import TradingEnvironment
   ```

3. **Verify `.coveragerc` isn't excluding too much:**
   ```bash
   # Check what's being excluded
   cat .coveragerc
   ```

### Issue: Tests are failing

```bash
# See which tests are failing
pytest -v

# Run a specific test file
pytest tests/test_environment.py -v

# Run a specific test function
pytest tests/test_environment.py::test_environment_initialization -v

# Get more detailed error output
pytest -vv
```

### Issue: Coverage XML not generated

1. **Check if pytest-cov is installed:**
   ```bash
   pip list | grep pytest-cov
   ```

2. **Verify coverage options:**
   ```bash
   pytest --cov=src --cov-report=xml -v
   ```

3. **Check file permissions:**
   ```bash
   ls -la coverage.xml
   ```

### Issue: SonarQube shows 0% coverage

1. **Verify coverage.xml exists:**
   ```bash
   ls -la rl-training/coverage.xml
   ```

2. **Check coverage.xml paths:**
   ```bash
   grep "<source>" rl-training/coverage.xml
   grep 'filename="' rl-training/coverage.xml | head -3
   ```

3. **Regenerate coverage:**
   ```bash
   rm rl-training/coverage.xml
   cd rl-training
   pytest --cov=src --cov-report=xml
   ```

## Quick Reference

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest -v` | Run tests with verbose output |
| `pytest --cov=src --cov-report=xml` | Generate coverage.xml for SonarQube |
| `pytest --cov=src --cov-report=html` | Generate HTML coverage report |
| `pytest --cov=src --cov-report=term` | Show coverage in terminal |
| `pytest -m unit` | Run only unit tests |
| `pytest -m "not slow"` | Exclude slow tests |
| `pytest -k "test_name"` | Run tests matching pattern |
| `pytest --collect-only` | List all tests without running |

## Additional Resources

- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py Documentation**: https://coverage.readthedocs.io/
- **SonarQube Python Plugin**: https://docs.sonarqube.org/latest/analysis/languages/python/

---

*Last Updated: 2025*
*Status: Production Ready* ✅
