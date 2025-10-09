# Dependency Management Guide

This document explains how to manage dependencies in the congenial-fortnight project and prevent import errors.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dependency Files](#dependency-files)
3. [Critical Dependencies](#critical-dependencies)
4. [Validation Tools](#validation-tools)
5. [Common Issues](#common-issues)
6. [CI/CD Integration](#cicd-integration)

## Quick Start

### Install All Dependencies

```bash
# Install TA-Lib system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y wget build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo ldconfig
cd ..

# Install Python dependencies
pip install -r requirements-tests.txt
pip install -r requirements.txt
```

### Validate Dependencies

```bash
# Run comprehensive validation
python validate_dependencies.py

# Quick check
python -c "import talib, pandas, numpy, lightgbm, sklearn, ta, pytest; print('✓ All critical imports OK')"
```

## Dependency Files

### `requirements.txt`

Main production dependencies for the trading system:

- **Django & REST Framework**: Web API and backend
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, Prophet
- **Data Processing**: pandas, numpy, scipy
- **Technical Analysis**: ta, TA-Lib
- **Visualization**: matplotlib, seaborn, plotly
- **APIs**: yfinance, fredapi
- **Utilities**: python-dotenv, requests

### `requirements-tests.txt`

Testing and development dependencies:

- **Testing**: pytest, pytest-cov, pytest-httpx
- **Validation**: jsonschema, responses
- **Core Libraries**: Subset of requirements.txt needed for tests

## Critical Dependencies

### TA-Lib (Technical Analysis Library)

**Why Critical**: Used by 4+ signal generation modules
- `scripts/day_trading_signals.py`
- `scripts/pip_based_signal_system.py`
- `scripts/candlestick_patterns.py`
- `scripts/slump_signals.py`

**Installation**:

TA-Lib requires system-level C library installation before the Python package:

```bash
# Ubuntu/Debian
sudo apt-get install -y wget build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo ldconfig

# Then install Python wrapper
pip install TA-Lib==0.4.28
```

**macOS**:
```bash
brew install ta-lib
pip install TA-Lib==0.4.28
```

**Windows**:
Download pre-built wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### fundamentals.py Module

**Why Critical**: Required by test modules
- `tests/test_fundamental_features_schema.py`
- `tests/test_network_failures.py`

**Location**: `/workspaces/congenial-fortnight/fundamentals.py`

**Purpose**: Provides unified interface for fetching fundamental data from various APIs

### NumPy Version Compatibility

**Critical Constraint**: `numpy<2.0.0` (due to darts dependency)

Current version: `numpy==1.26.4`

**Do NOT upgrade to numpy 2.x** - it will break the dependency chain:
- darts 0.30.0 requires numpy<2.0.0
- Multiple other packages depend on darts

## Validation Tools

### 1. Pre-Commit Validation Script

**File**: `validate_dependencies.py`

**Run before every commit**:
```bash
python validate_dependencies.py
```

**What it checks**:
- ✓ All critical packages can be imported
- ✓ All local modules exist
- ✓ Requirements files are valid
- ⚠️ Optional packages status

**Output**:
```
╔══════════════════════════════════════════════════════════════════╗
║               PRE-COMMIT DEPENDENCY VALIDATION                   ║
╚══════════════════════════════════════════════════════════════════╝

DEPENDENCY VALIDATION
======================================================================

✓ talib                         [CRITICAL] OK
✓ pandas                        [CRITICAL] OK
✓ numpy                         [CRITICAL] OK
✓ lightgbm                      [CRITICAL] OK
...

VALIDATION SUMMARY
======================================================================
✓ PASS   Requirements Files
✓ PASS   Python Dependencies
✓ PASS   Local Modules

✓ ALL VALIDATIONS PASSED
```

### 2. Pre-Commit Git Hook

**File**: `scripts/pre-commit-hook.sh`

**Install**:
```bash
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**What it does**:
- Automatically runs `validate_dependencies.py` before each commit
- Blocks commit if validation fails
- Can be bypassed with `git commit --no-verify`

### 3. GitHub Actions Workflow

**File**: `.github/workflows/validate_dependencies.yml`

**Runs on**:
- Every push to main or copilot/* branches
- Every pull request

**What it checks**:
- Installs TA-Lib system dependencies
- Installs all Python dependencies
- Verifies all imports can be resolved
- Checks for missing module references
- Runs pytest collection check

## Common Issues

### Issue 1: `ModuleNotFoundError: No module named 'talib'`

**Cause**: TA-Lib system library not installed

**Solution**:
```bash
# Install system library first
sudo apt-get install -y wget build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/ && ./configure --prefix=/usr && make && sudo make install

# Then install Python package
pip install TA-Lib==0.4.28
```

### Issue 2: `ModuleNotFoundError: No module named 'fundamentals'`

**Cause**: Missing fundamentals.py module

**Solution**:
```bash
# Verify file exists
ls -la fundamentals.py

# If missing, it should be in the repository
# Pull latest changes
git pull origin main
```

### Issue 3: NumPy version conflict

**Cause**: Trying to install numpy 2.x with darts

**Error**:
```
ERROR: Cannot install numpy==2.0.0 because darts 0.30.0 depends on numpy<2.0.0
```

**Solution**:
```bash
# Use numpy 1.26.4 (already in requirements.txt)
pip install numpy==1.26.4

# Or reinstall from requirements
pip install -r requirements.txt --force-reinstall
```

### Issue 4: pytest collection errors

**Cause**: Missing dependencies during test discovery

**Solution**:
```bash
# Install test dependencies
pip install -r requirements-tests.txt

# Install main dependencies
pip install -r requirements.txt

# Validate
python validate_dependencies.py

# Run tests
pytest -v
```

## CI/CD Integration

### GitHub Actions Setup

All workflows now include TA-Lib installation:

```yaml
- name: Install TA-Lib system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y wget build-essential
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    sudo ldconfig

- name: Install Python dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements-tests.txt
    pip install -r requirements.txt
```

### Modified Workflows

✓ `.github/workflows/dry_run.yml` - Added TA-Lib installation
✓ `.github/workflows/validate_dependencies.yml` - New comprehensive validation workflow

## Best Practices

1. **Before Committing**:
   ```bash
   python validate_dependencies.py
   ```

2. **After Pulling Changes**:
   ```bash
   pip install -r requirements.txt --upgrade
   python validate_dependencies.py
   ```

3. **When Adding New Dependencies**:
   - Add to appropriate requirements file
   - Update this documentation
   - Run validation script
   - Test in clean environment

4. **Version Pinning**:
   - Always pin exact versions (e.g., `numpy==1.26.4`)
   - Never use `>=` or `~=` for critical dependencies
   - Document version constraints

5. **Testing New Packages**:
   ```bash
   # Install in isolated environment
   python -m venv test_env
   source test_env/bin/activate
   pip install <new-package>
   
   # Verify compatibility
   pip install -r requirements.txt
   python validate_dependencies.py
   ```

## Troubleshooting

### Full Clean Reinstall

```bash
# Remove virtual environment
rm -rf .venv venv

# Create fresh environment
python -m venv .venv
source .venv/bin/activate

# Install TA-Lib system library
sudo apt-get install -y wget build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/ && ./configure --prefix=/usr && make && sudo make install
cd ..

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements-tests.txt
pip install -r requirements.txt

# Validate
python validate_dependencies.py
```

### Check Installed Versions

```bash
# List all installed packages
pip list

# Check specific package
pip show talib numpy pandas

# Verify imports work
python -c "import talib; print(f'TA-Lib version: {talib.__version__}')"
```

## Support

If you encounter dependency issues not covered here:

1. Run `python validate_dependencies.py` and share output
2. Check GitHub Actions logs for CI failures
3. Verify Python version: `python --version` (should be 3.11.x)
4. Check system: `uname -a`
5. Share full error message including traceback

## Version History

- **2025-10-08**: Added TA-Lib, fundamentals.py, validation tools
- **2025-10-08**: Fixed numpy version conflict (2.0.0 → 1.26.4)
- **2025-10-08**: Added comprehensive dependency validation
