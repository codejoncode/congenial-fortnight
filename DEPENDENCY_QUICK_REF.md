# Quick Reference: Dependency Management

## 🚀 Quick Start

### Install Everything
```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y wget build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/ && ./configure --prefix=/usr && make && sudo make install && cd ..

# Python dependencies
pip install -r requirements-tests.txt
pip install -r requirements.txt
```

### Validate Before Commit
```bash
python validate_dependencies.py
```

### Install Git Hook (One-Time)
```bash
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## 🔍 Quick Checks

### Test All Imports
```bash
python -c "import talib, pandas, numpy, lightgbm, sklearn, ta, pytest; print('✓ OK')"
```

### Check Package Versions
```bash
pip show talib numpy pandas | grep -E "(Name|Version)"
```

### List All Packages
```bash
pip list | grep -E "(talib|numpy|pandas|lightgbm|sklearn|ta|pytest)"
```

## ❌ Common Errors & Fixes

### Error: `ModuleNotFoundError: No module named 'talib'`
```bash
# Install TA-Lib C library first
sudo apt-get install -y wget build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/ && ./configure --prefix=/usr && make && sudo make install
pip install TA-Lib==0.4.28
```

### Error: `ModuleNotFoundError: No module named 'fundamentals'`
```bash
# File should exist in repository root
ls -la fundamentals.py
# If missing, pull latest changes
git pull origin copilot/vscode1759760951002
```

### Error: NumPy version conflict
```bash
# Use version 1.26.4 (not 2.x)
pip install numpy==1.26.4
```

## 📋 Pre-Commit Checklist

- [ ] Run `python validate_dependencies.py`
- [ ] All tests pass: `pytest tests/`
- [ ] No import errors
- [ ] Requirements files updated (if adding dependencies)
- [ ] Documentation updated (if changing dependencies)

## 🔧 Adding New Dependencies

```bash
# 1. Add to appropriate requirements file
echo "new-package==1.0.0" >> requirements.txt

# 2. Install and test
pip install new-package==1.0.0
python validate_dependencies.py

# 3. Test in clean environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
pytest tests/

# 4. Commit if successful
git add requirements.txt
git commit -m "feat: Add new-package dependency"
```

## 📚 Documentation

- **Full Guide**: `DEPENDENCY_MANAGEMENT.md`
- **Fix Details**: `TEST_INFRASTRUCTURE_FIX.md`
- **Validation Script**: `validate_dependencies.py`
- **Pre-Commit Hook**: `scripts/pre-commit-hook.sh`

## 🐛 Troubleshooting

### Full Clean Reinstall
```bash
rm -rf .venv venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-tests.txt
pip install -r requirements.txt
python validate_dependencies.py
```

### Check Python Version
```bash
python --version  # Should be 3.11.x
```

### Verify TA-Lib Installation
```bash
python -c "import talib; print(f'TA-Lib: {talib.__version__}')"
```

## 🚨 Critical Constraints

- **NumPy**: Must use `<2.0.0` (due to darts dependency)
- **TA-Lib**: Requires system C library installation
- **Python**: Version 3.11.x required

## 📞 Support

1. Read `DEPENDENCY_MANAGEMENT.md`
2. Run `python validate_dependencies.py`
3. Check GitHub Actions logs
4. Review `TEST_INFRASTRUCTURE_FIX.md`

---

**Last Updated**: October 8, 2025  
**Version**: 1.0
