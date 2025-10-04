# Configure Codespaces for Automated Training

---
scope: deployment
audience: developers
priority: high
status: todo
dependencies: [CFT_001_TRAIN_complete-training-to-93-percent]
---

## Definition of Done
- [ ] Codespaces auto-approve common development actions
- [ ] Environment variables automatically loaded
- [ ] Python dependencies pre-installed and cached  
- [ ] Training commands run without manual approval prompts
- [ ] FRED API key and other secrets properly configured
- [ ] Automated model training can run unattended

## Context
GitHub Codespaces currently requires manual approval for many actions during model training, which interrupts the automated training loop. We need to configure Codespaces settings and devcontainer to allow automated training to run smoothly without constant user intervention.

## Requirements
### MVP (Must Have)
- Devcontainer pre-configuration for all dependencies
- Auto-approval for Python package installations
- Environment variables automatically loaded from .env
- Port forwarding for any web interfaces
- Sufficient compute resources for training

### Future Enhancements (Nice to Have)
- GPU acceleration for deep learning models
- Persistent storage for model artifacts
- Automated backup of training results

## Implementation Steps

### Step 1: Update .devcontainer/devcontainer.json
```json
{
  "name": "Congenial Fortnight Trading System",
  "image": "mcr.microsoft.com/devcontainers/python:3.11-bullseye",
  
  "features": {
    "ghcr.io/devcontainers/features/common-utils:2": {
      "installZsh": true,
      "configureZshAsDefaultShell": true,
      "installOhMyZsh": true
    },
    "ghcr.io/devcontainers/features/node:1": {
      "nodeGypDependencies": true,
      "version": "18"
    }
  },

  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.pylint", 
        "ms-toolsai.jupyter",
        "GitHub.copilot",
        "ms-vscode.vscode-json"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": true,
        "jupyter.askForKernelRestart": false,
        "extensions.autoUpdate": true
      }
    }
  },

  "postCreateCommand": "bash .devcontainer/setup.sh",
  "postStartCommand": "echo 'Codespace ready for training!'",

  "forwardPorts": [8000, 3000, 8080],
  "portsAttributes": {
    "8000": {
      "label": "Django Backend",
      "onAutoForward": "silent"
    },
    "3000": {
      "label": "React Frontend", 
      "onAutoForward": "silent"
    }
  },

  "mounts": [
    "source=cft-models,target=/workspaces/congenial-fortnight/models,type=volume",
    "source=cft-data,target=/workspaces/congenial-fortnight/data,type=volume"
  ],

  "remoteEnv": {
    "PYTHONPATH": "/workspaces/congenial-fortnight",
    "ENVIRONMENT": "codespaces",
    "SKIP_MANUAL_PROMPTS": "true",
    "AUTO_INSTALL_DEPS": "true"
  }
}
```

### Step 2: Create .devcontainer/setup.sh
```bash
#!/bin/bash
set -e

echo "🚀 Setting up Congenial Fortnight Trading System..."

# Update system packages
sudo apt-get update -y
sudo apt-get install -y build-essential curl wget git

# Install Python dependencies  
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Install additional ML packages that might be needed
pip install --no-cache-dir \
    optuna \
    scipy \
    plotly \
    dash \
    streamlit

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "🔑 Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "FRED_API_KEY loaded: ${FRED_API_KEY:0:8}..."
fi

# Create necessary directories
mkdir -p models logs output data/backups

# Set permissions
chmod +x scripts/*.py
find scripts/ -name "*.py" -exec chmod +x {} \;

# Pre-compile Python bytecode for faster startup
python -m compileall scripts/

# Test critical imports
echo "🧪 Testing critical imports..."
python -c "
import pandas as pd
import numpy as np
import sklearn
import lightgbm as lgb
import xgboost as xgb
import optuna
print('✅ All critical packages imported successfully')
"

# Verify data files
echo "📊 Checking data files..."
python -c "
import os
data_files = ['EURUSD_Daily.csv', 'XAUUSD_Daily.csv', 'DGS10.csv'] 
for file in data_files:
    path = f'data/{file}'
    if os.path.exists(path):
        print(f'✅ {file} found')
    else:
        print(f'⚠️  {file} missing')
"

echo "✅ Codespace setup complete!"
```

### Step 3: Configure GitHub Codespaces Settings
Create `.github/codespaces/settings.json`:
```json
{
  "machine": "standardLinux32gb",
  "timeout": 240,
  "suspendTimeout": 60,
  "prebuild": true,
  
  "secrets": [
    "FRED_API_KEY",
    "FINNHUB_API_KEY", 
    "ALPHA_VANTAGE_API_KEY"
  ],
  
  "permissions": {
    "contents": "write",
    "metadata": "read",
    "pull_requests": "write"
  }
}
```

### Step 4: Add Training Automation Scripts
Create `scripts/codespaces_train.py`:
```python
#!/usr/bin/env python3
"""
Codespaces-optimized training script with auto-approval
"""
import os
import sys
import logging
import subprocess
from pathlib import Path

def setup_unattended_mode():
    """Configure environment for unattended training"""
    # Disable interactive prompts
    os.environ['DEBIAN_FRONTEND'] = 'noninteractive'
    os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    os.environ['PIP_NO_CACHE_DIR'] = '0'
    
    # Auto-approve pip installs
    os.environ['PIP_YES'] = '1'
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
def run_training():
    """Run the complete training pipeline"""
    print("🚀 Starting automated training in Codespaces...")
    
    # Apply environment fix
    print("1️⃣ Applying environment loading fix...")
    # This would apply the environment loading patch
    
    # Run diagnostic  
    print("2️⃣ Running training diagnostic...")
    result = subprocess.run(['python', 'training_diagnostic.py'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Diagnostic failed:", result.stderr)
        return False
        
    # Start training
    print("3️⃣ Starting training loop...")
    cmd = [
        'python', '-m', 'scripts.automated_training',
        '--pairs', 'EURUSD', 'XAUUSD', 
        '--target', '0.93',
        '--max-iterations', '100'
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

if __name__ == "__main__":
    setup_unattended_mode() 
    success = run_training()
    sys.exit(0 if success else 1)
```

### Step 5: Configure Auto-Approval for Dependencies
Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    
  - package-ecosystem: "npm"
    directory: "/frontend"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

### Step 6: Add Codespaces Pre-build Configuration
Update `.github/workflows/codespaces-prebuild.yml`:
```yaml
name: Codespaces Prebuild

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  prebuild:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
        
    - name: Install dependencies
      run: |
        pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Validate setup
      run: |
        python -c "import lightgbm, xgboost, optuna; print('✅ Dependencies valid')"
```

## Success Criteria
- **Unattended Training**: Training can run for hours without manual intervention
- **Fast Startup**: Codespace ready in <2 minutes with pre-built environment
- **Persistent Storage**: Model artifacts survive Codespace rebuilds
- **Resource Efficiency**: Optimal machine size for training workload
- **Error Handling**: Graceful failure and recovery mechanisms

## Testing Requirements
- [ ] Test complete training pipeline in fresh Codespace
- [ ] Verify environment variables load automatically
- [ ] Confirm model artifacts persist across Codespace sessions
- [ ] Validate training can run overnight without interruption

## Codespaces Resource Optimization
```bash
# Monitor resource usage during training
htop

# Check memory usage
free -h

# Monitor disk space
df -h

# Optimize Python memory usage
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
```

## Dependencies Check
Before implementing:
- [ ] CFT_001 environment loading fix identified
- [ ] CFT_002_DOCS_migrate-to-instructions.md completed
- [ ] Verify Codespaces billing limits sufficient for training workload
- [ ] Confirm all required secrets available in repository settings
- [ ] Test base devcontainer configuration works