#!/bin/bash
#
# Pre-commit hook to validate dependencies
# 
# To install this hook:
#   cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#

set -e

echo ""
echo "Running pre-commit validation..."
echo ""

# Run dependency validation
python validate_dependencies.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Pre-commit validation passed"
    echo ""
    exit 0
else
    echo ""
    echo "✗ Pre-commit validation failed"
    echo ""
    echo "Fix the issues above or use 'git commit --no-verify' to skip validation"
    echo ""
    exit 1
fi
