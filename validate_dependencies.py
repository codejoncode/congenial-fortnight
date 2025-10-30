#!/usr/bin/env python3
"""
Pre-Commit Dependency Validation

This script validates that all dependencies are properly installed and
all imports can be resolved before running tests or committing code.

Run this script before:
- Committing code
- Running CI/CD pipelines
- Deploying to production
"""

import sys
import importlib.util
import os
import subprocess
from typing import List, Tuple


def check_import(package_name: str) -> Tuple[bool, str]:
    """
    Check if a package can be imported.
    
    Args:
        package_name: Name of the package to import
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Handle packages with dashes in names
        import_name = package_name.replace('-', '_').lower()
        
        # Special cases
        if package_name == 'TA-Lib':
            import_name = 'talib'
        elif package_name == 'scikit-learn':
            import_name = 'sklearn'
        elif package_name == 'djangorestframework':
            import_name = 'rest_framework'
        elif package_name == 'djangorestframework-simplejwt':
            import_name = 'rest_framework_simplejwt'
        elif package_name == 'django-cors-headers':
            import_name = 'corsheaders'
        
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, f"Module '{import_name}' not found"
        
        # Try actual import
        __import__(import_name)
        return True, "OK"
    except Exception as e:
        return False, str(e)


def check_critical_dependencies() -> bool:
    """
    Check all critical dependencies required by the project.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    critical_packages = [
        # Core packages
        ('talib', 'TA-Lib', True),
        ('pandas', 'pandas', True),
        ('numpy', 'numpy', True),
        ('lightgbm', 'LightGBM', True),
        ('sklearn', 'scikit-learn', True),
        ('ta', 'ta', True),
        
        # Testing packages
        ('pytest', 'pytest', True),
        ('jsonschema', 'jsonschema', True),
        
        # Django packages
        ('django', 'Django', False),
        ('rest_framework', 'djangorestframework', False),
        
        # Optional packages
        ('xgboost', 'XGBoost', False),
        ('prophet', 'Prophet', False),
        ('shap', 'SHAP', False),
    ]
    
    print("=" * 70)
    print("DEPENDENCY VALIDATION")
    print("=" * 70)
    print()
    
    all_ok = True
    failed_critical = []
    failed_optional = []
    
    for import_name, display_name, is_critical in critical_packages:
        success, message = check_import(import_name)
        
        status = "✓" if success else "✗"
        criticality = "CRITICAL" if is_critical else "OPTIONAL"
        
        if success:
            print(f"{status} {display_name:30s} [{criticality:8s}] {message}")
        else:
            print(f"{status} {display_name:30s} [{criticality:8s}] {message}")
            if is_critical:
                failed_critical.append(display_name)
                all_ok = False
            else:
                failed_optional.append(display_name)
    
    print()
    print("=" * 70)
    
    if failed_critical:
        print(f"✗ CRITICAL FAILURES: {len(failed_critical)}")
        for pkg in failed_critical:
            print(f"  - {pkg}")
        print()
        print("These packages are required for the system to function.")
        print("Install them with: pip install -r requirements.txt")
        print()
        
        if 'TA-Lib' in failed_critical or 'talib' in failed_critical:
            print("Note: TA-Lib requires system-level installation:")
            print("  Ubuntu/Debian:")
            print("    sudo apt-get install -y build-essential wget")
            print("    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz")
            print("    tar -xzf ta-lib-0.4.0-src.tar.gz")
            print("    cd ta-lib/")
            print("    ./configure --prefix=/usr")
            print("    make && sudo make install")
            print("    pip install TA-Lib")
            print()
    
    if failed_optional:
        print(f"⚠  OPTIONAL FAILURES: {len(failed_optional)}")
        for pkg in failed_optional:
            print(f"  - {pkg}")
        print()
        print("These packages are optional. System will work without them.")
        print()
    
    if all_ok and not failed_optional:
        print("✓ ALL DEPENDENCIES OK")
        print()
        return True
    elif all_ok:
        print("✓ ALL CRITICAL DEPENDENCIES OK (some optional missing)")
        print()
        return True
    else:
        print("✗ DEPENDENCY CHECK FAILED")
        print()
        return False


def check_local_modules() -> bool:
    """
    Check that required local modules exist.
    
    Returns:
        True if all required modules exist, False otherwise
    """
    print("=" * 70)
    print("LOCAL MODULE VALIDATION")
    print("=" * 70)
    print()
    
    required_modules = [
        'fundamentals.py',
        'trading_system.py',
        'scripts/day_trading_signals.py',
        'scripts/pip_based_signal_system.py',
        'scripts/harmonic_patterns.py',
        'scripts/unified_signal_service.py',
    ]
    
    all_ok = True
    
    for module_path in required_modules:
        exists = os.path.exists(module_path)
        status = "✓" if exists else "✗"
        print(f"{status} {module_path}")
        if not exists:
            all_ok = False
    
    print()
    print("=" * 70)
    
    if all_ok:
        print("✓ ALL LOCAL MODULES FOUND")
    else:
        print("✗ SOME LOCAL MODULES MISSING")
    
    print()
    return all_ok


def check_requirements_files() -> bool:
    """
    Check that requirements files are in sync.
    
    Returns:
        True if requirements files are valid, False otherwise
    """
    print("=" * 70)
    print("REQUIREMENTS FILE VALIDATION")
    print("=" * 70)
    print()
    
    requirements_files = [
        'requirements.txt',
        'requirements-tests.txt'
    ]
    
    all_ok = True
    
    for req_file in requirements_files:
        if not os.path.exists(req_file):
            print(f"✗ {req_file} not found")
            all_ok = False
            continue
        
        try:
            with open(req_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                print(f"✓ {req_file} ({len(lines)} packages)")
        except Exception as e:
            print(f"✗ {req_file} - Error: {e}")
            all_ok = False
    
    print()
    print("=" * 70)
    
    if all_ok:
        print("✓ REQUIREMENTS FILES OK")
    else:
        print("✗ REQUIREMENTS FILES INVALID")
    
    print()
    return all_ok


def main():
    """Main validation function."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PRE-COMMIT DEPENDENCY VALIDATION" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    checks = [
        ("Requirements Files", check_requirements_files),
        ("Python Dependencies", check_critical_dependencies),
        ("Local Modules", check_local_modules),
    ]
    
    results = []
    
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    # Final summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")
        if not passed:
            all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print()
        print("You can proceed with:")
        print("  - Running tests: pytest -v")
        print("  - Committing code: git commit")
        print("  - Running CI/CD pipeline")
        print()
        return 0
    else:
        print("✗ VALIDATION FAILED")
        print()
        print("Please fix the issues above before:")
        print("  - Committing code")
        print("  - Running tests")
        print("  - Deploying to production")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
