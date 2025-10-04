# Migrate Documentation to .github/instructions Structure

---
scope: documentation  
audience: developers
priority: medium
status: todo
dependencies: [CFT_001_TRAIN_complete-training-to-93-percent]
---

## Definition of Done
- [ ] All root-level .md files converted to instruction format
- [ ] Files organized in .github/instructions/ directories  
- [ ] Cross-references updated with new CFT IDs
- [ ] README.md updated to point to instructions
- [ ] Archive outdated documentation
- [ ] Create master instruction index

## Context
The repository has 25+ markdown documentation files in the root directory. These need to be converted to the structured instruction format and organized in .github/instructions/ for better GitHub Copilot integration and maintainability.

## Current Documentation Files to Convert

### Training & Model Files
- `TRAINING_READINESS_SUMMARY.md` → `CFT_002_TRAIN_readiness-summary.md`
- `ROBUST_TRAINING_PLAN.md` → `CFT_003_TRAIN_robust-training-plan.md` 
- `Holloway_Algorithm_Implementation.md` → `CFT_004_MODEL_holloway-algorithm.md`
- `ENHANCED_REGULARIZATION_GUIDE.md` → `CFT_005_MODEL_regularization-guide.md`

### Data & Fundamentals
- `DATA_FUNDAMENTALS_MISSING.md` → `CFT_006_DATA_fundamentals-integration.md`
- `FUNDAMENTALS.md` → `CFT_007_DATA_fundamentals-sources.md`
- `Where_To_Get_Price_data.md` → `CFT_008_DATA_price-data-sources.md`

### Deployment & Infrastructure  
- `CLOUD_DEPLOYMENT_GUIDE.md` → `CFT_009_DEPLOY_cloud-deployment.md`
- `GOOGLE_CLOUD_DEPLOYMENT_GUIDE.md` → `CFT_010_DEPLOY_gcp-specific.md`
- `DOCKERFILE_FIX_COMPLETE.md` → `CFT_011_DEPLOY_docker-fixes.md`
- `BUILD_TIMEOUT_FIX.md` → `CFT_012_DEPLOY_build-timeout-fix.md`

### Frontend & API
- `FRONTEND_IMPLEMENTATION_COMPLETE.md` → `CFT_013_FRONTEND_implementation.md`
- `API_REFERENCE.md` → `CFT_014_API_reference-guide.md`

### System Updates & Maintenance
- `SYSTEM_UPDATES_DOCUMENTATION.md` → `CFT_015_MAINTAIN_system-updates.md`
- `CHANGELOG.md` → `CFT_016_MAINTAIN_changelog.md`

### Process & Planning
- `ENHANCEMENT_CHECKLIST.md` → `CFT_017_PLAN_enhancement-checklist.md`
- `COMPLETE-IMPLEMENTATION-GUIDE.md` → `CFT_018_PLAN_implementation-guide.md`
- `Lean_Six_Sigma_Roadmap.md` → `CFT_019_PLAN_lean-six-sigma.md`
- `FINALIZING_PROJECT_NEXT.md` → `CFT_020_PLAN_project-finalization.md`
- `NEXT_STEPS_TRADING_SYSTEM_RESCUE.md` → `CFT_021_PLAN_system-rescue.md`

### Archive (Outdated/Completed)
- `TRADING_SYSTEM_README.md` (superseded by README.md)
- `NOTIFICATION_SETUP.txt` (convert to proper .md)

## Implementation Steps

### Step 1: Create Directory Structure
```bash
mkdir -p .github/instructions/001-100_foundation
mkdir -p .github/instructions/101-200_optimization  
mkdir -p .github/instructions/201-300_deployment
mkdir -p .github/instructions/301-400_monitoring
mkdir -p .github/instructions/archive
```

### Step 2: Convert Each File
For each file, apply this conversion template:

```bash
# Example conversion script
convert_doc() {
    local original_file=$1
    local new_name=$2
    local category=$3
    
    echo "# $(basename ${original_file%.md} | tr '_' ' ')" > $new_name
    echo "" >> $new_name
    echo "---" >> $new_name
    echo "scope: $category" >> $new_name
    echo "audience: developers" >> $new_name
    echo "priority: medium" >> $new_name
    echo "status: migrated" >> $new_name
    echo "dependencies: []" >> $new_name
    echo "---" >> $new_name
    echo "" >> $new_name
    echo "## Migrated Content" >> $new_name
    echo "*This content was migrated from $original_file*" >> $new_name
    echo "" >> $new_name
    tail -n +2 $original_file >> $new_name
}
```

### Step 3: Directory Placement Rules
```
001-100_foundation/    # CFT_001-CFT_100
├── CFT_002_TRAIN_readiness-summary.md
├── CFT_003_TRAIN_robust-training-plan.md
├── CFT_004_MODEL_holloway-algorithm.md
├── CFT_005_MODEL_regularization-guide.md
├── CFT_006_DATA_fundamentals-integration.md
├── CFT_007_DATA_fundamentals-sources.md
└── CFT_008_DATA_price-data-sources.md

101-200_optimization/  # CFT_101-CFT_200
├── CFT_015_MAINTAIN_system-updates.md
└── CFT_016_MAINTAIN_changelog.md

201-300_deployment/    # CFT_201-CFT_300
├── CFT_009_DEPLOY_cloud-deployment.md
├── CFT_010_DEPLOY_gcp-specific.md
├── CFT_011_DEPLOY_docker-fixes.md
└── CFT_012_DEPLOY_build-timeout-fix.md

301-400_monitoring/    # CFT_301-CFT_400
├── CFT_013_FRONTEND_implementation.md
└── CFT_014_API_reference-guide.md
```

### Step 4: Create Master Index
```bash
# Create .github/instructions/README.md
cat > .github/instructions/README.md << 'EOF'
# Congenial Fortnight Trading System Instructions

## Foundation (001-100)
- [CFT_001](001-100_foundation/CFT_001_TRAIN_complete-training-to-93-percent.md) - Complete Training to 93%
- [CFT_002](001-100_foundation/CFT_002_TRAIN_readiness-summary.md) - Training Readiness Summary  
- [CFT_003](001-100_foundation/CFT_003_TRAIN_robust-training-plan.md) - Robust Training Plan
...

## Usage
Each instruction follows the standard template with:
- YAML front matter for GitHub Copilot optimization
- Definition of Done checklist
- Step-by-step implementation
- Testing requirements
- Dependencies tracking

## Status Legend
- 🔴 TODO - Not started
- 🟡 IN-PROGRESS - Currently working  
- 🟢 COMPLETE - Finished and validated
- ⚫ BLOCKED - Waiting on dependencies
EOF
```

### Step 5: Update Cross-References
```bash
# Find and update all internal documentation links
find . -name "*.md" -exec grep -l "\[.*\](.*\.md)" {} \; | \
while read file; do
    # Update links to point to new instruction paths
    sed -i 's|\[.*\](TRAINING_READINESS_SUMMARY.md)|[CFT_002](.github/instructions/001-100_foundation/CFT_002_TRAIN_readiness-summary.md)|g' $file
    # ... repeat for all migrated files
done
```

### Step 6: Update Root README.md
```bash
# Add instruction directory reference to main README
cat >> README.md << 'EOF'

## 📋 Project Instructions

Detailed implementation instructions are organized in [.github/instructions/](.github/instructions/) using the CFT (Congenial-Fortnight-Trading) system:

- **Foundation (001-100)**: Core training, data, and model setup
- **Optimization (101-200)**: Performance tuning and enhancement  
- **Deployment (201-300)**: Cloud deployment and automation
- **Monitoring (301-400)**: Testing, validation, and maintenance

See [Instructions Index](.github/instructions/README.md) for complete list.
EOF
```

## Success Criteria
- **Organization**: All docs moved to proper .github/instructions/ structure
- **Consistency**: All files follow CFT naming and template format
- **Accessibility**: Master index provides clear navigation
- **Integration**: GitHub Copilot can effectively use structured instructions
- **Maintenance**: Old root-level docs archived, not deleted

## Testing Requirements
- [ ] Verify all internal links work after migration
- [ ] Test GitHub Copilot suggestions improve with structured format
- [ ] Validate YAML front matter is properly formatted
- [ ] Confirm directory structure follows naming convention

## Automation Scripts
```bash
# Create migration automation script
cat > scripts/migrate_docs.py << 'EOF'
#!/usr/bin/env python3
"""
Automated documentation migration to .github/instructions structure
"""
import os, shutil, re
from pathlib import Path

def migrate_doc(source_file, cft_id, category, target_dir):
    """Convert and move a document to instruction format"""
    # Implementation here
    pass

if __name__ == "__main__":
    migrate_all_docs()
EOF
```

## Dependencies Check
Before starting migration:
- [ ] CFT_001 (Training to 93%) should be complete unless percentage received is thought to be good enough
- [ ] Verify no active PRs modifying root-level .md files  
- [ ] Backup current documentation state
- [ ] Ensure team awareness of new structure
- [ ] ensure no more .md's exist outside of the instructions folder here use find . -name "*.md" -type f to check 