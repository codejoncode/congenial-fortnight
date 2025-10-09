# Instruction Creation Guide

---
scope: documentation
audience: developers
priority: foundational
---

## Naming Convention for Instructions

### Format Structure
```
{project_id}_{sequential_id}_{category}_{task_description}.md
```

**Components:**
- `project_id`: **CFT** (Congenial-Fortnight-Trading)
- `sequential_id`: **001**, **002**, **003** (zero-padded, 3 digits)
- `category`: **TRAIN**, **DATA**, **MODEL**, **DEPLOY**, **TEST**, **FIX**
- `task_description`: Short hyphenated description

### Examples:
```
CFT_001_TRAIN_complete-training-setup.md
CFT_002_DATA_fundamental-schema-fix.md
CFT_003_MODEL_accuracy-target-93-percent.md
CFT_004_DEPLOY_cloud-run-automation.md
CFT_005_TEST_end-to-end-validation.md
```

## File Structure Template

### Header (Required)
```markdown
# [Task Title]

---
scope: [coding|data|deployment|testing]
audience: [developers|traders|devops]
priority: [critical|high|medium|low]  
status: [todo|in-progress|blocked|complete]
dependencies: [list of CFT IDs this depends on]
---

## Definition of Done
- [ ] Specific measurable outcome 1
- [ ] Specific measurable outcome 2  
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation updated

## Context
Brief description of why this task exists.

## Requirements
### MVP (Must Have)
- Critical requirement 1
- Critical requirement 2

### Future Enhancements (Nice to Have)  
- Enhancement 1
- Enhancement 2

## Implementation Steps
1. Step 1 with specific commands
2. Step 2 with expected outputs
3. Step 3 with validation

## Success Criteria
- Measurable outcome
- Performance target
- Quality gate

## Testing Requirements
- [ ] Unit tests for X
- [ ] Integration test for Y  
- [ ] End-to-end test for Z

## Dependencies Check
Before starting, verify these CFT tasks are complete:
- [ ] CFT_XXX_CATEGORY_task-name (if applicable)
```

## Placement Rules

### Directory Structure
```
.github/
├── instructions/
│   ├── 001-100_foundation/     # Setup, data, core training
│   ├── 101-200_optimization/   # Model tuning, accuracy improvement  
│   ├── 201-300_deployment/     # Cloud deployment, automation
│   ├── 301-400_monitoring/     # Testing, validation, maintenance
│   └── archive/                # Completed tasks
```

### Status Management
- **Active tasks**: Keep in appropriate numbered directory
- **Completed tasks**: Move to `archive/` with completion date
- **Blocked tasks**: Add `[BLOCKED]` prefix to filename

## GitHub Copilot Optimization

### YAML Front Matter Benefits
- `scope`: Helps Copilot suggest relevant code patterns
- `audience`: Influences suggestion complexity  
- `priority`: Affects suggestion urgency context

### Best Practices for AI
1. **Specific titles**: "Fix FRED API loading" vs "Fix data issues"
2. **Concrete DOD**: "Achieve 93% accuracy" vs "Improve accuracy"
3. **Command examples**: Include exact commands in implementation steps
4. **File limitations**: Max 300 lines per instruction file

## Automation Integration

### Auto-Status Updates
```bash
# Mark task as complete and archive
mv .github/instructions/001-100_foundation/CFT_001_TRAIN_setup.md \
   .github/instructions/archive/CFT_001_TRAIN_setup_completed_$(date +%Y%m%d).md
```

### Dependency Checking  
```bash
# Verify all dependencies are complete before starting
scripts/check_instruction_dependencies.py CFT_003_MODEL_accuracy-target
```

This guide ensures consistent, trackable, and AI-optimized instruction management for the entire project lifecycle.