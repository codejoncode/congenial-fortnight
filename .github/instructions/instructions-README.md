# .github/instructions Directory Setup and AI Agent Guidelines

---
scope: documentation
audience: ai_agents
priority: foundational
status: template
dependencies: []
---

## AI Agent Instructions for Using .github/instructions

### Purpose
This directory contains structured instructions optimized for AI agents to execute specific tasks within the Congenial Fortnight Trading System. Each instruction follows a standardized format for consistency and AI comprehension.

### Directory Structure
```
.github/
└── instructions/
    ├── README.md                          # This file - master index
    ├── 000_immediate/                     # Critical/urgent tasks
    │   └── CFT_000_IMMEDIATE_*.md
    ├── 001-100_foundation/                # Core setup, data, training
    │   ├── CFT_001_TRAIN_*.md
    │   ├── CFT_002_DATA_*.md
    │   └── CFT_003_MODEL_*.md
    ├── 101-200_optimization/              # Performance tuning
    │   └── CFT_1XX_OPTIMIZE_*.md
    ├── 201-300_deployment/                # Cloud deployment
    │   └── CFT_2XX_DEPLOY_*.md
    ├── 301-400_monitoring/                # Testing, validation
    │   └── CFT_3XX_TEST_*.md
    └── archive/                           # Completed tasks
        └── CFT_XXX_*_completed_YYYYMMDD.md
```

### Instruction File Format

Each instruction file follows this standardized template:

```markdown
# [Task Title]

---
scope: [training|data|deployment|testing|optimization]
audience: [ai_agents|developers|devops]
priority: [critical|high|medium|low]
status: [ready-to-execute|in-progress|blocked|complete]
dependencies: [CFT_XXX, CFT_YYY]
---

## AI Agent Context
Brief context for AI agents including why this task exists and what it accomplishes.

## Definition of Done
- [ ] Specific, measurable outcome 1
- [ ] Specific, measurable outcome 2
- [ ] Success criteria that can be automatically validated

## Context
Background information and current state.

## Implementation Steps
### Step 1: [Action] (estimated time)
```bash
# Exact commands to execute
command1
command2
```

Expected output:
- What the AI should see
- Success indicators
- Failure indicators

### Step 2: [Action] (estimated time)
[Continue with specific steps...]

## Success Criteria
- Measurable outcomes
- Performance targets
- Quality gates

## Monitoring During Execution
Watch for these indicators:
- ✅ Success patterns
- ❌ Failure patterns
- ⚠️ Warning patterns

## If Issues Occur
1. Issue: [Problem description]
   Solution: [Specific fix]
2. Issue: [Problem description]  
   Solution: [Alternative approach]

## AI Agent Notes
- Special considerations for AI execution
- Things to be careful about
- Expected behavior patterns
```

### Naming Convention
- Format: `CFT_{ID}_{CATEGORY}_{description}.md`
- ID: 3-digit sequential number (001, 002, 003...)
- Category: TRAIN, DATA, MODEL, DEPLOY, TEST, FIX, OPTIMIZE
- Description: Hyphenated lowercase description

### Categories and Ranges
- **000-099**: Immediate/critical tasks requiring urgent execution
- **001-100**: Foundation (core setup, data preparation, initial training)
- **101-200**: Optimization (performance tuning, accuracy improvement)
- **201-300**: Deployment (cloud deployment, automation, CI/CD)
- **301-400**: Monitoring (testing, validation, maintenance)

### AI Agent Execution Guidelines

#### Before Starting Any Task
1. **Check Dependencies**: Verify all listed CFT dependencies are complete
2. **Validate Environment**: Ensure required tools and access are available
3. **Backup State**: Create checkpoint before making changes
4. **Estimate Time**: Confirm you have sufficient time to complete the task

#### During Execution
1. **Follow Steps Exactly**: Execute implementation steps in order
2. **Monitor Output**: Watch for expected success/failure patterns
3. **Log Progress**: Document each step's completion status
4. **Handle Errors**: Use provided troubleshooting steps
5. **Validate Continuously**: Check success criteria after each major step

#### After Completion
1. **Verify Success Criteria**: Confirm all Definition of Done items are met
2. **Run Validation**: Execute any provided validation commands
3. **Update Status**: Mark instruction as complete with timestamp
4. **Archive**: Move completed instruction to archive/ directory
5. **Report Results**: Document outcomes and any deviations

### Special Instructions for AI Agents

#### Critical Tasks (CFT_000_*)
- Execute immediately when assigned
- Do not wait for human confirmation
- Monitor continuously during execution
- Report any blocking issues immediately

#### Training Tasks (CFT_0X*_TRAIN_*)
- Always backup existing models before retraining
- Monitor resource usage (CPU, memory, disk)
- Validate data quality before starting
- Log all training metrics and outcomes

#### Data Tasks (CFT_0X*_DATA_*)
- Verify data integrity before and after changes
- Create backups of original data
- Validate schema compliance
- Check for data consistency across timeframes

#### Deployment Tasks (CFT_2X*_DEPLOY_*)
- Test in staging environment first
- Verify all environment variables are set
- Validate service health after deployment
- Implement rollback plan if issues occur

### Error Handling for AI Agents

#### Common Error Patterns
1. **Permission Denied**: Check file permissions and user access
2. **Module Not Found**: Install missing dependencies
3. **File Not Found**: Verify file paths and existence
4. **API Errors**: Check API keys and network connectivity
5. **Memory Issues**: Monitor resource usage and optimize

#### Recovery Strategies
1. **Retry with Exponential Backoff**: For transient errors
2. **Alternative Approach**: Use backup methods provided in instructions
3. **Partial Completion**: Complete what's possible, document blockers
4. **Escalation**: Flag critical issues that require human intervention

### Monitoring and Validation

#### Success Indicators
```
✅ Expected log patterns
✅ File creation confirmations  
✅ Metric improvements
✅ Service health checks
```

#### Warning Indicators
```
⚠️ Resource usage spikes
⚠️ Slower than expected progress
⚠️ Non-critical errors
⚠️ Unusual output patterns
```

#### Failure Indicators
```
❌ Process crashes
❌ Data corruption
❌ Service failures
❌ Critical errors
```

### Integration with Development Tools

#### GitHub Copilot Optimization
- YAML front matter provides context for better suggestions
- Structured format enables pattern recognition
- Specific commands improve code completion accuracy

#### Automation Integration
- Instructions can trigger GitHub Actions workflows
- Status updates can be automated via commits
- Results can be logged to issue tracking systems

### Example Usage Workflow

1. **Receive Assignment**: AI agent is assigned CFT_004_TRAIN_clean-slate-retraining.md
2. **Pre-Execution Check**: Verify dependencies, environment, resources
3. **Execute Steps**: Follow Step 1 through Step 6 exactly as written
4. **Monitor Progress**: Watch for expected output patterns
5. **Handle Issues**: Use provided troubleshooting if errors occur
6. **Validate Success**: Confirm all Definition of Done criteria met
7. **Report Completion**: Update status and move to archive
8. **Clean Up**: Remove temporary files, restore clean state

This structured approach ensures consistent, reliable execution of complex tasks while providing clear guidance for both human developers and AI agents.

---

**Note for AI Agents**: Always read the entire instruction file before beginning execution. Pay special attention to the "AI Agent Context" and "AI Agent Notes" sections for task-specific guidance.