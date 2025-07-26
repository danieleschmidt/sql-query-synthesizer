# Autonomous Backlog Management System

## Overview

This autonomous backlog management system implements the full WSJF (Weighted Shortest Job First) prioritization methodology with automated discovery, TDD security integration, and comprehensive metrics reporting.

## Key Features

### 🔄 WSJF Scoring & Prioritization
- Automatically scores backlog items using `(value + time_criticality + risk_reduction) / effort`
- Applies aging multipliers to prevent stale items from languishing
- Supports configurable Fibonacci scales (1-2-3-5-8-13)

### 🔍 Automated Discovery
- Scans codebase for TODO/FIXME comments
- Detects failing tests and converts to backlog items
- Identifies security vulnerabilities using detect-secrets
- Monitors GitHub issues and project boards

### 🛡️ TDD + Security Integration
- Implements RED-GREEN-REFACTOR cycle with security validation
- Comprehensive security checklist covering:
  - Input validation and sanitization
  - Authentication and authorization controls
  - Secrets management via environment variables
  - Safe logging practices
  - Software Composition Analysis (SCA) with OWASP Dependency-Check
  - Static Application Security Testing (SAST) with GitHub CodeQL

### 🔧 Merge Conflict Automation
- Git rerere for automatic conflict resolution
- Smart merge drivers for package files
- Auto-rebase GitHub Actions
- Prometheus metrics for conflict tracking

### 📊 DORA Metrics & Reporting
- Deployment Frequency tracking
- Lead Time calculation
- Change Failure Rate monitoring
- Mean Time to Recovery (MTTR) measurement
- Daily status reports with backlog health metrics

## Quick Start

### 1. Basic Usage

```bash
# Run discovery and planning only (safe mode)
python3 run_autonomous_cycle.py --dry-run

# Run full autonomous cycle (max 5 iterations)
python3 run_autonomous_cycle.py --max-iterations 5

# Generate status report only
python3 autonomous_backlog_manager.py --status-report
```

### 2. Security Checklist

```bash
# Run security validation
python3 tdd_security_checklist.py

# Fail build on high severity issues
python3 tdd_security_checklist.py --fail-on-high
```

### 3. DORA Metrics

```bash
# Generate comprehensive metrics report
python3 dora_metrics.py

# Specify custom output directory
python3 dora_metrics.py --output-dir /path/to/reports
```

## Configuration

### Backlog Configuration (backlog.yml)

```yaml
config:
  wsjf:
    effort_scale: [1, 2, 3, 5, 8, 13]
    value_scale: [1, 2, 3, 5, 8, 13]
    time_criticality_scale: [1, 2, 3, 5, 8, 13]
    risk_reduction_scale: [1, 2, 3, 5, 8, 13]
    aging_multiplier_max: 2.0
    aging_days_threshold: 30
  statuses: ["NEW", "REFINED", "READY", "DOING", "PR", "DONE", "BLOCKED"]
  risk_tiers: ["low", "medium", "high"]
```

### Git Configuration (Automatic)

The system automatically configures:
- `git config rerere.enabled true`
- `git config rerere.autoupdate true`
- Merge drivers for package files (package-lock.json, poetry.lock)
- Auto-rebase hooks

## Backlog Item Structure

```yaml
- id: unique-identifier
  title: "Human-readable title"
  type: "security|feature|bug-fix|tech-debt|performance|documentation"
  description: "Detailed description"
  acceptance_criteria:
    - "✅ Criterion 1"
    - "❌ Criterion 2 (pending)"
  effort: 5        # Fibonacci scale: 1-2-3-5-8-13
  value: 8         # Business value: 1-2-3-5-8-13
  time_criticality: 3  # Urgency: 1-2-3-5-8-13
  risk_reduction: 5    # Risk mitigation: 1-2-3-5-8-13
  status: "READY"      # NEW|REFINED|READY|DOING|PR|DONE|BLOCKED
  risk_tier: "medium"  # low|medium|high
  created_at: "2025-07-26"
  links:
    - "file_path:line_number"
    - "BACKLOG.md:123"
  wsjf_score: 3.2     # Calculated automatically
  aging_multiplier: 1.0
  completed_at: null   # Set when status becomes DONE
  blocked_reason: null # Set when status becomes BLOCKED
```

## Macro Execution Loop

The autonomous system follows this cycle:

1. **Sync Repository** - Check git status and CI health
2. **Discover Tasks** - Scan for new TODO/FIXME, failing tests, security issues
3. **Score & Sort** - Calculate WSJF scores with aging factors
4. **Select Next Item** - Choose highest-priority READY item
5. **Execute Micro Cycle** - Run TDD + Security validation
6. **Merge & Log** - Complete task and update metrics
7. **Update Metrics** - Generate DORA and status reports
8. **Repeat** - Continue until no actionable items remain

## Micro Cycle (TDD + Security)

For each selected task:

1. **RED Phase** - Run security checklist and existing tests (should fail for new features)
2. **GREEN Phase** - Implement minimal code to pass tests
3. **REFACTOR Phase** - Clean up and optimize code
4. **Validate** - Final security checks, tests, and linting
5. **Merge** - Only if all validation passes

## Metrics & Reporting

### Daily Status Reports

Generated in `docs/status/YYYYMMDD.json` and `docs/status/YYYYMMDD.md`:

```json
{
  "timestamp": "2025-07-26T12:00:00",
  "completed_ids": ["task-1", "task-2"],
  "coverage_delta": "+2.3%",
  "ci_summary": "stable",
  "backlog_size_by_status": {"READY": 5, "DOING": 1, "DONE": 15},
  "avg_cycle_time": 18.5,
  "dora": {
    "deploy_frequency": 0.73,
    "lead_time": 2.67,
    "change_failure_rate": 13.6,
    "mean_time_to_recovery": 0.0
  },
  "rerere_auto_resolved_total": 3,
  "merge_driver_hits_total": 7,
  "ci_failure_rate": 15.0,
  "pr_backoff_state": "inactive"
}
```

### Prometheus Metrics

Exported for monitoring:
- `rerere_auto_resolved_total` - Conflicts resolved by rerere
- `merge_driver_hits_total` - Smart merge driver usage
- CI failure rate and PR throttling state

## CI Integration

### GitHub Actions Enhancement

The system enhances your existing CI with:

```yaml
- name: Security Checklist
  run: python tdd_security_checklist.py --fail-on-high
- name: Auto-rebase
  uses: ./.github/workflows/auto-rebase.yml
```

### Adaptive PR Throttling

- If CI failure rate > 30% → limit agent PRs to 2/day
- If CI failure rate < 10% → restore normal 5 PRs/day limit

## Exit Conditions

The autonomous system terminates when:
- Backlog is completely empty (all items DONE or BLOCKED)
- No READY items available for execution
- Maximum iterations reached
- Repository is not in clean state

## Continuous Improvement

The system includes meta-tasks for process improvement:
- Weekly conflict metrics analysis
- DORA trend monitoring
- Backlog health assessments
- Process optimization recommendations

## Security & Safety

### Risk Management
- High-risk items require human approval
- Security checklist must pass before any merge
- All changes are reversible via feature flags
- Maximum 5 agent PRs/day unless emergency

### Audit Trail
- All actions logged with structured format
- Complete DORA metrics history
- Security event tracking
- Backlog change attribution

## Advanced Configuration

### Custom Discovery Patterns

Add custom TODO patterns in `autonomous_backlog_manager.py`:

```python
# Search pattern split to avoid false positive detection
pattern = '|'.join(['CUS' + 'TOM', 'FIX' + 'ME', 'XXX', 'HACK'])
```

### External Tool Integration

Configure external security tools:
- OWASP Dependency-Check with cached NVD database
- GitHub CodeQL for SAST scanning
- detect-secrets for credential scanning

### Metrics Customization

Extend DORA metrics in `dora_metrics.py`:
- Custom deployment detection patterns
- Additional lead time calculation methods
- Enhanced failure categorization

## Troubleshooting

### Common Issues

1. **"No ready items found"** - All items are DONE/BLOCKED, or none meet safety criteria
2. **"Repository not clean"** - Uncommitted changes prevent execution
3. **"High-risk item requires approval"** - Human intervention needed for sensitive changes
4. **Missing tools** - Install ripgrep, pytest, detect-secrets for full functionality

### Debug Mode

Enable verbose logging:

```bash
export PYTHONPATH=/root/repo
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from run_autonomous_cycle import AutonomousCycleRunner
runner = AutonomousCycleRunner()
runner.run_macro_execution_loop(max_iterations=1)
"
```

## Best Practices

1. **Start with --dry-run** to understand recommendations
2. **Review security reports** before committing changes
3. **Monitor DORA metrics** for process health
4. **Keep acceptance criteria specific** and testable
5. **Use proper risk tiers** for sensitive changes
6. **Review blocked items regularly** for resolution
7. **Maintain clean git history** for accurate metrics

## Support

For issues or feature requests:
1. Check existing backlog items for duplicates
2. Review security checklist for compliance
3. Examine DORA metrics for trends
4. Create new backlog item with proper WSJF scoring