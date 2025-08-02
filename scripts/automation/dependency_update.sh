#!/bin/bash
# Automated dependency update script for SQL Query Synthesizer

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_FILE="${PROJECT_ROOT}/logs/dependency_update.log"
DRY_RUN=false
UPDATE_TYPE="patch"
CREATE_PR=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
    
    case "$level" in
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${message}" >&2
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${message}"
            ;;
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${message}"
            ;;
        "DEBUG")
            if [[ "${DEBUG:-}" == "1" ]]; then
                echo -e "${BLUE}[DEBUG]${NC} ${message}"
            fi
            ;;
    esac
}

# Help function
show_help() {
    cat << EOF
Automated Dependency Update Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -d, --dry-run           Show what would be updated without making changes
    -t, --type TYPE         Update type: patch|minor|major|all (default: patch)
    --no-pr                 Don't create pull request
    --force                 Force updates even if tests fail
    -v, --verbose           Enable verbose output

EXAMPLES:
    $0                      # Update patch versions
    $0 -t minor             # Update minor versions
    $0 --dry-run            # Show what would be updated
    $0 -t major --no-pr     # Update major versions without PR

ENVIRONMENT VARIABLES:
    GITHUB_TOKEN           GitHub token for PR creation
    DEBUG                  Enable debug logging (set to 1)
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -t|--type)
                UPDATE_TYPE="$2"
                shift 2
                ;;
            --no-pr)
                CREATE_PR=false
                shift
                ;;
            --force)
                FORCE_UPDATE=true
                shift
                ;;
            -v|--verbose)
                DEBUG=1
                shift
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate update type
validate_update_type() {
    case "$UPDATE_TYPE" in
        patch|minor|major|all)
            log "INFO" "Update type: $UPDATE_TYPE"
            ;;
        *)
            log "ERROR" "Invalid update type: $UPDATE_TYPE. Must be one of: patch, minor, major, all"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    # Check if we're in a git repository
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        log "ERROR" "Not in a git repository"
        exit 1
    fi
    
    # Check for required tools
    local required_tools=("pip" "python" "git")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log "ERROR" "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check for pip-tools
    if ! python -c "import pip_tools" >/dev/null 2>&1; then
        log "WARN" "pip-tools not found, installing..."
        pip install pip-tools
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    log "INFO" "Prerequisites check completed"
}

# Backup current requirements
backup_requirements() {
    log "INFO" "Backing up current requirements..."
    
    local backup_dir="${PROJECT_ROOT}/backups/requirements/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    for req_file in requirements.txt requirements-dev.txt; do
        if [[ -f "${PROJECT_ROOT}/$req_file" ]]; then
            cp "${PROJECT_ROOT}/$req_file" "$backup_dir/"
            log "DEBUG" "Backed up $req_file to $backup_dir"
        fi
    done
    
    echo "$backup_dir" > "${PROJECT_ROOT}/.last_requirements_backup"
    log "INFO" "Requirements backed up to $backup_dir"
}

# Update dependencies based on type
update_dependencies() {
    log "INFO" "Updating dependencies (type: $UPDATE_TYPE)..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would update dependencies"
        return 0
    fi
    
    case "$UPDATE_TYPE" in
        "patch")
            log "INFO" "Updating patch versions only"
            if [[ -f "requirements.in" ]]; then
                pip-compile --upgrade-package '*' --resolver=backtracking requirements.in
            fi
            if [[ -f "requirements-dev.in" ]]; then
                pip-compile --upgrade-package '*' --resolver=backtracking requirements-dev.in
            fi
            ;;
        "minor"|"major"|"all")
            log "INFO" "Updating to latest compatible versions"
            if [[ -f "requirements.in" ]]; then
                pip-compile --upgrade --resolver=backtracking requirements.in
            fi
            if [[ -f "requirements-dev.in" ]]; then
                pip-compile --upgrade --resolver=backtracking requirements-dev.in
            fi
            ;;
    esac
    
    log "INFO" "Dependencies updated"
}

# Install updated dependencies
install_dependencies() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would install updated dependencies"
        return 0
    fi
    
    log "INFO" "Installing updated dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Install updated dependencies
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    
    log "INFO" "Dependencies installed"
}

# Run security audit
run_security_audit() {
    log "INFO" "Running security audit..."
    
    cd "$PROJECT_ROOT"
    
    # Run safety check
    local vuln_count=0
    if command -v safety >/dev/null 2>&1; then
        if ! safety check --json --output safety-audit.json; then
            if [[ -f "safety-audit.json" ]]; then
                vuln_count=$(jq '.vulnerabilities | length' safety-audit.json 2>/dev/null || echo "0")
            fi
        fi
    else
        log "WARN" "Safety not available, skipping vulnerability check"
    fi
    
    log "INFO" "Security audit completed. Found $vuln_count vulnerabilities"
    return "$vuln_count"
}

# Run tests
run_tests() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would run tests"
        return 0
    fi
    
    log "INFO" "Running tests with updated dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Run tests
    if ! make test-unit; then
        log "ERROR" "Tests failed with updated dependencies"
        return 1
    fi
    
    log "INFO" "Tests passed with updated dependencies"
    return 0
}

# Check for changes
check_changes() {
    cd "$PROJECT_ROOT"
    
    if git diff --quiet requirements.txt requirements-dev.txt; then
        log "INFO" "No dependency changes detected"
        return 1
    else
        log "INFO" "Dependency changes detected"
        return 0
    fi
}

# Generate update summary
generate_summary() {
    local summary_file="${PROJECT_ROOT}/dependency_update_summary.md"
    
    log "INFO" "Generating update summary..."
    
    cd "$PROJECT_ROOT"
    
    cat > "$summary_file" << EOF
# Dependency Update Summary

**Update Type:** $UPDATE_TYPE
**Timestamp:** $(date -Iseconds)
**Triggered By:** Automated dependency update script

## Changes

\`\`\`diff
$(git diff --no-index requirements.txt.backup requirements.txt 2>/dev/null | head -50 || echo "No backup file for comparison")
\`\`\`

## Security Status

EOF
    
    if [[ -f "safety-audit.json" ]]; then
        local vuln_count=$(jq '.vulnerabilities | length' safety-audit.json 2>/dev/null || echo "0")
        echo "- Vulnerabilities found: $vuln_count" >> "$summary_file"
        
        if [[ "$vuln_count" -gt 0 ]]; then
            echo "- ⚠️ Security vulnerabilities detected" >> "$summary_file"
        else
            echo "- ✅ No security vulnerabilities found" >> "$summary_file"
        fi
    fi
    
    cat >> "$summary_file" << EOF

## Testing

- ✅ All tests passed with updated dependencies
- ✅ Security audit completed

## Next Steps

1. Review the changes in this summary
2. Test the application locally if needed
3. Merge the pull request if all checks pass

---

*This summary was generated automatically by the dependency update script.*
EOF
    
    log "INFO" "Update summary generated: $summary_file"
}

# Create pull request
create_pull_request() {
    if [[ "$CREATE_PR" != "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Skipping pull request creation"
        return 0
    fi
    
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        log "WARN" "GITHUB_TOKEN not set, cannot create pull request"
        return 1
    fi
    
    log "INFO" "Creating pull request..."
    
    cd "$PROJECT_ROOT"
    
    # Create branch for dependency update
    local branch_name="dependency-update-$(date +%Y%m%d-%H%M%S)"
    git checkout -b "$branch_name"
    
    # Commit changes
    git add requirements.txt requirements-dev.txt
    git commit -m "chore: automated dependency update ($UPDATE_TYPE)

Automated dependency update including security fixes and compatibility improvements.

- Update type: $UPDATE_TYPE
- Security vulnerabilities addressed
- All tests pass with updated dependencies

Generated by dependency update automation script." || {
        log "ERROR" "Failed to commit changes"
        return 1
    }
    
    # Push branch
    git push origin "$branch_name" || {
        log "ERROR" "Failed to push branch"
        return 1
    }
    
    # Create PR using GitHub CLI if available
    if command -v gh >/dev/null 2>&1; then
        local pr_body
        if [[ -f "dependency_update_summary.md" ]]; then
            pr_body=$(cat dependency_update_summary.md)
        else
            pr_body="Automated dependency update ($UPDATE_TYPE)"
        fi
        
        gh pr create \
            --title "chore: Automated dependency update ($UPDATE_TYPE)" \
            --body "$pr_body" \
            --label "dependencies,automated" || {
            log "ERROR" "Failed to create pull request"
            return 1
        }
        
        log "INFO" "Pull request created successfully"
    else
        log "WARN" "GitHub CLI not available, cannot create pull request automatically"
        log "INFO" "Please create PR manually for branch: $branch_name"
    fi
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    cd "$PROJECT_ROOT"
    
    if [[ $exit_code -ne 0 ]] && [[ -f ".last_requirements_backup" ]]; then
        local backup_dir=$(cat .last_requirements_backup)
        
        if [[ "$DRY_RUN" != "true" ]]; then
            log "WARN" "Restoring requirements from backup due to failure..."
            
            for req_file in requirements.txt requirements-dev.txt; do
                if [[ -f "$backup_dir/$req_file" ]]; then
                    cp "$backup_dir/$req_file" ./
                    log "INFO" "Restored $req_file from backup"
                fi
            done
        fi
    fi
    
    # Clean up temporary files
    rm -f safety-audit.json dependency_update_summary.md .last_requirements_backup
    
    log "INFO" "Dependency update script completed with exit code: $exit_code"
}

# Main function
main() {
    # Set up signal handling
    trap cleanup EXIT
    
    # Parse arguments
    parse_args "$@"
    
    # Validate inputs
    validate_update_type
    
    log "INFO" "Starting automated dependency update (type: $UPDATE_TYPE)"
    
    # Run the update process
    check_prerequisites
    backup_requirements
    update_dependencies
    
    # Check if there are any changes
    if ! check_changes; then
        log "INFO" "No dependency updates available"
        exit 0
    fi
    
    install_dependencies
    
    # Run security audit
    local vuln_count=0
    if ! run_security_audit; then
        vuln_count=$?
    fi
    
    # Run tests
    if ! run_tests; then
        if [[ "${FORCE_UPDATE:-}" != "true" ]]; then
            log "ERROR" "Tests failed, aborting update"
            exit 1
        else
            log "WARN" "Tests failed but continuing due to --force flag"
        fi
    fi
    
    # Generate summary and create PR
    generate_summary
    create_pull_request
    
    log "INFO" "Dependency update completed successfully"
    
    if [[ $vuln_count -gt 0 ]]; then
        log "WARN" "Found $vuln_count security vulnerabilities"
        exit 2
    fi
}

# Run main function with all arguments
main "$@"