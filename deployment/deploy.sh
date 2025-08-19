#!/bin/bash

# Production Deployment Script for SQL Query Synthesizer
# This script implements autonomous deployment with comprehensive checks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_ENV="${1:-production}"
IMAGE_TAG="${2:-latest}"
DRY_RUN="${DRY_RUN:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Deployment functions
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command '$cmd' not found"
            exit 1
        fi
    done
    
    # Check environment file
    if [[ ! -f "$SCRIPT_DIR/production.env" ]]; then
        error "Production environment file not found at $SCRIPT_DIR/production.env"
        error "Copy production.env.example and configure it"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

run_quality_gates() {
    log "Running autonomous quality gates..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment if it exists
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi
    
    # Run quality gates
    if python3 -c "
import asyncio
from sql_synthesizer.autonomous_sdlc import AutonomousQualityGateEngine

async def main():
    engine = AutonomousQualityGateEngine()
    results = await engine.execute_all_gates()
    
    print(f'Overall Score: {results[\"overall_score\"]:.3f}')
    print(f'All Passed: {results[\"overall_passed\"]}')
    
    if not results['overall_passed']:
        print('Quality gates failed!')
        exit(1)
    
    print('Quality gates passed!')

asyncio.run(main())
" 2>/dev/null; then
        success "Quality gates passed"
    else
        if [[ "$DRY_RUN" == "true" ]]; then
            warn "Quality gates would fail in actual deployment"
        else
            error "Quality gates failed - deployment aborted"
            exit 1
        fi
    fi
}

backup_current_deployment() {
    log "Creating backup of current deployment..."
    
    local backup_dir="$SCRIPT_DIR/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup database if running
    if docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" ps postgres | grep -q "Up"; then
        log "Backing up database..."
        docker-compose -f "$SCRIPT_DIR/docker-compose.production.yml" exec -T postgres \
            pg_dump -U sql_user sql_synthesizer > "$backup_dir/database.sql" || warn "Database backup failed"
    fi
    
    # Backup configuration
    cp -r "$SCRIPT_DIR"/*.yml "$backup_dir/" 2>/dev/null || true
    cp "$SCRIPT_DIR/production.env" "$backup_dir/" 2>/dev/null || true
    
    success "Backup created at $backup_dir"
}

build_and_test_image() {
    log "Building and testing application image..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    docker build -f Dockerfile.production -t "sql-synthesizer:$IMAGE_TAG" .
    
    # Run basic health check
    log "Testing image health..."
    local container_id
    container_id=$(docker run -d --rm \
        -e DATABASE_URL="sqlite:///tmp/test.db" \
        -e OPENAI_API_KEY="test" \
        "sql-synthesizer:$IMAGE_TAG")
    
    # Wait for startup
    sleep 10
    
    # Check health endpoint
    if docker exec "$container_id" curl -f http://localhost:5000/health &>/dev/null; then
        success "Image health check passed"
    else
        error "Image health check failed"
        docker logs "$container_id"
        docker stop "$container_id"
        exit 1
    fi
    
    docker stop "$container_id"
}

deploy_blue_green() {
    log "Starting blue-green deployment..."
    
    cd "$SCRIPT_DIR"
    
    # Load environment variables
    set -a
    source production.env
    set +a
    export IMAGE_TAG
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Would deploy with docker-compose"
        docker-compose -f docker-compose.production.yml config
        return
    fi
    
    # Deploy new version
    log "Deploying services..."
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if check_service_health; then
            success "All services are healthy"
            break
        fi
        
        attempt=$((attempt + 1))
        log "Health check attempt $attempt/$max_attempts..."
        sleep 10
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        error "Services failed to become healthy within timeout"
        rollback_deployment
        exit 1
    fi
}

check_service_health() {
    local healthy=true
    
    # Check main application
    if ! curl -sf http://localhost:5000/health &>/dev/null; then
        healthy=false
    fi
    
    # Check Redis
    if ! docker-compose -f docker-compose.production.yml exec -T redis redis-cli ping | grep -q PONG; then
        healthy=false
    fi
    
    # Check PostgreSQL
    if ! docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U sql_user -d sql_synthesizer &>/dev/null; then
        healthy=false
    fi
    
    $healthy
}

run_smoke_tests() {
    log "Running post-deployment smoke tests..."
    
    # Test basic API functionality
    local api_response
    api_response=$(curl -s -X POST http://localhost:5000/api/query \
        -H "Content-Type: application/json" \
        -d '{"question": "SELECT 1 as test"}' \
        -H "X-API-Key: ${API_KEY:-test}")
    
    if echo "$api_response" | jq -e '.sql' &>/dev/null; then
        success "API smoke test passed"
    else
        error "API smoke test failed"
        echo "Response: $api_response"
        return 1
    fi
    
    # Test metrics endpoint
    if curl -sf http://localhost:5000/metrics &>/dev/null; then
        success "Metrics endpoint test passed"
    else
        error "Metrics endpoint test failed"
        return 1
    fi
    
    # Test Grafana
    if curl -sf http://localhost:3000/api/health &>/dev/null; then
        success "Grafana test passed"
    else
        warn "Grafana test failed (may still be starting)"
    fi
}

rollback_deployment() {
    error "Rolling back deployment..."
    
    # Get latest backup
    local latest_backup
    latest_backup=$(find "$SCRIPT_DIR/backups" -type d -name "*_*" | sort | tail -1)
    
    if [[ -n "$latest_backup" ]]; then
        log "Restoring from backup: $latest_backup"
        
        # Restore database if backup exists
        if [[ -f "$latest_backup/database.sql" ]]; then
            docker-compose -f docker-compose.production.yml exec -T postgres \
                psql -U sql_user -d sql_synthesizer < "$latest_backup/database.sql"
        fi
        
        # Restart services
        docker-compose -f docker-compose.production.yml restart
        
        warn "Rollback completed"
    else
        error "No backup found for rollback"
    fi
}

cleanup_old_images() {
    log "Cleaning up old Docker images..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old sql-synthesizer images (keep last 3)
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}" | \
        grep "sql-synthesizer:" | \
        tail -n +4 | \
        awk '{print $2}' | \
        xargs -r docker rmi
    
    success "Image cleanup completed"
}

generate_deployment_report() {
    log "Generating deployment report..."
    
    local report_file="$SCRIPT_DIR/reports/deployment_$(date +%Y%m%d_%H%M%S).json"
    mkdir -p "$(dirname "$report_file")"
    
    cat > "$report_file" << EOF
{
  "deployment_id": "deploy_$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "environment": "$DEPLOY_ENV",
  "image_tag": "$IMAGE_TAG",
  "services": {
    "sql-synthesizer": "$(docker-compose -f docker-compose.production.yml ps -q sql-synthesizer)",
    "postgres": "$(docker-compose -f docker-compose.production.yml ps -q postgres)",
    "redis": "$(docker-compose -f docker-compose.production.yml ps -q redis)",
    "nginx": "$(docker-compose -f docker-compose.production.yml ps -q nginx)"
  },
  "health_status": "$(check_service_health && echo "healthy" || echo "unhealthy")",
  "deployed_by": "$(whoami)@$(hostname)"
}
EOF
    
    success "Deployment report saved to $report_file"
}

# Main deployment flow
main() {
    log "Starting autonomous deployment for environment: $DEPLOY_ENV"
    log "Image tag: $IMAGE_TAG"
    log "Dry run: $DRY_RUN"
    
    # Pre-deployment checks
    check_prerequisites
    run_quality_gates
    
    if [[ "$DRY_RUN" != "true" ]]; then
        backup_current_deployment
    fi
    
    # Build and deploy
    build_and_test_image
    deploy_blue_green
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Post-deployment validation
        if run_smoke_tests; then
            success "Deployment completed successfully"
            cleanup_old_images
            generate_deployment_report
        else
            error "Smoke tests failed"
            rollback_deployment
            exit 1
        fi
    else
        success "Dry run completed successfully"
    fi
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [environment] [image_tag]

Arguments:
  environment   Deployment environment (default: production)
  image_tag     Docker image tag (default: latest)

Environment Variables:
  DRY_RUN      Set to 'true' for dry run (default: false)

Examples:
  $0                           # Deploy to production with latest tag
  $0 staging v1.2.3           # Deploy to staging with specific tag
  DRY_RUN=true $0             # Dry run deployment

EOF
}

# Handle script arguments
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac