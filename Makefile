# SQL Synthesizer - Development and Production Makefile

.PHONY: help install dev test lint format security build docker clean docs

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# HELP & SETUP
# =============================================================================

help: ## Show this help message
	@echo "SQL Synthesizer - Available Commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\033[36m%-20s\033[0m %s\n", "Command", "Description"} /^[a-zA-Z_-]+:.*?##/ { printf "\033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

install: ## Install dependencies for development
	@echo "Installing development dependencies..."
	pip install -e .[dev]
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "✅ Development environment ready!"

install-prod: ## Install production dependencies only
	@echo "Installing production dependencies..."
	pip install -e .
	@echo "✅ Production dependencies installed!"

# =============================================================================
# DEVELOPMENT
# =============================================================================

dev: ## Start development server
	@echo "Starting development server..."
	python -m sql_synthesizer.webapp --debug

quick: ## Quick development cycle (format, lint, test)
	@echo "Running quick development cycle..."
	ruff --fix sql_synthesizer/
	pytest tests/ --tb=short -x
	@echo "✅ Quick cycle complete!"

check: ## Run all checks (lint, type check, security)
	@echo "Running all quality checks..."
	ruff sql_synthesizer/
	mypy sql_synthesizer/
	bandit -r sql_synthesizer/ -ll
	@echo "✅ All checks passed!"

dev-docker: ## Start development environment with Docker Compose
	@echo "Starting development environment..."
	docker-compose up --build

dev-background: ## Start development environment in background
	@echo "Starting development environment in background..."
	docker-compose up -d --build

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all tests
	@echo "Running test suite..."
	pytest tests/ -v --cov=sql_synthesizer --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	pytest tests/ -v -m "not integration" --cov=sql_synthesizer

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	pytest tests/ -v -m "integration"

test-security: ## Run security tests
	@echo "Running security tests..."
	pytest tests/ -v -m "security"
	bandit -r sql_synthesizer/

test-performance: ## Run performance tests
	@echo "Running performance tests..."
	pytest tests/ -v --benchmark-only

test-coverage: ## Generate detailed coverage report
	@echo "Generating coverage report..."
	pytest tests/ --cov=sql_synthesizer --cov-report=html --cov-report=xml
	@echo "Coverage report generated in htmlcov/"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run all linting tools
	@echo "Running linting tools..."
	pylint sql_synthesizer/
	ruff sql_synthesizer/
	mypy sql_synthesizer/

format: ## Format code with black and isort
	@echo "Formatting code..."
	black sql_synthesizer/ tests/
	isort sql_synthesizer/ tests/

format-check: ## Check code formatting without making changes
	@echo "Checking code formatting..."
	black --check sql_synthesizer/ tests/
	isort --check-only sql_synthesizer/ tests/

pre-commit: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

# =============================================================================
# SECURITY
# =============================================================================

security: ## Run comprehensive security checks
	@echo "Running security analysis..."
	bandit -r sql_synthesizer/ -f json -o security-report.json
	bandit -r sql_synthesizer/
	safety check
	@echo "Security analysis complete!"

security-audit: ## Run dependency vulnerability audit
	@echo "Auditing dependencies for vulnerabilities..."
	safety check --json --output safety-report.json
	safety check

secrets-scan: ## Scan for secrets in codebase
	@echo "Scanning for secrets..."
	detect-secrets scan --all-files --baseline .secrets.baseline

# =============================================================================
# BUILD & PACKAGING
# =============================================================================

build: ## Build Python package
	@echo "Building package..."
	python -m build
	@echo "✅ Package built in dist/"

build-wheel: ## Build wheel package only
	@echo "Building wheel..."
	python -m build --wheel

build-sdist: ## Build source distribution only
	@echo "Building source distribution..."
	python -m build --sdist

clean-build: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	@echo "✅ Build artifacts cleaned!"

# =============================================================================
# DOCKER
# =============================================================================

docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t sql-synthesizer:latest .

docker-build-dev: ## Build Docker image for development
	@echo "Building development Docker image..."
	docker build -t sql-synthesizer:dev --target builder .

docker-test: ## Test Docker image
	@echo "Testing Docker image..."
	docker run --rm sql-synthesizer:latest python -c "import sql_synthesizer; print('✅ Container test passed!')"

docker-scan: ## Scan Docker image for vulnerabilities
	@echo "Scanning Docker image for vulnerabilities..."
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image sql-synthesizer:latest

docker-push: ## Push Docker image to registry
	@echo "Pushing Docker image..."
	docker push sql-synthesizer:latest

docker-clean: ## Clean Docker artifacts
	@echo "Cleaning Docker artifacts..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "✅ Docker artifacts cleaned!"

# =============================================================================
# ENVIRONMENT MANAGEMENT
# =============================================================================

up: ## Start all services with Docker Compose
	@echo "Starting all services..."
	docker-compose up -d

down: ## Stop all services
	@echo "Stopping all services..."
	docker-compose down

restart: ## Restart all services
	@echo "Restarting all services..."
	docker-compose restart

logs: ## Show logs from all services
	docker-compose logs -f

logs-app: ## Show application logs only
	docker-compose logs -f sql-synthesizer

status: ## Show status of all services
	docker-compose ps

# =============================================================================
# DATABASE
# =============================================================================

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	python -c "from sql_synthesizer.database import migrate; migrate()"

db-seed: ## Seed database with sample data
	@echo "Seeding database..."
	python scripts/seed_database.py

db-reset: ## Reset database (WARNING: destructive)
	@echo "⚠️  Resetting database..."
	docker-compose down -v postgres
	docker-compose up -d postgres
	sleep 5
	$(MAKE) db-migrate db-seed

# =============================================================================
# MONITORING
# =============================================================================

metrics: ## Open Prometheus metrics in browser
	@echo "Opening Prometheus metrics..."
	python -c "import webbrowser; webbrowser.open('http://localhost:9090')"

grafana: ## Open Grafana dashboard in browser
	@echo "Opening Grafana dashboard..."
	python -c "import webbrowser; webbrowser.open('http://localhost:3000')"

health: ## Check application health
	@echo "Checking application health..."
	curl -f http://localhost:5000/health || echo "❌ Application not healthy"

load-test: ## Run load tests
	@echo "Running load tests..."
	docker-compose --profile test run --rm load-test

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Generate documentation
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html/
	@echo "📚 Documentation generated in docs/_build/html/"

docs-serve: ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8000..."
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation artifacts
	@echo "Cleaning documentation..."
	rm -rf docs/_build/
	@echo "✅ Documentation cleaned!"

# =============================================================================
# CLEANUP
# =============================================================================

clean: ## Clean all build artifacts and caches
	@echo "Cleaning all artifacts..."
	$(MAKE) clean-build
	$(MAKE) clean-cache
	$(MAKE) clean-test
	@echo "✅ All artifacts cleaned!"

clean-cache: ## Clean Python cache files
	@echo "Cleaning Python cache..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "✅ Python cache cleaned!"

clean-test: ## Clean test artifacts
	@echo "Cleaning test artifacts..."
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf coverage.xml
	@echo "✅ Test artifacts cleaned!"

clean-logs: ## Clean log files
	@echo "Cleaning log files..."
	rm -rf logs/*.log
	@echo "✅ Log files cleaned!"

# =============================================================================
# RELEASE
# =============================================================================

release-check: ## Check if ready for release
	@echo "Checking release readiness..."
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) security
	$(MAKE) build
	@echo "✅ Ready for release!"

release-tag: ## Create release tag (requires VERSION variable)
	@if [ -z "$(VERSION)" ]; then echo "Error: VERSION variable required. Use: make release-tag VERSION=1.0.0"; exit 1; fi
	@echo "Creating release tag v$(VERSION)..."
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)
	@echo "✅ Release tag v$(VERSION) created!"

# =============================================================================
# CI/CD HELPERS
# =============================================================================

ci-install: ## Install dependencies for CI environment
	@echo "Installing CI dependencies..."
	pip install -e .[dev]
	@echo "✅ CI dependencies installed!"

ci-test: ## Run tests suitable for CI environment
	@echo "Running CI test suite..."
	pytest tests/ -v --cov=sql_synthesizer --cov-report=xml --cov-fail-under=80

ci-security: ## Run security checks for CI
	@echo "Running CI security checks..."
	bandit -r sql_synthesizer/ -f json -o bandit-ci-report.json
	safety check --json --output safety-ci-report.json

# =============================================================================
# UTILITIES
# =============================================================================

deps-update: ## Update dependencies
	@echo "Updating dependencies..."
	pip-compile requirements.in
	pip-compile requirements-dev.in
	@echo "✅ Dependencies updated!"

deps-check: ## Check for outdated dependencies
	@echo "Checking for outdated dependencies..."
	pip list --outdated

version: ## Show current version
	@python -c "import sql_synthesizer; print(f'SQL Synthesizer v{sql_synthesizer.__version__}')"

info: ## Show project information
	@echo "=== SQL Synthesizer Project Information ==="
	@echo "Repository: $(shell git remote get-url origin 2>/dev/null || echo 'Not a git repository')"
	@echo "Branch: $(shell git branch --show-current 2>/dev/null || echo 'Unknown')"
	@echo "Python: $(shell python --version)"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker Compose: $(shell docker-compose --version 2>/dev/null || echo 'Not installed')"
	@$(MAKE) version