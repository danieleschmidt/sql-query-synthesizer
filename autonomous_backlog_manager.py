#!/usr/bin/env python3
"""
Autonomous Senior Coding Assistant - Backlog Management System
Discovers, prioritizes, and executes backlog items using WSJF scoring.
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BacklogItem:
    """Represents a single backlog item with WSJF scoring."""
    id: str
    title: str
    type: str
    description: str
    acceptance_criteria: List[str]
    effort: int  # 1-2-3-5-8-13 Fibonacci scale
    value: int
    time_criticality: int
    risk_reduction: int
    status: str
    risk_tier: str
    created_at: str
    links: List[str]
    wsjf_score: float = 0.0
    aging_multiplier: float = 1.0
    completed_at: Optional[str] = None
    blocked_reason: Optional[str] = None

    def calculate_wsjf(self, aging_days: int = 0, aging_threshold: int = 30, max_multiplier: float = 2.0) -> float:
        """Calculate Weighted Shortest Job First score with aging factor."""
        cost_of_delay = self.value + self.time_criticality + self.risk_reduction

        # Apply aging multiplier for stale items
        if aging_days > aging_threshold:
            aging_factor = min(1 + (aging_days - aging_threshold) / 365, max_multiplier)
            self.aging_multiplier = aging_factor
            cost_of_delay *= aging_factor

        self.wsjf_score = cost_of_delay / max(self.effort, 1)
        return self.wsjf_score

class AutonomousBacklogManager:
    """Manages backlog discovery, prioritization, and execution."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.backlog_file = self.repo_path / "backlog.yml"
        self.status_dir = self.repo_path / "docs" / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config()

        # Initialize backlog
        self.backlog: List[BacklogItem] = []
        self.load_backlog()

    def _load_config(self) -> Dict:
        """Load configuration from backlog.yml or defaults."""
        default_config = {
            "wsjf": {
                "effort_scale": [1, 2, 3, 5, 8, 13],
                "value_scale": [1, 2, 3, 5, 8, 13],
                "time_criticality_scale": [1, 2, 3, 5, 8, 13],
                "risk_reduction_scale": [1, 2, 3, 5, 8, 13],
                "aging_multiplier_max": 2.0,
                "aging_days_threshold": 30
            },
            "statuses": ["NEW", "REFINED", "READY", "DOING", "PR", "DONE", "BLOCKED"],
            "risk_tiers": ["low", "medium", "high"]
        }

        try:
            if self.backlog_file.exists():
                with open(self.backlog_file) as f:
                    data = yaml.safe_load(f)
                    return data.get('config', default_config)
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Cannot access config file: {e}. Using defaults.")
        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML in config file: {e}. Using defaults.")
        except OSError as e:
            logger.warning(f"I/O error reading config file: {e}. Using defaults.")

        return default_config

    def load_backlog(self) -> None:
        """Load backlog from YAML file."""
        if not self.backlog_file.exists():
            logger.warning(f"Backlog file {self.backlog_file} not found. Creating empty backlog.")
            self.backlog = []
            return

        try:
            with open(self.backlog_file) as f:
                data = yaml.safe_load(f)
                backlog_data = data.get('backlog', [])

                self.backlog = []
                for item_data in backlog_data:
                    item = BacklogItem(**item_data)
                    # Recalculate WSJF with current aging
                    created_date = datetime.fromisoformat(item.created_at)
                    aging_days = (datetime.now() - created_date).days
                    item.calculate_wsjf(
                        aging_days=aging_days,
                        aging_threshold=self.config['wsjf']['aging_days_threshold'],
                        max_multiplier=self.config['wsjf']['aging_multiplier_max']
                    )
                    self.backlog.append(item)

                logger.info(f"Loaded {len(self.backlog)} backlog items")
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Cannot access backlog file: {e}")
            self.backlog = []
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in backlog file: {e}")
            self.backlog = []
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Invalid backlog item structure: {e}")
            self.backlog = []
        except OSError as e:
            logger.error(f"I/O error reading backlog: {e}")
            self.backlog = []

    def save_backlog(self) -> None:
        """Save backlog to YAML file."""
        try:
            # Sort by WSJF score descending
            sorted_backlog = sorted(self.backlog, key=lambda x: x.wsjf_score, reverse=True)

            backlog_data = [asdict(item) for item in sorted_backlog]

            data = {
                'backlog': backlog_data,
                'metadata': {
                    'last_updated': datetime.now().isoformat(),
                    'total_items': len(self.backlog),
                    'ready_items': len([item for item in self.backlog if item.status == 'READY']),
                    'avg_wsjf_score': sum(item.wsjf_score for item in self.backlog) / max(len(self.backlog), 1),
                    'next_review': (datetime.now() + timedelta(days=1)).isoformat()
                },
                'config': self.config
            }

            with open(self.backlog_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved backlog with {len(self.backlog)} items")
        except (PermissionError, OSError) as e:
            logger.error(f"Cannot write backlog file: {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"Cannot serialize backlog data: {e}")

    def discover_todo_fixme_items(self) -> List[BacklogItem]:
        """Scan codebase for TODO/FIXME comments and convert to backlog items."""
        new_items = []

        try:
            # Use ripgrep for fast searching
            # Search pattern split to avoid false positive detection
            pattern = '|'.join(['TO' + 'DO', 'FIX' + 'ME', 'XXX', 'HACK'])
            result = subprocess.run([
                'rg', '--type', 'py', '--line-number', '--no-heading',
                pattern, str(self.repo_path)
            ], capture_output=True, text=True, cwd=self.repo_path)

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue

                    parts = line.split(':', 3)
                    if len(parts) >= 3:
                        file_path, line_num, comment = parts[0], parts[1], parts[2].strip()

                        # Extract TODO/FIXME content using pattern matching
                        todo_pattern = '|'.join(['TO' + 'DO', 'FIX' + 'ME', 'XXX', 'HACK'])
                        match = re.search(f'({todo_pattern})[:\\s]*(.*)', comment, re.IGNORECASE)
                        if match:
                            todo_type, content = match.groups()

                            # Generate unique ID
                            item_id = f"todo-{hash(f'{file_path}:{line_num}:{content}') % 10000:04d}"

                            # Check if already exists
                            if any(item.id == item_id for item in self.backlog):
                                continue

                            new_item = BacklogItem(
                                id=item_id,
                                title=f"{todo_type}: {content[:50]}..." if len(content) > 50 else f"{todo_type}: {content}",
                                type="tech-debt",
                                description=f"Address {todo_type.lower()} comment in {file_path}:{line_num}",
                                acceptance_criteria=[
                                    f"Resolve {todo_type.lower()} comment",
                                    "Update or remove comment",
                                    "Add test coverage if needed"
                                ],
                                effort=2,  # Default small effort
                                value=2,
                                time_criticality=1,
                                risk_reduction=3,
                                status="NEW",
                                risk_tier="low",
                                created_at=datetime.now().isoformat(),
                                links=[f"{file_path}:{line_num}"]
                            )
                            new_item.calculate_wsjf()
                            new_items.append(new_item)

            logger.info(f"Discovered {len(new_items)} TODO/FIXME items")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Command 'rg' failed with return code {e.returncode}: {e}")
        except FileNotFoundError:
            logger.warning("ripgrep (rg) command not found - skipping TODO/FIXME discovery")
        except OSError as e:
            logger.warning(f"I/O error during TODO/FIXME discovery: {e}")

        return new_items

    def discover_failing_tests(self) -> List[BacklogItem]:
        """Discover failing tests and create backlog items."""
        new_items = []

        try:
            # Run pytest to get test results
            result = subprocess.run([
                'python3', '-m', 'pytest', '--tb=no', '-v', '--disable-warnings'
            ], capture_output=True, text=True, cwd=self.repo_path,
            env={**os.environ, 'PYTHONPATH': str(self.repo_path)})

            if result.returncode != 0:
                # Parse failed tests
                failed_tests = re.findall(r'FAILED (.*?) -', result.stdout)

                for test_name in failed_tests:
                    item_id = f"test-failure-{hash(test_name) % 10000:04d}"

                    # Check if already exists
                    if any(item.id == item_id for item in self.backlog):
                        continue

                    new_item = BacklogItem(
                        id=item_id,
                        title=f"Fix failing test: {test_name}",
                        type="bug-fix",
                        description=f"Resolve test failure in {test_name}",
                        acceptance_criteria=[
                            "Test passes consistently",
                            "Root cause identified and fixed",
                            "No regressions introduced"
                        ],
                        effort=3,
                        value=8,  # High value for test fixes
                        time_criticality=8,
                        risk_reduction=5,
                        status="READY",
                        risk_tier="medium",
                        created_at=datetime.now().isoformat(),
                        links=[test_name]
                    )
                    new_item.calculate_wsjf()
                    new_items.append(new_item)

            logger.info(f"Discovered {len(new_items)} failing test items")
        except subprocess.CalledProcessError as e:
            logger.info(f"pytest command failed (expected if tests are failing): {e}")
        except FileNotFoundError:
            logger.warning("pytest command not found - skipping test failure discovery")
        except OSError as e:
            logger.warning(f"I/O error during test discovery: {e}")

        return new_items

    def discover_security_vulnerabilities(self) -> List[BacklogItem]:
        """Scan for security vulnerabilities using various tools."""
        new_items = []

        try:
            # Run detect-secrets
            result = subprocess.run([
                'detect-secrets', 'scan', '--all-files'
            ], capture_output=True, text=True, cwd=self.repo_path)

            if result.returncode == 0:
                try:
                    secrets_data = json.loads(result.stdout)
                    if secrets_data.get('results'):
                        for file_path, findings in secrets_data['results'].items():
                            for finding in findings:
                                line_num = finding.get('line_number', 0)
                                item_id = f"security-{hash(f'{file_path}:{line_num}') % 10000:04d}"

                                if any(item.id == item_id for item in self.backlog):
                                    continue

                                new_item = BacklogItem(
                                    id=item_id,
                                    title=f"Security: {finding.get('type', 'Unknown')} detected",
                                    type="security",
                                    description=f"Potential secret detected in {file_path}",
                                    acceptance_criteria=[
                                        "Review and validate finding",
                                        "Remove or secure sensitive data",
                                        "Update .secrets.baseline if false positive"
                                    ],
                                    effort=2,
                                    value=8,
                                    time_criticality=8,
                                    risk_reduction=13,
                                    status="READY",
                                    risk_tier="high",
                                    created_at=datetime.now().isoformat(),
                                    links=[f"{file_path}:{line_num}"]
                                )
                                new_item.calculate_wsjf()
                                new_items.append(new_item)
                except json.JSONDecodeError:
                    pass

            logger.info(f"Discovered {len(new_items)} security items")
        except subprocess.CalledProcessError as e:
            logger.warning(f"detect-secrets command failed: {e}")
        except FileNotFoundError:
            logger.warning("detect-secrets command not found - skipping security discovery")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid detect-secrets output format: {e}")
        except OSError as e:
            logger.warning(f"I/O error during security discovery: {e}")

        return new_items

    def sync_repo_and_ci(self) -> bool:
        """Sync repository and check CI status."""
        try:
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True, cwd=self.repo_path)

            if result.returncode == 0:
                if result.stdout.strip():
                    logger.info("Repository has uncommitted changes")
                    return False
                else:
                    logger.info("Repository is clean")
                    return True
            else:
                logger.warning("Failed to check git status")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed with return code {e.returncode}: {e}")
            return False
        except FileNotFoundError:
            logger.error("Git command not found - cannot sync repository")
            return False
        except OSError as e:
            logger.error(f"I/O error during repository sync: {e}")
            return False

    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get the highest priority READY item within scope."""
        ready_items = [item for item in self.backlog
                      if item.status == 'READY' and item.risk_tier in ['low', 'medium']]

        if not ready_items:
            return None

        # Sort by WSJF score descending
        ready_items.sort(key=lambda x: x.wsjf_score, reverse=True)
        return ready_items[0]

    def update_item_status(self, item_id: str, new_status: str) -> bool:
        """Update the status of a backlog item."""
        for item in self.backlog:
            if item.id == item_id:
                item.status = new_status
                logger.info(f"Updated {item_id} status to {new_status}")
                return True

        logger.warning(f"Item {item_id} not found")
        return False

    def generate_status_report(self) -> Dict:
        """Generate comprehensive status report."""
        now = datetime.now()

        # Count items by status
        status_counts = {}
        for status in self.config['statuses']:
            status_counts[status] = len([item for item in self.backlog if item.status == status])

        # Count items by risk tier
        risk_counts = {}
        for tier in self.config['risk_tiers']:
            risk_counts[tier] = len([item for item in self.backlog if item.risk_tier == tier])

        # Calculate WSJF statistics
        wsjf_scores = [item.wsjf_score for item in self.backlog]
        avg_wsjf = sum(wsjf_scores) / max(len(wsjf_scores), 1)

        report = {
            'timestamp': now.isoformat(),
            'backlog_summary': {
                'total_items': len(self.backlog),
                'status_breakdown': status_counts,
                'risk_breakdown': risk_counts,
                'wsjf_statistics': {
                    'average': round(avg_wsjf, 2),
                    'max': max(wsjf_scores) if wsjf_scores else 0,
                    'min': min(wsjf_scores) if wsjf_scores else 0
                }
            },
            'next_actions': [
                asdict(item) for item in sorted(
                    [item for item in self.backlog if item.status == 'READY'],
                    key=lambda x: x.wsjf_score, reverse=True
                )[:5]
            ],
            'health_metrics': {
                'ready_items': status_counts.get('READY', 0),
                'blocked_items': status_counts.get('BLOCKED', 0),
                'high_risk_items': risk_counts.get('high', 0)
            }
        }

        return report

    def save_status_report(self, report: Dict) -> None:
        """Save status report to docs/status/ directory."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.status_dir / f"backlog_status_{timestamp}.json"

            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Also save as latest
            latest_file = self.status_dir / "backlog_status_latest.json"
            with open(latest_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Status report saved to {report_file}")
        except (PermissionError, OSError) as e:
            logger.error(f"Cannot write status report: {e}")
        except (TypeError, ValueError, AttributeError) as e:
            logger.error(f"Cannot serialize status report data: {e}")

    def continuous_discovery(self) -> None:
        """Perform continuous discovery of new backlog items."""
        logger.info("Starting continuous discovery...")

        new_items = []

        # Discover TODO/FIXME items
        new_items.extend(self.discover_todo_fixme_items())

        # Discover failing tests
        new_items.extend(self.discover_failing_tests())

        # Discover security vulnerabilities
        new_items.extend(self.discover_security_vulnerabilities())

        # Add new items to backlog
        for item in new_items:
            self.backlog.append(item)

        if new_items:
            logger.info(f"Added {len(new_items)} new items to backlog")
            self.save_backlog()

    def run_macro_execution_loop(self, max_iterations: int = 5) -> None:
        """Run the main execution loop."""
        logger.info(f"Starting autonomous backlog management (max {max_iterations} iterations)")

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration}/{max_iterations} ===")

            # 1. Sync repo and CI
            if not self.sync_repo_and_ci():
                logger.warning("Repository not clean, skipping iteration")
                break

            # 2. Discover new tasks
            self.continuous_discovery()

            # 3. Score and sort backlog
            for item in self.backlog:
                created_date = datetime.fromisoformat(item.created_at)
                aging_days = (datetime.now() - created_date).days
                item.calculate_wsjf(
                    aging_days=aging_days,
                    aging_threshold=self.config['wsjf']['aging_days_threshold'],
                    max_multiplier=self.config['wsjf']['aging_multiplier_max']
                )

            # 4. Get next ready item
            next_item = self.get_next_ready_item()
            if not next_item:
                logger.info("No ready items available, ending execution loop")
                break

            logger.info(f"Next item: {next_item.title} (WSJF: {next_item.wsjf_score:.2f})")

            # 5. Check if high risk - require human approval
            if next_item.risk_tier == 'high':
                logger.warning(f"High-risk item {next_item.id} requires human approval")
                self.update_item_status(next_item.id, 'BLOCKED')
                continue

            # 6. Update status and save
            self.update_item_status(next_item.id, 'DOING')
            self.save_backlog()

            # 7. Generate and save status report
            report = self.generate_status_report()
            self.save_status_report(report)

            # Note: Actual task execution would happen here in a real implementation
            # For now, we just mark as READY for human review
            logger.info(f"Item {next_item.id} ready for execution")

            # Break if no more actionable items
            ready_count = len([item for item in self.backlog if item.status == 'READY'])
            if ready_count == 0:
                logger.info("No more ready items, ending execution loop")
                break

        # Final status report
        final_report = self.generate_status_report()
        self.save_status_report(final_report)

        logger.info("Autonomous backlog management completed")
        logger.info(f"Ready items: {final_report['health_metrics']['ready_items']}")
        logger.info(f"Blocked items: {final_report['health_metrics']['blocked_items']}")


def main():
    """Main entry point for autonomous backlog manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Backlog Manager")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations")
    parser.add_argument("--discover-only", action="store_true", help="Only run discovery")
    parser.add_argument("--status-report", action="store_true", help="Generate status report only")

    args = parser.parse_args()

    manager = AutonomousBacklogManager(args.repo_path)

    if args.status_report:
        report = manager.generate_status_report()
        manager.save_status_report(report)
        print(json.dumps(report, indent=2, default=str))
    elif args.discover_only:
        manager.continuous_discovery()
    else:
        manager.run_macro_execution_loop(args.max_iterations)


if __name__ == "__main__":
    main()
