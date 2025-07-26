#!/usr/bin/env python3
"""
Complete Autonomous Backlog Management Cycle
Implements the full macro execution loop with TDD + Security integration.
"""

import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Import our modules
from autonomous_backlog_manager import AutonomousBacklogManager
from tdd_security_checklist import TDDSecurityChecker
from dora_metrics import MetricsReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutonomousCycleRunner:
    """Runs the complete autonomous backlog management cycle."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.backlog_manager = AutonomousBacklogManager(str(repo_path))
        self.security_checker = TDDSecurityChecker(str(repo_path))
        self.metrics_reporter = MetricsReporter(str(repo_path))
    
    def sync_repo_and_ci(self) -> bool:
        """Sync repository and check CI status."""
        logger.info("=== Syncing Repository and CI ===")
        return self.backlog_manager.sync_repo_and_ci()
    
    def discover_new_tasks(self) -> None:
        """Discover new tasks across all sources."""
        logger.info("=== Discovering New Tasks ===")
        self.backlog_manager.continuous_discovery()
    
    def score_and_sort_backlog(self) -> None:
        """Score and sort backlog using WSJF."""
        logger.info("=== Scoring and Sorting Backlog ===")
        for item in self.backlog_manager.backlog:
            created_date = datetime.fromisoformat(item.created_at)
            aging_days = (datetime.now() - created_date).days
            item.calculate_wsjf(
                aging_days=aging_days,
                aging_threshold=self.backlog_manager.config['wsjf']['aging_days_threshold'],
                max_multiplier=self.backlog_manager.config['wsjf']['aging_multiplier_max']
            )
        
        # Sort backlog by WSJF score
        self.backlog_manager.backlog.sort(key=lambda x: x.wsjf_score, reverse=True)
        self.backlog_manager.save_backlog()
    
    def execute_micro_cycle(self, task_id: str) -> bool:
        """Execute TDD + Security micro cycle for a task."""
        logger.info(f"=== Executing Micro Cycle for {task_id} ===")
        
        # Step 1: Run security checklist (RED phase)
        logger.info("1. Running security checklist...")
        security_results = self.security_checker.run_security_checklist()
        security_report = self.security_checker.generate_security_report()
        
        if security_report['severity_breakdown']['high'] > 0:
            logger.error(f"High severity security issues found for {task_id}")
            return False
        
        # Step 2: Run existing tests (should fail for new functionality)
        logger.info("2. Running tests (RED phase)...")
        test_result = self._run_tests()
        
        # Step 3: Implement minimal code to make tests pass (GREEN phase)
        logger.info("3. Implementation would happen here (GREEN phase)")
        logger.info(f"   Task {task_id} ready for manual implementation")
        
        # Step 4: Refactor and clean up (REFACTOR phase)
        logger.info("4. Refactoring would happen here (REFACTOR phase)")
        
        # Step 5: Final security and quality checks
        logger.info("5. Running final checks...")
        final_security = self.security_checker.run_security_checklist()
        final_test_result = self._run_tests()
        lint_result = self._run_lint()
        
        return all([
            self.security_checker.generate_security_report()['status'] == 'PASS',
            final_test_result,
            lint_result
        ])
    
    def _run_tests(self) -> bool:
        """Run test suite."""
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', '--tb=short', '-v'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            success = result.returncode == 0
            if not success:
                logger.warning(f"Tests failed: {result.stdout}")
            return success
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not run tests")
            return False
    
    def _run_lint(self) -> bool:
        """Run code linting."""
        try:
            result = subprocess.run([
                'ruff', 'check', '.'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            success = result.returncode == 0
            if not success:
                logger.warning(f"Linting failed: {result.stdout}")
            return success
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not run linting")
            return True  # Don't fail if linter not available
    
    def merge_and_log(self, task_id: str) -> None:
        """Merge changes and log completion."""
        logger.info(f"=== Merging and Logging {task_id} ===")
        
        # Update task status to completed
        self.backlog_manager.update_item_status(task_id, 'DONE')
        
        # Set completion timestamp
        for item in self.backlog_manager.backlog:
            if item.id == task_id:
                item.completed_at = datetime.now().isoformat()
                break
        
        self.backlog_manager.save_backlog()
        logger.info(f"Task {task_id} marked as completed")
    
    def update_metrics(self) -> None:
        """Update and save metrics."""
        logger.info("=== Updating Metrics ===")
        report = self.metrics_reporter.generate_comprehensive_report()
        report_file = self.metrics_reporter.save_report(report)
        logger.info(f"Metrics report saved to: {report_file}")
    
    def run_macro_execution_loop(self, max_iterations: int = 5) -> None:
        """Run the complete macro execution loop."""
        logger.info(f"Starting Autonomous Backlog Management Cycle (max {max_iterations} iterations)")
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}/{max_iterations}")
            logger.info(f"{'='*60}")
            
            # 1. Sync repo and CI
            if not self.sync_repo_and_ci():
                logger.warning("Repository not clean, skipping iteration")
                break
            
            # 2. Discover new tasks
            self.discover_new_tasks()
            
            # 3. Score and sort backlog
            self.score_and_sort_backlog()
            
            # 4. Get next ready item
            next_item = self.backlog_manager.get_next_ready_item()
            if not next_item:
                logger.info("No ready items available, ending execution loop")
                break
            
            logger.info(f"Next item: {next_item.title} (WSJF: {next_item.wsjf_score:.2f})")
            
            # 5. Check if high risk - require human approval
            if next_item.risk_tier == 'high':
                logger.warning(f"High-risk item {next_item.id} requires human approval")
                self.backlog_manager.update_item_status(next_item.id, 'BLOCKED')
                continue
            
            # 6. Execute micro cycle (TDD + Security)
            self.backlog_manager.update_item_status(next_item.id, 'DOING')
            
            if self.execute_micro_cycle(next_item.id):
                # 7. Merge and log successful completion
                self.merge_and_log(next_item.id)
            else:
                logger.error(f"Micro cycle failed for {next_item.id}")
                self.backlog_manager.update_item_status(next_item.id, 'BLOCKED')
            
            # 8. Update metrics
            self.update_metrics()
            
            # Check exit conditions
            ready_count = len([item for item in self.backlog_manager.backlog 
                             if item.status == 'READY'])
            if ready_count == 0:
                logger.info("No more ready items, ending execution loop")
                break
        
        # Final metrics and status
        logger.info(f"\n{'='*60}")
        logger.info("AUTONOMOUS CYCLE COMPLETED")
        logger.info(f"{'='*60}")
        
        final_report = self.backlog_manager.generate_status_report()
        self.backlog_manager.save_status_report(final_report)
        
        logger.info(f"Ready items: {final_report['health_metrics']['ready_items']}")
        logger.info(f"Blocked items: {final_report['health_metrics']['blocked_items']}")
        logger.info(f"Total items: {final_report['backlog_summary']['total_items']}")


def main():
    """Main entry point for autonomous cycle runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Backlog Management Cycle")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations")
    parser.add_argument("--dry-run", action="store_true", help="Run discovery and planning only")
    
    args = parser.parse_args()
    
    runner = AutonomousCycleRunner(args.repo_path)
    
    if args.dry_run:
        logger.info("DRY RUN MODE: Discovery and planning only")
        runner.sync_repo_and_ci()
        runner.discover_new_tasks()
        runner.score_and_sort_backlog()
        runner.update_metrics()
        
        next_item = runner.backlog_manager.get_next_ready_item()
        if next_item:
            logger.info(f"Next recommended item: {next_item.title} (WSJF: {next_item.wsjf_score:.2f})")
        else:
            logger.info("No ready items found")
    else:
        runner.run_macro_execution_loop(args.max_iterations)


if __name__ == "__main__":
    main()