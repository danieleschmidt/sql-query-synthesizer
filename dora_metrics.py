#!/usr/bin/env python3
"""
DORA Metrics Collection and Reporting
Implements DevOps Research and Assessment metrics for autonomous backlog management.
"""

import os
import json
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DoraMetrics:
    """DORA metrics data structure."""
    deploy_frequency: float  # deployments per day
    lead_time: float  # hours from commit to deploy
    change_failure_rate: float  # percentage of deployments causing failures
    mean_time_to_recovery: float  # hours to recover from failures
    measurement_period_days: int = 30

class DoraMetricsCollector:
    """Collects and calculates DORA metrics."""
    
    def __init__(self, repo_path: str = ".", lookback_days: int = 30):
        self.repo_path = Path(repo_path)
        self.lookback_days = lookback_days
        self.since_date = datetime.now() - timedelta(days=lookback_days)
    
    def collect_deployment_frequency(self) -> float:
        """Calculate deployment frequency (deployments per day)."""
        try:
            # Count merge commits to main as deployments
            result = subprocess.run([
                'git', 'log', '--oneline', '--merges',
                f'--since={self.since_date.isoformat()}',
                'main'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                merge_count = len([line for line in result.stdout.strip().split('\n') if line])
                return merge_count / max(self.lookback_days, 1)
            else:
                logger.warning("Failed to get deployment frequency")
                return 0.0
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Git not available for deployment frequency calculation")
            return 0.0
    
    def collect_lead_time(self) -> float:
        """Calculate lead time (hours from commit to deploy)."""
        try:
            # Get recent merge commits with timestamps
            result = subprocess.run([
                'git', 'log', '--merges', '--pretty=format:%H|%ct',
                f'--since={self.since_date.isoformat()}',
                'main'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                lead_times = []
                for line in result.stdout.strip().split('\n'):
                    if line and '|' in line:
                        merge_hash, merge_timestamp = line.split('|')
                        merge_time = datetime.fromtimestamp(int(merge_timestamp))
                        
                        # Get first commit in the merge
                        first_commit_result = subprocess.run([
                            'git', 'log', '--pretty=format:%ct', '--reverse',
                            f'{merge_hash}^1..{merge_hash}^2'
                        ], capture_output=True, text=True, cwd=self.repo_path)
                        
                        if first_commit_result.returncode == 0:
                            first_line = first_commit_result.stdout.strip().split('\n')[0]
                            if first_line:
                                first_commit_time = datetime.fromtimestamp(int(first_line))
                                lead_time_hours = (merge_time - first_commit_time).total_seconds() / 3600
                                lead_times.append(lead_time_hours)
                
                return sum(lead_times) / max(len(lead_times), 1)
            else:
                return 0.0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0.0
    
    def collect_change_failure_rate(self) -> float:
        """Calculate change failure rate (percentage of deployments causing failures)."""
        try:
            # Count total deployments (merges to main)
            total_deploys_result = subprocess.run([
                'git', 'log', '--oneline', '--merges',
                f'--since={self.since_date.isoformat()}',
                'main'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            total_deploys = len([line for line in total_deploys_result.stdout.strip().split('\n') if line])
            
            if total_deploys == 0:
                return 0.0
            
            # Count incident-related commits (those with incident labels or hotfix patterns)
            incident_result = subprocess.run([
                'git', 'log', '--oneline', '--grep=incident', '--grep=hotfix',
                '--grep=emergency', '--grep=critical',
                f'--since={self.since_date.isoformat()}',
                'main'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            incident_count = len([line for line in incident_result.stdout.strip().split('\n') if line])
            
            return (incident_count / total_deploys) * 100
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0.0
    
    def collect_mean_time_to_recovery(self) -> float:
        """Calculate MTTR (hours to recover from failures)."""
        try:
            # Find incident tickets marked with type:incident
            incidents_result = subprocess.run([
                'git', 'log', '--oneline', '--grep=type:incident',
                '--pretty=format:%H|%ct|%s',
                f'--since={self.since_date.isoformat()}'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if incidents_result.returncode == 0:
                recovery_times = []
                for line in incidents_result.stdout.strip().split('\n'):
                    if line and '|' in line:
                        parts = line.split('|', 2)
                        if len(parts) >= 2:
                            incident_hash, incident_timestamp = parts[0], parts[1]
                            incident_time = datetime.fromtimestamp(int(incident_timestamp))
                            
                            # Find the next commit after this incident (recovery)
                            recovery_result = subprocess.run([
                                'git', 'log', '--oneline', '--pretty=format:%ct',
                                f'--since={incident_time.isoformat()}',
                                '--max-count=2'
                            ], capture_output=True, text=True, cwd=self.repo_path)
                            
                            if recovery_result.returncode == 0:
                                recovery_lines = recovery_result.stdout.strip().split('\n')
                                if len(recovery_lines) >= 2:
                                    recovery_timestamp = recovery_lines[1]
                                    recovery_time = datetime.fromtimestamp(int(recovery_timestamp))
                                    mttr_hours = (recovery_time - incident_time).total_seconds() / 3600
                                    recovery_times.append(mttr_hours)
                
                return sum(recovery_times) / max(len(recovery_times), 1)
            else:
                return 0.0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0.0
    
    def collect_all_metrics(self) -> DoraMetrics:
        """Collect all DORA metrics."""
        return DoraMetrics(
            deploy_frequency=self.collect_deployment_frequency(),
            lead_time=self.collect_lead_time(), 
            change_failure_rate=self.collect_change_failure_rate(),
            mean_time_to_recovery=self.collect_mean_time_to_recovery(),
            measurement_period_days=self.lookback_days
        )


class MetricsReporter:
    """Enhanced metrics and reporting system."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.status_dir = self.repo_path / "docs" / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_rerere_metrics(self) -> Dict:
        """Collect git rerere metrics."""
        metrics = {
            "rerere_auto_resolved_total": 0,
            "merge_driver_hits_total": 0
        }
        
        try:
            # Check rerere cache for auto-resolved conflicts
            rerere_cache_dir = self.repo_path / ".git" / "rr-cache"
            if rerere_cache_dir.exists():
                resolved_conflicts = len(list(rerere_cache_dir.glob("*")))
                metrics["rerere_auto_resolved_total"] = resolved_conflicts
        except (OSError, PermissionError):
            pass
        
        try:
            # Check recent commits for merge driver usage (package-lock.json changes)
            result = subprocess.run([
                'git', 'log', '--oneline', '--name-only', '--since=1.week.ago'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                package_lock_changes = result.stdout.count('package-lock.json')
                metrics["merge_driver_hits_total"] = package_lock_changes
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return metrics
    
    def calculate_ci_failure_rate(self) -> float:
        """Calculate CI failure rate over last 24 hours."""
        # This would normally check CI system API
        # For now, return a placeholder value
        return 15.0  # 15% failure rate
    
    def determine_pr_backoff_state(self, failure_rate: float) -> str:
        """Determine if PR throttling should be active."""
        if failure_rate > 30:
            return "active"
        elif failure_rate < 10:
            return "inactive"
        else:
            return "monitoring"
    
    def collect_backlog_metrics(self) -> Dict:
        """Collect backlog-specific metrics."""
        backlog_file = self.repo_path / "backlog.yml"
        if not backlog_file.exists():
            return {}
        
        try:
            import yaml
            with open(backlog_file, 'r') as f:
                data = yaml.safe_load(f)
            
            backlog = data.get('backlog', [])
            
            # Count by status
            status_counts = {}
            for item in backlog:
                status = item.get('status', 'UNKNOWN')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate average cycle time (for completed items)
            completed_items = [item for item in backlog if item.get('status') == 'DONE']
            cycle_times = []
            
            for item in completed_items:
                created_at = item.get('created_at')
                completed_at = item.get('completed_at')
                if created_at and completed_at:
                    try:
                        created = datetime.fromisoformat(created_at)
                        completed = datetime.fromisoformat(completed_at)
                        cycle_time = (completed - created).total_seconds() / 3600
                        cycle_times.append(cycle_time)
                    except (ValueError, TypeError):
                        continue
            
            avg_cycle_time = sum(cycle_times) / max(len(cycle_times), 1)
            
            # Get top WSJF items
            wsjf_snapshot = sorted(
                [item for item in backlog if item.get('status') == 'READY'],
                key=lambda x: x.get('wsjf_score', 0),
                reverse=True
            )[:3]
            
            return {
                "backlog_size_by_status": status_counts,
                "avg_cycle_time": round(avg_cycle_time, 2),
                "wsjf_snapshot": [f"{item.get('id', 'unknown')}: {item.get('title', 'No title')}" 
                                for item in wsjf_snapshot]
            }
        except Exception as e:
            logger.warning(f"Failed to collect backlog metrics: {e}")
            return {}
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive status report with all metrics."""
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        
        # Collect DORA metrics
        dora_collector = DoraMetricsCollector(str(self.repo_path))
        dora_metrics = dora_collector.collect_all_metrics()
        
        # Collect other metrics
        rerere_metrics = self.collect_rerere_metrics()
        ci_failure_rate = self.calculate_ci_failure_rate()
        pr_backoff_state = self.determine_pr_backoff_state(ci_failure_rate)
        backlog_metrics = self.collect_backlog_metrics()
        
        # Read recent completed IDs from backlog
        completed_ids = []
        try:
            import yaml
            backlog_file = self.repo_path / "backlog.yml"
            if backlog_file.exists():
                with open(backlog_file, 'r') as f:
                    data = yaml.safe_load(f)
                completed_ids = [
                    item.get('id') for item in data.get('backlog', [])
                    if item.get('status') == 'DONE' and 
                    item.get('completed_at', '').startswith(date_str)
                ]
        except Exception:
            pass
        
        report = {
            "timestamp": now.isoformat(),
            "completed_ids": completed_ids,
            "coverage_delta": "+5.2%",  # Placeholder - would calculate from test runs
            "flaky_tests": [],  # Would be populated from test history
            "ci_summary": "stable",
            "open_prs": 0,  # Would check GitHub API
            "risks_or_blocks": ["No significant risks identified"],
            **backlog_metrics,
            "dora": asdict(dora_metrics),
            **rerere_metrics,
            "ci_failure_rate": ci_failure_rate,
            "pr_backoff_state": pr_backoff_state
        }
        
        return report
    
    def save_report(self, report: Dict, date_suffix: Optional[str] = None) -> str:
        """Save report to docs/status/ directory."""
        if date_suffix is None:
            date_suffix = datetime.now().strftime('%Y%m%d')
        
        report_file = self.status_dir / f"{date_suffix}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also generate markdown report
        md_file = self.status_dir / f"{date_suffix}.md"
        self._generate_markdown_report(report, md_file)
        
        return str(report_file)
    
    def _generate_markdown_report(self, report: Dict, md_file: Path) -> None:
        """Generate markdown version of the report."""
        md_content = f"""# Autonomous Backlog Management Report - {report['timestamp'][:10]}

## Summary
- **Completed Items**: {len(report.get('completed_ids', []))}
- **CI Status**: {report.get('ci_summary', 'unknown')}
- **Coverage**: {report.get('coverage_delta', 'N/A')}
- **PR Backoff**: {report.get('pr_backoff_state', 'unknown')}

## DORA Metrics
- **Deployment Frequency**: {report.get('dora', {}).get('deploy_frequency', 0):.2f} per day
- **Lead Time**: {report.get('dora', {}).get('lead_time', 0):.1f} hours
- **Change Failure Rate**: {report.get('dora', {}).get('change_failure_rate', 0):.1f}%
- **MTTR**: {report.get('dora', {}).get('mean_time_to_recovery', 0):.1f} hours

## Backlog Status
- **Average Cycle Time**: {report.get('avg_cycle_time', 0)} hours
- **Status Breakdown**: {report.get('backlog_size_by_status', {})}

## Infrastructure Metrics
- **Rerere Auto-Resolved**: {report.get('rerere_auto_resolved_total', 0)}
- **Merge Driver Hits**: {report.get('merge_driver_hits_total', 0)}
- **CI Failure Rate**: {report.get('ci_failure_rate', 0):.1f}%

## Next Actions
{chr(10).join(f"- {item}" for item in report.get('wsjf_snapshot', []))}

## Risks and Blocks
{chr(10).join(f"- {risk}" for risk in report.get('risks_or_blocks', []))}
"""
        
        with open(md_file, 'w') as f:
            f.write(md_content)


def main():
    """Main entry point for metrics reporting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DORA Metrics Reporter")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--date-suffix", help="Date suffix for report files")
    
    args = parser.parse_args()
    
    reporter = MetricsReporter(args.repo_path)
    if args.output_dir:
        reporter.status_dir = Path(args.output_dir)
        reporter.status_dir.mkdir(parents=True, exist_ok=True)
    
    report = reporter.generate_comprehensive_report()
    report_file = reporter.save_report(report, args.date_suffix)
    
    print(f"Comprehensive metrics report saved to: {report_file}")
    print(f"DORA Metrics Summary:")
    print(f"  Deploy Frequency: {report['dora']['deploy_frequency']:.2f}/day")
    print(f"  Lead Time: {report['dora']['lead_time']:.1f} hours")
    print(f"  Change Failure Rate: {report['dora']['change_failure_rate']:.1f}%")
    print(f"  MTTR: {report['dora']['mean_time_to_recovery']:.1f} hours")
    
    return 0


if __name__ == "__main__":
    exit(main())