#!/usr/bin/env python3
"""
Test suite for autonomous backlog management system.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from autonomous_backlog_manager import AutonomousBacklogManager, BacklogItem
from tdd_security_checklist import TDDSecurityChecker
from dora_metrics import DoraMetricsCollector, MetricsReporter


class TestAutonomousBacklogManager(unittest.TestCase):
    """Test autonomous backlog manager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = AutonomousBacklogManager(self.temp_dir)
    
    def test_backlog_item_wsjf_calculation(self):
        """Test WSJF score calculation."""
        item = BacklogItem(
            id="test-1",
            title="Test Item",
            type="test",
            description="Test description",
            acceptance_criteria=["Test criterion"],
            effort=5,
            value=8,
            time_criticality=5,
            risk_reduction=8,
            status="NEW",
            risk_tier="low",
            created_at="2025-07-26",
            links=[]
        )
        
        score = item.calculate_wsjf()
        expected_score = (8 + 5 + 8) / 5  # 4.2
        self.assertAlmostEqual(score, expected_score, places=1)
    
    def test_aging_multiplier(self):
        """Test aging multiplier for stale items."""
        item = BacklogItem(
            id="test-2",
            title="Old Item",
            type="test",
            description="Test description",
            acceptance_criteria=["Test criterion"],
            effort=3,
            value=5,
            time_criticality=3,
            risk_reduction=2,
            status="NEW",
            risk_tier="low",
            created_at="2025-07-26",
            links=[]
        )
        
        # Test with aging > threshold
        score_aged = item.calculate_wsjf(aging_days=40, aging_threshold=30, max_multiplier=2.0)
        score_normal = item.calculate_wsjf(aging_days=20, aging_threshold=30, max_multiplier=2.0)
        
        self.assertGreater(score_aged, score_normal)
        self.assertGreater(item.aging_multiplier, 1.0)
    
    def test_backlog_loading_with_blocked_items(self):
        """Test loading backlog with blocked items."""
        # Create a sample backlog file
        backlog_data = {
            'backlog': [{
                'id': 'test-blocked',
                'title': 'Blocked Item',
                'type': 'test',
                'description': 'Test',
                'acceptance_criteria': ['Test'],
                'effort': 2,
                'value': 3,
                'time_criticality': 1,
                'risk_reduction': 2,
                'status': 'BLOCKED',
                'risk_tier': 'low',
                'created_at': '2025-07-26',
                'links': [],
                'wsjf_score': 3.0,
                'aging_multiplier': 1.0,
                'completed_at': None,
                'blocked_reason': 'Test blocking reason'
            }],
            'config': self.manager.config
        }
        
        backlog_file = Path(self.temp_dir) / "backlog.yml"
        import yaml
        with open(backlog_file, 'w') as f:
            yaml.dump(backlog_data, f)
        
        # Load and verify
        manager = AutonomousBacklogManager(self.temp_dir)
        self.assertEqual(len(manager.backlog), 1)
        self.assertEqual(manager.backlog[0].status, 'BLOCKED')
        self.assertEqual(manager.backlog[0].blocked_reason, 'Test blocking reason')


class TestSecurityChecker(unittest.TestCase):
    """Test TDD security checker functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checker = TDDSecurityChecker(self.temp_dir)
    
    def test_security_checklist_execution(self):
        """Test security checklist runs without errors."""
        results = self.checker.run_security_checklist()
        
        # Should have at least basic checks
        self.assertGreater(len(results), 0)
        
        # All results should have required fields
        for result in results:
            self.assertTrue(hasattr(result, 'check_name'))
            self.assertTrue(hasattr(result, 'passed'))
            self.assertTrue(hasattr(result, 'details'))
            self.assertTrue(hasattr(result, 'severity'))
    
    def test_security_report_generation(self):
        """Test security report generation."""
        self.checker.run_security_checklist()
        report = self.checker.generate_security_report()
        
        # Check report structure
        self.assertIn('timestamp', report)
        self.assertIn('summary', report)
        self.assertIn('severity_breakdown', report)
        self.assertIn('results', report)
        self.assertIn('status', report)
        
        # Check summary fields
        summary = report['summary']
        self.assertIn('total_checks', summary)
        self.assertIn('passed', summary)
        self.assertIn('failed', summary)
        self.assertIn('pass_rate', summary)


class TestDoraMetrics(unittest.TestCase):
    """Test DORA metrics collection."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = DoraMetricsCollector(self.temp_dir)
        self.reporter = MetricsReporter(self.temp_dir)
    
    @patch('subprocess.run')
    def test_deployment_frequency_calculation(self, mock_run):
        """Test deployment frequency calculation."""
        # Mock git log output
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="commit1 Merge pull request #1\ncommit2 Merge pull request #2\n"
        )
        
        freq = self.collector.collect_deployment_frequency()
        self.assertGreaterEqual(freq, 0)
    
    @patch('subprocess.run')
    def test_lead_time_calculation(self, mock_run):
        """Test lead time calculation."""
        # Mock git log output
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123|1643723400\n"  # Mock hash and timestamp
        )
        
        lead_time = self.collector.collect_lead_time()
        self.assertGreaterEqual(lead_time, 0)
    
    def test_metrics_report_generation(self):
        """Test comprehensive metrics report generation."""
        report = self.reporter.generate_comprehensive_report()
        
        # Check required fields
        self.assertIn('timestamp', report)
        self.assertIn('dora', report)
        self.assertIn('ci_failure_rate', report)
        self.assertIn('pr_backoff_state', report)
        
        # Check DORA metrics structure
        dora = report['dora']
        self.assertIn('deploy_frequency', dora)
        self.assertIn('lead_time', dora)
        self.assertIn('change_failure_rate', dora)
        self.assertIn('mean_time_to_recovery', dora)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_dry_run_execution(self):
        """Test dry-run execution without errors."""
        from run_autonomous_cycle import AutonomousCycleRunner
        
        runner = AutonomousCycleRunner(self.temp_dir)
        
        # Should complete without exceptions
        try:
            runner.sync_repo_and_ci()
            runner.discover_new_tasks()
            runner.score_and_sort_backlog()
            runner.update_metrics()
            success = True
        except Exception as e:
            success = False
            print(f"Integration test failed: {e}")
        
        self.assertTrue(success, "Dry-run execution should complete without errors")
    
    def test_status_report_creation(self):
        """Test status report file creation."""
        reporter = MetricsReporter(self.temp_dir)
        report = reporter.generate_comprehensive_report()
        report_file = reporter.save_report(report, "test")
        
        # Check files were created
        self.assertTrue(Path(report_file).exists())
        
        # Check JSON format
        with open(report_file, 'r') as f:
            loaded_report = json.load(f)
        
        self.assertEqual(loaded_report['timestamp'], report['timestamp'])
        
        # Check markdown file
        md_file = Path(self.temp_dir) / "docs" / "status" / "test.md"
        self.assertTrue(md_file.exists())


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)