"""Tests for specific exception handling improvements."""

import pytest
import yaml
import subprocess
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from autonomous_backlog_manager import AutonomousBacklogManager


class TestSpecificExceptionHandling:
    """Test specific exception handling instead of broad Exception catches."""
    
    def test_config_loading_yaml_exceptions(self):
        """Test that config loading handles YAML-specific exceptions."""
        manager = AutonomousBacklogManager()
        
        # Test yaml.YAMLError specifically
        with patch('yaml.safe_load') as mock_yaml:
            mock_yaml.side_effect = yaml.YAMLError("Invalid YAML syntax")
            config = manager._load_config()
            
            # Should return default config on YAML error
            assert isinstance(config, dict)
            assert 'wsjf' in config
    
    def test_backlog_loading_file_exceptions(self):
        """Test backlog loading handles file-specific exceptions."""
        manager = AutonomousBacklogManager()
        
        # Test FileNotFoundError specifically
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = FileNotFoundError("File not found")
            manager.load_backlog()
            # Should handle FileNotFoundError and create empty backlog
            assert manager.backlog == []
        
        # Test PermissionError specifically  
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")
            manager.load_backlog()
            assert manager.backlog == []
    
    def test_subprocess_specific_exceptions(self):
        """Test subprocess operations handle specific exceptions."""
        manager = AutonomousBacklogManager()
        
        # Test subprocess.CalledProcessError
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'rg', 'command failed')
            items = manager.discover_todo_fixme_items()
            # Should handle CalledProcessError and return empty list
            assert items == []
        
        # Test FileNotFoundError for missing command
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("Command not found")
            items = manager.discover_todo_fixme_items()
            assert items == []
    
    def test_json_serialization_exceptions(self):
        """Test JSON operations handle specific exceptions."""
        manager = AutonomousBacklogManager()
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('json.dump') as mock_json:
                mock_json.side_effect = TypeError("Object is not JSON serializable")
                
                # Should handle TypeError specifically without crashing
                manager.save_status_report({'test': 'data'})