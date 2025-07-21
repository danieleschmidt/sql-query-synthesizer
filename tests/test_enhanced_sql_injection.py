"""Comprehensive tests for enhanced SQL injection prevention."""

import pytest
from sql_synthesizer.services.enhanced_query_validator import EnhancedQueryValidatorService
from sql_synthesizer.user_experience import UserFriendlyError


class TestEnhancedSQLInjectionPrevention:
    """Test enhanced SQL injection prevention capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = EnhancedQueryValidatorService(
            max_question_length=1000,
            allowed_tables=['users', 'orders', 'products', 'customers']
        )
    
    def test_basic_safe_questions(self):
        """Test that safe questions pass validation."""
        safe_questions = [
            "Show me all users",
            "Count the orders",
            "List products with price > 100",
            "Find customers from California",
            "What is the average order value?",
            "Show top 10 selling products"
        ]
        
        for question in safe_questions:
            # Should not raise any exceptions
            result = self.validator.validate_question(question)
            assert isinstance(result, str)
            assert len(result.strip()) > 0
    
    def test_basic_sql_injection_patterns(self):
        """Test detection of basic SQL injection patterns."""
        malicious_inputs = [
            "users'; DROP TABLE users; --",
            "1 OR 1=1",
            "'; UPDATE users SET password='hacked' WHERE 1=1; --",
            "UNION SELECT username, password FROM admin_users",
            "1; EXEC xp_cmdshell('format c:'); --",
            "'; INSERT INTO users VALUES ('attacker', 'password'); --"
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(UserFriendlyError):
                self.validator.validate_question(malicious_input)
    
    def test_sophisticated_sql_injection_attempts(self):
        """Test detection of sophisticated SQL injection techniques."""
        sophisticated_attacks = [
            # Time-based blind injection
            "users WHERE id=1 AND (SELECT COUNT(*) FROM sysobjects)>0 WAITFOR DELAY '00:00:05'",
            # Boolean-based blind injection with encoding
            "users WHERE id=1 AND ASCII(SUBSTRING((SELECT TOP 1 username FROM users),1,1))>65",
            # Second-order injection
            "users WHERE name='O''Reilly' UNION SELECT credit_card FROM payments",
            # SQL injection with comments
            "users/*comment*/WHERE/**/id=1/**/UNION/**/SELECT/**/password/**/FROM/**/admin",
            # Hex encoding attempt
            "users WHERE id=0x41 UNION SELECT 0x31",
            # Stacked queries
            "users WHERE id=1; INSERT INTO logs VALUES ('breach attempt')",
            # Function-based injection
            "users WHERE id=CONVERT(int, (SELECT COUNT(*) FROM information_schema.tables))"
        ]
        
        for attack in sophisticated_attacks:
            with pytest.raises(UserFriendlyError):
                self.validator.validate_question(attack)
    
    def test_context_aware_validation(self):
        """Test context-aware validation based on available tables."""
        # Valid references to existing tables
        valid_queries = [
            "Show users table data",
            "Count orders by status", 
            "List all products",
            "Find customers in orders"
        ]
        
        for query in valid_queries:
            result = self.validator.validate_question(query)
            assert isinstance(result, str)
        
        # Invalid references to non-existent tables (with explicit SQL patterns)
        invalid_queries = [
            "SELECT * FROM admin_secrets",
            "SELECT * FROM system_config", 
            "SELECT * FROM credit_cards"
        ]
        
        for query in invalid_queries:
            with pytest.raises(UserFriendlyError):
                self.validator.validate_question(query)
    
    def test_ast_based_sql_validation(self):
        """Test AST-based SQL statement validation."""
        # Valid SELECT statements
        valid_sql = [
            "SELECT * FROM users",
            "SELECT name, email FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM orders GROUP BY status",
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        ]
        
        for sql in valid_sql:
            result = self.validator.validate_sql_statement(sql)
            assert result == sql  # Should return unchanged if valid
        
        # Invalid SQL statements (non-SELECT)
        invalid_sql = [
            "DROP TABLE users",
            "UPDATE users SET password = 'hacked'",
            "INSERT INTO users VALUES ('attacker', 'evil@hack.com')",
            "DELETE FROM orders WHERE id > 0",
            "CREATE TABLE backdoor (id int)",
            "ALTER TABLE users ADD COLUMN backdoor text"
        ]
        
        for sql in invalid_sql:
            with pytest.raises(ValueError, match="Only SELECT statements"):
                self.validator.validate_sql_statement(sql)
    
    def test_semantic_analysis_detection(self):
        """Test semantic analysis for injection detection."""
        # Queries that are structurally valid but semantically suspicious
        suspicious_queries = [
            # Tautology-based injections
            "SELECT * FROM users WHERE 1=1 OR 'a'='a'",
            "SELECT * FROM users WHERE 'x'='x' AND id > 0",
            # Always-true conditions
            "SELECT * FROM users WHERE id = id",
            "SELECT * FROM users WHERE 1 < 2",
            # Suspicious function usage
            "SELECT * FROM users WHERE LENGTH(password) > 0 AND 1=1",
            # Information gathering attempts
            "SELECT * FROM users WHERE database() LIKE '%'",
            "SELECT * FROM users WHERE version() IS NOT NULL"
        ]
        
        for query in suspicious_queries:
            with pytest.raises(UserFriendlyError):
                self.validator.validate_sql_statement(query)
    
    def test_encoding_and_obfuscation_detection(self):
        """Test detection of encoded and obfuscated injection attempts."""
        obfuscated_attacks = [
            # URL encoding
            "users%20WHERE%20id%3D1%20UNION%20SELECT%20password",
            # Double encoding
            "%2527%2520OR%2520%25271%2527%253D%25271",
            # Unicode encoding
            "users\\u0027\\u0020OR\\u0020\\u0027\\u0031\\u0027\\u003D\\u0027\\u0031",
            # Mixed case evasion
            "UsErS wHeRe Id=1 uNiOn SeLeCt PaSsWoRd",
            # Whitespace evasion
            "users\twhere\nid=1\runion\fselect\vpassword"
        ]
        
        for attack in obfuscated_attacks:
            # Should decode and detect the attack
            with pytest.raises(UserFriendlyError):
                self.validator.validate_question(attack)
    
    def test_performance_with_large_inputs(self):
        """Test validation performance with large inputs."""
        # Large but safe input
        large_safe_input = "Show me all users " + "with valid data " * 100
        
        # Should handle large inputs efficiently
        result = self.validator.validate_question(large_safe_input)
        assert isinstance(result, str)
        
        # Large malicious input
        large_malicious_input = "users WHERE id=1 " + "OR 1=1 " * 50
        
        with pytest.raises(UserFriendlyError):
            self.validator.validate_question(large_malicious_input)
    
    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        edge_cases = [
            # Empty and whitespace
            ("", UserFriendlyError),
            ("   ", UserFriendlyError), 
            ("\t\n", UserFriendlyError),
            # Very long input
            ("a" * 2000, UserFriendlyError),  # Exceeds max_question_length
            # Special characters
            ("users with $pecial ch@racters!", str),
            # Numbers and dates
            ("orders from 2023-01-01 to 2023-12-31", str),
            # Legitimate quotes in names
            ("customers with name O'Connor", str)
        ]
        
        for input_text, expected_result in edge_cases:
            if expected_result == UserFriendlyError:
                with pytest.raises(UserFriendlyError):
                    self.validator.validate_question(input_text)
            else:
                result = self.validator.validate_question(input_text)
                assert isinstance(result, expected_result)
    
    def test_allowlist_validation(self):
        """Test allowlist-based validation for tables and columns."""
        # Initialize validator with specific allowlists
        validator = EnhancedQueryValidatorService(
            allowed_tables=['users', 'orders'],
            allowed_columns=['id', 'name', 'email', 'status', 'total']
        )
        
        # Valid table and column references
        valid_references = [
            "SELECT id, name FROM users",
            "SELECT total FROM orders WHERE status = 'completed'",
            "SELECT users.name, orders.total FROM users JOIN orders"
        ]
        
        for sql in valid_references:
            result = validator.validate_sql_statement(sql)
            assert result == sql
        
        # Invalid table references
        with pytest.raises(UserFriendlyError):
            validator.validate_sql_statement("SELECT * FROM products")
        
        # Invalid column references
        with pytest.raises(UserFriendlyError):
            validator.validate_sql_statement("SELECT password FROM users")
    
    def test_rate_limiting_and_ddos_protection(self):
        """Test protection against rapid-fire injection attempts."""
        # Simulate rapid injection attempts (this would be handled at the service layer)
        validator = EnhancedQueryValidatorService(
            max_validation_attempts_per_minute=10
        )
        
        # Normal usage should work
        for i in range(5):
            result = validator.validate_question(f"Show users batch {i}")
            assert isinstance(result, str)
        
        # Rapid malicious attempts should be rate limited
        malicious_attempts = ["'; DROP TABLE users; --"] * 20
        
        # First few should trigger normal validation errors
        for i in range(5):
            with pytest.raises(UserFriendlyError):
                validator.validate_question(malicious_attempts[i])
        
        # After rate limit, should get rate limiting error
        # Note: This would require actual rate limiting implementation
        # For now, we test the validation logic itself
    
    def test_integration_with_existing_validator(self):
        """Test integration with existing QueryValidatorService."""
        from sql_synthesizer.services.query_validator_service import QueryValidatorService
        
        # Test that enhanced validator can be used as drop-in replacement
        enhanced_validator = EnhancedQueryValidatorService()
        legacy_validator = QueryValidatorService()
        
        safe_question = "Show me all users"
        
        # Both should handle safe questions
        enhanced_result = enhanced_validator.validate_question(safe_question)
        legacy_result = legacy_validator.validate_question(safe_question)
        
        assert isinstance(enhanced_result, str)
        assert isinstance(legacy_result, str)
        
        # Enhanced should catch more sophisticated attacks
        sophisticated_attack = "users WHERE ASCII(SUBSTRING((SELECT username FROM admin),1,1))>65"
        
        # Enhanced validator should catch this
        with pytest.raises(ValueError):
            enhanced_validator.validate_question(sophisticated_attack)
        
        # Legacy might miss it (depending on implementation)
        # This test documents the improvement