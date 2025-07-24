"""Enhanced query validation service with comprehensive SQL injection prevention."""

import re
import logging
import urllib.parse
import unicodedata
import time
from typing import List, Set, Optional, Dict, Any
from collections import defaultdict

try:
    import sqlparse
    from sqlparse.sql import IdentifierList, Identifier, Function
    from sqlparse.tokens import Keyword, DML
except ImportError:
    sqlparse = None

from ..user_experience import (
    create_empty_question_error,
    create_question_too_long_error,
    create_unsafe_input_error,
    create_invalid_sql_error,
    create_multiple_statements_error,
    create_invalid_table_error,
)
from .. import metrics
from ..security_audit import security_audit_logger, SecurityEventType, SecurityEventSeverity

logger = logging.getLogger(__name__)

# Enhanced SQL injection patterns with more sophisticated detection
ENHANCED_SQL_INJECTION_PATTERNS = [
    # Basic injection patterns
    r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|GRANT|REVOKE|TRUNCATE|EXEC)",
    r"--\s*[^'\s]",  # SQL comments
    r"/\*.*?\*/",    # Multi-line comments
    r"\*/.*?/\*",    # Nested comments
    
    # Union-based injection
    r"UNION\s+(ALL\s+)?SELECT",
    r"UNION\s+(ALL\s+)?INSERT",
    
    # Boolean-based injection
    r"(OR|AND)\s+1\s*[=!<>]+\s*1",
    r"(OR|AND)\s+['\"]?[a-zA-Z]+['\"]?\s*[=!<>]+\s*['\"]?[a-zA-Z]+['\"]?",
    
    # Time-based injection
    r"WAITFOR\s+DELAY",
    r"BENCHMARK\s*\(",
    r"SLEEP\s*\(",
    r"pg_sleep\s*\(",
    
    # Information gathering
    r"(SELECT|UNION).*?(information_schema|sys\.|mysql\.|pg_catalog)",
    r"(SELECT|UNION).*?(database\s*\(|version\s*\(|user\s*\()",
    r"(SELECT|UNION).*?(@@version|@@servername|@@database)",
    
    # Stacked queries
    r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)",
    
    # Function-based injection
    r"(ASCII|CHAR|SUBSTRING|LENGTH|CONVERT|CAST)\s*\(",
    r"(HEX|UNHEX|MD5|SHA1|ENCRYPT)\s*\(",
    
    # Advanced evasion techniques
    r"['\"](\s*(OR|AND)\s*['\"]|\s*\|\|\s*['\"])",  # Quote breaking
    r"\b(CHR|CONCAT|CONCATENATE)\s*\(",  # String manipulation
    r"[0-9]+\s*[=!<>]+\s*[0-9]+\s*(OR|AND)",  # Numeric tautologies
    
    # Blind injection patterns
    r"(IF|CASE)\s*\(.*(SELECT|database|user)",
    r"(LIKE|RLIKE|REGEXP)\s*['\"][%_].*[%_]['\"]",
    
    # Command execution attempts
    r"(xp_cmdshell|sp_executesql|exec\s+master)",
    r"(load_file|into\s+outfile|into\s+dumpfile)",
    
    # Encoding evasion
    r"(\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|%[0-9a-fA-F]{2})",
]

# Compile patterns for performance
COMPILED_ENHANCED_PATTERNS = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in ENHANCED_SQL_INJECTION_PATTERNS]

# Suspicious SQL functions that shouldn't appear in user queries
SUSPICIOUS_FUNCTIONS = {
    'database', 'version', 'user', 'current_user', 'session_user',
    'system_user', 'connection_id', 'load_file', 'into_outfile',
    'benchmark', 'sleep', 'waitfor', 'pg_sleep', 'extractvalue',
    'updatexml', 'exp', 'ascii', 'char', 'hex', 'unhex', 'md5', 'sha1'
}

# Always-true/false conditions that indicate tautology-based injection
TAUTOLOGY_PATTERNS = [
    r"\b1\s*=\s*1\b",
    r"\b0\s*=\s*0\b", 
    r"\b['\"]([^'\"]*)\1\s*=\s*\1([^'\"]*)\1\b",  # 'a'='a'
    r"\b\w+\s*=\s*\w+\b.*\bOR\b.*\b\w+\s*=\s*\w+\b",  # id=id OR name=name
]

COMPILED_TAUTOLOGY_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in TAUTOLOGY_PATTERNS]


class EnhancedQueryValidatorService:
    """Enhanced query validation service with comprehensive SQL injection prevention."""
    
    def __init__(
        self, 
        max_question_length: int = 1000,
        allowed_tables: Optional[List[str]] = None,
        allowed_columns: Optional[List[str]] = None,
        max_validation_attempts_per_minute: int = 60
    ):
        """Initialize the enhanced validator service.
        
        Args:
            max_question_length: Maximum allowed length for user questions
            allowed_tables: List of allowed table names (None = allow all discovered tables)
            allowed_columns: List of allowed column names (None = allow all discovered columns)
            max_validation_attempts_per_minute: Rate limiting for validation attempts
        """
        self.max_question_length = max_question_length
        self.allowed_tables = set(allowed_tables) if allowed_tables else None
        self.allowed_columns = set(allowed_columns) if allowed_columns else None
        self.max_validation_attempts = max_validation_attempts_per_minute
        
        # Rate limiting tracking
        self.validation_attempts: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"Enhanced SQL injection prevention initialized with {len(COMPILED_ENHANCED_PATTERNS)} patterns")
    
    def validate_question(self, question: str, client_id: Optional[str] = None) -> str:
        """Validate a natural language question for safety and policy compliance.
        
        Args:
            question: The user's natural language question
            client_id: Optional client identifier for rate limiting
            
        Returns:
            str: The sanitized question
            
        Raises:
            ValueError: If the question is unsafe or violates policy
        """
        # Rate limiting check
        if client_id:
            self._check_rate_limit(client_id)
        
        # Basic validation
        if not question or not question.strip():
            raise create_empty_question_error()
        
        question = question.strip()
        
        if len(question) > self.max_question_length:
            raise create_question_too_long_error(self.max_question_length)
        
        # Decode any URL/Unicode encoding to detect obfuscated attacks
        decoded_question = self._decode_input(question)
        
        # Check for SQL injection patterns
        if self._contains_sql_injection(decoded_question):
            metrics.record_query_error("sql_injection_detected")
            security_audit_logger.log_sql_injection_attempt(
                malicious_input=question,
                detection_method="pattern_matching",
                client_id=client_id,
                original_input=decoded_question
            )
            raise create_unsafe_input_error()
        
        # Check for unauthorized table/column references
        if self._contains_unauthorized_references(decoded_question):
            metrics.record_query_error("unauthorized_reference_detected")
            security_audit_logger.log_event(
                event_type=SecurityEventType.UNSAFE_INPUT_DETECTED,
                severity=SecurityEventSeverity.HIGH,
                message="Unauthorized table/column reference detected",
                client_id=client_id,
                malicious_input=question,
                detection_method="unauthorized_reference",
                decoded_input=decoded_question
            )
            raise create_unsafe_input_error()
        
        # Semantic analysis for suspicious patterns
        if self._contains_suspicious_patterns(decoded_question):
            metrics.record_query_error("suspicious_pattern_detected")
            security_audit_logger.log_event(
                event_type=SecurityEventType.UNSAFE_INPUT_DETECTED,
                severity=SecurityEventSeverity.MEDIUM,
                message="Suspicious pattern detected in user input",
                client_id=client_id,
                malicious_input=question,
                detection_method="semantic_analysis",
                decoded_input=decoded_question
            )
            raise create_unsafe_input_error()
        
        return question
    
    def validate_sql_statement(self, sql: str) -> str:
        """Validate a SQL statement for safety and policy compliance.
        
        Args:
            sql: The SQL statement to validate
            
        Returns:
            str: The validated SQL statement
            
        Raises:
            ValueError: If the SQL is unsafe or violates policy
        """
        if not sql or not sql.strip():
            raise create_invalid_sql_error("SQL statement cannot be empty")
        
        sql = sql.strip()
        
        # Parse SQL using sqlparse if available
        if sqlparse:
            try:
                parsed = sqlparse.parse(sql)
                if not parsed:
                    raise create_invalid_sql_error("Unable to parse SQL statement")
                
                statement = parsed[0]
                
                # Check if it's a SELECT statement
                if not self._is_select_statement(statement):
                    raise create_invalid_sql_error("Only SELECT statements are allowed")
                
                # Validate table and column references
                self._validate_sql_references(statement)
                
                # Check for suspicious semantic patterns
                if self._contains_sql_suspicious_patterns(sql):
                    security_audit_logger.log_sql_injection_attempt(
                        malicious_input=sql,
                        detection_method="ast_semantic_analysis",
                        sql_statement=sql
                    )
                    raise create_unsafe_input_error()
                
            except sqlparse.exceptions.SQLParseError as e:
                raise create_invalid_sql_error(f"SQL parsing error: {e}")
        else:
            # Fallback validation without sqlparse
            logger.warning("sqlparse not available, using basic SQL validation")
            if not sql.upper().strip().startswith('SELECT'):
                raise create_invalid_sql_error("Only SELECT statements are allowed")
        
        # Additional injection pattern checking
        if self._contains_sql_injection(sql):
            security_audit_logger.log_sql_injection_attempt(
                malicious_input=sql,
                detection_method="sql_pattern_matching",
                sql_statement=sql
            )
            raise create_unsafe_input_error()
        
        return sql
    
    def _decode_input(self, text: str) -> str:
        """Decode URL encoding, Unicode escapes, and other obfuscation attempts.
        
        Args:
            text: The input text to decode
            
        Returns:
            str: The decoded text
        """
        try:
            # URL decode
            decoded = urllib.parse.unquote(text)
            decoded = urllib.parse.unquote_plus(decoded)  # Handle + encoding
            
            # Unicode normalization to handle Unicode escapes
            decoded = unicodedata.normalize('NFKC', decoded)
            
            # Decode common Unicode escapes
            try:
                decoded = decoded.encode('utf-8').decode('unicode_escape')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass  # Keep original if decoding fails
            
            return decoded
        except UnicodeDecodeError:
            # Unicode decoding failed, return original
            return text
        except ValueError:
            # URL decoding failed, return original
            return text
        except Exception:
            # Any other decoding error, return original
            return text
    
    def _contains_sql_injection(self, text: str) -> bool:
        """Check if text contains SQL injection patterns using enhanced detection.
        
        Args:
            text: The text to check
            
        Returns:
            bool: True if injection patterns are detected
        """
        # Check against enhanced patterns
        for pattern in COMPILED_ENHANCED_PATTERNS:
            if pattern.search(text):
                logger.warning(f"Enhanced SQL injection pattern detected: {pattern.pattern[:50]}...")
                return True
        
        # Check for tautology-based injection
        for pattern in COMPILED_TAUTOLOGY_PATTERNS:
            if pattern.search(text):
                logger.warning(f"Tautology injection pattern detected: {pattern.pattern}")
                return True
        
        return False
    
    def _contains_unauthorized_references(self, text: str) -> bool:
        """Check for references to unauthorized tables or columns.
        
        Args:
            text: The text to check
            
        Returns:
            bool: True if unauthorized references are found
        """
        if not self.allowed_tables and not self.allowed_columns:
            return False  # No restrictions configured
        
        text_lower = text.lower()
        
        # Check for unauthorized table references - only flag actual SQL statements
        if self.allowed_tables:
            # Only check if text looks like actual SQL statement (has SQL structure indicators)
            # More restrictive criteria for SQL detection
            sql_structure_patterns = [
                r'select\s+.*\s+from\s+',  # SELECT ... FROM pattern
                r'insert\s+into\s+',       # INSERT INTO pattern  
                r'update\s+\w+\s+set',     # UPDATE table SET pattern
                r'delete\s+from\s+',       # DELETE FROM pattern
                r'alter\s+table\s+',       # ALTER TABLE pattern
                r'drop\s+table\s+'         # DROP TABLE pattern
            ]
            
            is_sql_statement = any(re.search(pattern, text_lower) for pattern in sql_structure_patterns)
            
            if is_sql_statement:
                # Look for explicit table references in SQL context
                sql_table_patterns = [
                    r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:where|join|group|order|limit|;|\s*$)',
                    r'join\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:on|where|group|order|limit|;|\s*$)', 
                    r'update\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*set',
                    r'insert\s+into\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|values)'
                ]
                
                for pattern in sql_table_patterns:
                    matches = re.findall(pattern, text_lower)
                    for table in matches:
                        if table not in self.allowed_tables:
                            logger.warning(f"Unauthorized table reference detected: {table}")
                            return True
        
        return False
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check for suspicious patterns that may indicate injection attempts.
        
        Args:
            text: The text to check
            
        Returns:
            bool: True if suspicious patterns are found
        """
        text_lower = text.lower()
        
        # Check for suspicious function usage
        for func in SUSPICIOUS_FUNCTIONS:
            if f'{func}(' in text_lower:
                logger.warning(f"Suspicious function detected: {func}")
                return True
        
        # Check for information schema references
        info_schema_patterns = [
            'information_schema', 'sys.', 'mysql.', 'pg_catalog',
            'syscolumns', 'sysobjects', 'systables'
        ]
        
        for pattern in info_schema_patterns:
            if pattern in text_lower:
                logger.warning(f"Information schema reference detected: {pattern}")
                return True
        
        # Check for suspicious keywords combinations
        suspicious_combinations = [
            ('union', 'select'),
            ('order', 'by', 'sleep'),
            ('where', '1=1'),
            ('or', 'exists'),
            ('and', 'ascii'),
            ('convert', 'select'),
            ('cast', 'select')
        ]
        
        for combo in suspicious_combinations:
            if all(word in text_lower for word in combo):
                logger.warning(f"Suspicious keyword combination detected: {' + '.join(combo)}")
                return True
        
        return False
    
    def _is_select_statement(self, statement) -> bool:
        """Check if the parsed statement is a SELECT statement.
        
        Args:
            statement: Parsed SQL statement from sqlparse
            
        Returns:
            bool: True if it's a SELECT statement
        """
        for token in statement.tokens:
            if token.ttype is Keyword and token.normalized == 'SELECT':
                return True
            elif token.ttype is DML:
                return token.normalized == 'SELECT'
        return False
    
    def _validate_sql_references(self, statement) -> None:
        """Validate table and column references in parsed SQL.
        
        Args:
            statement: Parsed SQL statement from sqlparse
            
        Raises:
            ValueError: If unauthorized references are found
        """
        if not self.allowed_tables and not self.allowed_columns:
            return  # No restrictions configured
        
        # Extract table and column references
        tables, columns = self._extract_sql_references(statement)
        
        # Validate table references
        if self.allowed_tables:
            for table in tables:
                if table not in self.allowed_tables:
                    raise create_invalid_table_error(table, list(self.allowed_tables))
        
        # Validate column references  
        if self.allowed_columns:
            for column in columns:
                if column not in self.allowed_columns:
                    raise create_unsafe_input_error()
    
    def _extract_sql_references(self, statement) -> tuple[Set[str], Set[str]]:
        """Extract table and column references from parsed SQL.
        
        Args:
            statement: Parsed SQL statement from sqlparse
            
        Returns:
            tuple: (set of table names, set of column names)
        """
        tables = set()
        columns = set()
        
        # Convert to string and use regex patterns for more reliable extraction
        sql_str = str(statement).lower()
        
        # Extract table names using regex patterns
        table_patterns = [
            r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'join\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'update\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'insert\s+into\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, sql_str)
            for match in matches:
                # Skip aliases and keywords
                if match not in ['on', 'where', 'set', 'values', 'as']:
                    tables.add(match)
        
        # For column extraction, we'll be more conservative and not extract
        # since it's complex and error-prone. Focus on table validation.
        # Column validation can be added later with more sophisticated parsing.
        
        return tables, columns
    
    def _contains_sql_suspicious_patterns(self, sql: str) -> bool:
        """Check for suspicious patterns specific to SQL statements.
        
        Args:
            sql: The SQL statement to check
            
        Returns:
            bool: True if suspicious patterns are found
        """
        sql_lower = sql.lower()
        
        # Check for always-true conditions
        always_true_patterns = [
            r'\b1\s*=\s*1\b',
            r'\b0\s*=\s*0\b',
            r"'\s*'\s*=\s*'\s*'",
            r'"\s*"\s*=\s*"\s*"'
        ]
        
        for pattern in always_true_patterns:
            if re.search(pattern, sql_lower):
                return True
        
        # Check for function-based information gathering
        info_functions = ['database()', 'version()', 'user()', 'current_user()']
        for func in info_functions:
            if func in sql_lower:
                return True
        
        return False
    
    def _check_rate_limit(self, client_id: str) -> None:
        """Check if client has exceeded rate limits.
        
        Args:
            client_id: Client identifier for rate limiting
            
        Raises:
            ValueError: If rate limit is exceeded
        """
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old attempts
        self.validation_attempts[client_id] = [
            timestamp for timestamp in self.validation_attempts[client_id]
            if timestamp > minute_ago
        ]
        
        # Check current rate
        if len(self.validation_attempts[client_id]) >= self.max_validation_attempts:
            security_audit_logger.log_rate_limit_exceeded(
                client_identifier=client_id,
                limit_type="validation_attempts_per_minute",
                current_rate=len(self.validation_attempts[client_id]),
                limit_threshold=self.max_validation_attempts,
                time_window=60
            )
            raise create_unsafe_input_error()
        
        # Record this attempt
        self.validation_attempts[client_id].append(current_time)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics and configuration.
        
        Returns:
            Dict containing validation statistics
        """
        return {
            'max_question_length': self.max_question_length,
            'allowed_tables_count': len(self.allowed_tables) if self.allowed_tables else None,
            'allowed_columns_count': len(self.allowed_columns) if self.allowed_columns else None,
            'enhanced_patterns_count': len(COMPILED_ENHANCED_PATTERNS),
            'tautology_patterns_count': len(COMPILED_TAUTOLOGY_PATTERNS),
            'suspicious_functions_count': len(SUSPICIOUS_FUNCTIONS),
            'rate_limit_per_minute': self.max_validation_attempts,
            'sqlparse_available': sqlparse is not None
        }