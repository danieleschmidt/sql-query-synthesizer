"""
Comprehensive Security Validation for Quantum SDLC Systems

Advanced security scanning, vulnerability assessment, and compliance validation
for the quantum-inspired autonomous SDLC framework.
"""

import ast
import re
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import tempfile
import os


class SecurityLevel(Enum):
    """Security assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    SQL_INJECTION = "sql_injection"
    CODE_INJECTION = "code_injection"
    PATH_TRAVERSAL = "path_traversal"
    INSECURE_RANDOM = "insecure_random"
    HARDCODED_SECRETS = "hardcoded_secrets"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    INFORMATION_DISCLOSURE = "information_disclosure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSECURE_CONFIG = "insecure_config"
    DEPENDENCY_VULNERABILITY = "dependency_vulnerability"


@dataclass
class SecurityFinding:
    """Represents a security finding"""
    vulnerability_type: VulnerabilityType
    severity: SecurityLevel
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    remediation: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityReport:
    """Comprehensive security assessment report"""
    scan_timestamp: float
    project_path: str
    total_files_scanned: int
    scan_duration: float
    findings: List[SecurityFinding]
    summary: Dict[str, Any]
    compliance_status: Dict[str, Any]
    recommendations: List[str]


class QuantumSecurityScanner:
    """
    Advanced security scanner for quantum SDLC systems
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Security patterns to detect
        self.security_patterns = self._load_security_patterns()
        
        # Files to scan
        self.scan_extensions = {'.py', '.js', '.ts', '.sql', '.yaml', '.yml', '.json', '.env'}
        self.exclude_patterns = {
            '.git', '__pycache__', 'node_modules', '.pytest_cache',
            'venv', '.venv', 'env', '.env', 'build', 'dist'
        }
        
        # Compliance frameworks
        self.compliance_frameworks = {
            'OWASP_TOP_10': self._get_owasp_top_10_rules(),
            'CWE_TOP_25': self._get_cwe_top_25_rules(),
            'NIST_CYBERSECURITY': self._get_nist_rules(),
            'GDPR': self._get_gdpr_rules()
        }
    
    def _load_security_patterns(self) -> Dict[VulnerabilityType, List[Dict[str, Any]]]:
        """Load security detection patterns"""
        
        patterns = {
            VulnerabilityType.SQL_INJECTION: [
                {
                    'pattern': r'(?i)(execute|exec|query|select|insert|update|delete).*\+.*["\']',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Potential SQL injection via string concatenation',
                    'cwe_id': 'CWE-89'
                },
                {
                    'pattern': r'(?i)\.format\([^)]*\).*(?:execute|exec|query)',
                    'severity': SecurityLevel.MEDIUM,
                    'description': 'String formatting in SQL query may allow injection',
                    'cwe_id': 'CWE-89'
                },
                {
                    'pattern': r'f["\'][^"\']*\{[^}]+\}[^"\']*["\'].*(?:execute|exec|query)',
                    'severity': SecurityLevel.MEDIUM,
                    'description': 'F-string in SQL query may allow injection',
                    'cwe_id': 'CWE-89'
                }
            ],
            
            VulnerabilityType.CODE_INJECTION: [
                {
                    'pattern': r'(?i)(eval|exec)\s*\(',
                    'severity': SecurityLevel.CRITICAL,
                    'description': 'Dynamic code execution can lead to code injection',
                    'cwe_id': 'CWE-94'
                },
                {
                    'pattern': r'(?i)subprocess\.(call|run|Popen).*shell\s*=\s*True',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Shell injection vulnerability via subprocess',
                    'cwe_id': 'CWE-78'
                },
                {
                    'pattern': r'(?i)os\.system\s*\(',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Command injection via os.system',
                    'cwe_id': 'CWE-78'
                }
            ],
            
            VulnerabilityType.PATH_TRAVERSAL: [
                {
                    'pattern': r'(?i)(open|file|read).*\+.*["\'][^"\']*\.\./[^"\']*["\']',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Path traversal vulnerability',
                    'cwe_id': 'CWE-22'
                },
                {
                    'pattern': r'(?i)os\.path\.join\([^)]*\.\.[^)]*\)',
                    'severity': SecurityLevel.MEDIUM,
                    'description': 'Potential path traversal in path construction',
                    'cwe_id': 'CWE-22'
                }
            ],
            
            VulnerabilityType.HARDCODED_SECRETS: [
                {
                    'pattern': r'(?i)(password|passwd|pwd)\s*[:=]\s*["\'][^"\']{6,}["\']',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Hardcoded password detected',
                    'cwe_id': 'CWE-798'
                },
                {
                    'pattern': r'(?i)(api[_-]?key|apikey|secret[_-]?key)\s*[:=]\s*["\'][^"\']{10,}["\']',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Hardcoded API key or secret detected',
                    'cwe_id': 'CWE-798'
                },
                {
                    'pattern': r'(?i)(token|auth[_-]?token)\s*[:=]\s*["\'][^"\']{20,}["\']',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Hardcoded authentication token detected',
                    'cwe_id': 'CWE-798'
                },
                {
                    'pattern': r'(?i)(private[_-]?key|priv[_-]?key)\s*[:=]\s*["\'][^"\']{50,}["\']',
                    'severity': SecurityLevel.CRITICAL,
                    'description': 'Hardcoded private key detected',
                    'cwe_id': 'CWE-798'
                }
            ],
            
            VulnerabilityType.INSECURE_RANDOM: [
                {
                    'pattern': r'(?i)random\.random\(\)',
                    'severity': SecurityLevel.MEDIUM,
                    'description': 'Insecure random number generation for security purposes',
                    'cwe_id': 'CWE-338'
                },
                {
                    'pattern': r'(?i)random\.choice\(',
                    'severity': SecurityLevel.MEDIUM,
                    'description': 'Potentially insecure random choice for security purposes',
                    'cwe_id': 'CWE-338'
                }
            ],
            
            VulnerabilityType.WEAK_CRYPTOGRAPHY: [
                {
                    'pattern': r'(?i)hashlib\.(md5|sha1)\(',
                    'severity': SecurityLevel.MEDIUM,
                    'description': 'Weak cryptographic hash function',
                    'cwe_id': 'CWE-327'
                },
                {
                    'pattern': r'(?i)DES|3DES|RC4',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Weak encryption algorithm',
                    'cwe_id': 'CWE-327'
                }
            ],
            
            VulnerabilityType.UNSAFE_DESERIALIZATION: [
                {
                    'pattern': r'(?i)pickle\.loads?\(',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Unsafe deserialization with pickle',
                    'cwe_id': 'CWE-502'
                },
                {
                    'pattern': r'(?i)yaml\.load\([^,)]*\)',
                    'severity': SecurityLevel.HIGH,
                    'description': 'Unsafe YAML deserialization',
                    'cwe_id': 'CWE-502'
                }
            ],
            
            VulnerabilityType.INFORMATION_DISCLOSURE: [
                {
                    'pattern': r'(?i)(print|log|debug).*(?:password|secret|key|token)',
                    'severity': SecurityLevel.MEDIUM,
                    'description': 'Sensitive information may be logged or printed',
                    'cwe_id': 'CWE-200'
                },
                {
                    'pattern': r'(?i)traceback\.print_exc\(\)',
                    'severity': SecurityLevel.LOW,
                    'description': 'Stack trace may leak sensitive information',
                    'cwe_id': 'CWE-209'
                }
            ]
        }
        
        return patterns
    
    def _get_owasp_top_10_rules(self) -> List[Dict[str, Any]]:
        """OWASP Top 10 compliance rules"""
        return [
            {'id': 'A01', 'name': 'Broken Access Control', 'severity': SecurityLevel.HIGH},
            {'id': 'A02', 'name': 'Cryptographic Failures', 'severity': SecurityLevel.HIGH},
            {'id': 'A03', 'name': 'Injection', 'severity': SecurityLevel.HIGH},
            {'id': 'A04', 'name': 'Insecure Design', 'severity': SecurityLevel.MEDIUM},
            {'id': 'A05', 'name': 'Security Misconfiguration', 'severity': SecurityLevel.MEDIUM},
            {'id': 'A06', 'name': 'Vulnerable Components', 'severity': SecurityLevel.HIGH},
            {'id': 'A07', 'name': 'Identification and Authentication Failures', 'severity': SecurityLevel.HIGH},
            {'id': 'A08', 'name': 'Software and Data Integrity Failures', 'severity': SecurityLevel.HIGH},
            {'id': 'A09', 'name': 'Security Logging and Monitoring Failures', 'severity': SecurityLevel.MEDIUM},
            {'id': 'A10', 'name': 'Server-Side Request Forgery (SSRF)', 'severity': SecurityLevel.MEDIUM}
        ]
    
    def _get_cwe_top_25_rules(self) -> List[Dict[str, Any]]:
        """CWE Top 25 most dangerous software errors"""
        return [
            {'id': 'CWE-79', 'name': 'Cross-site Scripting', 'severity': SecurityLevel.HIGH},
            {'id': 'CWE-89', 'name': 'SQL Injection', 'severity': SecurityLevel.HIGH},
            {'id': 'CWE-20', 'name': 'Improper Input Validation', 'severity': SecurityLevel.HIGH},
            {'id': 'CWE-125', 'name': 'Out-of-bounds Read', 'severity': SecurityLevel.HIGH},
            {'id': 'CWE-78', 'name': 'OS Command Injection', 'severity': SecurityLevel.HIGH},
            {'id': 'CWE-787', 'name': 'Out-of-bounds Write', 'severity': SecurityLevel.CRITICAL},
            {'id': 'CWE-22', 'name': 'Path Traversal', 'severity': SecurityLevel.HIGH},
            {'id': 'CWE-352', 'name': 'Cross-Site Request Forgery', 'severity': SecurityLevel.MEDIUM},
            {'id': 'CWE-434', 'name': 'Unrestricted Upload', 'severity': SecurityLevel.HIGH},
            {'id': 'CWE-94', 'name': 'Code Injection', 'severity': SecurityLevel.CRITICAL}
        ]
    
    def _get_nist_rules(self) -> List[Dict[str, Any]]:
        """NIST Cybersecurity Framework compliance rules"""
        return [
            {'id': 'ID.AM', 'name': 'Asset Management', 'severity': SecurityLevel.MEDIUM},
            {'id': 'PR.AC', 'name': 'Access Control', 'severity': SecurityLevel.HIGH},
            {'id': 'PR.DS', 'name': 'Data Security', 'severity': SecurityLevel.HIGH},
            {'id': 'DE.AE', 'name': 'Anomalies and Events', 'severity': SecurityLevel.MEDIUM},
            {'id': 'RS.RP', 'name': 'Response Planning', 'severity': SecurityLevel.MEDIUM},
            {'id': 'RC.RP', 'name': 'Recovery Planning', 'severity': SecurityLevel.MEDIUM}
        ]
    
    def _get_gdpr_rules(self) -> List[Dict[str, Any]]:
        """GDPR compliance rules"""
        return [
            {'id': 'Art.25', 'name': 'Data Protection by Design', 'severity': SecurityLevel.HIGH},
            {'id': 'Art.32', 'name': 'Security of Processing', 'severity': SecurityLevel.HIGH},
            {'id': 'Art.33', 'name': 'Data Breach Notification', 'severity': SecurityLevel.MEDIUM},
            {'id': 'Art.35', 'name': 'Data Protection Impact Assessment', 'severity': SecurityLevel.MEDIUM}
        ]
    
    async def scan_project(self, project_path: Path) -> SecurityReport:
        """Perform comprehensive security scan of project"""
        
        start_time = time.time()
        self.logger.info(f"🔍 Starting security scan of {project_path}")
        
        findings: List[SecurityFinding] = []
        files_scanned = 0
        
        # Scan files
        for file_path in self._get_scannable_files(project_path):
            try:
                file_findings = await self._scan_file(file_path)
                findings.extend(file_findings)
                files_scanned += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to scan {file_path}: {str(e)}")
        
        # Dependency scan
        dependency_findings = await self._scan_dependencies(project_path)
        findings.extend(dependency_findings)
        
        # Configuration scan
        config_findings = await self._scan_configurations(project_path)
        findings.extend(config_findings)
        
        scan_duration = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(findings)
        
        # Check compliance
        compliance_status = self._check_compliance(findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings, summary)
        
        report = SecurityReport(
            scan_timestamp=start_time,
            project_path=str(project_path),
            total_files_scanned=files_scanned,
            scan_duration=scan_duration,
            findings=findings,
            summary=summary,
            compliance_status=compliance_status,
            recommendations=recommendations
        )
        
        self.logger.info(
            f"✅ Security scan completed: {len(findings)} findings in "
            f"{scan_duration:.2f}s ({files_scanned} files scanned)"
        )
        
        return report
    
    def _get_scannable_files(self, project_path: Path) -> List[Path]:
        """Get list of files to scan"""
        
        scannable_files = []
        
        for file_path in project_path.rglob('*'):
            if file_path.is_file():
                # Check extension
                if file_path.suffix in self.scan_extensions:
                    # Check if file should be excluded
                    if not any(exclude in str(file_path) for exclude in self.exclude_patterns):
                        scannable_files.append(file_path)
        
        return scannable_files
    
    async def _scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan individual file for security issues"""
        
        findings = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            # Pattern-based scanning
            for vuln_type, patterns in self.security_patterns.items():
                for pattern_info in patterns:
                    pattern = pattern_info['pattern']
                    
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            finding = SecurityFinding(
                                vulnerability_type=vuln_type,
                                severity=pattern_info['severity'],
                                title=f"{vuln_type.value.replace('_', ' ').title()} Detected",
                                description=pattern_info['description'],
                                file_path=str(file_path),
                                line_number=line_num,
                                code_snippet=line.strip(),
                                remediation=self._get_remediation(vuln_type),
                                cwe_id=pattern_info.get('cwe_id'),
                                confidence=0.8,
                                metadata={'pattern': pattern}
                            )
                            findings.append(finding)
            
            # AST-based analysis for Python files
            if file_path.suffix == '.py':
                ast_findings = await self._ast_analysis(file_path, content)
                findings.extend(ast_findings)
            
        except Exception as e:
            self.logger.warning(f"Error scanning {file_path}: {str(e)}")
        
        return findings
    
    async def _ast_analysis(self, file_path: Path, content: str) -> List[SecurityFinding]:
        """Perform AST-based analysis for Python files"""
        
        findings = []
        
        try:
            tree = ast.parse(content)
            
            # Check for dangerous function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check function name
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        # Dangerous functions
                        if func_name in ['eval', 'exec']:
                            finding = SecurityFinding(
                                vulnerability_type=VulnerabilityType.CODE_INJECTION,
                                severity=SecurityLevel.CRITICAL,
                                title="Dangerous Function Call",
                                description=f"Use of {func_name}() can lead to code injection",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=self._get_node_source(node, content),
                                remediation="Avoid using eval() and exec(). Consider safer alternatives.",
                                cwe_id="CWE-94",
                                confidence=0.9
                            )
                            findings.append(finding)
                    
                    # Check method calls
                    elif isinstance(node.func, ast.Attribute):
                        attr_name = node.func.attr
                        
                        # Dangerous methods
                        if attr_name in ['system'] and isinstance(node.func.value, ast.Name):
                            if node.func.value.id == 'os':
                                finding = SecurityFinding(
                                    vulnerability_type=VulnerabilityType.CODE_INJECTION,
                                    severity=SecurityLevel.HIGH,
                                    title="Command Injection Risk",
                                    description="os.system() can lead to command injection",
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    code_snippet=self._get_node_source(node, content),
                                    remediation="Use subprocess with proper argument handling",
                                    cwe_id="CWE-78",
                                    confidence=0.9
                                )
                                findings.append(finding)
                
                # Check for hardcoded strings that look like secrets
                elif isinstance(node, ast.Str):
                    if len(node.s) > 20 and self._looks_like_secret(node.s):
                        finding = SecurityFinding(
                            vulnerability_type=VulnerabilityType.HARDCODED_SECRETS,
                            severity=SecurityLevel.MEDIUM,
                            title="Potential Hardcoded Secret",
                            description="String value looks like a hardcoded secret",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            code_snippet=self._get_node_source(node, content),
                            remediation="Use environment variables or secure key management",
                            cwe_id="CWE-798",
                            confidence=0.6
                        )
                        findings.append(finding)
        
        except SyntaxError:
            # Invalid Python syntax, skip AST analysis
            pass
        except Exception as e:
            self.logger.warning(f"AST analysis failed for {file_path}: {str(e)}")
        
        return findings
    
    def _get_node_source(self, node: ast.AST, content: str) -> str:
        """Get source code for AST node"""
        lines = content.split('\n')
        if hasattr(node, 'lineno') and node.lineno <= len(lines):
            return lines[node.lineno - 1].strip()
        return ""
    
    def _looks_like_secret(self, value: str) -> bool:
        """Check if string looks like a secret/key/token"""
        
        # Check for high entropy (randomness)
        unique_chars = len(set(value.lower()))
        entropy = unique_chars / len(value) if value else 0
        
        if entropy > 0.7:  # High entropy strings
            return True
        
        # Check for common secret patterns
        secret_patterns = [
            r'^[A-Za-z0-9+/]{40,}={0,2}$',  # Base64-like
            r'^[A-Fa-f0-9]{32,}$',          # Hex strings
            r'^[A-Za-z0-9_-]{32,}$',        # API keys
        ]
        
        return any(re.match(pattern, value) for pattern in secret_patterns)
    
    async def _scan_dependencies(self, project_path: Path) -> List[SecurityFinding]:
        """Scan dependencies for known vulnerabilities"""
        
        findings = []
        
        # Check Python requirements
        requirements_files = [
            project_path / 'requirements.txt',
            project_path / 'requirements-dev.txt',
            project_path / 'pyproject.toml'
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                try:
                    # Parse dependencies
                    deps = self._parse_dependencies(req_file)
                    
                    # Check each dependency (simplified check)
                    for dep_name, dep_version in deps.items():
                        if self._is_vulnerable_dependency(dep_name, dep_version):
                            finding = SecurityFinding(
                                vulnerability_type=VulnerabilityType.DEPENDENCY_VULNERABILITY,
                                severity=SecurityLevel.HIGH,
                                title=f"Vulnerable Dependency: {dep_name}",
                                description=f"Dependency {dep_name} version {dep_version} has known vulnerabilities",
                                file_path=str(req_file),
                                line_number=1,
                                code_snippet=f"{dep_name}=={dep_version}",
                                remediation=f"Update {dep_name} to the latest secure version",
                                cwe_id="CWE-1104",
                                confidence=0.7
                            )
                            findings.append(finding)
                
                except Exception as e:
                    self.logger.warning(f"Failed to scan dependencies in {req_file}: {str(e)}")
        
        return findings
    
    def _parse_dependencies(self, file_path: Path) -> Dict[str, str]:
        """Parse dependencies from file"""
        
        deps = {}
        
        try:
            content = file_path.read_text()
            
            if file_path.name == 'pyproject.toml':
                # Basic TOML parsing for dependencies
                lines = content.split('\n')
                in_dependencies = False
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('[project') and 'dependencies' in line:
                        in_dependencies = True
                        continue
                    elif line.startswith('[') and in_dependencies:
                        in_dependencies = False
                        continue
                    
                    if in_dependencies and '=' in line:
                        # Extract dependency name and version
                        match = re.match(r'["\']([^"\']+)["\']', line)
                        if match:
                            dep_spec = match.group(1)
                            if '==' in dep_spec:
                                name, version = dep_spec.split('==', 1)
                                deps[name] = version
                            elif '>=' in dep_spec:
                                name, version = dep_spec.split('>=', 1)
                                deps[name] = f">={version}"
            
            else:
                # Parse requirements.txt format
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '==' in line:
                            name, version = line.split('==', 1)
                            deps[name.strip()] = version.strip()
                        elif '>=' in line:
                            name, version = line.split('>=', 1)
                            deps[name.strip()] = f">={version.strip()}"
        
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path}: {str(e)}")
        
        return deps
    
    def _is_vulnerable_dependency(self, name: str, version: str) -> bool:
        """Check if dependency has known vulnerabilities (simplified)"""
        
        # Known vulnerable packages (simplified list)
        known_vulnerabilities = {
            'pillow': ['<8.3.2'],
            'django': ['<3.2.15', '<4.0.7'],
            'flask': ['<2.0.3'],
            'requests': ['<2.20.0'],
            'pyyaml': ['<5.4'],
            'jinja2': ['<2.11.3'],
            'urllib3': ['<1.26.5']
        }
        
        if name.lower() in known_vulnerabilities:
            vulnerable_versions = known_vulnerabilities[name.lower()]
            
            # Simple version comparison (in real implementation, use proper version parsing)
            for vuln_version in vulnerable_versions:
                if version.startswith(vuln_version.replace('<', '').replace('>=', '')):
                    return True
        
        return False
    
    async def _scan_configurations(self, project_path: Path) -> List[SecurityFinding]:
        """Scan configuration files for security issues"""
        
        findings = []
        
        # Configuration files to check
        config_files = [
            project_path / '.env',
            project_path / '.env.example',
            project_path / 'config.yml',
            project_path / 'config.yaml',
            project_path / 'docker-compose.yml',
            project_path / 'docker-compose.yaml'
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    content = config_file.read_text()
                    
                    # Check for insecure configurations
                    insecure_patterns = [
                        (r'(?i)debug\s*[:=]\s*true', 'Debug mode enabled in production'),
                        (r'(?i)ssl\s*[:=]\s*false', 'SSL/TLS disabled'),
                        (r'(?i)verify_ssl\s*[:=]\s*false', 'SSL verification disabled'),
                        (r'(?i)secure\s*[:=]\s*false', 'Security feature disabled'),
                        (r'(?i)host\s*[:=]\s*["\']0\.0\.0\.0["\']', 'Service bound to all interfaces')
                    ]
                    
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        for pattern, description in insecure_patterns:
                            if re.search(pattern, line):
                                finding = SecurityFinding(
                                    vulnerability_type=VulnerabilityType.INSECURE_CONFIG,
                                    severity=SecurityLevel.MEDIUM,
                                    title="Insecure Configuration",
                                    description=description,
                                    file_path=str(config_file),
                                    line_number=line_num,
                                    code_snippet=line.strip(),
                                    remediation="Review and secure configuration settings",
                                    cwe_id="CWE-16",
                                    confidence=0.8
                                )
                                findings.append(finding)
                
                except Exception as e:
                    self.logger.warning(f"Failed to scan {config_file}: {str(e)}")
        
        return findings
    
    def _get_remediation(self, vuln_type: VulnerabilityType) -> str:
        """Get remediation advice for vulnerability type"""
        
        remediation_map = {
            VulnerabilityType.SQL_INJECTION: (
                "Use parameterized queries or ORM methods. "
                "Avoid string concatenation in SQL queries."
            ),
            VulnerabilityType.CODE_INJECTION: (
                "Avoid eval() and exec(). Use subprocess with proper argument handling. "
                "Validate and sanitize all user input."
            ),
            VulnerabilityType.PATH_TRAVERSAL: (
                "Validate file paths. Use os.path.join() safely. "
                "Implement proper access controls."
            ),
            VulnerabilityType.HARDCODED_SECRETS: (
                "Move secrets to environment variables or secure key management systems. "
                "Never commit secrets to version control."
            ),
            VulnerabilityType.INSECURE_RANDOM: (
                "Use cryptographically secure random number generators "
                "like secrets module for security purposes."
            ),
            VulnerabilityType.WEAK_CRYPTOGRAPHY: (
                "Use strong cryptographic algorithms like SHA-256, AES-256. "
                "Avoid MD5, SHA-1, DES, and RC4."
            ),
            VulnerabilityType.UNSAFE_DESERIALIZATION: (
                "Use safe serialization formats like JSON. "
                "Validate and sanitize deserialized data."
            ),
            VulnerabilityType.INFORMATION_DISCLOSURE: (
                "Avoid logging sensitive information. "
                "Implement proper error handling without exposing internals."
            ),
            VulnerabilityType.INSECURE_CONFIG: (
                "Review configuration settings. "
                "Disable debug mode in production. Enable security features."
            ),
            VulnerabilityType.DEPENDENCY_VULNERABILITY: (
                "Update dependencies to secure versions. "
                "Monitor dependencies for known vulnerabilities."
            )
        }
        
        return remediation_map.get(vuln_type, "Review code for security issues and follow secure coding practices.")
    
    def _generate_summary(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Generate security summary"""
        
        summary = {
            'total_findings': len(findings),
            'by_severity': {
                SecurityLevel.CRITICAL.value: 0,
                SecurityLevel.HIGH.value: 0,
                SecurityLevel.MEDIUM.value: 0,
                SecurityLevel.LOW.value: 0
            },
            'by_type': {},
            'risk_score': 0.0,
            'top_issues': []
        }
        
        # Count by severity
        for finding in findings:
            summary['by_severity'][finding.severity.value] += 1
        
        # Count by type
        for finding in findings:
            vuln_type = finding.vulnerability_type.value
            summary['by_type'][vuln_type] = summary['by_type'].get(vuln_type, 0) + 1
        
        # Calculate risk score (0-100)
        severity_weights = {
            SecurityLevel.CRITICAL: 10,
            SecurityLevel.HIGH: 5,
            SecurityLevel.MEDIUM: 2,
            SecurityLevel.LOW: 1
        }
        
        total_weight = sum(
            summary['by_severity'][sev.value] * weight
            for sev, weight in severity_weights.items()
        )
        
        # Normalize to 0-100 scale
        summary['risk_score'] = min(100.0, total_weight)
        
        # Top issues
        sorted_findings = sorted(findings, key=lambda f: (
            severity_weights[f.severity], f.confidence
        ), reverse=True)
        
        summary['top_issues'] = [
            {
                'type': f.vulnerability_type.value,
                'severity': f.severity.value,
                'title': f.title,
                'file': f.file_path,
                'line': f.line_number
            }
            for f in sorted_findings[:10]
        ]
        
        return summary
    
    def _check_compliance(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Check compliance with security frameworks"""
        
        compliance = {}
        
        for framework_name, rules in self.compliance_frameworks.items():
            framework_compliance = {
                'score': 100.0,
                'violations': [],
                'status': 'COMPLIANT'
            }
            
            # Check for violations (simplified)
            violations = []
            
            for finding in findings:
                if finding.cwe_id:
                    # Check if this CWE relates to any framework rules
                    for rule in rules:
                        if (finding.severity == SecurityLevel.CRITICAL or 
                            finding.severity == SecurityLevel.HIGH):
                            violations.append({
                                'rule_id': rule['id'],
                                'rule_name': rule['name'],
                                'violation': finding.title,
                                'severity': finding.severity.value,
                                'file': finding.file_path,
                                'line': finding.line_number
                            })
            
            framework_compliance['violations'] = violations[:10]  # Top 10
            
            # Calculate compliance score
            if violations:
                violation_penalty = len(violations) * 5  # 5 points per violation
                framework_compliance['score'] = max(0.0, 100.0 - violation_penalty)
                
                if framework_compliance['score'] < 70.0:
                    framework_compliance['status'] = 'NON_COMPLIANT'
                elif framework_compliance['score'] < 90.0:
                    framework_compliance['status'] = 'PARTIALLY_COMPLIANT'
            
            compliance[framework_name] = framework_compliance
        
        return compliance
    
    def _generate_recommendations(self, findings: List[SecurityFinding], 
                                summary: Dict[str, Any]) -> List[str]:
        """Generate security recommendations"""
        
        recommendations = []
        
        # High-level recommendations based on findings
        if summary['by_severity'][SecurityLevel.CRITICAL.value] > 0:
            recommendations.append(
                f"🚨 CRITICAL: Address {summary['by_severity'][SecurityLevel.CRITICAL.value]} "
                "critical security vulnerabilities immediately."
            )
        
        if summary['by_severity'][SecurityLevel.HIGH.value] > 5:
            recommendations.append(
                f"⚠️ HIGH PRIORITY: {summary['by_severity'][SecurityLevel.HIGH.value]} "
                "high-severity issues require immediate attention."
            )
        
        # Type-specific recommendations
        type_counts = summary['by_type']
        
        if type_counts.get('sql_injection', 0) > 0:
            recommendations.append(
                "🛡️ Implement parameterized queries to prevent SQL injection attacks."
            )
        
        if type_counts.get('hardcoded_secrets', 0) > 0:
            recommendations.append(
                "🔐 Move hardcoded secrets to environment variables or secure key management."
            )
        
        if type_counts.get('code_injection', 0) > 0:
            recommendations.append(
                "⚡ Eliminate use of eval() and exec() functions to prevent code injection."
            )
        
        if type_counts.get('weak_cryptography', 0) > 0:
            recommendations.append(
                "🔒 Upgrade to stronger cryptographic algorithms (SHA-256, AES-256)."
            )
        
        if type_counts.get('dependency_vulnerability', 0) > 0:
            recommendations.append(
                "📦 Update vulnerable dependencies to secure versions."
            )
        
        # Risk score recommendations
        if summary['risk_score'] > 80:
            recommendations.append(
                "🔴 URGENT: Risk score is very high. Implement comprehensive security review."
            )
        elif summary['risk_score'] > 50:
            recommendations.append(
                "🟡 MODERATE RISK: Consider security hardening and regular security reviews."
            )
        
        # General recommendations
        recommendations.extend([
            "✅ Implement automated security scanning in CI/CD pipeline.",
            "📚 Provide security training for development team.",
            "🔄 Establish regular security review processes.",
            "📋 Implement security logging and monitoring.",
            "🎯 Consider penetration testing for critical applications."
        ])
        
        return recommendations
    
    def export_report(self, report: SecurityReport, output_path: Path) -> Path:
        """Export security report to JSON file"""
        
        # Convert report to serializable format
        report_dict = {
            'scan_timestamp': report.scan_timestamp,
            'project_path': report.project_path,
            'total_files_scanned': report.total_files_scanned,
            'scan_duration': report.scan_duration,
            'findings': [
                {
                    'vulnerability_type': finding.vulnerability_type.value,
                    'severity': finding.severity.value,
                    'title': finding.title,
                    'description': finding.description,
                    'file_path': finding.file_path,
                    'line_number': finding.line_number,
                    'code_snippet': finding.code_snippet,
                    'remediation': finding.remediation,
                    'cwe_id': finding.cwe_id,
                    'cvss_score': finding.cvss_score,
                    'confidence': finding.confidence,
                    'metadata': finding.metadata
                }
                for finding in report.findings
            ],
            'summary': report.summary,
            'compliance_status': report.compliance_status,
            'recommendations': report.recommendations,
            'metadata': {
                'scanner_version': '1.0.0',
                'export_time': time.time()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"📋 Security report exported to: {output_path}")
        
        return output_path
    
    def print_summary(self, report: SecurityReport):
        """Print security report summary to console"""
        
        print("\n" + "="*80)
        print("🔒 SECURITY SCAN REPORT")
        print("="*80)
        
        print(f"Project: {report.project_path}")
        print(f"Scan Duration: {report.scan_duration:.2f}s")
        print(f"Files Scanned: {report.total_files_scanned}")
        print(f"Total Findings: {report.summary['total_findings']}")
        print(f"Risk Score: {report.summary['risk_score']:.1f}/100")
        
        print("\n📊 FINDINGS BY SEVERITY:")
        for severity, count in report.summary['by_severity'].items():
            if count > 0:
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")
                print(f"  {icon} {severity.upper()}: {count}")
        
        print("\n🔍 FINDINGS BY TYPE:")
        for vuln_type, count in report.summary['by_type'].items():
            if count > 0:
                print(f"  • {vuln_type.replace('_', ' ').title()}: {count}")
        
        if report.summary['top_issues']:
            print("\n⚠️ TOP ISSUES:")
            for i, issue in enumerate(report.summary['top_issues'][:5], 1):
                print(f"  {i}. {issue['title']} ({issue['severity'].upper()})")
                print(f"     {issue['file']}:{issue['line']}")
        
        print("\n✅ COMPLIANCE STATUS:")
        for framework, status in report.compliance_status.items():
            score = status['score']
            status_icon = "✅" if score >= 90 else "⚠️" if score >= 70 else "❌"
            print(f"  {status_icon} {framework}: {score:.1f}% ({status['status']})")
        
        if report.recommendations:
            print("\n📋 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


async def main():
    """Main security validation function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("SecurityValidation")
    
    # Initialize scanner
    scanner = QuantumSecurityScanner(logger)
    
    # Scan current project
    project_path = Path.cwd()
    report = await scanner.scan_project(project_path)
    
    # Print summary
    scanner.print_summary(report)
    
    # Export detailed report
    report_path = project_path / "security_report.json"
    scanner.export_report(report, report_path)
    
    return report


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())