#!/usr/bin/env python3
"""
Repository health monitoring script for SQL Query Synthesizer.

This script monitors various aspects of repository health including:
- Branch protection status
- Security configuration
- CI/CD pipeline health
- Documentation completeness
- Code quality metrics
"""

import json
import os
import sys
import subprocess
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

class RepositoryHealthChecker:
    """Monitors and reports on repository health metrics."""
    
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path or ".")
        self.repo_name = "danieleschmidt/sql-query-synthesizer"
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.health_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_score": 0,
            "categories": {},
            "recommendations": []
        }
    
    def check_branch_protection(self) -> Dict[str, Any]:
        """Check branch protection configuration."""
        if not self.github_token:
            return {"status": "skipped", "reason": "No GitHub token available"}
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            response = requests.get(
                f"https://api.github.com/repos/{self.repo_name}/branches/main/protection",
                headers=headers
            )
            
            if response.status_code == 200:
                protection = response.json()
                
                checks = {
                    "require_pull_request_reviews": bool(protection.get("required_pull_request_reviews")),
                    "require_status_checks": bool(protection.get("required_status_checks")),
                    "enforce_admins": bool(protection.get("enforce_admins")),
                    "restrict_pushes": bool(protection.get("restrictions")),
                    "require_signed_commits": bool(protection.get("required_signatures"))
                }
                
                score = sum(checks.values()) / len(checks) * 100
                
                return {
                    "status": "healthy" if score >= 80 else "warning",
                    "score": score,
                    "checks": checks,
                    "details": protection
                }
            elif response.status_code == 404:
                return {
                    "status": "unhealthy",
                    "score": 0,
                    "message": "Branch protection not configured"
                }
            else:
                return {
                    "status": "error",
                    "message": f"API error: {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check branch protection: {e}"
            }
    
    def check_security_configuration(self) -> Dict[str, Any]:
        """Check repository security configuration."""
        if not self.github_token:
            return {"status": "skipped", "reason": "No GitHub token available"}
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        checks = {}
        
        try:
            # Check if security features are enabled
            response = requests.get(
                f"https://api.github.com/repos/{self.repo_name}",
                headers=headers
            )
            
            if response.status_code == 200:
                repo_data = response.json()
                checks["security_and_analysis"] = bool(repo_data.get("security_and_analysis"))
            
            # Check for security policy
            security_policy_path = self.repo_path / "SECURITY.md"
            checks["security_policy"] = security_policy_path.exists()
            
            # Check for dependabot configuration
            dependabot_config = self.repo_path / ".github" / "dependabot.yml"
            checks["dependabot_config"] = dependabot_config.exists()
            
            # Check for GitHub Actions security workflows
            workflows_dir = self.repo_path / ".github" / "workflows"
            security_workflows = []
            if workflows_dir.exists():
                for workflow_file in workflows_dir.glob("*.yml"):
                    with open(workflow_file, 'r') as f:
                        content = f.read().lower()
                        if any(keyword in content for keyword in ["security", "bandit", "safety", "trivy"]):
                            security_workflows.append(workflow_file.name)
            
            checks["security_workflows"] = len(security_workflows) > 0
            checks["security_workflow_count"] = len(security_workflows)
            
            score = sum(1 for v in checks.values() if isinstance(v, bool) and v) / \
                   sum(1 for v in checks.values() if isinstance(v, bool)) * 100
            
            return {
                "status": "healthy" if score >= 75 else "warning",
                "score": score,
                "checks": checks
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check security configuration: {e}"
            }
    
    def check_ci_cd_health(self) -> Dict[str, Any]:
        """Check CI/CD pipeline health."""
        if not self.github_token:
            return {"status": "skipped", "reason": "No GitHub token available"}
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            # Get recent workflow runs
            response = requests.get(
                f"https://api.github.com/repos/{self.repo_name}/actions/runs?per_page=50",
                headers=headers
            )
            
            if response.status_code == 200:
                runs_data = response.json()
                runs = runs_data.get("workflow_runs", [])
                
                if not runs:
                    return {
                        "status": "warning",
                        "score": 50,
                        "message": "No recent workflow runs found"
                    }
                
                # Calculate success rate
                total_runs = len(runs)
                successful_runs = sum(1 for run in runs if run["conclusion"] == "success")
                success_rate = (successful_runs / total_runs) * 100 if total_runs > 0 else 0
                
                # Check for different workflow types
                workflow_types = set(run["name"] for run in runs)
                
                checks = {
                    "has_ci_workflows": len(workflow_types) > 0,
                    "success_rate_good": success_rate >= 90,
                    "recent_activity": len([r for r in runs if r["created_at"] > "2025-07-26"]) > 0,
                    "multiple_workflows": len(workflow_types) >= 2
                }
                
                return {
                    "status": "healthy" if success_rate >= 90 else "warning",
                    "score": success_rate,
                    "success_rate": success_rate,
                    "total_runs": total_runs,
                    "successful_runs": successful_runs,
                    "workflow_types": list(workflow_types),
                    "checks": checks
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to fetch workflow runs: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check CI/CD health: {e}"
            }
    
    def check_documentation_completeness(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        required_docs = [
            "README.md",
            "LICENSE",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "CHANGELOG.md"
        ]
        
        optional_docs = [
            "docs/API.md",
            "docs/DEVELOPMENT.md",
            "docs/DEPLOYMENT.md",
            "docs/ARCHITECTURE.md",
            "PROJECT_CHARTER.md"
        ]
        
        checks = {}
        
        # Check required documentation
        for doc in required_docs:
            doc_path = self.repo_path / doc
            checks[f"required_{doc.lower().replace('.', '_')}"] = doc_path.exists()
        
        # Check optional documentation
        for doc in optional_docs:
            doc_path = self.repo_path / doc
            checks[f"optional_{doc.lower().replace('/', '_').replace('.', '_')}"] = doc_path.exists()
        
        # Check docs directory structure
        docs_dir = self.repo_path / "docs"
        checks["has_docs_directory"] = docs_dir.exists()
        
        if docs_dir.exists():
            md_files = list(docs_dir.glob("**/*.md"))
            checks["docs_file_count"] = len(md_files)
            checks["comprehensive_docs"] = len(md_files) >= 10
        
        # Calculate score
        required_score = sum(1 for k, v in checks.items() if k.startswith("required_") and v) / \
                        sum(1 for k in checks.keys() if k.startswith("required_")) * 100
        
        optional_score = sum(1 for k, v in checks.items() if k.startswith("optional_") and v) / \
                        sum(1 for k in checks.keys() if k.startswith("optional_")) * 100 if \
                        any(k.startswith("optional_") for k in checks.keys()) else 0
        
        overall_score = (required_score * 0.7) + (optional_score * 0.3)
        
        return {
            "status": "healthy" if overall_score >= 80 else "warning",
            "score": overall_score,
            "required_score": required_score,
            "optional_score": optional_score,
            "checks": checks
        }
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        checks = {}
        
        # Check for quality configuration files
        quality_files = [
            ".pre-commit-config.yaml",
            "pyproject.toml",
            ".editorconfig",
            ".gitignore"
        ]
        
        for file in quality_files:
            file_path = self.repo_path / file
            checks[f"has_{file.replace('.', '_').replace('-', '_')}"] = file_path.exists()
        
        # Check test coverage
        try:
            result = subprocess.run(
                ["coverage", "report", "--format=total"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            if result.returncode == 0:
                coverage = int(result.stdout.strip())
                checks["test_coverage"] = coverage
                checks["good_coverage"] = coverage >= 80
        except Exception:
            checks["test_coverage"] = None
            checks["good_coverage"] = False
        
        # Check for test directory
        test_dirs = ["tests", "test"]
        checks["has_tests"] = any((self.repo_path / test_dir).exists() for test_dir in test_dirs)
        
        # Count test files
        test_files = []
        for test_dir in test_dirs:
            test_path = self.repo_path / test_dir
            if test_path.exists():
                test_files.extend(list(test_path.glob("**/test_*.py")))
        
        checks["test_file_count"] = len(test_files)
        checks["comprehensive_tests"] = len(test_files) >= 20
        
        # Calculate score
        boolean_checks = {k: v for k, v in checks.items() if isinstance(v, bool)}
        score = sum(boolean_checks.values()) / len(boolean_checks) * 100 if boolean_checks else 0
        
        return {
            "status": "healthy" if score >= 75 else "warning",
            "score": score,
            "checks": checks
        }
    
    def check_dependency_health(self) -> Dict[str, Any]:
        """Check dependency health and security."""
        checks = {}
        
        # Check for requirements files
        req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
        for req_file in req_files:
            req_path = self.repo_path / req_file
            checks[f"has_{req_file.replace('.', '_').replace('-', '_')}"] = req_path.exists()
        
        # Run safety check if available
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            vulnerabilities = 0
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities = len(safety_data)
                except json.JSONDecodeError:
                    pass
            
            checks["vulnerability_count"] = vulnerabilities
            checks["no_vulnerabilities"] = vulnerabilities == 0
            
        except Exception:
            checks["safety_check_available"] = False
        
        # Check for dependency management tools
        tools = ["Pipfile", "poetry.lock", "requirements.in"]
        for tool in tools:
            tool_path = self.repo_path / tool
            checks[f"uses_{tool.replace('.', '_')}"] = tool_path.exists()
        
        score = sum(1 for v in checks.values() if isinstance(v, bool) and v) / \
               sum(1 for v in checks.values() if isinstance(v, bool)) * 100
        
        return {
            "status": "healthy" if score >= 75 else "warning",
            "score": score,
            "checks": checks
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on health check results."""
        recommendations = []
        
        for category, results in self.health_report["categories"].items():
            if results["status"] in ["warning", "unhealthy"]:
                if category == "branch_protection":
                    recommendations.append("Configure branch protection rules for main branch")
                elif category == "security":
                    recommendations.append("Enable GitHub security features and add security workflows")
                elif category == "ci_cd":
                    recommendations.append("Improve CI/CD pipeline reliability and add more workflow types")
                elif category == "documentation":
                    recommendations.append("Add missing required documentation files")
                elif category == "code_quality":
                    recommendations.append("Improve test coverage and add quality configuration files")
                elif category == "dependencies":
                    recommendations.append("Address dependency vulnerabilities and improve dependency management")
        
        return recommendations
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive repository health check."""
        print("Running repository health check...")
        
        categories = {
            "branch_protection": self.check_branch_protection,
            "security": self.check_security_configuration,
            "ci_cd": self.check_ci_cd_health,
            "documentation": self.check_documentation_completeness,
            "code_quality": self.check_code_quality,
            "dependencies": self.check_dependency_health
        }
        
        total_score = 0
        valid_categories = 0
        
        for name, check_func in categories.items():
            print(f"Checking {name}...")
            result = check_func()
            self.health_report["categories"][name] = result
            
            if "score" in result and result["status"] != "skipped":
                total_score += result["score"]
                valid_categories += 1
        
        # Calculate overall score
        self.health_report["overall_score"] = total_score / valid_categories if valid_categories > 0 else 0
        
        # Generate recommendations
        self.health_report["recommendations"] = self.generate_recommendations()
        
        return self.health_report
    
    def generate_report(self) -> str:
        """Generate human-readable health report."""
        report = []
        report.append(f"# Repository Health Report")
        report.append(f"Generated: {self.health_report['timestamp']}")
        report.append(f"Repository: {self.repo_name}")
        report.append(f"Overall Score: {self.health_report['overall_score']:.1f}/100")
        report.append("")
        
        # Overall status
        score = self.health_report['overall_score']
        if score >= 90:
            status = "🟢 Excellent"
        elif score >= 75:
            status = "🟡 Good"
        elif score >= 60:
            status = "🟠 Needs Improvement"
        else:
            status = "🔴 Poor"
        
        report.append(f"Status: {status}")
        report.append("")
        
        # Category breakdown
        report.append("## Category Breakdown")
        for category, results in self.health_report["categories"].items():
            status_emoji = {
                "healthy": "🟢",
                "warning": "🟡",
                "unhealthy": "🔴",
                "error": "❌",
                "skipped": "⏭️"
            }.get(results["status"], "❓")
            
            score_text = f" ({results['score']:.1f}/100)" if "score" in results else ""
            report.append(f"- {status_emoji} {category.replace('_', ' ').title()}{score_text}")
        
        report.append("")
        
        # Recommendations
        if self.health_report["recommendations"]:
            report.append("## Recommendations")
            for i, rec in enumerate(self.health_report["recommendations"], 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for category, results in self.health_report["categories"].items():
            report.append(f"### {category.replace('_', ' ').title()}")
            
            if "checks" in results:
                for check, value in results["checks"].items():
                    if isinstance(value, bool):
                        emoji = "✅" if value else "❌"
                        report.append(f"- {emoji} {check.replace('_', ' ').title()}")
                    elif isinstance(value, (int, float)):
                        report.append(f"- {check.replace('_', ' ').title()}: {value}")
            
            if "message" in results:
                report.append(f"- Message: {results['message']}")
            
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function to run repository health check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check repository health")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    
    args = parser.parse_args()
    
    checker = RepositoryHealthChecker(args.repo_path)
    health_report = checker.run_health_check()
    
    if args.json:
        output = json.dumps(health_report, indent=2)
    else:
        output = checker.generate_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()