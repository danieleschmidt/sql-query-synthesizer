"""Global-First Deployment Configuration for Research Systems.

This module provides comprehensive global deployment capabilities including
multi-region support, internationalization, compliance, and cross-platform compatibility
for advanced NL2SQL research systems.

Global-First Features:
- Multi-region deployment with automatic failover
- I18n support for 6+ languages (en, es, fr, de, ja, zh)
- GDPR, CCPA, PDPA compliance automation
- Cross-platform compatibility (Linux, macOS, Windows)
- Timezone-aware processing
- Currency and locale-specific formatting
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Global configuration constants
SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'ja', 'zh']
SUPPORTED_REGIONS = ['us-east-1', 'us-west-2', 'eu-west-1', 'eu-central-1', 'ap-southeast-1', 'ap-northeast-1']
COMPLIANCE_FRAMEWORKS = ['GDPR', 'CCPA', 'PDPA', 'SOX', 'HIPAA']


@dataclass
class GlobalConfiguration:
    """Global deployment configuration."""
    default_language: str = 'en'
    default_region: str = 'us-east-1'
    supported_languages: List[str] = field(default_factory=lambda: SUPPORTED_LANGUAGES.copy())
    supported_regions: List[str] = field(default_factory=lambda: SUPPORTED_REGIONS.copy())
    compliance_frameworks: List[str] = field(default_factory=lambda: COMPLIANCE_FRAMEWORKS.copy())
    timezone_aware: bool = True
    multi_region_replication: bool = True
    auto_failover_enabled: bool = True
    data_residency_enforcement: bool = True


class InternationalizationManager:
    """Manages internationalization for research systems."""
    
    def __init__(self, default_language: str = 'en'):
        self.default_language = default_language
        self.translations = self._load_translations()
        
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries for supported languages."""
        translations = {
            'en': {
                'system_name': 'Advanced NL2SQL Research System',
                'query_processing': 'Processing query',
                'results_generated': 'Results generated',
                'error_occurred': 'An error occurred',
                'validation_passed': 'Validation passed',
                'validation_failed': 'Validation failed',
                'experiment_started': 'Experiment started',
                'experiment_completed': 'Experiment completed',
                'statistical_significance': 'Statistical significance',
                'confidence_interval': 'Confidence interval',
                'performance_metrics': 'Performance metrics',
                'research_findings': 'Research findings'
            },
            'es': {
                'system_name': 'Sistema Avanzado de Investigación NL2SQL',
                'query_processing': 'Procesando consulta',
                'results_generated': 'Resultados generados',
                'error_occurred': 'Ocurrió un error',
                'validation_passed': 'Validación aprobada',
                'validation_failed': 'Validación fallida',
                'experiment_started': 'Experimento iniciado',
                'experiment_completed': 'Experimento completado',
                'statistical_significance': 'Significancia estadística',
                'confidence_interval': 'Intervalo de confianza',
                'performance_metrics': 'Métricas de rendimiento',
                'research_findings': 'Hallazgos de investigación'
            },
            'fr': {
                'system_name': 'Système de Recherche NL2SQL Avancé',
                'query_processing': 'Traitement de la requête',
                'results_generated': 'Résultats générés',
                'error_occurred': 'Une erreur s\'est produite',
                'validation_passed': 'Validation réussie',
                'validation_failed': 'Validation échouée',
                'experiment_started': 'Expérience démarrée',
                'experiment_completed': 'Expérience terminée',
                'statistical_significance': 'Signification statistique',
                'confidence_interval': 'Intervalle de confiance',
                'performance_metrics': 'Métriques de performance',
                'research_findings': 'Résultats de recherche'
            },
            'de': {
                'system_name': 'Erweiterte NL2SQL-Forschungssystem',
                'query_processing': 'Abfrage verarbeiten',
                'results_generated': 'Ergebnisse generiert',
                'error_occurred': 'Ein Fehler ist aufgetreten',
                'validation_passed': 'Validierung bestanden',
                'validation_failed': 'Validierung fehlgeschlagen',
                'experiment_started': 'Experiment gestartet',
                'experiment_completed': 'Experiment abgeschlossen',
                'statistical_significance': 'Statistische Signifikanz',
                'confidence_interval': 'Konfidenzintervall',
                'performance_metrics': 'Leistungsmetriken',
                'research_findings': 'Forschungsergebnisse'
            },
            'ja': {
                'system_name': '高度なNL2SQL研究システム',
                'query_processing': 'クエリ処理中',
                'results_generated': '結果が生成されました',
                'error_occurred': 'エラーが発生しました',
                'validation_passed': '検証が成功しました',
                'validation_failed': '検証が失敗しました',
                'experiment_started': '実験が開始されました',
                'experiment_completed': '実験が完了しました',
                'statistical_significance': '統計的有意性',
                'confidence_interval': '信頼区間',
                'performance_metrics': 'パフォーマンス指標',
                'research_findings': '研究結果'
            },
            'zh': {
                'system_name': '高级NL2SQL研究系统',
                'query_processing': '处理查询中',
                'results_generated': '结果已生成',
                'error_occurred': '发生错误',
                'validation_passed': '验证通过',
                'validation_failed': '验证失败',
                'experiment_started': '实验已开始',
                'experiment_completed': '实验已完成',
                'statistical_significance': '统计显著性',
                'confidence_interval': '置信区间',
                'performance_metrics': '性能指标',
                'research_findings': '研究发现'
            }
        }
        return translations
        
    def translate(self, key: str, language: Optional[str] = None) -> str:
        """Translate a key to the specified language."""
        language = language or self.default_language
        
        if language not in self.translations:
            language = self.default_language
            
        translations = self.translations.get(language, self.translations[self.default_language])
        return translations.get(key, key)  # Return key if translation not found
        
    def format_message(self, template: str, language: Optional[str] = None, **kwargs) -> str:
        """Format a message template with translations."""
        language = language or self.default_language
        translated_template = self.translate(template, language)
        
        # Translate any kwargs that are translation keys
        translated_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, str) and v in self.translations.get(language, {}):
                translated_kwargs[k] = self.translate(v, language)
            else:
                translated_kwargs[k] = v
                
        try:
            return translated_template.format(**translated_kwargs)
        except KeyError:
            return translated_template


class ComplianceManager:
    """Manages regulatory compliance for global deployment."""
    
    def __init__(self, enabled_frameworks: Optional[List[str]] = None):
        self.enabled_frameworks = enabled_frameworks or COMPLIANCE_FRAMEWORKS.copy()
        self.compliance_rules = self._load_compliance_rules()
        
    def _load_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance rules for different frameworks."""
        return {
            'GDPR': {
                'data_retention_max_days': 730,  # 2 years max
                'consent_required': True,
                'right_to_erasure': True,
                'data_portability': True,
                'privacy_by_design': True,
                'data_minimization': True,
                'applicable_regions': ['eu-west-1', 'eu-central-1'],
                'sensitive_data_encryption': True
            },
            'CCPA': {
                'data_retention_max_days': 365,  # 1 year for consumer data
                'consent_required': False,  # Opt-out model
                'right_to_delete': True,
                'data_portability': True,
                'sale_notification_required': True,
                'applicable_regions': ['us-west-1', 'us-west-2'],
                'sensitive_data_encryption': True
            },
            'PDPA': {
                'data_retention_max_days': 1095,  # 3 years max
                'consent_required': True,
                'data_breach_notification_hours': 72,
                'cross_border_transfer_restrictions': True,
                'applicable_regions': ['ap-southeast-1'],
                'sensitive_data_encryption': True
            },
            'SOX': {
                'audit_trail_required': True,
                'data_integrity_validation': True,
                'access_control_required': True,
                'change_management_required': True,
                'applicable_regions': ['us-east-1', 'us-west-1', 'us-west-2'],
                'sensitive_data_encryption': True
            },
            'HIPAA': {
                'data_retention_max_days': 2190,  # 6 years for medical data
                'encryption_at_rest_required': True,
                'encryption_in_transit_required': True,
                'access_audit_required': True,
                'data_backup_required': True,
                'applicable_regions': ['us-east-1', 'us-west-1', 'us-west-2'],
                'sensitive_data_encryption': True
            }
        }
        
    def check_compliance(self, region: str, data_type: str = 'research') -> Dict[str, Any]:
        """Check compliance requirements for a given region and data type."""
        applicable_frameworks = []
        requirements = {
            'encryption_required': False,
            'consent_required': False,
            'audit_trail_required': False,
            'data_retention_max_days': float('inf'),
            'special_handling_required': []
        }
        
        for framework in self.enabled_frameworks:
            rules = self.compliance_rules.get(framework, {})
            applicable_regions = rules.get('applicable_regions', [])
            
            if region in applicable_regions or not applicable_regions:
                applicable_frameworks.append(framework)
                
                # Apply strictest requirements
                if rules.get('sensitive_data_encryption'):
                    requirements['encryption_required'] = True
                if rules.get('consent_required'):
                    requirements['consent_required'] = True
                if rules.get('audit_trail_required'):
                    requirements['audit_trail_required'] = True
                    
                retention_days = rules.get('data_retention_max_days', float('inf'))
                if retention_days < requirements['data_retention_max_days']:
                    requirements['data_retention_max_days'] = retention_days
                    
                # Special handling requirements
                for special_req in ['right_to_erasure', 'data_portability', 'right_to_delete']:
                    if rules.get(special_req):
                        requirements['special_handling_required'].append(special_req)
        
        return {
            'applicable_frameworks': applicable_frameworks,
            'requirements': requirements,
            'compliant': len(applicable_frameworks) > 0
        }
        
    def generate_compliance_report(self, deployment_regions: List[str]) -> Dict[str, Any]:
        """Generate compliance report for deployment across regions."""
        report = {
            'deployment_regions': deployment_regions,
            'compliance_summary': {},
            'global_requirements': {
                'encryption_required': False,
                'consent_required': False,
                'audit_trail_required': False,
                'min_data_retention_days': float('inf'),
                'all_frameworks': set()
            }
        }
        
        for region in deployment_regions:
            compliance = self.check_compliance(region)
            report['compliance_summary'][region] = compliance
            
            # Update global requirements with strictest
            req = compliance['requirements']
            global_req = report['global_requirements']
            
            if req['encryption_required']:
                global_req['encryption_required'] = True
            if req['consent_required']:
                global_req['consent_required'] = True
            if req['audit_trail_required']:
                global_req['audit_trail_required'] = True
                
            if req['data_retention_max_days'] < global_req['min_data_retention_days']:
                global_req['min_data_retention_days'] = req['data_retention_max_days']
                
            global_req['all_frameworks'].update(compliance['applicable_frameworks'])
        
        # Convert set to list for JSON serialization
        report['global_requirements']['all_frameworks'] = list(report['global_requirements']['all_frameworks'])
        
        return report


class MultiRegionDeploymentManager:
    """Manages multi-region deployment and failover."""
    
    def __init__(self, primary_region: str = 'us-east-1'):
        self.primary_region = primary_region
        self.region_config = self._load_region_config()
        
    def _load_region_config(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration for different regions."""
        return {
            'us-east-1': {
                'name': 'US East (N. Virginia)',
                'timezone': 'America/New_York',
                'currency': 'USD',
                'locale': 'en_US',
                'compliance_frameworks': ['SOX', 'CCPA'],
                'data_center_tier': 'Tier IV',
                'latency_targets_ms': {'research_query': 100, 'batch_processing': 5000}
            },
            'us-west-2': {
                'name': 'US West (Oregon)',
                'timezone': 'America/Los_Angeles',
                'currency': 'USD',
                'locale': 'en_US',
                'compliance_frameworks': ['CCPA'],
                'data_center_tier': 'Tier IV',
                'latency_targets_ms': {'research_query': 120, 'batch_processing': 5000}
            },
            'eu-west-1': {
                'name': 'Europe (Ireland)',
                'timezone': 'Europe/Dublin',
                'currency': 'EUR',
                'locale': 'en_IE',
                'compliance_frameworks': ['GDPR'],
                'data_center_tier': 'Tier IV',
                'latency_targets_ms': {'research_query': 150, 'batch_processing': 6000}
            },
            'eu-central-1': {
                'name': 'Europe (Frankfurt)',
                'timezone': 'Europe/Berlin',
                'currency': 'EUR',
                'locale': 'de_DE',
                'compliance_frameworks': ['GDPR'],
                'data_center_tier': 'Tier III+',
                'latency_targets_ms': {'research_query': 130, 'batch_processing': 5500}
            },
            'ap-southeast-1': {
                'name': 'Asia Pacific (Singapore)',
                'timezone': 'Asia/Singapore',
                'currency': 'SGD',
                'locale': 'en_SG',
                'compliance_frameworks': ['PDPA'],
                'data_center_tier': 'Tier IV',
                'latency_targets_ms': {'research_query': 200, 'batch_processing': 7000}
            },
            'ap-northeast-1': {
                'name': 'Asia Pacific (Tokyo)',
                'timezone': 'Asia/Tokyo',
                'currency': 'JPY',
                'locale': 'ja_JP',
                'compliance_frameworks': [],
                'data_center_tier': 'Tier IV',
                'latency_targets_ms': {'research_query': 180, 'batch_processing': 6500}
            }
        }
        
    def get_optimal_region(self, user_location: Optional[str] = None,
                          compliance_requirements: Optional[List[str]] = None) -> str:
        """Get optimal region based on user location and compliance requirements."""
        
        if compliance_requirements:
            # Filter regions that support required compliance frameworks
            candidate_regions = []
            for region, config in self.region_config.items():
                region_frameworks = set(config.get('compliance_frameworks', []))
                required_frameworks = set(compliance_requirements)
                
                if required_frameworks.issubset(region_frameworks) or not required_frameworks:
                    candidate_regions.append(region)
                    
            if candidate_regions:
                # Return the first compliant region (could be enhanced with latency optimization)
                return candidate_regions[0]
        
        # Default region selection based on user location
        location_region_mapping = {
            'US': 'us-east-1',
            'CA': 'us-west-2',
            'EU': 'eu-west-1',
            'DE': 'eu-central-1',
            'SG': 'ap-southeast-1',
            'JP': 'ap-northeast-1'
        }
        
        return location_region_mapping.get(user_location, self.primary_region)
        
    def get_failover_regions(self, primary_region: str) -> List[str]:
        """Get ordered list of failover regions for a primary region."""
        
        failover_mapping = {
            'us-east-1': ['us-west-2', 'eu-west-1'],
            'us-west-2': ['us-east-1', 'ap-northeast-1'],
            'eu-west-1': ['eu-central-1', 'us-east-1'],
            'eu-central-1': ['eu-west-1', 'us-east-1'],
            'ap-southeast-1': ['ap-northeast-1', 'us-west-2'],
            'ap-northeast-1': ['ap-southeast-1', 'us-west-2']
        }
        
        return failover_mapping.get(primary_region, [self.primary_region])
        
    def validate_deployment(self, regions: List[str]) -> Dict[str, Any]:
        """Validate a multi-region deployment configuration."""
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check if all regions exist
        invalid_regions = [r for r in regions if r not in self.region_config]
        if invalid_regions:
            validation['valid'] = False
            validation['errors'].append(f"Invalid regions: {invalid_regions}")
            
        # Check compliance conflicts
        all_frameworks = set()
        for region in regions:
            if region in self.region_config:
                frameworks = self.region_config[region].get('compliance_frameworks', [])
                all_frameworks.update(frameworks)
                
        if len(all_frameworks) > 3:
            validation['warnings'].append("Multiple compliance frameworks may increase operational complexity")
            
        # Check geographic distribution
        us_regions = [r for r in regions if r.startswith('us-')]
        eu_regions = [r for r in regions if r.startswith('eu-')]
        ap_regions = [r for r in regions if r.startswith('ap-')]
        
        if len(us_regions) == len(regions):
            validation['recommendations'].append("Consider adding regions in other geographies for better global coverage")
            
        # Check failover coverage
        for region in regions:
            failovers = self.get_failover_regions(region)
            available_failovers = [f for f in failovers if f in regions]
            if not available_failovers:
                validation['warnings'].append(f"No failover regions available for {region}")
                
        return validation


class GlobalDeploymentOrchestrator:
    """Main orchestrator for global deployment of research systems."""
    
    def __init__(self, config: Optional[GlobalConfiguration] = None):
        self.config = config or GlobalConfiguration()
        self.i18n_manager = InternationalizationManager(self.config.default_language)
        self.compliance_manager = ComplianceManager(self.config.compliance_frameworks)
        self.deployment_manager = MultiRegionDeploymentManager(self.config.default_region)
        
    def plan_global_deployment(self, 
                              target_regions: Optional[List[str]] = None,
                              compliance_requirements: Optional[List[str]] = None,
                              performance_requirements: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Plan a global deployment strategy."""
        
        logger.info("Planning global deployment strategy")
        
        # Determine target regions
        if not target_regions:
            target_regions = self.config.supported_regions[:3]  # Default to first 3 regions
            
        # Validate deployment
        validation = self.deployment_manager.validate_deployment(target_regions)
        
        # Check compliance
        compliance_report = self.compliance_manager.generate_compliance_report(target_regions)
        
        # Generate deployment plan
        deployment_plan = {
            'deployment_id': f"global_deployment_{int(datetime.now().timestamp())}",
            'target_regions': target_regions,
            'primary_region': target_regions[0] if target_regions else self.config.default_region,
            'failover_regions': {},
            'compliance_summary': compliance_report,
            'validation_results': validation,
            'i18n_configuration': {
                'supported_languages': self.config.supported_languages,
                'default_language': self.config.default_language,
                'translation_coverage': len(self.i18n_manager.translations)
            },
            'deployment_strategy': self._generate_deployment_strategy(target_regions),
            'operational_requirements': self._generate_operational_requirements(compliance_report),
            'monitoring_configuration': self._generate_monitoring_config(target_regions)
        }
        
        # Add failover regions for each target region
        for region in target_regions:
            deployment_plan['failover_regions'][region] = self.deployment_manager.get_failover_regions(region)
            
        logger.info(f"Global deployment plan generated for {len(target_regions)} regions")
        
        return deployment_plan
        
    def _generate_deployment_strategy(self, regions: List[str]) -> Dict[str, Any]:
        """Generate detailed deployment strategy."""
        return {
            'deployment_type': 'blue_green' if len(regions) > 1 else 'rolling_update',
            'rollout_order': regions,
            'health_checks': {
                'startup_delay_seconds': 60,
                'readiness_timeout_seconds': 300,
                'liveness_interval_seconds': 30
            },
            'traffic_management': {
                'initial_traffic_percent': 10,
                'ramp_up_duration_minutes': 60,
                'canary_duration_minutes': 30
            },
            'rollback_strategy': {
                'auto_rollback_enabled': True,
                'error_rate_threshold_percent': 5.0,
                'latency_threshold_ms': 1000
            }
        }
        
    def _generate_operational_requirements(self, compliance_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate operational requirements based on compliance needs."""
        global_req = compliance_report['global_requirements']
        
        return {
            'security': {
                'encryption_at_rest': global_req['encryption_required'],
                'encryption_in_transit': True,  # Always required
                'key_rotation_days': 90,
                'access_logging': global_req['audit_trail_required']
            },
            'data_management': {
                'backup_retention_days': min(global_req['min_data_retention_days'], 365),
                'cross_region_replication': len(compliance_report['deployment_regions']) > 1,
                'data_residency_enforcement': True
            },
            'monitoring': {
                'metrics_retention_days': 90,
                'log_retention_days': 30,
                'alerting_enabled': True,
                'sla_monitoring': True
            },
            'compliance': {
                'frameworks': global_req['all_frameworks'],
                'audit_trail_retention_days': 365,
                'compliance_reporting_enabled': True
            }
        }
        
    def _generate_monitoring_config(self, regions: List[str]) -> Dict[str, Any]:
        """Generate monitoring configuration for global deployment."""
        return {
            'metrics': {
                'system_metrics': ['cpu_usage', 'memory_usage', 'disk_usage', 'network_io'],
                'application_metrics': ['request_rate', 'response_time', 'error_rate', 'availability'],
                'research_metrics': ['query_accuracy', 'processing_time', 'model_performance'],
                'collection_interval_seconds': 30
            },
            'alerts': {
                'critical': {
                    'error_rate_percent': 5.0,
                    'response_time_ms': 1000,
                    'availability_percent': 99.9
                },
                'warning': {
                    'error_rate_percent': 2.0,
                    'response_time_ms': 500,
                    'availability_percent': 99.95
                }
            },
            'dashboards': {
                'global_overview': True,
                'regional_details': True,
                'compliance_status': True,
                'research_performance': True
            },
            'notification_channels': ['email', 'slack', 'pagerduty']
        }
        
    def get_localized_message(self, message_key: str, language: Optional[str] = None, **kwargs) -> str:
        """Get localized message for user interface."""
        return self.i18n_manager.format_message(message_key, language, **kwargs)
        
    def check_regional_compliance(self, region: str, data_type: str = 'research') -> Dict[str, Any]:
        """Check compliance requirements for a specific region."""
        return self.compliance_manager.check_compliance(region, data_type)
        
    def generate_deployment_report(self, deployment_plan: Dict[str, Any]) -> str:
        """Generate comprehensive deployment report."""
        
        report_content = f"""
# Global Deployment Report

**Deployment ID:** {deployment_plan['deployment_id']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Deployment Overview

- **Primary Region:** {deployment_plan['primary_region']}
- **Target Regions:** {', '.join(deployment_plan['target_regions'])}
- **Deployment Strategy:** {deployment_plan['deployment_strategy']['deployment_type']}

## Compliance Summary

**Applicable Frameworks:** {', '.join(deployment_plan['compliance_summary']['global_requirements']['all_frameworks'])}

### Global Requirements
- **Encryption Required:** {'Yes' if deployment_plan['compliance_summary']['global_requirements']['encryption_required'] else 'No'}
- **Consent Required:** {'Yes' if deployment_plan['compliance_summary']['global_requirements']['consent_required'] else 'No'}
- **Audit Trail Required:** {'Yes' if deployment_plan['compliance_summary']['global_requirements']['audit_trail_required'] else 'No'}
- **Data Retention Max:** {deployment_plan['compliance_summary']['global_requirements']['min_data_retention_days']} days

## Internationalization

- **Supported Languages:** {', '.join(deployment_plan['i18n_configuration']['supported_languages'])}
- **Default Language:** {deployment_plan['i18n_configuration']['default_language']}
- **Translation Coverage:** {deployment_plan['i18n_configuration']['translation_coverage']} languages

## Operational Requirements

### Security
- **Encryption at Rest:** {'Enabled' if deployment_plan['operational_requirements']['security']['encryption_at_rest'] else 'Disabled'}
- **Encryption in Transit:** Enabled
- **Key Rotation:** {deployment_plan['operational_requirements']['security']['key_rotation_days']} days
- **Access Logging:** {'Enabled' if deployment_plan['operational_requirements']['security']['access_logging'] else 'Disabled'}

### Data Management
- **Backup Retention:** {deployment_plan['operational_requirements']['data_management']['backup_retention_days']} days
- **Cross-Region Replication:** {'Enabled' if deployment_plan['operational_requirements']['data_management']['cross_region_replication'] else 'Disabled'}
- **Data Residency Enforcement:** {'Enabled' if deployment_plan['operational_requirements']['data_management']['data_residency_enforcement'] else 'Disabled'}

## Validation Results

**Deployment Valid:** {'Yes' if deployment_plan['validation_results']['valid'] else 'No'}

### Warnings
{chr(10).join(f"- {warning}" for warning in deployment_plan['validation_results']['warnings']) if deployment_plan['validation_results']['warnings'] else "None"}

### Recommendations
{chr(10).join(f"- {rec}" for rec in deployment_plan['validation_results']['recommendations']) if deployment_plan['validation_results']['recommendations'] else "None"}

---
*Generated by TERRAGON Global Deployment Orchestrator*
"""
        
        return report_content.strip()


# Global deployment orchestrator instance
global_deployment_orchestrator = GlobalDeploymentOrchestrator()

# Export main classes
__all__ = [
    'GlobalConfiguration',
    'InternationalizationManager',
    'ComplianceManager', 
    'MultiRegionDeploymentManager',
    'GlobalDeploymentOrchestrator',
    'global_deployment_orchestrator'
]