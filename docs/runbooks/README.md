# Operational Runbooks

Comprehensive operational procedures for the SQL Query Synthesizer.

## Runbook Structure

Each runbook follows a standard format:
- **Overview** - What the procedure accomplishes
- **Prerequisites** - Required access, tools, and knowledge
- **Steps** - Detailed step-by-step instructions
- **Verification** - How to verify success
- **Rollback** - How to revert changes if needed
- **Troubleshooting** - Common issues and solutions

## Available Runbooks

### Incident Response
- [High Error Rate Response](incident-response.md#high-error-rate)
- [Database Connection Failures](incident-response.md#database-failures)
- [Security Incident Response](incident-response.md#security-incidents)
- [Performance Degradation](incident-response.md#performance-issues)

### Maintenance Procedures  
- [Database Maintenance](maintenance.md#database-maintenance)
- [Cache Maintenance](maintenance.md#cache-maintenance)
- [Log Rotation](maintenance.md#log-rotation)
- [Certificate Renewal](maintenance.md#certificate-renewal)

### Deployment Operations
- [Production Deployment](deployment-ops.md#production-deployment)
- [Rollback Procedures](deployment-ops.md#rollback)
- [Blue-Green Deployment](deployment-ops.md#blue-green)
- [Database Migrations](deployment-ops.md#migrations)

### Monitoring & Alerting
- [Alert Investigation](monitoring-ops.md#alert-investigation)
- [Dashboard Creation](monitoring-ops.md#dashboard-creation)
- [Metric Troubleshooting](monitoring-ops.md#metric-troubleshooting)

## Quick Reference

### Emergency Contacts
```yaml
# On-call rotation
primary_oncall: "@team-lead"
secondary_oncall: "@senior-developer"
escalation: "@engineering-manager"

# External contacts
infrastructure: "infrastructure@company.com"
security: "security@company.com"
```

### Service URLs
```yaml
production:
  app: "https://sql-synthesizer.company.com"
  grafana: "https://grafana.company.com"
  prometheus: "https://prometheus.company.com"

staging:
  app: "https://sql-synthesizer-staging.company.com"
  grafana: "https://grafana-staging.company.com"
```

### Key Commands
```bash
# Health check
curl -f https://sql-synthesizer.company.com/health

# View logs
kubectl logs -f deployment/sql-synthesizer -n production

# Scale service
kubectl scale deployment sql-synthesizer --replicas=5 -n production

# Database status
docker exec postgres pg_isready -U postgres
```

## Escalation Procedures

### Severity Levels

#### P0 - Critical (Service Down)
- **Response Time**: 15 minutes
- **Escalation**: Immediate to on-call engineer
- **Communication**: Incident channel + status page
- **Examples**: Total service outage, data corruption

#### P1 - High (Major Functionality Impacted)  
- **Response Time**: 1 hour
- **Escalation**: On-call engineer
- **Communication**: Incident channel
- **Examples**: High error rate, performance degradation

#### P2 - Medium (Minor Functionality Impacted)
- **Response Time**: 4 hours
- **Escalation**: During business hours
- **Communication**: Team channel
- **Examples**: Non-critical feature broken, monitoring gaps

#### P3 - Low (Minimal Impact)
- **Response Time**: Next business day
- **Escalation**: Standard triage
- **Communication**: Issue tracker
- **Examples**: Documentation gaps, minor UI issues

### Communication Templates

#### Incident Notification
```
🚨 INCIDENT: [SEVERITY] - [BRIEF_DESCRIPTION]

Impact: [USER_IMPACT_DESCRIPTION]
Status: [INVESTIGATING|IDENTIFIED|FIXING|MONITORING|RESOLVED]
ETA: [ESTIMATED_RESOLUTION_TIME]
Owner: [INCIDENT_COMMANDER]

Updates will be posted every [X] minutes.
```

#### Status Update
```
📊 UPDATE: [INCIDENT_TITLE]

Current Status: [DETAILED_STATUS_UPDATE]
Actions Taken: [WHAT_HAS_BEEN_DONE]
Next Steps: [WHAT_WILL_BE_DONE_NEXT]
ETA: [UPDATED_ETA]
```

#### Resolution Notice
```
✅ RESOLVED: [INCIDENT_TITLE]

Resolution: [HOW_THE_ISSUE_WAS_RESOLVED]
Root Cause: [BRIEF_ROOT_CAUSE]
Duration: [TOTAL_INCIDENT_DURATION]
Post-Mortem: [LINK_TO_POST_MORTEM_DOC]
```

## Post-Incident Procedures

### Post-Mortem Template
1. **Incident Summary**
   - Timeline of events
   - Impact assessment
   - Response timeline

2. **Root Cause Analysis**
   - Technical root cause
   - Contributing factors
   - Why detection was delayed

3. **Action Items**
   - Immediate fixes
   - Preventive measures
   - Process improvements

4. **Lessons Learned**
   - What went well
   - What could be improved
   - Knowledge gaps identified

### Improvement Tracking
- Action items added to project backlog
- Process updates to runbooks
- Monitoring improvements
- Training needs assessment

## Training & Knowledge Transfer

### New Team Member Onboarding
1. Review all runbooks
2. Shadow incident response
3. Practice deployment procedures
4. Complete maintenance tasks under supervision

### Regular Training
- Monthly incident response drills
- Quarterly runbook reviews
- Annual disaster recovery testing
- Cross-team knowledge sharing

## Runbook Maintenance

### Review Schedule
- **Monthly**: Update emergency contacts and URLs
- **Quarterly**: Review and test all procedures
- **Annually**: Complete runbook overhaul

### Change Management
- All runbook changes require peer review
- Test procedures in staging environment
- Update training materials when procedures change
- Communicate changes to entire team

For detailed procedures, see the specific runbook files in this directory.