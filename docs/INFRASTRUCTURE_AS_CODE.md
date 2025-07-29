# Infrastructure as Code Templates

Comprehensive Infrastructure as Code (IaC) templates for deploying SQL Query Synthesizer across multiple cloud providers and environments.

## Deployment Architecture

### Multi-Tier Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │  Web Tier       │    │  Data Tier      │
│   (ALB/NGINX)   │───▶│  (SQL Synth)    │───▶│  (PostgreSQL)   │
│                 │    │  Auto Scaling   │    │  Read Replicas  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CDN/WAF       │    │  Cache Tier     │    │  Monitoring     │
│   (CloudFlare)  │    │  (Redis)        │    │  (Prometheus)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## AWS Infrastructure

### 1. Terraform Configuration

#### Main Infrastructure (`terraform/aws/main.tf`)
```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "sql-synthesizer-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-lock-table"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "sql-synthesizer"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "sql-synthesizer-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = {
    Terraform = "true"
    Environment = var.environment
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "sql-synthesizer-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = var.environment == "production"
  
  tags = {
    Environment = var.environment
  }
}

# Security Groups
resource "aws_security_group" "alb" {
  name_prefix = "sql-synthesizer-alb-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "sql-synthesizer-alb-sg"
  }
}
```

#### ECS Cluster Configuration (`terraform/aws/ecs.tf`)
```hcl
# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "sql-synthesizer-cluster"
  
  configuration {
    execute_command_configuration {
      logging = "OVERRIDE"
      
      log_configuration {
        cloud_watch_encryption_enabled = true
        cloud_watch_log_group_name     = aws_cloudwatch_log_group.ecs.name
      }
    }
  }
  
  tags = {
    Environment = var.environment
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "app" {
  family                   = "sql-synthesizer"
  network_mode            = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                     = var.app_cpu
  memory                  = var.app_memory
  execution_role_arn      = aws_iam_role.ecs_execution_role.arn
  task_role_arn          = aws_iam_role.ecs_task_role.arn
  
  container_definitions = jsonencode([
    {
      name  = "sql-synthesizer"
      image = "${var.ecr_repository_url}:${var.app_version}"
      
      portMappings = [
        {
          containerPort = 5000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "DATABASE_URL"
          value = "postgresql://${aws_rds_cluster.main.master_username}:${random_password.db_password.result}@${aws_rds_cluster.main.endpoint}:5432/${aws_rds_cluster.main.database_name}"
        },
        {
          name  = "REDIS_URL"
          value = "redis://${aws_elasticache_replication_group.redis.configuration_endpoint_address}:6379"
        }
      ]
      
      secrets = [
        {
          name      = "OPENAI_API_KEY"
          valueFrom = aws_ssm_parameter.openai_api_key.arn
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.app.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:5000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# ECS Service
resource "aws_ecs_service" "app" {
  name            = "sql-synthesizer-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.app_count
  launch_type     = "FARGATE"
  
  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets         = module.vpc.private_subnets
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "sql-synthesizer"
    container_port   = 5000
  }
  
  depends_on = [aws_lb_listener.app]
  
  lifecycle {
    ignore_changes = [desired_count]
  }
  
  tags = {
    Environment = var.environment
  }
}
```

#### RDS Database Configuration (`terraform/aws/rds.tf`)
```hcl
# RDS Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "sql-synthesizer-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = {
    Name = "SQL Synthesizer DB subnet group"
  }
}

# RDS Cluster (Aurora PostgreSQL)
resource "aws_rds_cluster" "main" {
  cluster_identifier      = "sql-synthesizer-cluster"
  engine                 = "aurora-postgresql"
  engine_version         = "15.3"
  availability_zones     = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  database_name          = "sql_synthesizer"
  master_username        = "dbadmin"
  master_password        = random_password.db_password.result
  backup_retention_period = 30
  preferred_backup_window = "07:00-09:00"
  preferred_maintenance_window = "sun:05:00-sun:07:00"
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  storage_encrypted = true
  kms_key_id       = aws_kms_key.rds.arn
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  deletion_protection = var.environment == "production"
  skip_final_snapshot = var.environment != "production"
  
  tags = {
    Environment = var.environment
  }
}

# RDS Cluster Instances
resource "aws_rds_cluster_instance" "cluster_instances" {
  count              = var.rds_instance_count
  identifier         = "sql-synthesizer-${count.index}"
  cluster_identifier = aws_rds_cluster.main.id
  instance_class     = var.rds_instance_class
  engine             = aws_rds_cluster.main.engine
  engine_version     = aws_rds_cluster.main.engine_version
  
  performance_insights_enabled = true
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_enhanced_monitoring.arn
  
  tags = {
    Environment = var.environment
  }
}

# Database Password
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "aws_ssm_parameter" "db_password" {
  name  = "/sql-synthesizer/${var.environment}/db_password"
  type  = "SecureString"
  value = random_password.db_password.result
  
  tags = {
    Environment = var.environment
  }
}
```

### 2. Auto Scaling Configuration
```hcl
# Auto Scaling Target
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# Auto Scaling Policy - CPU
resource "aws_appautoscaling_policy" "ecs_policy_cpu" {
  name               = "cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

# Auto Scaling Policy - Memory
resource "aws_appautoscaling_policy" "ecs_policy_memory" {
  name               = "memory-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value = 80.0
  }
}
```

## Kubernetes Deployment

### 1. Helm Chart (`helm/sql-synthesizer/Chart.yaml`)
```yaml
apiVersion: v2
name: sql-synthesizer
description: A Helm chart for SQL Query Synthesizer
type: application
version: 0.2.2
appVersion: "0.2.2"

dependencies:
  - name: postgresql
    version: "12.1.9"
    repository: "https://charts.bitnami.com/bitnami"
    condition: postgresql.enabled
  - name: redis
    version: "17.4.3"
    repository: "https://charts.bitnami.com/bitnami"
    condition: redis.enabled
```

### 2. Deployment Configuration (`helm/sql-synthesizer/templates/deployment.yaml`)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "sql-synthesizer.fullname" . }}
  labels:
    {{- include "sql-synthesizer.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "sql-synthesizer.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
      labels:
        {{- include "sql-synthesizer.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "sql-synthesizer.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 5000
              protocol: TCP
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ include "sql-synthesizer.fullname" . }}
                  key: database-url
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "sql-synthesizer.fullname" . }}
                  key: openai-api-key
            - name: REDIS_URL
              value: "redis://{{ include "sql-synthesizer.fullname" . }}-redis-master:6379"
          envFrom:
            - configMapRef:
                name: {{ include "sql-synthesizer.fullname" . }}
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
              readOnly: true
      volumes:
        - name: config-volume
          configMap:
            name: {{ include "sql-synthesizer.fullname" . }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

### 3. Horizontal Pod Autoscaler (`helm/sql-synthesizer/templates/hpa.yaml`)
```yaml
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "sql-synthesizer.fullname" . }}
  labels:
    {{- include "sql-synthesizer.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "sql-synthesizer.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
{{- end }}
```

## Docker Optimization

### 1. Multi-stage Dockerfile
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r sqlsynth && useradd -r -g sqlsynth sqlsynth

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/sqlsynth/.local

# Set up application
WORKDIR /app
COPY . .

# Change ownership to non-root user
RUN chown -R sqlsynth:sqlsynth /app

# Switch to non-root user
USER sqlsynth

# Add local packages to PATH
ENV PATH=/home/sqlsynth/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Start application
CMD ["python", "-m", "sql_synthesizer.webapp"]
```

### 2. Docker Compose for Development
```yaml
# docker-compose.development.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    ports:
      - "5000:5000"
    volumes:
      - .:/app
      - /app/__pycache__
    environment:
      - FLASK_ENV=development
      - DATABASE_URL=postgresql://postgres:password@db:5432/sql_synthesizer
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=sql_synthesizer
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app

volumes:
  postgres_data:
  redis_data:
```

## Monitoring and Observability

### 1. Prometheus Configuration (`monitoring/prometheus.yml`)
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'sql-synthesizer'
    static_configs:
      - targets: ['app:5000']
    metrics_path: '/metrics'
    scrape_interval: 15s
    
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 2. Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "SQL Query Synthesizer Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(flask_http_request_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(flask_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Database Connection Pool",
        "type": "graph",
        "targets": [
          {
            "expr": "sqlalchemy_pool_checked_out_connections",
            "legendFormat": "Active Connections"
          }
        ]
      }
    ]
  }
}
```

## Deployment Scripts

### 1. Automated Deployment Script (`scripts/deploy.sh`)
```bash
#!/bin/bash
set -euo pipefail

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

echo "Deploying SQL Query Synthesizer v${VERSION} to ${ENVIRONMENT}"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    echo "Error: Invalid environment. Use 'staging' or 'production'"
    exit 1
fi

# Build and tag Docker image
docker build -t sql-synthesizer:${VERSION} .
docker tag sql-synthesizer:${VERSION} ${ECR_REGISTRY}/sql-synthesizer:${VERSION}

# Push to container registry
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}
docker push ${ECR_REGISTRY}/sql-synthesizer:${VERSION}

# Deploy with Terraform
cd terraform/aws
terraform init
terraform workspace select ${ENVIRONMENT} || terraform workspace new ${ENVIRONMENT}
terraform plan -var="app_version=${VERSION}" -var="environment=${ENVIRONMENT}"
terraform apply -var="app_version=${VERSION}" -var="environment=${ENVIRONMENT}" -auto-approve

# Wait for deployment to complete
echo "Waiting for deployment to stabilize..."
aws ecs wait services-stable --cluster sql-synthesizer-cluster --services sql-synthesizer-service

# Run smoke tests
./scripts/smoke-tests.sh ${ENVIRONMENT}

echo "Deployment completed successfully!"
```

### 2. Database Migration Script (`scripts/migrate.sh`)
```bash
#!/bin/bash
set -euo pipefail

ENVIRONMENT=${1:-staging}

echo "Running database migrations for ${ENVIRONMENT} environment"

# Get database credentials from AWS Parameter Store
DB_PASSWORD=$(aws ssm get-parameter --name "/sql-synthesizer/${ENVIRONMENT}/db_password" --with-decryption --query 'Parameter.Value' --output text)
DB_HOST=$(aws rds describe-db-clusters --db-cluster-identifier sql-synthesizer-cluster --query 'DBClusters[0].Endpoint' --output text)

# Run migrations
export DATABASE_URL="postgresql://dbadmin:${DB_PASSWORD}@${DB_HOST}:5432/sql_synthesizer"

# Install migration dependencies
pip install alembic

# Run migrations
alembic upgrade head

echo "Database migrations completed successfully"
```

## Environment-Specific Configurations

### Production Values (`terraform/aws/environments/production.tfvars`)
```hcl
environment = "production"
aws_region = "us-west-2"

# Application
app_cpu = 1024
app_memory = 2048
app_count = 3
min_capacity = 2
max_capacity = 10

# Database
rds_instance_class = "db.r6g.large"
rds_instance_count = 2

# Monitoring
enable_detailed_monitoring = true
log_retention_days = 30
```

### Staging Values (`terraform/aws/environments/staging.tfvars`)
```hcl
environment = "staging"
aws_region = "us-west-2"

# Application
app_cpu = 512
app_memory = 1024
app_count = 1
min_capacity = 1
max_capacity = 3

# Database
rds_instance_class = "db.t4g.medium"
rds_instance_count = 1

# Monitoring
enable_detailed_monitoring = false
log_retention_days = 7
```

This Infrastructure as Code framework provides a comprehensive, production-ready deployment solution for SQL Query Synthesizer across multiple environments and cloud providers, with proper security, monitoring, and scalability built in.