# Terraform Infrastructure for SQL Synthesizer
# Production-ready cloud deployment configuration

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
  
  # Backend configuration for state management
  backend "s3" {
    bucket         = var.terraform_state_bucket
    key            = "sql-synthesizer/terraform.tfstate"
    region         = var.aws_region
    dynamodb_table = var.terraform_lock_table
    encrypt        = true
  }
}

# Provider configurations
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "sql-synthesizer"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner_team
    }
  }
}

# Local values for resource naming
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = var.owner_team
  }
}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"
  
  name_prefix = local.name_prefix
  vpc_cidr    = var.vpc_cidr
  
  availability_zones = var.availability_zones
  private_subnets    = var.private_subnets
  public_subnets     = var.public_subnets
  
  enable_nat_gateway = var.enable_nat_gateway
  enable_vpn_gateway = var.enable_vpn_gateway
  
  tags = local.common_tags
}

# ECS Cluster for containerized application
module "ecs" {
  source = "./modules/ecs"
  
  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  
  private_subnet_ids = module.vpc.private_subnet_ids
  public_subnet_ids  = module.vpc.public_subnet_ids
  
  # Application configuration
  app_image_uri    = var.app_image_uri
  app_port         = var.app_port
  app_cpu          = var.app_cpu
  app_memory       = var.app_memory
  app_desired_count = var.app_desired_count
  
  # Environment variables
  environment_variables = var.environment_variables
  secret_variables     = var.secret_variables
  
  # Load balancer configuration
  certificate_arn = var.ssl_certificate_arn
  domain_name     = var.domain_name
  
  tags = local.common_tags
}

# RDS PostgreSQL database
module "database" {
  source = "./modules/rds"
  
  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  
  private_subnet_ids = module.vpc.private_subnet_ids
  
  # Database configuration
  engine_version    = var.db_engine_version
  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  storage_encrypted = true
  
  database_name = var.database_name
  username      = var.db_username
  
  # Security
  allowed_security_groups = [module.ecs.app_security_group_id]
  
  # Backup and maintenance
  backup_retention_period = var.db_backup_retention_period
  backup_window          = var.db_backup_window
  maintenance_window     = var.db_maintenance_window
  
  # Monitoring
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  tags = local.common_tags
}

# ElastiCache Redis for caching
module "redis" {
  source = "./modules/elasticache"
  
  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  
  private_subnet_ids = module.vpc.private_subnet_ids
  
  # Redis configuration
  node_type         = var.redis_node_type
  num_cache_nodes   = var.redis_num_nodes
  engine_version    = var.redis_engine_version
  
  # Security
  allowed_security_groups = [module.ecs.app_security_group_id]
  
  # Backup
  snapshot_retention_limit = var.redis_snapshot_retention_limit
  snapshot_window         = var.redis_snapshot_window
  
  tags = local.common_tags
}

# S3 bucket for application assets and logs
resource "aws_s3_bucket" "app_assets" {
  bucket = "${local.name_prefix}-assets"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "app_assets" {
  bucket = aws_s3_bucket.app_assets.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app_assets" {
  bucket = aws_s3_bucket.app_assets.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "app_assets" {
  bucket = aws_s3_bucket.app_assets.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "app_logs" {
  name              = "/aws/ecs/${local.name_prefix}"
  retention_in_days = var.log_retention_days
  
  tags = local.common_tags
}

# Secrets Manager for sensitive configuration
resource "aws_secretsmanager_secret" "app_secrets" {
  name = "${local.name_prefix}-secrets"
  
  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    database_url = module.database.connection_string
    redis_url    = module.redis.connection_string
    openai_api_key = var.openai_api_key
  })
  
  lifecycle {
    ignore_changes = [secret_string]
  }
}

# CloudWatch Dashboard for monitoring
resource "aws_cloudwatch_dashboard" "app_dashboard" {
  dashboard_name = "${local.name_prefix}-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ServiceName", "${local.name_prefix}-service", "ClusterName", "${local.name_prefix}-cluster"],
            [".", "MemoryUtilization", ".", ".", ".", "."]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "ECS Service Metrics"
        }
      },
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", module.database.instance_id],
            [".", "DatabaseConnections", ".", "."],
            [".", "ReadLatency", ".", "."],
            [".", "WriteLatency", ".", "."]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "RDS Metrics"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "${local.name_prefix}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ECS CPU utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ServiceName = "${local.name_prefix}-service"
    ClusterName = "${local.name_prefix}-cluster"
  }
  
  tags = local.common_tags
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "${local.name_prefix}-alerts"
  
  tags = local.common_tags
}

# Output values
output "application_url" {
  description = "URL of the deployed application"
  value       = "https://${var.domain_name}"
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = module.database.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.redis.endpoint
  sensitive   = true
}

output "s3_bucket_name" {
  description = "S3 bucket name for assets"
  value       = aws_s3_bucket.app_assets.bucket
}