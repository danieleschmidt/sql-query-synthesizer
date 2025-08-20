
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "regions" {
  description = "List of AWS regions for deployment"
  type        = list(string)
  default     = ["us-east-1", "eu-west-1", "ap-southeast-1", "ap-northeast-1"]
}

# Multi-region EKS clusters
module "eks_clusters" {
  source = "./modules/eks"
  
  for_each = toset(var.regions)
  
  region       = each.value
  cluster_name = "sql-synthesizer-${each.value}"
  
  node_groups = {
    main = {
      min_size       = 2
      max_size       = 100
      desired_size   = 3
      instance_types = ["t3.medium"]
    }
  }
  
  tags = {
    Environment = "production"
    Project     = "sql-synthesizer"
    Region      = each.value
  }
}

# Global Load Balancer
resource "aws_route53_zone" "main" {
  name = "sql-synthesizer.com"
  
  tags = {
    Environment = "production"
  }
}

# WAF for DDoS protection
resource "aws_wafv2_web_acl" "sql_synthesizer" {
  name  = "sql-synthesizer-waf"
  scope = "CLOUDFRONT"
  
  default_action {
    allow {}
  }
  
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    action {
      block {}
    }
    
    statement {
      rate_based_statement {
        limit              = 10000
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      sampled_requests_enabled   = true
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
    }
  }
}

# CloudFront distribution for global CDN
resource "aws_cloudfront_distribution" "sql_synthesizer" {
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  web_acl_id          = aws_wafv2_web_acl.sql_synthesizer.arn
  
  origin {
    domain_name = "api.sql-synthesizer.com"
    origin_id   = "sql-synthesizer-api"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "sql-synthesizer-api"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type"]
      
      cookies {
        forward = "none"
      }
    }
  }
  
  geo_restriction {
    restriction_type = "none"
  }
  
  viewer_certificate {
    acm_certificate_arn = aws_acm_certificate.ssl_cert.arn
    ssl_support_method  = "sni-only"
  }
}

# SSL Certificate
resource "aws_acm_certificate" "ssl_cert" {
  domain_name       = "*.sql-synthesizer.com"
  validation_method = "DNS"
  
  lifecycle {
    create_before_destroy = true
  }
}
