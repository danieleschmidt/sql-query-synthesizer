# Multi-stage build for SQL Synthesizer
# Production-ready container with security and performance optimizations
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user for security
RUN groupadd -r sqlsynthuser && useradd -r -g sqlsynthuser sqlsynthuser

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/sqlsynthuser/.local

# Copy source code
COPY . .

# Set ownership to non-root user
RUN chown -R sqlsynthuser:sqlsynthuser /app

# Switch to non-root user
USER sqlsynthuser

# Add local bin to PATH
ENV PATH=/home/sqlsynthuser/.local/bin:$PATH

# Set Python path
ENV PYTHONPATH=/app

# Expose port (configurable via environment)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)" || exit 1

# Default command
CMD ["python", "-m", "sql_synthesizer.webapp", "--port", "5000", "--host", "0.0.0.0"]