"""Gunicorn configuration for SQL Query Synthesizer production deployment."""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"
worker_connections = 1000
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", 65))

# Restart workers after this many requests, to help prevent memory leaks
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", 50))

# Preload the application
preload_app = os.getenv("GUNICORN_PRELOAD", "true").lower() == "true"

# Threading
threads = int(os.getenv("GUNICORN_THREADS", 2))

# Logging
accesslog = "/app/logs/gunicorn_access.log"
errorlog = "/app/logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "sql-synthesizer"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = "/app/tmp"

# SSL (if certificates are provided)
keyfile = os.getenv("SSL_KEYFILE")
certfile = os.getenv("SSL_CERTFILE")

# Worker tmp directory
worker_tmp_dir = "/dev/shm"

# Graceful timeout
graceful_timeout = 30

# Limit the allowed size of an HTTP request header field
limit_request_field_size = 8190

# Limit the number of HTTP request header fields  
limit_request_fields = 100

# Limit the allowed size of an HTTP request line
limit_request_line = 4094

# The maximum size of HTTP request body
max_requests_jitter = 50

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("SQL Query Synthesizer server is ready. Listening on: %s", server.address)

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info("Worker received SIGINT or SIGQUIT. Shutting down gracefully.")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)
    
def pre_exec(server):
    """Called just before exec()."""
    server.log.info("Forked child, re-executing.")

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.info("Worker received SIGABRT signal")

def on_exit(server):
    """Called just before exiting."""
    server.log.info("SQL Query Synthesizer server is shutting down.")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading SQL Query Synthesizer server configuration.")

# Custom application
def application(environ, start_response):
    """Default application."""
    from sql_synthesizer.webapp import create_app
    app = create_app()
    return app(environ, start_response)