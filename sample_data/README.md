# Sample Data for SQL Synthesizer Demo

This directory contains sample data and example queries for demonstrating the SQL Synthesizer in a demo environment.

## Files

### `demo_queries.txt`
Contains a comprehensive list of example natural language queries that demonstrate the capabilities of the SQL Synthesizer. These queries cover:

- Basic data retrieval
- Aggregations and counting
- Join operations across tables
- Time-based filtering
- Complex analytical queries
- Industry-specific examples

## Using in Demo Environment

### For Demo Deployment
1. These sample queries can be displayed in the web interface as suggestions
2. They serve as examples for users to understand the system's capabilities
3. Can be used for automated testing of the demo environment

### Integration with Web Interface
The demo queries can be integrated into the web interface by:

1. **Displaying example queries** on the main page
2. **Quick-fill buttons** that populate the input field
3. **Random query suggestions** for user inspiration
4. **Category-based query browsing**

### Sample Database Setup
For a complete demo, you would typically want to:

1. Create a sample SQLite database with realistic data
2. Include tables like:
   - `users` (customer information)
   - `orders` (transaction data)
   - `products` (inventory)
   - `categories` (product categories)

## Demo Environment Considerations

### Rate Limiting
The production application already includes rate limiting for public access:
- Configurable via `QUERY_AGENT_RATE_LIMIT_PER_MINUTE`
- Default: 60 requests per minute per client

### Security
- API key authentication can be enabled
- CSRF protection is built-in
- Input validation prevents SQL injection

### Monitoring
- Health checks available at `/health`
- Prometheus metrics at `/metrics`
- Security audit logging built-in

## Containerization

The Dockerfile in the project root can be used to containerize the application for demo deployment:

```bash
# Build the container
docker build -t sql-synthesizer .

# Run the demo
docker run -p 5000:5000 sql-synthesizer
```

## Environment Variables for Demo

Key environment variables for demo deployment:
- `QUERY_AGENT_WEBAPP_PORT=5000`
- `QUERY_AGENT_RATE_LIMIT_PER_MINUTE=30` (more restrictive for public demo)
- `QUERY_AGENT_API_KEY_REQUIRED=false` (for easy demo access)
- `DATABASE_URL=sqlite:///demo.db` (demo database)

## Extending the Demo

To enhance the demo experience:
1. Add more industry-specific query examples
2. Create interactive tutorials
3. Implement query history
4. Add export functionality for results