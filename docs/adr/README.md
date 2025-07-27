# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the SQL Synthesizer project.

## What are ADRs?

Architecture Decision Records (ADRs) are documents that capture important architectural decisions made during the development of this project, along with their context and consequences.

## ADR Format

Each ADR follows this structure:

```markdown
# ADR-XXXX: [Title]

## Status
[Proposed | Accepted | Rejected | Deprecated | Superseded by ADR-YYYY]

## Context
[What is the issue that we're seeing that is motivating this decision or change?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult to do because of this change?]
```

## Current ADRs

- [ADR-0001: Service Layer Architecture](./0001-service-layer-architecture.md) - Decision to implement service layer pattern
- [ADR-0002: Multi-Backend Caching Strategy](./0002-multi-backend-caching-strategy.md) - Caching architecture decisions
- [ADR-0003: Security-First Design](./0003-security-first-design.md) - Security architecture principles

## Creating New ADRs

1. Create a new file with format `XXXX-short-title.md`
2. Use the next sequential number
3. Follow the standard ADR template
4. Update this README with a link to the new ADR