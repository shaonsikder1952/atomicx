# Contributing to AtomicX

Thank you for your interest in AtomicX.

## Contribution Policy

This is a **private research repository**. Contributions are currently **by invitation only**.

## For Collaborators

If you've been granted collaborator access:

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/[username]/atomicx.git
   cd atomicx
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[all]"
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize database**
   ```bash
   # Start PostgreSQL with TimescaleDB
   brew install timescaledb  # macOS
   psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
   
   # Run migrations
   python run_migration.py
   ```

### Code Standards

- **Python**: 3.11+ required
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Google-style docstrings for all public functions
- **Formatting**: Code will be auto-formatted (configuration in pyproject.toml)
- **Imports**: Organize imports (stdlib → third-party → local)

### Commit Messages

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

**Examples**:
```
feat(causal): Add NOTEARS algorithm implementation
fix(swarm): Correct agent initialization bug
docs(readme): Update installation instructions
```

### Pull Request Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following code standards

3. **Test locally**
   ```bash
   pytest tests/
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create PR on GitHub with clear description

6. **Code review** - Wait for review and address feedback

### Architecture Guidelines

- **Layer separation**: Respect the 14-layer architecture boundaries
- **Domain-agnostic**: Keep variable definitions and layer logic domain-agnostic
- **Evolution-first**: All agents must implement performance tracking
- **Async-first**: Use async/await for I/O operations
- **Type safety**: Pydantic models for data structures

### Testing

- Write tests for new features
- Maintain test coverage above 80%
- Use pytest fixtures for common setup
- Integration tests for layer interactions

### Documentation

- Update README.md if changing architecture
- Add docstrings to new functions
- Update `docs/` for major feature additions
- Keep `docs/example_domains.py` updated if adding domain examples

## Questions?

Contact the repository maintainer through GitHub issues or direct message.

## License

By contributing, you agree that your contributions will be licensed under the same proprietary license as the project.
