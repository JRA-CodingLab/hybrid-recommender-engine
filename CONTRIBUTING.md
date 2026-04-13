# Contributing

Thanks for your interest in contributing to **Hybrid Recommender Engine**!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/hybrid-recommender-engine.git
   cd hybrid-recommender-engine
   ```
3. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run the test suite:
   ```bash
   pytest -v --cov=src
   ```
4. Lint your code:
   ```bash
   ruff check src/ tests/
   ```
5. Commit with a descriptive message:
   ```bash
   git commit -m "feat: add your feature description"
   ```
6. Push and open a Pull Request against `main`

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints for all function signatures
- Format with `ruff format`
- Keep functions focused and under ~50 lines where possible

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `refactor:` — code change without feature/fix
- `chore:` — maintenance tasks

## Reporting Issues

Open an issue on GitHub with:

- A clear title
- Steps to reproduce (if applicable)
- Expected vs. actual behaviour
- Environment details (Python version, OS)

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
