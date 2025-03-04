# Contributing to LongitudinalLLM

Thank you for considering contributing to LongitudinalLLM! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and inclusive when contributing to this project. We welcome contributions from everyone who wishes to improve the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with the following information:

1. A clear, descriptive title
2. A detailed description of the bug
3. Steps to reproduce the issue
4. Expected behavior
5. Actual behavior
6. Screenshots or logs if applicable
7. Environment information (OS, Python version, etc.)

### Suggesting Features

We welcome feature suggestions! Please create an issue with:

1. A clear, descriptive title
2. A detailed description of the proposed feature
3. Any relevant examples or use cases
4. If possible, a high-level implementation approach

### Pull Requests

Follow these steps to submit a pull request:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Pull Request Guidelines

When submitting a pull request:

- Ensure your code follows the project's style guidelines
- Write or update tests for the changes you make
- Update documentation as needed
- Keep your pull requests focused and specific
- Reference any relevant issues in your PR description

## Development Setup

See [docs/development.md](docs/development.md) for detailed instructions on setting up your development environment.

## Code Style Guidelines

We follow PEP 8 style guidelines for Python code. You can use the following tools:

- `black` for code formatting
- `flake8` for linting
- `isort` for import sorting
- `mypy` for type checking

Run all formatting tools with:

```bash
make format
```

## Testing

All new features or bug fixes should include tests. To run tests:

```bash
make test
```

## Documentation

When adding or modifying features, please update the relevant documentation:

- Update the README.md if necessary
- Update or add documentation in the docs/ directory
- Add docstrings to public functions, classes, and methods

## Commit Messages

Write clear and meaningful commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Fix bug" not "Fixes bug")
- Reference issues and pull requests where appropriate

## Release Process

The maintainers will handle the release process, including:

1. Updating the version number
2. Creating release notes
3. Publishing the release

## Questions?

If you have any questions about contributing, please open an issue or contact the maintainers.

Thank you for contributing to LongitudinalLLM!
