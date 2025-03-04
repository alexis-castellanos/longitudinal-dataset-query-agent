# Development Guide

This guide provides information for developers who want to contribute to or extend LongitudinalLLM.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/longitudinal-llm.git
cd longitudinal-llm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

4. Install development dependencies:
```bash
pip install pytest pytest-cov black flake8 mypy isort
```

## Project Structure

The project is organized as follows:

```
longitudinal-llm/
├── app.py                      # Main Streamlit application
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_models.py          # Pydantic models for data structures
│   ├── query_processor.py      # LLM query processing logic
│   ├── data_manager.py         # Dataset handling and transformations
│   └── utils.py                # Helper functions
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_data_models.py
│   ├── test_query_processor.py
│   └── test_data_manager.py
└── data/                       # Sample data and user data
    └── sample/                 # Sample datasets for demo
```

## Development Workflow

### Code Style

We follow PEP 8 guidelines. You can format your code with:

```bash
make format
```

### Running Tests

Run the test suite:

```bash
make test
```

This will run all tests and generate a coverage report.

### Adding New Features

1. **Create a new branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Implement your changes**:
   - Add new functionality
   - Write tests for your code
   - Update documentation

3. **Run tests and linting**:
```bash
make test
make lint
```

4. **Submit a pull request**

## Extending Functionality

### Adding Support for New Data Sources

To add support for a new data source type:

1. Extend the `DataManager` class in `src/data_manager.py`:
```python
class CustomDataManager(DataManager):
    def _load_custom_format(self, file_path):
        # Custom loading logic
        pass
```

2. Update the `_load_datasets_from_dir` method to detect and load your format

### Enhancing LLM Capabilities

To improve query understanding:

1. Modify prompts in the `QueryProcessor` class
2. Add more example queries and improve parsing logic

### Adding New Query Types

To add support for new query types:

1. Update the `ParsedQuery` model in `data_models.py`
2. Add parsing logic in `QueryProcessor.parse_query()`
3. Implement execution in `QueryProcessor.process_query()`

## Working with Ollama

LongitudinalLLM uses [Ollama](https://ollama.com/) for local LLM inference.

### Using Different Models

To use a different model:

1. Pull the model with Ollama:
```bash
ollama pull mistral
```

2. Update the model name in your code:
```python
query_processor = QueryProcessor(model_name="mistral")
```

### Creating a Custom Model

You can create a custom Ollama model with a Modelfile:

```
FROM llama3
SYSTEM "You are an assistant specialized in querying longitudinal datasets."
```

Pull the model:
```bash
ollama create longitudinal-llm-agent -f Modelfile
```

Use it in your code:
```python
query_processor = QueryProcessor(model_name="longitudinal-llm-agent")
```

## UI Customization

To customize the Streamlit UI:

1. Modify `app.py` to add new UI elements
2. Create a `.streamlit/config.toml` file for theme customization:
```toml
[theme]
primaryColor = "#007BFF"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## Deployment

### Local Deployment

For local deployment:

```bash
streamlit run app.py
```

### Docker Deployment

Build and run with Docker:

```bash
docker build -t longitudinal-llm .
docker run -p 8501:8501 longitudinal-llm
```

### Cloud Deployment

For cloud deployment (e.g., using Streamlit Sharing):

1. Push your code to GitHub
2. Connect your repository to Streamlit Sharing
3. Deploy your app

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Getting Help

If you need help or want to discuss development:

1. Open an issue on GitHub
2. Check the existing documentation
3. Contact the maintainers
