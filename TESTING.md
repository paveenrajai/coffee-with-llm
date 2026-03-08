# Testing Before Publishing

This guide covers how to test your package before publishing to PyPI.

## 1. Run All Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=ask_llm --cov-report=html
```

## 2. Check Package Structure

```bash
# Verify package can be discovered
python -c "import ask_llm; print(ask_llm.__version__ if hasattr(ask_llm, '__version__') else 'OK')"

# Check imports work
python -c "from ask_llm import AskLLM, AskLLMError, ConfigurationError, APIError, ValidationError; print('All imports OK')"
```

## 3. Build the Package

```bash
# Install build tools
pip install build twine

# Build source distribution and wheel
python -m build

# This creates:
# - dist/ask_llm-0.1.0.tar.gz (source distribution)
# - dist/ask_llm-0.1.0-py3-none-any.whl (wheel)
```

## 4. Check Built Package

```bash
# Check the built package for common issues
twine check dist/*

# This checks:
# - Package metadata
# - File permissions
# - File names
# - Package structure
```

## 5. Test Installation from Built Package

```bash
# Create a clean virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from the built wheel
pip install dist/ask_llm-0.1.0-py3-none-any.whl

# Or install from source distribution
pip install dist/ask_llm-0.1.0.tar.gz

# Verify installation
python -c "import ask_llm; print(ask_llm.__version__ if hasattr(ask_llm, '__version__') else 'OK')"
python -c "from ask_llm import AskLLM; print('Import successful')"

# Run tests after installation
pip install pytest pytest-asyncio
pytest tests/

# Clean up
deactivate
rm -rf test_env
```

## 6. Test Package Metadata

```bash
# Check what will be displayed on PyPI
python -m build --sdist --wheel
python -c "
import tomli
with open('pyproject.toml', 'rb') as f:
    data = tomli.load(f)
    print('Package name:', data['project']['name'])
    print('Version:', data['project']['version'])
    print('Description:', data['project']['description'])
    print('Authors:', data['project']['authors'])
    print('License:', data['project']['license'])
"
```

## 7. Verify README Renders Correctly

```bash
# Check if README.md will render correctly on PyPI
# PyPI uses reStructuredText or Markdown
# Your README.md should be in Markdown format (which PyPI supports)
```

## 8. Test in Different Python Versions (Optional)

```bash
# Test with Python 3.10
python3.10 -m venv test_env_310
source test_env_310/bin/activate
pip install dist/ask_llm-0.1.0-py3-none-any.whl
python -c "import ask_llm; print('Python 3.10 OK')"
deactivate

# Test with Python 3.11, 3.12, 3.13 similarly
```

## 9. Dry Run Upload (Test PyPI)

```bash
# Upload to Test PyPI first (recommended!)
# Create account at https://test.pypi.org/

# Upload to Test PyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Test install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ ask-llm
```

## 10. Final Checklist

Before publishing to production PyPI:

- [ ] All tests pass (`pytest`)
- [ ] Package builds without errors (`python -m build`)
- [ ] `twine check` passes
- [ ] Package installs correctly in clean environment
- [ ] All imports work after installation
- [ ] README.md renders correctly
- [ ] Version number is correct
- [ ] Author/email information is correct
- [ ] License file is included
- [ ] No sensitive information in package
- [ ] Tested on Test PyPI (optional but recommended)

## Common Issues

### Missing Files
If files are missing, check `MANIFEST.in`:
```bash
cat MANIFEST.in
```

### Import Errors After Installation
Check that `__init__.py` files exist in all packages:
```bash
find ask_llm -name "__init__.py"
```

### Version Issues
Make sure version in `pyproject.toml` matches what you want to publish.

