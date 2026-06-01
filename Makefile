.PHONY: install hooks test lint typecheck format build clean

install:
	pip install -e ".[dev]"

hooks:
	pre-commit install

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=nvsonar --cov-report=term-missing

lint:
	black --check src/ tests/
	isort --check-only src/ tests/

typecheck:
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
