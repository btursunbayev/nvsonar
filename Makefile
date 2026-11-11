.PHONY: install hooks test lint typecheck format build clean go-build go-test go-lint

install:
	pip install -e ".[dev]"

hooks:
	pre-commit install

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=nvsonar --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

build:
	python -m build

go-build:
	cd exporter-go && go build -o bin/nvsonar-exporter ./cmd/nvsonar-exporter

go-test:
	cd exporter-go && go test ./...

go-lint:
	cd exporter-go && test -z "$$(gofmt -l .)" && go vet ./...

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info exporter-go/bin/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
