.DEFAULT_GOAL := help

SHELL=/bin/bash

UNAME=$(shell uname -s)

.PHONY: install
install:  ## Install a virtual environment
	@poetry install -vv

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	@poetry run pip install pre-commit
	@poetry run pre-commit install
	@poetry run pre-commit run --all-files

.PHONY: test
test: install ## Run tests
	@poetry run pytest

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@git clean -X -d -f

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort

.PHONY: jupyter
jupyter: install ## Run jupyter lab
	@poetry run pip install jupyterlab
	@poetry run jupyter lab
