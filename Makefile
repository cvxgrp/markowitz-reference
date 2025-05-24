# Makefile for the Markowitz reference implementation
# This Makefile provides commands for setting up the development environment,
# running experiments, and maintaining code quality.

.DEFAULT_GOAL := help

##@ Development Setup

venv: ## Create a virtual environment using uv
	# Download and install uv package manager
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	# Create a virtual environment
	@uv venv


.PHONY: install
install: venv ## Install a virtual environment
	# Upgrade pip to the latest version
	@uv pip install --upgrade pip
	# Install dependencies from requirements.txt
	@uv pip install -r requirements.txt

##@ Code Quality

.PHONY: fmt
fmt: venv ## Run autoformatting and linting
	# Install pre-commit for managing git hooks
	@uv pip install pre-commit
	# Install pre-commit hooks in the git repository
	@uv run pre-commit install
	# Run all pre-commit hooks on all files
	@uv run pre-commit run --all-files

##@ Dependencies

.PHONY: freeze
freeze: install  ## Freeze all requirements
	# Create a frozen requirements file with exact versions
	@uv pip freeze > requirements_frozen.txt

##@ Cleanup

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	# Remove all files ignored by git
	@git clean -X -d -f

##@ Experiments

.PHONY: experiments
experiments: install ## Run all experiment
	# Execute the experiments script
	@uv run python experiments.py

##@ Help

.PHONY: help
help:  ## Display this help screen
	# Display a header for the help message
	@echo -e "\033[1mAvailable commands:\033[0m"
	# Parse the Makefile to extract targets and their descriptions
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
