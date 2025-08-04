# Makefile for the Markowitz reference implementation
# This Makefile provides commands for setting up the development environment,
# running experiments, and maintaining code quality.

.DEFAULT_GOAL := help

##@ Development Setup

uv: ## Create a virtual environment using uv
	# Download and install uv package manager
	@curl -LsSf https://astral.sh/uv/install.sh | sh

##@ Code Quality

.PHONY: fmt
fmt: uv ## Run autoformatting and linting
	# Run all pre-commit hooks on all files
	@uvx run pre-commit run --all-files

##@ Cleanup

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	# Remove all files ignored by git
	@git clean -X -d -f

##@ Experiments

.PHONY: experiments
experiments: uv ## Run all experiment
	# Execute the experiments script
	@uv run experiments.py

##@ Help

.PHONY: help
help:  ## Display this help screen
	# Display a header for the help message
	@echo -e "\033[1mAvailable commands:\033[0m"
	# Parse the Makefile to extract targets and their descriptions
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
