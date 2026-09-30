USER_NAME := $(shell whoami)
export UV_CACHE_DIR = /goinfre/$(USER_NAME)/.cache/uv
export HF_HOME = /goinfre/$(USER_NAME)/.cache/huggingface

KEEP = .git .gitignore README.md uv.lock cache.py\
		pyproject.toml Makefile needed.md src __init__.py\
		UnansweredQuestions AnsweredQuestions data raw datasets\
		cli.py models.py utils.py __main__.py vllm-0.10.1 llm.py server.py\

ARG ?= ""
VENV_DIR = .venv

install:
	uv sync

run: install
	uv run python3 -m src ${ARG}

debug:
	uv run python3 -m pdb src/__main__.py

clean:
	rm -rf $(UV_CACHE_DIR)
	rm -rf $(HF_HOME)
	rm -rf .venv
	find . -path ./.git -prune -o -maxdepth 2 ! -name "." $(foreach item,$(KEEP),! -name "$(item)") -exec rm -rf {} +  2>/dev/null || true

re: clean install

lint:
	uv run python3 -m flake8 . --exclude=$(VENV_DIR), data/raw/vllm-0.10.1/
	uv run python3 -m mypy . --exclude='($(VENV_DIR)| data/raw/vllm-0.10.1/)' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	uv run python3 -m flake8 . --exclude=$(VENV_DIR),data/raw/vllm-0.10.1/
	uv run python3 -m mypy . --exclude='($(VENV_DIR)|data/raw/vllm-0.10.1/)' --strict

.PHONY: install run debug clean lint lint-strict re
