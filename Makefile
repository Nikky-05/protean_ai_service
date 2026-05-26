.PHONY: install install-dev run test lint format docker

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -v

lint:
	ruff check app tests

format:
	ruff check --fix app tests
	ruff format app tests

docker:
	docker compose up --build
