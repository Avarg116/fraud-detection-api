# Short commands for common tasks

THRESHOLD ?= 0.99

.PHONY: help test train compare up down logs dev psql

help:
	@echo make test      - run the tests
	@echo make train     - train the model and save its threshold ($(THRESHOLD))
	@echo make compare   - logistic regression vs LightGBM with bootstrap
	@echo make up        - build and start Postgres + API in Docker (API on :8020)
	@echo make down      - stop the containers (data is kept)
	@echo make logs      - follow the API logs
	@echo make dev       - run the API locally with auto-reload (:8010, needs Postgres up)
	@echo make psql      - open a SQL shell in the database

test:
	uv run pytest


train:
	uv run python -m fraud.training.train
	uv run python -m fraud.training.threshold --set $(THRESHOLD)

compare:
	uv run python -m fraud.training.compare

up:
	docker compose up -d --build --wait

down:
	docker compose down

logs:
	docker compose logs -f api

dev:
	uv run uvicorn fraud.api.app:app --reload --port 8010 --env-file .env

psql:
	docker compose exec postgres sh -c "psql -U $$POSTGRES_USER -d $$POSTGRES_DB"
