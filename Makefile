.PHONY: install build serve dev test e2e lint

install: ## Install Python and site dependencies
	uv sync
	cd site && npm ci

build: ## Build the lesson site into site/dist
	cd site && npm run build

serve: ## Serve the built site and API on HOST:PORT (default 127.0.0.1:8000)
	uv run kd-course-serve

dev: ## Live-reloading site for lesson authors (no API)
	cd site && npm run dev

lint:
	uv run ruff check .
	uv run ruff format --check .

test: lint ## Python tests, site unit tests and type check
	uv run pytest -q
	cd site && npm test && npm run check

e2e: build ## Browser smoke test against the real server
	cd site && npx playwright test --trace off
