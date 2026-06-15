.PHONY: help install backend-install frontend-install demo api web test test-backend test-frontend lint format build

help:
	@echo "NeuroForge — research/simulation only (not a medical device)"
	@echo ""
	@echo "  make install           Install backend + frontend dependencies"
	@echo "  make demo              Run the headless closed-loop CLI demo"
	@echo "  make api               Start the FastAPI server (http://localhost:8000)"
	@echo "  make web               Start the Vite dev server (http://localhost:5173)"
	@echo "  make test              Run backend + frontend test suites"
	@echo "  make lint              Lint backend (ruff) + typecheck frontend"
	@echo "  make format            Auto-format backend (ruff --fix + black)"
	@echo "  make build             Production build of the frontend"

install: backend-install frontend-install

backend-install:
	cd backend && pip install -e ".[dev]"

frontend-install:
	cd frontend && npm install

demo:
	cd backend && python -m neuroforge.cli demo

api:
	cd backend && uvicorn neuroforge.api.app:app --reload

web:
	cd frontend && npm run dev

test: test-backend test-frontend

test-backend:
	cd backend && pytest

test-frontend:
	cd frontend && npm test

lint:
	cd backend && ruff check neuroforge tests
	cd frontend && npm run typecheck

format:
	cd backend && ruff check --fix neuroforge tests && black neuroforge tests

build:
	cd frontend && npm run build
