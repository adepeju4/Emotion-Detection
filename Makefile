.PHONY: install-api
install-api:
	@if command -v python3.11 >/dev/null 2>&1; then \
		python3.11 -m venv .venv; \
	elif command -v python3.12 >/dev/null 2>&1; then \
		python3.12 -m venv .venv; \
	else \
		python3 -m venv .venv; \
	fi
	.venv/bin/python -m pip install --upgrade pip setuptools wheel
	.venv/bin/pip install -r requirements-api.txt

.PHONY: dev
dev:
	@if [ -d ".venv" ]; then \
		.venv/bin/python -m uvicorn api:app --reload --reload-exclude .venv --port 8000; \
	elif [ -d "venv" ]; then \
		venv/bin/python -m uvicorn api:app --reload --reload-exclude .venv --port 8000; \
	else \
		@echo "Error: No virtual environment found. Run 'make install-api' first."; \
		exit 1; \
	fi

.PHONY: frontend
frontend:
	@echo "Starting React frontend..."
	cd client && npm run dev

install-frontend:
	cd client && npm install

