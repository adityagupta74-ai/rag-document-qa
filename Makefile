.PHONY: install db down run seed examples large eval lint test

install:        ## Install Python dependencies
	pip install -r requirements.txt

db:             ## Start Postgres + pgvector (Docker)
	docker compose up -d

down:           ## Stop the database
	docker compose down

run:            ## Run the API with autoreload
	uvicorn app.main:app --reload

seed:           ## Ingest a sample doc and run a demo query
	python -m scripts.seed

examples:       ## Ingest the bundled example docs (handbook, FAQ, PDF)
	python -m scripts.ingest_examples

large:          ## Download a large public-domain book (Gutenberg) for scale testing
	python -m scripts.fetch_large_sample

eval:           ## Measure retrieval quality (hit-rate@k + MRR) vs a vector-only baseline
	python -m scripts.eval

lint:           ## Lint the codebase with ruff
	ruff check .

test:           ## Run the test suite
	pytest -q
