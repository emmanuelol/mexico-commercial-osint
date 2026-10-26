.PHONY: install notebook ingest resolve score test check-blacklist build migrate-cells clean demo

PYTHON = $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

demo:
	$(PYTHON) examples/01_quickstart_pipeline.py

install:
	pip install -e ".[dev]"
	pre-commit install --install-hooks

notebook:
	jupyter lab notebooks/

ingest:
	jupyter nbconvert --to notebook --execute notebooks/01_data_ingest.ipynb --inplace

resolve:
	jupyter nbconvert --to notebook --execute notebooks/02_entity_resolution.ipynb --inplace

score:
	jupyter nbconvert --to notebook --execute notebooks/03_scoring_export.ipynb --inplace

test:
	pytest tests/ -v --cov=src/mx_osint

check-blacklist:
	bash scripts/check_blacklist.sh

build:
	docker build -t mexico-commercial-osint:latest .

migrate-cells:
	python scripts/migrate_cells.py --notebooks notebooks/ --target src/mx_osint/

clean:
	rm -rf data/raw/* data/processed/* .pytest_cache .coverage
