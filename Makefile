.PHONY: lint format typecheck run test clean

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy bot.py config.py main.py data/ execution/ risk/ theo/ utils/

run:
	python main.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
