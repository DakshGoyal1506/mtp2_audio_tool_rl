setup:
	python -m pip install -e .
	python -m pip install -r requirements/base.txt

setup-audio:
	python -m pip install -r requirements/audio.txt

setup-rl:
	python -m pip install -r requirements/rl.txt

check:
	python scripts/setup/check_repo_setup.py

imports:
	python scripts/setup/check_imports.py

test:
	pytest -q

status:
	git status --short
