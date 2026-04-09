.PHONY: docs docs-live docs-linkcheck

docs:
	python -m pip install -r docs/requirements.txt
	sphinx-build -W -b html docs docs/_build/html

docs-live:
	python -m pip install -r docs/requirements.txt
	sphinx-autobuild docs docs/_build/html

docs-linkcheck:
	python -m pip install -r docs/requirements.txt
	sphinx-build -W -b linkcheck docs docs/_build/linkcheck
