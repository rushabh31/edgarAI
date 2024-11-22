SHELL:=/bin/bash
VENV_NAME:=edgar-venv

## ----------------------------------------------------------------------
## Makefile with recipes
## ----------------------------------------------------------------------

help:	## Show help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

########################################################
# Local developement recipes
########################################################

venv:	## Create python environment, install pre-commit hooks, install dbt dps, and create .env

	@if [ -d "/opt/python/3.10.13" ]; then \
		echo "/opt/python/3.10.13 found. Building venv using this python."; \
		/opt/python/3.10.13/bin/python3 -m venv $(VENV_NAME); \
	else \
		echo "/opt/python/3.10.13 not found. Building venv using python3."; \
		python3 -m venv $(VENV_NAME); \
	fi
	source $(VENV_NAME)/bin/activate; \
	pip install wheel;\
	pip install -r requirements.txt; \
	pre-commit install; \
	python -m ipykernel install --name $(VENV_NAME) --display-name $(VENV_NAME) --user;

