# Neurons agentic workflow


## Local set up

### Build
- Install python (see `pyproject.toml for supported python versions)
- Install poetry (https://python-poetry.org/docs/#installation)
- Install environment: `poetry install`
- Add API keys as an environmment variables 
    GOOGLE_API_KEY
    LANGSMITH_API_KEY
    LANGSMITH_TRACING 
    LANGSMITH_PROJECT

### Run
- `poetry run python ./main.py`
- You can access the docs through the browser

## To deploy
This repo is synchronized with a service in Render. The configuration is found in render.yaml.
It deploys automatically for every commit.


## Overview
TODO: Write overview of what this is doing.



