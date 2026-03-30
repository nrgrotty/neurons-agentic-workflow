# Neurons agentic workflow


## Local set up (for development)

### Build
- Install python (see `pyproject.toml for supported python versions)
- Install poetry (https://python-poetry.org/docs/#installation)
- Install environment: `poetry install`
- Add API keys as an environmment variables GOOGLE_API_KEY and LANGSMITH_API_KEY
- Add other environmental variables: LANGSMITH_TRACING and LANGSMITH_PROJECT

### Run
- `poetry run python ./main.py` in one terminal
- `poetry run streamlit run frontend.py` in  another terminal

## To deploy

### Install docker

- `sudo apt update`
- `sudo apt install -y docker.io docker-compose`
- `sudo systemctl enable docker`
- `sudo usermod -aG docker $USER`

### Build
- Set environment variables in .env file
    - GOOGLE_API_KEY
    - LANGSMITH_API_KEY
    - LANGSMITH_TRACING
    - LANGSMITH_PROJECT
- `docker-compose build`

### Deploy
- `docker-compose up -d`

### Tear down
- `docker-compose down`

## Overview
Agentic workflow to edit creative images based on recommendations.

The pipeline takes as input: 

- 1 creative image
- 1 set of brand guidelines
- List of recommendations

and it outputs:

- Edited variants for each of the recommendations, together with a selected best variant.
- Audit log

It runs each of the recommendations in parallel through an agentic workflow that:
- Creates a plan to edit the recommendation and a plan to evaluate them.
- Spawns 3 variants per recommendation, with an editor, critic and refiner, that implement an "evalutor-optimizer" loop
- Uses the evaluation plan to rank the different variants and select the best


## Notes
The assignment is being suggested as a part of a job interview process for "AI engineer" in Neurons. 

I focused on making the agentic workflow backend. Frontend is implemented with vibe coding, just for demo purposes. 

Tracing and monitoring is done using langsmith. You can link a langsmith project through the environment variables (LANGSMITH_PROJECT and LANGSMITH_API). If you prefer to run it without traces, then set LANGSMITH_TRACING to false. 