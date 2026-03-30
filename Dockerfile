# Single image used by both the backend and frontend services.
# The docker-compose.yml overrides the CMD for each service.
FROM python:3.13-slim

WORKDIR /app

# Install Poetry (2.x)
RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"

# Copy dependency manifests first for better layer caching
COPY pyproject.toml poetry.lock ./

# Install production dependencies only — skip the root package so the
# source-copy below is not required at this layer.
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction

# Copy source code (invalidates only this layer on code changes)
COPY . .

EXPOSE 8000 8501
