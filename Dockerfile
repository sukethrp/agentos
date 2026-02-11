FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS-level dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy application source
COPY src ./src
COPY examples ./examples
COPY plugins ./plugins

EXPOSE 8000

CMD ["uvicorn", "agentos.web.app:app", "--host", "0.0.0.0", "--port", "8000"]

