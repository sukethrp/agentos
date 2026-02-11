#!/usr/bin/env bash
set -e

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Please install Docker first." >&2
  exit 1
fi

COMPOSE_CMD="docker compose"
if ! $COMPOSE_CMD version >/dev/null 2>&1; then
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
  else
    echo "Neither 'docker compose' nor 'docker-compose' was found on PATH." >&2
    exit 1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Building and starting AgentOS with Docker..."
$COMPOSE_CMD up -d --build

echo
echo "AgentOS is starting in Docker."
echo "Once the container is healthy, access the web UI at: http://localhost:8000"

