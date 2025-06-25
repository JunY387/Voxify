#!/bin/bash
set -e

echo "Building backend base image..."
docker build -f backend/Dockerfile.base -t backend-base:latest backend

echo "Building compose services..."
docker compose -f docker-compose.yml build

echo "Running database initialization..."
docker compose -f docker-compose.yml run --rm db-init

echo "Running tests..."
docker compose -f docker-compose.yml run --rm tests

echo "Cleaning up..."
docker compose -f docker-compose.yml down --volumes

echo "All tests passed!"