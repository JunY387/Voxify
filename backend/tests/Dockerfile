FROM backend-base:latest

WORKDIR /backend

COPY backend/api ./api
COPY backend/data ./data
COPY backend/database ./database

COPY backend/tests ./tests

# Run tests by default (override with docker run if needed)
CMD ["python", "-m", "unittest", "discover", "-s", "test"]