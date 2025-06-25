FROM backend-base:latest

WORKDIR /backend

COPY ../database ./database
COPY ../scripts ./scripts

VOLUME ["/backend/data"]

CMD ["python", "scripts/init_db.py"]