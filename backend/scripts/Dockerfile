FROM backend-base:latest

WORKDIR /backend/scripts

COPY backend/scripts/init_db.py .
COPY backend/database ../database
COPY backend/data ../data 

CMD ["python", "init_db.py"]