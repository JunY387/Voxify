FROM backend-base:latest

WORKDIR /app/backend

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# TODO: Consider removing curl, git, build-essential if not needed.
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsqlite3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV FLASK_APP=api/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

# Default command for dev — can override in docker-compose or with a production server like gunicorn
CMD ["flask", "run"]

# Production
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]