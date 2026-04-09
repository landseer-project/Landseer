# Database Setup

Landseer defaults to SQLite and works without any DB setup.

Use MySQL if you need cross-run analytics or shared persistence.

## MySQL via Docker

```bash
docker run -d --name landseer-mysql \
  -e MYSQL_ROOT_PASSWORD=rootpass \
  -e MYSQL_DATABASE=landseer_pipeline \
  -e MYSQL_USER=landseer \
  -e MYSQL_PASSWORD=landseer \
  -p 3306:3306 \
  mysql:8.0
```

Set env vars before starting backend:

```bash
export LANDSEER_DB_TYPE=mysql
export LANDSEER_DB_HOST=localhost
export LANDSEER_DB_PORT=3306
export LANDSEER_DB_NAME=landseer_pipeline
export LANDSEER_DB_USER=landseer
export LANDSEER_DB_PASSWORD=landseer
```

Then run backend normally.

For legacy detailed notes, see `docs/DATABASE_SETUP.md`.
