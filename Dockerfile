FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TIANLONG_DATA_DIR=/data
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[web]' && useradd --create-home --uid 1000 player \
    && mkdir /data && chown player:player /data
USER player
EXPOSE 8000
CMD ["python", "-m", "tianlong.runtime.web", "--host", "0.0.0.0"]
