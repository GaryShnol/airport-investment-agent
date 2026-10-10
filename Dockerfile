# Airport Investment Agent: Streamlit app. The DB is not in the image: on start, src.bootstrap
# downloads the seed DB (SEED_DB_URL) into the /app/data volume, or uses a DB already there.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    APP_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

RUN useradd --create-home --uid 10001 app
WORKDIR /app

# Dependencies first (cached while pyproject.toml and uv.lock are unchanged).
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY app.py ./
COPY src/ src/
COPY config/ config/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/data && chown -R app:app /app/data

USER app
VOLUME ["/app/data"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['APP_PORT'] + '/_stcore/health', timeout=4)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
