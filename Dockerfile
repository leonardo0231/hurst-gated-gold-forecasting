FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HGE_TRADING_MODE=offline
WORKDIR /app
RUN pip install --no-cache-dir uv==0.8.14
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY configs ./configs
COPY scripts ./scripts
RUN mkdir -p data artifacts models reports
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]
CMD ["/app/.venv/bin/uvicorn", "hge_gold.api:app", "--host", "0.0.0.0", "--port", "8000"]
