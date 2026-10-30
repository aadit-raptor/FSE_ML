# API image (FastAPI + the model). The web app deploys separately (Vercel).
#
#   docker build -t fse-api .
#   docker run -p 8000:8000 fse-api
#
# One image runs everything, the live sliders included (PLAN.md 5.8): no
# feature needs PyTorch or any other ML package.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY analytics ./analytics
COPY api ./api
COPY benchmarks ./benchmarks
COPY companies ./companies
COPY core ./core
COPY db ./db
COPY economy ./economy
COPY jobs ./jobs
COPY lbo_engine ./lbo_engine
COPY library ./library
COPY ml ./ml
COPY research ./research
COPY simulation ./simulation
COPY teams ./teams
COPY uploads ./uploads
COPY validation ./validation
# The words of an investment memo, in every language the web has (PLAN.md 6.3)
COPY web/messages ./web/messages

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health', timeout=4)"

# Render (and most hosts) set PORT; --proxy-headers because TLS ends at the host's proxy.
# --no-access-log: the API writes its own JSON request log with request IDs (api/observability.py)
# --no-server-header: no "server: uvicorn" advertising the stack (PLAN.md 1.7)
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --no-access-log --no-server-header"]
