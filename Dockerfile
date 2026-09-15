# API image (FastAPI + the model). The web app deploys separately (Vercel).
#
#   docker build -t fse-api .
#   docker run -p 8000:8000 fse-api
#
# INSTALL_ML=true adds the optional ML layer (deal risk, live sliders, macro
# regime). It makes the image ~1 GB larger and needs more than 512 MB of RAM
# at run time because of torch; the app works without it.
FROM python:3.12-slim

ARG INSTALL_ML=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
 && if [ "$INSTALL_ML" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install -r requirements-ml.txt; \
    fi

COPY analytics ./analytics
COPY api ./api
COPY core ./core
COPY lbo_engine ./lbo_engine
COPY ml ./ml
COPY simulation ./simulation

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health', timeout=4)"

# Render (and most hosts) set PORT; --proxy-headers because TLS ends at the host's proxy
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
