# ──────────────────────────────────────────────────────────────────────────────
# Lead-Ops · Dockerfile
# ──────────────────────────────────────────────────────────────────────────────
# Optimized for Hugging Face Spaces (8GB RAM limit, port 7860)
# Base: python:3.11-slim (~150MB)
# Final image target: < 500MB
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System setup ─────────────────────────────────────────────────────────────

# Prevent Python from buffering stdout/stderr (important for HF logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Create app directory
WORKDIR /app

# ── Install dependencies first (cache layer) ────────────────────────────────

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Copy application code ───────────────────────────────────────────────────

COPY . .

# ── Seed the database at build time ─────────────────────────────────────────
# This pre-populates master.db so the container starts fast (<60s)

RUN python database_init.py && \
    python scripts/dirty_seeder.py && \
    python scripts/interaction_generator.py

# ── Verify database was created ─────────────────────────────────────────────

RUN python -c "from pathlib import Path; \
    db = Path('master.db'); \
    assert db.exists(), 'master.db not created'; \
    size_mb = db.stat().st_size / (1024*1024); \
    print(f'master.db size: {size_mb:.2f} MB'); \
    assert size_mb < 100, f'DB too large: {size_mb:.2f} MB'"

# Create an unprivileged runtime user
RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Health check ─────────────────────────────────────────────────────────────

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:7860/health'); r.raise_for_status()" || exit 1

# ── Expose port ──────────────────────────────────────────────────────────────

EXPOSE 7860

# ── Start the server ─────────────────────────────────────────────────────────

CMD ["python", "app.py"]
