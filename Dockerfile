FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DJANGO_SETTINGS_MODULE=forex_signal.settings
ENV PORT=8080
ENV NODE_ENV=production

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory and a non-root user for safer runtime
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser || true

# Copy requirements first for better caching
COPY requirements.txt .
# Upgrade pip and install as non-root where possible; suppress root warning explicitly
RUN python -m pip install --upgrade pip --no-cache-dir \
    && python -m pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy project files
COPY . .

# Build React frontend as root (node/npm will run inside container during build)
WORKDIR /app/frontend
RUN npm ci --only=production --silent && npm run build

# Ensure ownership and switch to non-root user for runtime
WORKDIR /app
RUN chown -R appuser:appuser /app
USER appuser

# Collect static files (run as non-root user)
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/signals/health/ || exit 1

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "forex_signal.wsgi:application"]