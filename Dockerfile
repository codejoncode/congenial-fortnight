# Multi-stage build for optimization
FROM node:20-alpine AS frontend-builder

# Build frontend first
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production --silent
COPY frontend/ ./
RUN npm run build

# Main Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DJANGO_SETTINGS_MODULE=forex_signal.settings
ENV PORT=8080
ENV NODE_ENV=production

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files (excluding frontend)
COPY --exclude=frontend . .

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/ || exit 1

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "forex_signal.wsgi:application"]