FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=forex_signal.settings
ENV PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Build React frontend
WORKDIR /app/frontend
RUN npm install && npm run build

# Move back to root and copy built frontend to static files
WORKDIR /app
RUN mkdir -p forex_signal/static && cp -r frontend/build/* forex_signal/static/

# Create necessary directories
RUN mkdir -p models data/raw output logs

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/ || exit 1

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "forex_signal.wsgi:application"]