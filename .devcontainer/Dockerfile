# Using a base Python image. You can specify a version, e.g., python:3.10
FROM python:3.12-slim

# Avoid writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# Install system dependencies that might be needed for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Git is useful for version control
    git \
    # Procps is useful for process management (e.g., ps)
    procps \
    # Curl and wget are useful for downloading files
    curl \
    wget \
    # Build-essential contains compilers and tools for building from source
    build-essential \
    # CMake is required by some packages like xgboost
    cmake \
    # git-lfs for large file storage
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# (Optional) Copy requirements.txt and install dependencies during build
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Keep the container running
CMD ["sleep", "infinity"]
