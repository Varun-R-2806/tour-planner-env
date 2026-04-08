# Using Python 3.10 as base (OpenEnv recommended)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Installation of system level dependencies if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set local working directory
WORKDIR /app

# Copy the entire environment folder to the container
COPY . .

# Install the environment as a package (handles all dependencies from pyproject.toml)
RUN pip install --no-cache-dir -e .

# Expose port (default 8000)
EXPOSE 8000

# Server execution entry point
CMD ["python", "-m", "tour_planner_env.server.app"]
