# Use an official Python image as the base.
# "3.11-slim" = Python 3.11 on a slim Debian-based OS (smaller image, faster pull).
FROM python:3.11-slim

# Optional: prevent Python from writing .pyc and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
# All following COPY/RUN commands are relative to this path.
# SageMaker will run commands with this as current directory so "params.yaml" and "src/" are here.
WORKDIR /opt/ml/code

# Copy dependency list first. Docker caches layers; if requirements.txt doesn't change,
# "pip install" won't re-run on every build (faster rebuilds).
COPY requirements.txt .

# Install Python dependencies. --no-cache-dir keeps the image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (source code and config).
# This includes src/, params.yaml. Add sql/ if split or other scripts need it.
COPY src ./src
COPY params.yaml .

# If your split or other scripts need SQL files, uncomment:
# COPY sql ./sql

# Default command: run Python. SageMaker will override with e.g. "python -m src.model.train".
# Using "python" as entrypoint lets SageMaker pass the full command as arguments.
ENTRYPOINT ["python"]
CMD ["-m", "src.model.train"]