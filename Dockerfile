# Use an official, minimal Python image (slim)
FROM python:3.12-slim

# Disable bytecode caching and force unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security purposes
RUN adduser --disabled-password --gecos '' api-user

# Set working directory
WORKDIR /home/api-user/projet

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models (only required elements in production)
COPY app/ ./app/
COPY models_export/ ./models_export/

# Set permissions for the new user
RUN chown -R api-user:api-user /home/api-user/projet
USER api-user

# Expose API listening port (Render injects its own port through the PORT variable)
ENV PORT=8000
EXPOSE 8000

# Container health check on the /health route (used by Docker; Render uses healthCheckPath)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')" || exit 1

# Command to run the ASGI server (shell form so that $PORT is expanded)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}