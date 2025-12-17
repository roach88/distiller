# Distiller Pipeline Components
# Contains all Python scripts for the training data pipeline

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir pyyaml

# Copy scripts
COPY scripts/ /app/scripts/
COPY schema/ /app/schema/
COPY config/ /app/config/

# Make scripts executable
RUN chmod +x /app/scripts/*.py

# Default command (overridden by component spec)
CMD ["python", "--version"]
