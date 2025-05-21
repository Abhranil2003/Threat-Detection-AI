
# syntax=docker/dockerfile:1

FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment file from new location
COPY .github/workflows/environment.yml /tmp/environment.yml

# Create environment
RUN conda env create -f /tmp/environment.yml

# Make sure conda activates the environment
SHELL ["conda", "run", "-n", "threat-detection-ai", "/bin/bash", "-c"]

# Copy the rest of the app
COPY . /app

# Default command
CMD ["conda", "run", "--no-capture-output", "-n", "threat-detection-ai", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
