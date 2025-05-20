# ----------- Stage 1: Build Environment -----------
FROM continuumio/miniconda3:23.3.1 AS builder

WORKDIR /app

# Copy only environment.yml to cache Conda layer
COPY environment.yml .

# Create environment (cached if environment.yml hasn't changed)
RUN conda env create -f environment.yml

# Copy all project files
COPY . .

# ----------- Stage 2: Slim Runtime Environment -----------
FROM continuumio/miniconda3:23.3.1

WORKDIR /app

# Copy env from builder
COPY --from=builder /opt/conda/envs/threat-detection-ai /opt/conda/envs/threat-detection-ai
COPY --from=builder /app /app

# Set shell to use conda environment
SHELL ["conda", "run", "-n", "threat-detection-ai", "/bin/bash", "-c"]

# Expose FastAPI port
EXPOSE 8000

# Entrypoint
CMD ["conda", "run", "--no-capture-output", "-n", "threat-detection-ai", "uvicorn", "app.api.v1.api:app", "--host", "0.0.0.0", "--port", "8000"]
