# Start from Miniconda base image
FROM continuumio/miniconda3:23.3.1

# Set working directory
WORKDIR /app

# Copy only environment.yml first to leverage Docker layer caching
COPY environment.yml ./

# Create the environment (this step will be cached unless environment.yml changes)
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "threat-detection-ai", "/bin/bash", "-c"]

# Copy the rest of the application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command (uses conda environment)
CMD ["conda", "run", "--no-capture-output", "-n", "threat-detection-ai", "uvicorn", "app.api.v1.api:app", "--host", "0.0.0.0", "--port", "8000"]