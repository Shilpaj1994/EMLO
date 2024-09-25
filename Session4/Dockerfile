# Image
FROM ubuntu:latest

# Install uv
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Requirements
RUN uv pip install -r Requirements.txt

# Set the working directory
WORKDIR /workspace/

# Copy Package to the container
COPY src/ /workspace/

