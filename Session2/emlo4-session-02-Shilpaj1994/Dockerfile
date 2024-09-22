# Image to be used
FROM python:3.9-slim

# Install dependencies (torch and torchvision for CPU only)
RUN pip install --no-cache-dir torch==1.12.1 torchvision==0.13.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html "numpy<2"
# RUN pip install torch==2.3.1+cpu torchvision==0.18.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && rm -rf /root/.cache/pip

# Workspace Directory
WORKDIR /workspace

# Copy trian.py to the Working directory
COPY train.py /workspace/

# Run the training script
CMD ["python3", "train.py"] 