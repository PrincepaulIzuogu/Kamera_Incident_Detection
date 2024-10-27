# Backend/Dockerfile

# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set the timezone environment variable to avoid interactive prompts
ENV TZ=America/New_York  # Change this to your preferred timezone

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    libatlas-base-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set the timezone configuration
RUN echo "$TZ" > /etc/timezone && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Clone OpenPose and install it
RUN git clone --recursive https://github.com/CMU-Perceptual-Computing-Lab/openpose.git /openpose

# Set OpenPose as the working directory
WORKDIR /openpose

# Build OpenPose
RUN bash ./scripts/ubuntu/install_openpose.sh

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY ../../ ./

# Set environment variables for OpenPose
ENV PYTHONPATH="/openpose/python:${PYTHONPATH}"

# Set the command to run your backend application
CMD ["python", "src/data_preprocessing.py"]
