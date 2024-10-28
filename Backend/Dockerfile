# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variable for timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Set working directory
WORKDIR /app

# Install system dependencies and set timezone
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    python3-venv \   # Install python3-venv for creating virtual environments
    libatlas-base-dev \
    lsb-release \
    tzdata \
    libopencv-dev \
    libboost-all-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Cleanup previous installations
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv venv

# Upgrade pip and install Python dependencies within the virtual environment
RUN ./venv/bin/pip install --no-cache-dir --upgrade pip \
    && ./venv/bin/pip install --no-cache-dir -r requirements.txt

# Cleanup unnecessary files
RUN rm -rf /tmp/*

# Copy your application files
COPY ../../ ./ 

# Copy your data folder to the container
COPY data /app/data

# Set the command to run your application using Uvicorn
CMD ["./venv/bin/uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
