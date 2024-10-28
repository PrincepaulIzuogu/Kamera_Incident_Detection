# Use a Python base image
FROM python:3.8-slim

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
    libatlas-base-dev \
    lsb-release \
    tzdata \
    libopencv-dev \
    libboost-all-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    python3-venv \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /app/venv

# Upgrade pip
RUN /app/venv/bin/pip install --no-cache-dir --upgrade pip

# Copy requirements.txt before installing dependencies
COPY ./requirements.txt ./  # Ensure requirements.txt is copied before installing

# Install Python dependencies within the virtual environment
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Cleanup unnecessary files
RUN rm -rf /tmp/*

# Copy your application files
COPY . .

# Set the command to run your application using Uvicorn
CMD ["/app/venv/bin/uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
