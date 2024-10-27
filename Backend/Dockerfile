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
    libatlas-base-dev \
    lsb-release \
    tzdata \
    libopencv-dev \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt  # Ensure pip3 is used

# Copy your application files
COPY ../../ ./

# Copy your data folder to the container
COPY data /app/data

# Set the command to run your application using Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
