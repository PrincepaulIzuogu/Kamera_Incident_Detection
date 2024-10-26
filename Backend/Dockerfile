# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    libatlas-base-dev \
    libboost-all-dev \
    libopencv-dev \
    && apt-get clean

# Clone OpenPose repository
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git /openpose

# Set up OpenPose environment
WORKDIR /openpose
RUN git submodule update --init --recursive

# Back to app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into /app (going back two directories)
COPY ../../ .  
# Set the command to run your backend application
CMD ["python", "src/data_preprocessing.py"]
