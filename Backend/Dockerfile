# Use a Python base image
FROM python:3.8-slim

# Set environment variable for timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Set working directory
WORKDIR /app

# Install essential system dependencies
RUN apt-get update && apt-get install -y \
    git \ 
    build-essential \ 
    libopencv-dev \  
    python3-venv \  
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /app/venv

# Upgrade pip
RUN /app/venv/bin/pip install --no-cache-dir --upgrade pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt ./
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Cleanup unnecessary files
RUN rm -rf /tmp/*

# Copy your application files
COPY . .

# Set the command to run your application (example)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
