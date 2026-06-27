# Use an official Python 3.11 slim image as the base
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for MuJoCo, Pygame, and OpenGL rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libegl1 \
    libegl-mesa0 \
    libosmesa6 \
    libosmesa6-dev \
    patchelf \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    xvfb \
    xauth \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless OpenGL rendering
ENV MUJOCO_GL="egl"
ENV PYOPENGL_PLATFORM="egl"
ENV SDL_VIDEODRIVER="dummy"
ENV SDL_AUDIODRIVER="dummy"

# Set working directory inside the container
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Step 1: Install MuJoCo physics simulation environment and graphics packages (as per README.md)
RUN pip install --no-cache-dir pygame mujoco==2.3.3 gymnasium-robotics==1.2.2 xmltodict pyyaml

# Step 2: Install Safety-Gymnasium without standard gymnasium dependencies
RUN pip install --no-cache-dir safety-gymnasium --no-deps

# Step 3: Copy requirements.txt and install remaining core libraries
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Default command: run an interactive bash shell
CMD ["bash"]
