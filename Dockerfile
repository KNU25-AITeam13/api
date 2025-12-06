FROM python:3.13-slim

# Install system dependencies for OpenCV, wget, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --no-dev

# Download Depth Pro checkpoint (빌드 타임)
# 이미 포함되어 있으면 스킵됨
RUN mkdir -p /app/ai/depth_pro/checkpoints && \
    if [ ! -f /app/ai/depth_pro/checkpoints/depth_pro.pt ]; then \
        echo "Downloading Depth Pro checkpoint..." && \
        wget -q --show-progress \
            https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
            -O /app/ai/depth_pro/checkpoints/depth_pro.pt && \
        echo "Depth Pro checkpoint downloaded successfully!"; \
    else \
        echo "Depth Pro checkpoint already exists, skipping download."; \
    fi

# Expose port
EXPOSE 80

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--port", "80"]