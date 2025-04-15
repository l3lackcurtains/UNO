FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Install Python dependencies in fewer layers
RUN pip3 install -r requirements.txt

EXPOSE 8686

CMD ["python3", "app.py"]