FROM python:3.11-slim

WORKDIR /app

# Install kubectl and configure in-cluster access
RUN apt-get update && apt-get install -y curl && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set kubectl to use in-cluster config
ENV KUBERNETES_SERVICE_HOST=kubernetes.default.svc.cluster.local
ENV KUBERNETES_SERVICE_PORT=443

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 5005

# Single ENTRYPOINT
ENTRYPOINT ["python", "app.py"]

