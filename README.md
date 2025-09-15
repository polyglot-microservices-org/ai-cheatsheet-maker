# 🧠 AI Cheatsheet Maker

> **AI-powered cheatsheet generator using AWS Bedrock and Claude 3 Sonnet**

## 📋 Overview

The AI Cheatsheet Maker is a Python Flask microservice that generates comprehensive cheatsheets on any topic using AWS Bedrock's Claude 3 Sonnet model. It provides intelligent, contextual content generation for developers, students, and professionals.

## 🏗️ Architecture

### Technology Stack
- **Language**: Python 3.9+
- **Framework**: Flask with CORS support
- **AI Service**: AWS Bedrock (Claude 3 Sonnet)
- **Container**: Docker with multi-stage build
- **Orchestration**: Kubernetes with HPA

### Service Details
- **Port**: 5005
- **Health Check**: `/healthz`
- **Main Endpoint**: `/cheatsheet` (POST)
- **Docker Image**: `yaswanthmitta/multiapp-bedrock-app`

## 🚀 API Documentation

### Generate Cheatsheet
**POST** `/cheatsheet`

Generate a comprehensive cheatsheet on any topic.

#### Request Body
```json
{
  "topic": "Python"
}
```

#### Response
```json
{
  "cheatsheet": "# Python Cheatsheet\n\n## 1. Variables and Data Types\n..."
}
```

#### Error Response
```json
{
  "error": "Topic is required"
}
```

### Health Check
**GET** `/healthz`

Returns service health status.

#### Response
```json
{
  "status": "ok"
}
```

## 🔧 Configuration

### Environment Variables
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

### AWS Bedrock Setup
1. Enable AWS Bedrock in your region
2. Request access to Claude 3 Sonnet model
3. Configure IAM permissions for Bedrock access

## 🐳 Docker Configuration

### Dockerfile Features
- Multi-stage build for optimization
- Non-root user for security
- Health check integration
- Minimal Python base image

### Build and Run
```bash
# Build image
docker build -t ai-cheatsheet-maker .

# Run container
docker run -p 5005:5005 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_DEFAULT_REGION=us-east-1 \
  ai-cheatsheet-maker
```

## ☸️ Kubernetes Deployment

### Resources
- **Deployment**: Manages pod replicas with rolling updates
- **Service**: ClusterIP for internal communication
- **HPA**: Auto-scaling based on CPU/memory usage
- **ConfigMap**: Configuration management
- **Secret**: AWS credentials (bedrock-secrets)

### Key Features
- **Rolling Updates**: Zero-downtime deployments
- **Health Probes**: Liveness and readiness checks
- **Resource Limits**: CPU and memory constraints
- **Auto-scaling**: HPA with 1-10 replicas

### Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=bedrock-app
kubectl logs -l app=bedrock-app
```

## 🔄 CI/CD Pipeline

### Continuous Integration (CI)
**Trigger**: Push to any branch with changes in `app/` directory

**Steps**:
1. **Checkout Code**: Get latest source code
2. **Docker Login**: Authenticate with Docker Hub
3. **Build & Push**: Create image with Git SHA tag
4. **Update Manifests**: Inject new image tag into Kubernetes files
5. **Commit Changes**: Push updated manifests with `[skip ci]`

### Continuous Deployment (CD)
**Trigger**: Successful CI completion or manual dispatch

**Steps**:
1. **Checkout Code**: Get updated manifests
2. **Deploy to K8s**: Apply all Kubernetes resources
3. **Rolling Update**: Kubernetes handles zero-downtime deployment

### Image Tagging Strategy
```bash
# Format: yaswanthmitta/multiapp-bedrock-app:<git-sha>
yaswanthmitta/multiapp-bedrock-app:a1b2c3d4e5f6
```

## 🔒 Security Implementation

### Container Security
- **Non-root User**: Runs as user ID 1000
- **Read-only Filesystem**: Immutable container layers
- **Resource Limits**: Prevents resource exhaustion
- **Health Checks**: Monitors container health

### Network Security
- **Network Policies**: Controlled ingress/egress traffic
- **Service Mesh**: Secure service-to-service communication
- **Secret Management**: AWS credentials in Kubernetes secrets

### AWS Security
- **IAM Roles**: Least privilege access to Bedrock
- **Encryption**: Data encrypted in transit and at rest
- **VPC**: Network isolation in AWS

## 📊 Monitoring & Observability

### Metrics Collection
- **Prometheus**: Custom metrics for request count, latency
- **Health Checks**: Kubernetes liveness/readiness probes
- **Application Logs**: Structured logging with Flask

### Key Metrics
- Request rate and response time
- Error rate and success rate
- AWS Bedrock API usage
- Resource utilization (CPU/Memory)

### Alerts
- High error rate (>5%)
- Slow response time (>10s)
- AWS API throttling
- Pod restart frequency

## 🧪 Testing

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Run application
python app.py

# Test endpoint
curl -X POST http://localhost:5005/cheatsheet \
  -H "Content-Type: application/json" \
  -d '{"topic": "Docker"}'
```

### Integration Testing
```bash
# Test with Kubernetes
kubectl port-forward svc/bedrock-app 5005:5005

# Test health check
curl http://localhost:5005/healthz

# Test cheatsheet generation
curl -X POST http://localhost:5005/cheatsheet \
  -H "Content-Type: application/json" \
  -d '{"topic": "Kubernetes"}'
```

## 🚨 Troubleshooting

### Common Issues

#### AWS Bedrock Access Denied
```bash
# Check IAM permissions
aws bedrock list-foundation-models --region us-east-1

# Verify model access
aws bedrock get-foundation-model --model-identifier anthropic.claude-3-sonnet-20240229-v1:0
```

#### Pod CrashLoopBackOff
```bash
# Check pod logs
kubectl logs -l app=bedrock-app

# Check events
kubectl describe pod <pod-name>

# Verify secrets
kubectl get secret bedrock-secrets -o yaml
```

#### Slow Response Times
- Check AWS Bedrock quotas and limits
- Monitor Prometheus metrics for bottlenecks
- Verify network connectivity to AWS

### Debug Commands
```bash
# Check service status
kubectl get svc bedrock-app

# View pod details
kubectl describe deployment bedrock-app

# Check HPA status
kubectl get hpa bedrock-app-hpa

# View logs
kubectl logs -f deployment/bedrock-app
```

## 📈 Performance Optimization

### Scaling Configuration
```yaml
# HPA settings
minReplicas: 1
maxReplicas: 10
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

### Resource Tuning
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

## 🔗 Integration

### Frontend Integration
The service integrates with the frontend through Nginx reverse proxy:
```javascript
// Frontend API call
const response = await fetch('/api/bedrock/cheatsheet', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ topic: 'Python' })
});
```

### Service Mesh
- **Ingress**: Nginx routes `/api/bedrock/*` to this service
- **Service Discovery**: Kubernetes DNS resolution
- **Load Balancing**: Kubernetes service load balancing

## 📚 Dependencies

### Python Packages
```txt
Flask==2.3.3
flask-cors==4.0.0
boto3==1.28.85
python-dotenv==1.0.0
```

### AWS Services
- **AWS Bedrock**: AI model hosting
- **Claude 3 Sonnet**: Text generation model
- **IAM**: Access management

## 🏷️ Tags
`python` `flask` `aws-bedrock` `claude-3` `ai` `microservices` `kubernetes` `docker` `cicd`

---

**🌟 This service demonstrates enterprise-grade AI integration with modern DevOps practices!**