# MedClinic Deployment Guide

Complete instructions for deploying MedClinic in different environments.

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Edge Device Deployment](#edge-device-deployment)
5. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites
- Python 3.10 or higher
- 8GB+ RAM
- 10GB free disk space

### Installation

**Windows:**
```bash
# Run the batch setup script
setup.bat

# Activate virtual environment
venv\Scripts\activate.bat

# Start the server
python main.py
```

**macOS/Linux:**
```bash
# Run the bash setup script
bash setup.sh

# Activate virtual environment
source venv/bin/activate

# Start the server
python main.py
```

### First Run
On first run, MedGemma and Whisper models will auto-download (~5GB total). This takes 3-5 minutes. Models are cached for subsequent runs.

### Testing
```bash
# In a new terminal, with venv activated:
python test_api.py
```

---

## Docker Deployment

### Prerequisites
- Docker installed (https://www.docker.com)
- 8GB+ RAM allocated to Docker
- 15GB free disk space

### Build & Run

```bash
# Build image (first time only)
docker build -t medclinic:latest .

# Run container
docker run -p 8000:8000 medclinic:latest

# Or use docker-compose (simpler)
docker-compose up
```

Access at: http://localhost:8000

### Docker Compose (Recommended)

```yaml
# docker-compose.yml is pre-configured
# Just run:
docker-compose up -d
docker-compose logs -f

# Stop:
docker-compose down
```

### GPU Acceleration (Optional)
If you have NVIDIA GPU:

```bash
# Install nvidia-docker
curl https://get.docker.com | sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list

# Uncomment GPU config in docker-compose.yml
# Then:
docker-compose up
```

---

## Production Deployment

### Option 1: AWS EC2

```bash
# 1. Launch EC2 instance (t3.large minimum)
# 2. SSH in:
ssh -i your-key.pem ec2-user@your-instance

# 3. Install Docker
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker

# 4. Clone repo and deploy
git clone https://github.com/your-org/medclinic.git
cd medclinic
docker-compose up -d

# 5. Setup reverse proxy (nginx)
sudo yum install nginx -y
# Edit /etc/nginx/nginx.conf to proxy to localhost:8000
```

### Option 2: Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project/medclinic

# Deploy
gcloud run deploy medclinic \
  --image gcr.io/your-project/medclinic \
  --platform managed \
  --memory 8Gi \
  --timeout 600
```

### Option 3: Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry your-registry --image medclinic:latest .

# Deploy
az container create \
  --resource-group your-rg \
  --name medclinic \
  --image your-registry.azurecr.io/medclinic:latest \
  --memory 8 \
  --ports 8000
```

### Production Checklist

- [ ] Set up SSL/TLS (use Let's Encrypt)
- [ ] Configure firewall to restrict access
- [ ] Set up logging and monitoring
- [ ] Configure database for audit logs
- [ ] Add authentication (OAuth2/SAML)
- [ ] Enable HIPAA compliance logging
- [ ] Set up health checks and auto-restart
- [ ] Configure backups
- [ ] Test disaster recovery

---

## Edge Device Deployment

### NVIDIA Jetson (Jetson Nano / AGX Orin)

```bash
# 1. Flash Jetpack OS to device
# 2. SSH into Jetson:
ssh user@jetson-ip

# 3. Install dependencies
sudo apt update
sudo apt install python3-pip python3-dev
pip3 install -r requirements-jetson.txt  # Optimized for Jetson

# 4. Run
python3 main.py
```

### Mobile (iOS/Android - Future)

Using ONNX Runtime for mobile:

```bash
# Convert models to ONNX format
python convert_to_onnx.py

# Deploy via React Native / Flutter
# Implementation provided in separate branch
```

### Raspberry Pi 4 (Coming Soon)

Note: MedGemma-2B is too large for Pi 4. Use quantized version:

```bash
pip install -r requirements-pi.txt
python main.py  # Uses 8-bit quantized model
```

---

## Environment Configuration

### Environment Variables

Create `.env` file (optional):

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Models
MEDGEMMA_MODEL=google/medgemma-2b
WHISPER_MODEL=base
DEVICE=auto  # auto, cuda, cpu

# Logging
LOG_LEVEL=INFO

# Security (optional)
API_KEY=your-secret-key
```

### Model Configuration

Edit in `main.py`:

```python
# Change model size (smaller = faster, less accurate)
whisper_model_size = "tiny"  # tiny, base, small, medium, large
medgemma_model = "google/medgemma-2b"  # Currently only 2B available
```

---

## Monitoring & Logs

### Docker Logs

```bash
docker-compose logs -f medclinic
```

### Local Logs

```bash
# Server writes to stdout
# Redirect to file:
python main.py > medclinic.log 2>&1

# View logs
tail -f medclinic.log
```

### Health Monitoring

```bash
# Check health every 30 seconds
watch -n 30 'curl -s http://localhost:8000/health | jq'
```

---

## Troubleshooting

### Models Fail to Download

**Error**: `urllib.error.URLError: <urlopen error Temporary failure in name resolution>`

**Solution**:
```bash
# Download models manually
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/medgemma-2b')"
python -c "import whisper; whisper.load_model('base')"

# Or use offline mode:
pip install transformers[offline]
```

### Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Use CPU instead of GPU
# Set in code or environment:
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Or reduce batch size
max_batch_size = 1
```

### Slow Inference

**Expected**: First request takes 30-60s (model loading)

**Optimize**:
- Use GPU if available
- Use smaller Whisper model: `whisper.load_model("tiny")`
- Consider quantization (int8)

### CORS Errors

**Error**: `Access to XMLHttpRequest blocked by CORS`

**Check**: CORS is enabled by default. If issues persist:

```python
# In main.py, modify CORS settings:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Kill process on port 8000
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port:
python main.py --port 8001
```

---

## Performance Tuning

### For CPU Inference
```python
import torch
torch.set_num_threads(8)  # Match core count
os.environ['OMP_NUM_THREADS'] = '8'
```

### For GPU Inference
```python
import torch
torch.backends.cudnn.benchmark = True  # Optimize for GPU
device = torch.device('cuda')
```

### Memory Optimization
```python
# Use 8-bit quantization (experimental)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-2b",
    load_in_8bit=True
)
```

---

## Support & Contributing

For issues or questions:
1. Check logs: `docker-compose logs medclinic`
2. Run tests: `python test_api.py`
3. Open GitHub issue with error details
4. Submit PRs for improvements

---

**Last Updated**: February 2026  
**Version**: 1.0.0
