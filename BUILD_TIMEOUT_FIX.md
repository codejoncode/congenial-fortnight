# Updated Google Cloud Build Commands

## 🚀 Deploy with Extended Timeout

### Option 1: Using Cloud Build Configuration File (Recommended)
```bash
gcloud builds submit --config cloudbuild.yaml
```

### Option 2: Using Command Line with Custom Timeout
```bash
gcloud builds submit --timeout=3600s --machine-type=E2_HIGHCPU_8
```

### Option 3: Manual Docker Build and Deploy (Fallback)
```bash
# Build and tag the image
docker build -t gcr.io/YOUR_PROJECT_ID/congenial-fortnight:v1 .

# Push to registry
docker push gcr.io/YOUR_PROJECT_ID/congenial-fortnight:v1

# Deploy to Cloud Run
gcloud run deploy congenial-fortnight \
    --image gcr.io/YOUR_PROJECT_ID/congenial-fortnight:v1 \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 10 \
    --timeout 300
```

## 🛠 Build Optimizations Applied

1. **Timeout Extended**: From 30 minutes to 60 minutes
2. **Machine Type**: E2_HIGHCPU_8 for faster builds
3. **Docker Caching**: Added layer caching for faster rebuilds
4. **Multi-stage Build**: Separate frontend build stage
5. **Dockerignore**: Exclude unnecessary files from build context
6. **Disk Size**: Increased to 100GB for ML models

## 📊 Expected Build Time Reduction

- **Before**: 25-35 minutes (often timing out)
- **After**: 15-25 minutes (with caching on subsequent builds)

## 🔧 Troubleshooting

If you still get timeouts, try:

1. **Use Local Docker Build:**
   ```bash
   docker build -t your-image .
   docker tag your-image gcr.io/PROJECT_ID/congenial-fortnight:v1
   docker push gcr.io/PROJECT_ID/congenial-fortnight:v1
   ```

2. **Split the Process:**
   - Build frontend locally
   - Copy build artifacts
   - Build only the Python backend in Cloud

3. **Use Artifact Registry (newer than Container Registry):**
   ```bash
   gcloud builds submit --region=us-central1 --timeout=3600s
   ```

## ✅ Ready to Deploy!

Your `cloudbuild.yaml` now includes:
- ✅ 60-minute timeout
- ✅ High-CPU machine type
- ✅ Docker layer caching
- ✅ Multi-stage build optimization
- ✅ Improved build context with .dockerignore

Run: `gcloud builds submit --config cloudbuild.yaml`