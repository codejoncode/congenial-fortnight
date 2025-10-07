# Google Cloud Run Deployment Guide for Forex Trading System

This guide provides comprehensive instructions for deploying the automated forex trading system to Google Cloud Run, including troubleshooting common issues encountered during setup.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Google Cloud Project Setup](#google-cloud-project-setup)
- [Environment Configuration](#environment-configuration)
- [Local Development Setup](#local-development-setup)
- [Deployment Configuration](#deployment-configuration)
- [Building and Deploying](#building-and-deploying)
- [Troubleshooting](#troubleshooting)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Cost Optimization](#cost-optimization)

## Prerequisites

### Required Accounts and Tools
- Google Cloud Platform account with billing enabled
- GitHub account with repository access
- Gmail account for notifications (with app password)
- FRED API key from Federal Reserve Economic Data
- Git installed locally
- Python 3.10+ installed locally
- Node.js and npm installed locally

### Required Permissions
- Cloud Build Admin
- Cloud Run Admin
- Storage Admin
- Service Account User
- Logs Viewer

## Google Cloud Project Setup

### 1. Create Google Cloud Project
```bash
# In Cloud Shell or locally with gcloud CLI
gcloud projects create YOUR_PROJECT_NAME
gcloud config set project YOUR_PROJECT_NAME
```

### 2. Enable Required APIs
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable logging.googleapis.com
```

### 3. Create Service Account (Optional but Recommended)
```bash
gcloud iam service-accounts create forex-deployer \
  --description="Service account for forex trading system deployment" \
  --display-name="Forex Deployer"

gcloud projects add-iam-policy-binding YOUR_PROJECT_NAME \
  --member="serviceAccount:forex-deployer@YOUR_PROJECT_NAME.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"

gcloud projects add-iam-policy-binding YOUR_PROJECT_NAME \
  --member="serviceAccount:forex-deployer@YOUR_PROJECT_NAME.iam.gserviceaccount.com" \
  --role="roles/run.admin"
```

## Environment Configuration

### Gmail Setup for Notifications
1. Enable 2-factor authentication on your Gmail account
2. Generate an app password:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
   - Save the 16-character password

### FRED API Key
1. Register at https://fred.stlouisfed.org/docs/api/api_key.html
2. Generate API key
3. Save the key securely

### Environment Variables
Create the following substitutions in your `cloudbuild.yaml`:

```yaml
substitutions:
  _REGION: us-central1
  _FRED_API_KEY: your_fred_api_key_here
  _GMAIL_USERNAME: your_gmail@gmail.com
  _GMAIL_APP_PASSWORD: your_16_char_app_password
  _NOTIFICATION_EMAIL: your_notification@email.com
  _NOTIFICATION_SMS: "1234567890"  # Note: quotes required for YAML
```

## Local Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/congenial-fortnight.git
cd congenial-fortnight
```

### 2. Set Up Python Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Set Up Django
```bash
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py check
```

### 4. Build Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 5. Test Locally
```bash
python manage.py runserver
# In another terminal:
cd frontend && npm start
```

## Deployment Configuration

### Cloud Build Configuration (`cloudbuild.yaml`)

```yaml
options:
  logging: LEGACY

steps:
  # Build the Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      [
        'build',
        '-t',
        'gcr.io/$PROJECT_ID/congenial-fortnight:v1',
        '.',
      ]

  # Push the image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      [
        'push',
        'gcr.io/$PROJECT_ID/congenial-fortnight:v1',
      ]

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      [
        'run',
        'deploy',
        'congenial-fortnight',
        '--image',
        'gcr.io/$PROJECT_ID/congenial-fortnight:v1',
        '--region',
        '${_REGION}',
        '--platform',
        'managed',
        '--allow-unauthenticated',
        '--port',
        '8080',
        '--memory',
        '2Gi',
        '--cpu',
        '1',
        '--max-instances',
        '10',
        '--concurrency',
        '80',
        '--timeout',
        '300',
        '--set-env-vars',
        'FRED_API_KEY=${_FRED_API_KEY},GMAIL_USERNAME=${_GMAIL_USERNAME},GMAIL_APP_PASSWORD=${_GMAIL_APP_PASSWORD},NOTIFICATION_EMAIL=${_NOTIFICATION_EMAIL},NOTIFICATION_SMS=${_NOTIFICATION_SMS}',
      ]

  # Run automated training job after deployment
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      [
        'run',
        'jobs',
        'create',
        'automated-training',
        '--image',
        'gcr.io/$PROJECT_ID/congenial-fortnight:v1',
        '--region',
        '${_REGION}',
        '--set-env-vars',
        'FRED_API_KEY=${_FRED_API_KEY},GMAIL_USERNAME=${_GMAIL_USERNAME},GMAIL_APP_PASSWORD=${_GMAIL_APP_PASSWORD},NOTIFICATION_EMAIL=${_NOTIFICATION_EMAIL},NOTIFICATION_SMS=${_NOTIFICATION_SMS}',
        '--memory',
        '4Gi',
        '--cpu',
        '2',
        '--max-retries',
        '3',
        '--task-timeout',
        '3600',
        '--command',
        'python,scripts/automated_training.py,--target,0.85,--max-iterations,50',
      ]
    waitFor: ['-']

images:
  - 'gcr.io/$PROJECT_ID/congenial-fortnight:v1'

substitutions:
  _REGION: us-central1
  _FRED_API_KEY: your_fred_api_key
  _GMAIL_USERNAME: your_gmail@gmail.com
  _GMAIL_APP_PASSWORD: your_app_password
  _NOTIFICATION_EMAIL: your_notification@email.com
  _NOTIFICATION_SMS: "1234567890"

timeout: '1800s'
```

### Dockerfile Configuration

```dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=forex_signal.settings
ENV PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Build React frontend
WORKDIR /app/frontend
RUN npm install && npm run build

# Move back to root and copy built frontend to static files
WORKDIR /app
RUN mkdir -p forex_signal/static && cp -r frontend/build/* forex_signal/static/

# Create necessary directories
RUN mkdir -p models data output logs

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/ || exit 1

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "forex_signal.wsgi:application"]
```

## Building and Deploying

### 1. Commit and Push Changes
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2. Deploy to Cloud Run
```bash
# In Cloud Shell
gcloud config set project YOUR_PROJECT_NAME
git clone https://github.com/YOUR_USERNAME/congenial-fortnight.git
cd congenial-fortnight
gcloud builds submit --config cloudbuild.yaml .
```

### 3. Monitor Deployment
```bash
# Check build status
gcloud builds list --limit=5

# Get build logs
gcloud builds log BUILD_ID

# Check Cloud Run service
gcloud run services list
gcloud run services describe congenial-fortnight

# Check automated training job
gcloud run jobs list
gcloud run jobs describe automated-training
```

### 4. Execute Training Job
```bash
gcloud run jobs execute automated-training
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Authentication Errors
**Error:** `Request is missing required authentication credential`

**Solution:**
```bash
gcloud config set project YOUR_PROJECT_NAME
gcloud auth login
```

#### 2. GitHub Authentication Issues
**Error:** `403 Forbidden` or `Repository not found`

**Solution:**
- Create GitHub Personal Access Token with `repo` scope
- Use token in clone URL: `https://YOUR_TOKEN@github.com/username/repo.git`

#### 3. YAML Parsing Errors
**Error:** Issues with phone numbers or special characters in YAML

**Solution:** Quote string values containing special characters:
```yaml
_NOTIFICATION_SMS: "7084652230"
```

#### 4. Docker Build Failures
**Error:** Build step fails during Docker image creation

**Solutions:**
- Check Node.js/npm compatibility
- Ensure all dependencies are listed in requirements.txt
- Verify frontend build works locally
- Check for missing system dependencies

**Debug Steps:**
```bash
# Enable logging in cloudbuild.yaml
options:
  logging: LEGACY

# Get detailed logs
gcloud builds log $(gcloud builds list --limit=1 --format="value(id)")
```

#### 5. Cloud Run Deployment Failures
**Error:** Service deployment fails

**Solutions:**
- Verify image exists in Container Registry
- Check service account permissions
- Validate environment variables
- Ensure port 8080 is exposed

#### 6. Training Job Issues
**Error:** Automated training job fails

**Solutions:**
- Check Cloud Run job logs
- Verify environment variables are set
- Ensure sufficient memory/CPU allocation
- Validate model files exist

### Log Analysis

#### Accessing Build Logs
```bash
# For builds with LEGACY logging
gcloud builds log BUILD_ID

# For real-time logs during build
gcloud builds submit --config cloudbuild.yaml . --stream
```

#### Cloud Run Logs
```bash
# Service logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=congenial-fortnight" --limit=50

# Job logs
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=automated-training" --limit=50
```

## Monitoring and Maintenance

### Health Checks
- The application includes a health check endpoint at `/health/`
- Cloud Run automatically monitors container health
- Configure uptime checks in Cloud Monitoring

### Performance Monitoring
```bash
# View service metrics
gcloud run services describe congenial-fortnight --region=us-central1

# Monitor resource usage
gcloud logging read "resource.type=cloud_run_revision" --filter="resource.labels.service_name=congenial-fortnight"
```

### Updating the Application
```bash
# Make changes locally
git add .
git commit -m "Update description"
git push origin main

# Redeploy
gcloud builds submit --config cloudbuild.yaml .
```

### Backup and Recovery
- Model files are stored in Cloud Storage (configure in application)
- Database backups should be configured separately
- Use Cloud Build triggers for automated deployments

## Cost Optimization

### Cloud Run Pricing
- Pay only for actual usage (CPU time)
- Configure appropriate memory/CPU limits
- Set concurrency limits to control costs
- Use Cloud Run minimum instances if needed

### Storage Costs
- Model files in Container Registry count toward storage costs
- Consider using Cloud Storage for large model files
- Clean up old container images regularly

### Monitoring Costs
- Cloud Logging charges for data volume
- Set appropriate log retention policies
- Use log sampling for high-volume applications

### Cost Monitoring
```bash
# View billing information
gcloud billing accounts list
gcloud billing projects link PROJECT_ID --billing-account=BILLING_ACCOUNT_ID

# Monitor costs in Cloud Console
# Billing → Reports
```

## Security Best Practices

### Environment Variables
- Never commit secrets to version control
- Use Cloud Build substitutions for sensitive data
- Rotate API keys and passwords regularly

### Service Accounts
- Use minimal required permissions
- Rotate service account keys
- Monitor service account usage

### Network Security
- Cloud Run services are automatically secured
- Use VPC if additional network isolation needed
- Configure appropriate firewall rules

## Support and Resources

### Google Cloud Documentation
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Container Registry Documentation](https://cloud.google.com/container-registry/docs)

### Community Resources
- [Google Cloud Community](https://cloud.google.com/community)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-run)

### Getting Help
1. Check Cloud Build and Cloud Run logs
2. Review Google Cloud status page
3. Consult documentation and community forums
4. Create support tickets for billing/account issues

---

**Last Updated:** October 1, 2025
**Version:** 1.0
**Author:** Deployment Guide for Forex Trading System</content>
<parameter name="filePath">c:\users\jonat\documents\codejoncode\congenial-fortnight\GOOGLE_CLOUD_DEPLOYMENT_GUIDE.md