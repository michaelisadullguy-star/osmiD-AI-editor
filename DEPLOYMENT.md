# Deployment Guide

## Pre-Deployment Checklist

### ⚠️ CRITICAL: Do not deploy until ALL items are checked ✅

#### 1. Testing Requirements
- [ ] All unit tests passing (`pytest tests/ -v`)
- [ ] Coordinate conversion tests passing (critical!)
- [ ] Integration tests completed
- [ ] Manual testing of coordinate accuracy (100+ features reviewed)
- [ ] Test coverage ≥ 70% (`pytest --cov`)

#### 2. Security Requirements
- [ ] All security fixes applied
- [ ] Dependency vulnerabilities scanned (`safety check`)
- [ ] No secrets in repository (`git secrets --scan`)
- [ ] `.env` file properly configured
- [ ] File permissions set correctly (`.env` should be 600)

#### 3. OSM Community Compliance
- [ ] Read [OSM Automated Edits Code of Conduct](https://wiki.openstreetmap.org/wiki/Automated_Edits_code_of_conduct)
- [ ] Dry-run mode tested extensively
- [ ] Human review process implemented
- [ ] Changeset comments configured
- [ ] Contact information added to profile
- [ ] Local mapping community consulted (if applicable)

#### 4. Configuration
- [ ] MAPBOX_ACCESS_TOKEN set in `.env`
- [ ] OSM credentials configured securely
- [ ] Model checkpoint validated and tested
- [ ] Logging directory created and writable
- [ ] Rate limits configured appropriately

#### 5. Documentation
- [ ] README.md reviewed and up to date
- [ ] SECURITY.md reviewed
- [ ] API documentation generated (if applicable)
- [ ] User guide created
- [ ] Troubleshooting guide available

## Installation

### 1. System Requirements

- **Python**: 3.8 or higher (3.10+ recommended)
- **OS**: Linux, macOS, or Windows
- **GPU**: Optional (CUDA-compatible for training)
- **RAM**: Minimum 8GB (16GB+ recommended for training)
- **Disk Space**: 10GB+ for data and models

### 2. Clone Repository

```bash
git clone https://github.com/michaelisadullguy-star/osmiD-AI-editor.git
cd osmiD-AI-editor
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
# Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For development (includes testing tools)
pip install -r requirements.txt pytest pytest-cov
```

### 5. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your credentials
nano .env  # or use your preferred editor
```

Required environment variables:
```bash
MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
OSM_API_URL=https://api.openstreetmap.org/api/0.6
```

### 6. Verify Installation

```bash
# Run tests to verify installation
pytest tests/ -v

# Check for dependency vulnerabilities
safety check

# Verify Python version
python --version  # Should be 3.8+
```

## Usage Workflow

### Phase 1: Data Acquisition (Training Data)

```bash
# 1. Download OSM data for cities
python -m part1_data_acquisition.osm_downloader

# 2. Download Mapbox satellite imagery
python -m part1_data_acquisition.mapbox_imagery

# 3. Correlate features with imagery
python -m part1_data_acquisition.feature_correlator

# Or run all steps:
python run_data_acquisition.py
```

### Phase 2: Model Training

```bash
# Train the model
python -m part2_training.train

# Monitor training (in separate terminal)
tensorboard --logdir=./logs
```

Training will:
- Use Paris, London, New York, Hong Kong, Moscow for training
- Use Tokyo, Singapore for validation
- Run for 100 epochs (configurable in `config.yaml`)
- Save checkpoints to `./models/checkpoints/`

### Phase 3: Feature Detection & Mapping

#### IMPORTANT: Start with Dry-Run Mode!

```bash
# Run in dry-run mode (no actual OSM uploads)
python -m part3_executable.gui_application
```

In the GUI:
1. **Test Mode First**:
   - Check "Dry Run Mode" checkbox
   - Enter test polygon (small area!)
   - Review detected features WITHOUT uploading

2. **After Extensive Testing**:
   - Manual review of 100+ detected features
   - Verify coordinate accuracy
   - Check feature classification accuracy
   - Only then proceed to actual uploads

## Production Deployment

### Step 1: Create Production Configuration

```bash
# Create production config
cp config.yaml config.production.yaml

# Edit production settings
nano config.production.yaml
```

Production settings:
```yaml
# Production configuration
dry_run: false  # Only after extensive testing!
require_review: true  # Always require human review
max_features_per_upload: 50  # Limit upload size
log_level: INFO
```

### Step 2: Set Up Logging

```bash
# Create logs directory
mkdir -p logs

# Set permissions
chmod 755 logs

# Configure log rotation (example for systemd)
sudo nano /etc/logrotate.d/osmid-ai-editor
```

Example logrotate configuration:
```
/path/to/osmiD-AI-editor/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 user group
}
```

### Step 3: Run as Service (Optional)

For long-running deployments, create a systemd service:

```bash
sudo nano /etc/systemd/system/osmid-ai-editor.service
```

Example service file:
```ini
[Unit]
Description=osmiD-AI-editor Service
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/osmiD-AI-editor
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python -m part3_executable.gui_application
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable osmid-ai-editor
sudo systemctl start osmid-ai-editor
sudo systemctl status osmid-ai-editor
```

## Monitoring & Maintenance

### Check Logs

```bash
# View recent logs
tail -f logs/osmid.log

# Search for errors
grep ERROR logs/osmid.log

# View specific component logs
grep "data_acquisition" logs/osmid.log
```

### Monitor API Usage

- **Mapbox**: Check usage at https://account.mapbox.com/
- **OSM**: Monitor changeset comments for disputes
- **Rate Limits**: Review logs for rate limit warnings

### Regular Maintenance

**Weekly**:
- Review logs for errors
- Check disk space usage
- Verify backup integrity

**Monthly**:
- Update dependencies: `pip install --upgrade -r requirements.txt`
- Run security scan: `safety check`
- Review OSM changesets for reversion/disputes
- Rotate API tokens

**Quarterly**:
- Review and update model with new training data
- Performance audit
- Security audit

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**2. CUDA Out of Memory**
```bash
# Reduce batch size in config.yaml
batch_size: 4  # Instead of 8

# Or train on CPU
device: 'cpu'
```

**3. API Authentication Failed**
```bash
# Verify credentials in .env
cat .env | grep MAPBOX

# Test Mapbox token
curl "https://api.mapbox.com/v4/mapbox.satellite/0/0/0.png?access_token=YOUR_TOKEN"

# Check OSM credentials (use development API for testing)
```

**4. Coordinate Accuracy Issues**
```bash
# Run coordinate tests
pytest tests/test_coordinates.py -v

# Verify bounding box configuration
python -c "from common import CITIES; print(CITIES['paris'])"
```

### Getting Help

1. **Check Documentation**: README.md, SECURITY.md, CODE_REVIEW.md
2. **Run Tests**: `pytest tests/ -v` to diagnose issues
3. **Review Logs**: `tail -f logs/osmid.log`
4. **GitHub Issues**: https://github.com/michaelisadullguy-star/osmiD-AI-editor/issues

## Rollback Procedure

If issues are discovered after deployment:

1. **Stop Service Immediately**:
   ```bash
   sudo systemctl stop osmid-ai-editor
   ```

2. **Revert to Previous Version**:
   ```bash
   git checkout <previous-commit>
   pip install -r requirements.txt
   ```

3. **Review OSM Changesets**:
   - Identify problematic changesets
   - Contact OSM community if needed
   - Revert incorrect edits manually

4. **Investigate Root Cause**:
   - Review logs
   - Run tests
   - Fix issues before redeploying

## Performance Optimization

### Training Performance

```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES=0

# Increase workers for data loading
num_workers: 8  # in config.yaml

# Use mixed precision training (PyTorch 1.6+)
# Add to training script:
from torch.cuda.amp import autocast, GradScaler
```

### Inference Performance

```bash
# Batch processing for multiple images
# Use smaller image sizes if accuracy permits
img_size: [256, 256]  # Instead of [512, 512]

# Cache model in memory for multiple inferences
```

## Security Hardening

### Production Security

```bash
# Set restrictive file permissions
chmod 600 .env
chmod 700 logs/

# Disable debug mode
export DEBUG=false

# Use HTTPS only
# Verify SSL certificates enabled
```

### API Security

```bash
# Rotate Mapbox token regularly
# Limit API token permissions
# Monitor API usage for anomalies
# Set up IP whitelisting if possible
```

## Compliance & Legal

### OSM Compliance

- **Always** include proper changeset comments
- **Always** attribute osmiD-AI-editor as the tool
- **Never** upload unreviewed features
- **Respond promptly** to community feedback
- **Document** your mapping methodology

### Data Usage

- Mapbox imagery: Follow Mapbox TOS
- OSM data: Licensed under ODbL
- Respect API rate limits
- Comply with GDPR/privacy laws if applicable

## Support

For production support:

- **Email**: [support@example.com]
- **Documentation**: https://github.com/michaelisadullguy-star/osmiD-AI-editor
- **OSM Community**: https://community.openstreetmap.org/

---

**Last Updated**: 2025-11-16
**Version**: 2.0.0 (Security-hardened)
