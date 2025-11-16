# Security Guide

## Overview

This document describes the security improvements made to osmiD-AI-editor and best practices for secure deployment.

## Critical Security Fixes

### 1. Secure Credential Handling

**Issue**: Passwords and API tokens were handled in plaintext.

**Fix**:
- API tokens no longer exposed in URL query strings
- Use environment variables for sensitive configuration
- Credentials cleared from memory after use

**Best Practices**:
```python
# ❌ DON'T: Token in URL
url = f"https://api.example.com?token={token}"

# ✅ DO: Token in Authorization header
headers = {'Authorization': f'Bearer {token}'}
response = requests.get(url, headers=headers)
```

### 2. Input Validation

**Issue**: No validation of user inputs (coordinates, file paths, etc.)

**Fix**:
- All coordinates validated for valid lat/lon ranges
- File paths validated to prevent path traversal
- Polygons validated for geometric correctness
- Email addresses validated

**Usage**:
```python
from common.validation import validate_coordinates, validate_file_path

# Validate coordinates
validate_coordinates(lat=48.8566, lon=2.3522)  # OK
validate_coordinates(lat=999, lon=0)  # Raises ValueError

# Validate file paths (prevents path traversal)
model_path = validate_model_path(user_input)
```

### 3. Coordinate Order Consistency

**Issue**: Mixing (lat, lon) and (lon, lat) order could map features to wrong locations.

**Fix**:
- Explicit `LatLon` and `LonLat` types
- Type system prevents confusion
- Comprehensive tests for coordinate conversion

**Usage**:
```python
from common.coordinates import LatLon, LonLat

# OSM API uses (lat, lon)
osm_coord = LatLon(lat=48.8566, lon=2.3522)

# GeoJSON uses [lon, lat]
geojson_coord = osm_coord.to_geojson()  # [2.3522, 48.8566]
```

### 4. Safe Deserialization

**Issue**: PyTorch models loaded without safety checks (arbitrary code execution risk).

**Fix**:
```python
# ❌ DON'T:
checkpoint = torch.load(model_path)

# ✅ DO:
checkpoint = torch.load(model_path, weights_only=True)
```

### 5. Rate Limiting & Retry Logic

**Issue**: No exponential backoff for API failures.

**Fix**:
```python
from common.retry import retry_with_backoff, RateLimiter

@retry_with_backoff(max_retries=4, base_delay=2.0)
def api_call():
    # Will retry with exponential backoff
    return requests.get(url)

# Rate limiting
limiter = RateLimiter(calls_per_second=2.0)
for item in items:
    with limiter:
        api_call(item)
```

## Deployment Security Checklist

Before deploying to production:

- [ ] **Environment Variables**: Set MAPBOX_ACCESS_TOKEN in .env
- [ ] **File Permissions**: Restrict access to .env file (chmod 600)
- [ ] **API Tokens**: Rotate Mapbox access tokens regularly
- [ ] **OSM Credentials**: Never commit credentials to git
- [ ] **Dry-Run Mode**: Test extensively in dry-run mode first
- [ ] **Model Files**: Validate model checkpoints are from trusted sources
- [ ] **Dependency Scanning**: Run `safety check` before deployment
- [ ] **SSL Verification**: Ensure HTTPS certificate verification enabled
- [ ] **Logging**: Review logs for sensitive data leakage
- [ ] **Rate Limits**: Configure appropriate rate limits for APIs

## Testing Security

Run security-focused tests:

```bash
# Run all tests
pytest tests/ -v

# Run only security tests
pytest tests/ -v -m security

# Run critical coordinate conversion tests
pytest tests/test_coordinates.py -v
```

## Vulnerability Reporting

To report security vulnerabilities:

1. **Do not** create a public GitHub issue
2. Email: [your-security-email@example.com]
3. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Updates

- **2025-11-16**: Major security overhaul
  - Fixed plaintext credential handling
  - Added input validation
  - Implemented retry logic
  - Created comprehensive test suite

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OSM Automated Edits Code of Conduct](https://wiki.openstreetmap.org/wiki/Automated_Edits_code_of_conduct)
- [Mapbox API Security](https://docs.mapbox.com/help/troubleshooting/how-to-use-mapbox-securely/)
