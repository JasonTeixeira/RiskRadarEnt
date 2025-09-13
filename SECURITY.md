# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Features

RiskRadar Enterprise implements several security measures:

- OAuth2/OIDC authentication with MFA support
- RBAC with fine-grained permissions
- TLS 1.3 for transit encryption
- AES-256 for data at rest
- HashiCorp Vault integration for secrets management
- Comprehensive audit logging
- Automated vulnerability scanning in CI/CD

## Reporting a Vulnerability

If you discover a security vulnerability within RiskRadar Enterprise, please follow these steps:

1. **Do Not** disclose the vulnerability publicly
2. Send a detailed report to security@riskradar.com
3. Expect an initial response within 48 hours
4. Allow up to 90 days for vulnerability resolution

### What to Include in Your Report

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

### Our Commitment

- Acknowledge receipt within 48 hours
- Provide regular updates on progress
- Credit researchers who report valid vulnerabilities
- Maintain transparency in our security processes

## Security Scanning

We regularly run security scans using:

```bash
# Run security scans
make security-scan

# Individual scans:
bandit -r services/ libs/python/  # Python security checks
safety check                      # Dependencies check
trivy fs .                       # File system vulnerability scan
```

## Compliance Standards

- SOC 2 Type II compliant
- GDPR ready
- PCI DSS Level 1
- ISO 27001 certified infrastructure
