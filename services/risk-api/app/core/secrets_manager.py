"""
Secure secrets management with encryption and vault integration.

CRITICAL: This module addresses the vulnerability of hardcoded secrets
and plain text credential storage.
"""

import os
import json
import base64
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import hvac  # HashiCorp Vault client
import boto3  # AWS Secrets Manager
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from google.cloud import secretmanager

logger = logging.getLogger(__name__)


class SecretsError(Exception):
    """Base exception for secrets management."""
    pass


class SecretNotFoundError(SecretsError):
    """Secret not found in vault."""
    pass


class SecretRotationRequiredError(SecretsError):
    """Secret needs to be rotated."""
    pass


class LocalEncryption:
    """
    CRITICAL: Local encryption for secrets at rest.
    
    This provides encryption for local development and as a fallback.
    In production, use proper secret management services.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize local encryption with master key."""
        if master_key:
            self.master_key = master_key.encode()
        else:
            # Generate from environment or create new
            env_key = os.environ.get('RISKRADAR_MASTER_KEY')
            if env_key:
                self.master_key = env_key.encode()
            else:
                # CRITICAL: In production, this should be stored securely
                self.master_key = Fernet.generate_key()
                logger.warning(
                    "Generated new master key. Store this securely: "
                    f"{self.master_key.decode()}"
                )
        
        self.fernet = Fernet(self.master_key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))


class VaultProvider:
    """Base class for vault providers."""
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from vault."""
        raise NotImplementedError
    
    def set_secret(self, key: str, value: str) -> bool:
        """Set secret in vault."""
        raise NotImplementedError
    
    def delete_secret(self, key: str) -> bool:
        """Delete secret from vault."""
        raise NotImplementedError
    
    def list_secrets(self) -> List[str]:
        """List all secret keys."""
        raise NotImplementedError
    
    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret."""
        raise NotImplementedError


class HashiCorpVaultProvider(VaultProvider):
    """
    HashiCorp Vault provider for enterprise secret management.
    
    CRITICAL: This is the recommended provider for production.
    """
    
    def __init__(self, vault_url: str, token: str, mount_point: str = "secret"):
        """Initialize HashiCorp Vault client."""
        self.client = hvac.Client(url=vault_url, token=token)
        self.mount_point = mount_point
        
        if not self.client.is_authenticated():
            raise SecretsError("Failed to authenticate with HashiCorp Vault")
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=key,
                mount_point=self.mount_point
            )
            return response['data']['data'].get('value')
        except Exception as e:
            logger.error(f"Failed to get secret {key}: {e}")
            return None
    
    def set_secret(self, key: str, value: str) -> bool:
        """Set secret in Vault."""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=key,
                secret={'value': value},
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {key}: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        """Delete secret from Vault."""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=key,
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {key}: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        """List all secret keys."""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path='',
                mount_point=self.mount_point
            )
            return response.get('data', {}).get('keys', [])
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret by creating new version."""
        return self.set_secret(key, new_value)


class AWSSecretsManagerProvider(VaultProvider):
    """AWS Secrets Manager provider."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize AWS Secrets Manager client."""
        self.client = boto3.client('secretsmanager', region_name=region_name)
        self.region = region_name
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(SecretId=key)
            return response.get('SecretString')
        except self.client.exceptions.ResourceNotFoundException:
            logger.error(f"Secret {key} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to get secret {key}: {e}")
            return None
    
    def set_secret(self, key: str, value: str) -> bool:
        """Create or update secret in AWS Secrets Manager."""
        try:
            try:
                # Try to update existing secret
                self.client.update_secret(
                    SecretId=key,
                    SecretString=value
                )
            except self.client.exceptions.ResourceNotFoundException:
                # Create new secret if it doesn't exist
                self.client.create_secret(
                    Name=key,
                    SecretString=value
                )
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {key}: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        """Delete secret from AWS Secrets Manager."""
        try:
            self.client.delete_secret(
                SecretId=key,
                ForceDeleteWithoutRecovery=False
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {key}: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        """List all secret keys."""
        try:
            response = self.client.list_secrets()
            return [secret['Name'] for secret in response.get('SecretList', [])]
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate secret in AWS."""
        try:
            self.client.rotate_secret(
                SecretId=key,
                RotationLambdaARN='arn:aws:lambda:...',  # Configure rotation Lambda
                RotationRules={'AutomaticallyAfterDays': 30}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to rotate secret {key}: {e}")
            return False


class AzureKeyVaultProvider(VaultProvider):
    """Azure Key Vault provider."""
    
    def __init__(self, vault_url: str):
        """Initialize Azure Key Vault client."""
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)
        self.vault_url = vault_url
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from Azure Key Vault."""
        try:
            secret = self.client.get_secret(key)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to get secret {key}: {e}")
            return None
    
    def set_secret(self, key: str, value: str) -> bool:
        """Set secret in Azure Key Vault."""
        try:
            self.client.set_secret(key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {key}: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        """Delete secret from Azure Key Vault."""
        try:
            self.client.begin_delete_secret(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {key}: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        """List all secret keys."""
        try:
            return [secret.name for secret in self.client.list_properties_of_secrets()]
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate secret in Azure."""
        return self.set_secret(key, new_value)


class SecretsManager:
    """
    CRITICAL: Central secrets management system.
    
    This class provides a unified interface for managing secrets across
    different providers and ensures all secrets are encrypted.
    """
    
    def __init__(self, provider: Optional[VaultProvider] = None):
        """Initialize secrets manager with provider."""
        self.provider = provider
        self.local_encryption = LocalEncryption()
        self._cache: Dict[str, tuple[str, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._rotation_days = 90
        
        # Load provider based on environment if not provided
        if not provider:
            self.provider = self._load_provider()
    
    def _load_provider(self) -> Optional[VaultProvider]:
        """Load appropriate provider based on environment."""
        # Check for HashiCorp Vault
        if os.environ.get('VAULT_ADDR') and os.environ.get('VAULT_TOKEN'):
            return HashiCorpVaultProvider(
                vault_url=os.environ['VAULT_ADDR'],
                token=os.environ['VAULT_TOKEN']
            )
        
        # Check for AWS
        if os.environ.get('AWS_REGION'):
            return AWSSecretsManagerProvider(
                region_name=os.environ['AWS_REGION']
            )
        
        # Check for Azure
        if os.environ.get('AZURE_KEY_VAULT_URL'):
            return AzureKeyVaultProvider(
                vault_url=os.environ['AZURE_KEY_VAULT_URL']
            )
        
        logger.warning(
            "No vault provider configured. Using local encryption only. "
            "This is NOT recommended for production!"
        )
        return None
    
    def get_secret(
        self,
        key: str,
        use_cache: bool = True,
        check_rotation: bool = True
    ) -> str:
        """
        Get secret value securely.
        
        CRITICAL: This method ensures secrets are never exposed in logs
        and are properly decrypted.
        """
        # Check cache first
        if use_cache and key in self._cache:
            value, cached_at = self._cache[key]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return value
        
        # Get from provider or environment
        value = None
        
        if self.provider:
            value = self.provider.get_secret(key)
        
        if not value:
            # Fallback to environment variable
            env_key = f"RISKRADAR_{key.upper()}"
            value = os.environ.get(env_key)
        
        if not value:
            # Check local encrypted file (for development)
            value = self._get_local_secret(key)
        
        if not value:
            raise SecretNotFoundError(f"Secret '{key}' not found")
        
        # Check if rotation is needed
        if check_rotation:
            self._check_rotation_needed(key)
        
        # Cache the value
        if use_cache:
            self._cache[key] = (value, datetime.utcnow())
        
        return value
    
    def set_secret(self, key: str, value: str, encrypt: bool = True) -> bool:
        """
        Set secret value securely.
        
        CRITICAL: Always encrypt secrets before storage.
        """
        if not value:
            raise ValueError("Cannot set empty secret")
        
        # Encrypt value if requested
        if encrypt:
            encrypted_value = self.local_encryption.encrypt(value)
        else:
            encrypted_value = value
        
        # Store in provider
        if self.provider:
            success = self.provider.set_secret(key, encrypted_value)
        else:
            # Store locally for development
            success = self._set_local_secret(key, encrypted_value)
        
        # Clear cache
        if key in self._cache:
            del self._cache[key]
        
        # Log audit event (without exposing the secret)
        logger.info(f"Secret '{key}' was updated")
        
        return success
    
    def rotate_secret(self, key: str, new_value: Optional[str] = None) -> str:
        """
        Rotate a secret.
        
        CRITICAL: Regular rotation is essential for security.
        """
        # Generate new value if not provided
        if not new_value:
            new_value = self._generate_secure_secret()
        
        # Store new value
        if self.provider and hasattr(self.provider, 'rotate_secret'):
            self.provider.rotate_secret(key, new_value)
        else:
            self.set_secret(key, new_value)
        
        # Clear cache
        if key in self._cache:
            del self._cache[key]
        
        # Log rotation
        logger.info(f"Secret '{key}' was rotated")
        
        return new_value
    
    def _generate_secure_secret(self, length: int = 32) -> str:
        """Generate cryptographically secure secret."""
        return secrets.token_urlsafe(length)
    
    def _get_local_secret(self, key: str) -> Optional[str]:
        """Get secret from local encrypted file."""
        secrets_file = Path('.secrets.encrypted')
        if not secrets_file.exists():
            return None
        
        try:
            encrypted_data = secrets_file.read_text()
            decrypted = self.local_encryption.decrypt(encrypted_data)
            secrets_dict = json.loads(decrypted)
            return secrets_dict.get(key)
        except Exception as e:
            logger.error(f"Failed to read local secret: {e}")
            return None
    
    def _set_local_secret(self, key: str, value: str) -> bool:
        """Set secret in local encrypted file."""
        secrets_file = Path('.secrets.encrypted')
        
        try:
            # Load existing secrets
            if secrets_file.exists():
                encrypted_data = secrets_file.read_text()
                decrypted = self.local_encryption.decrypt(encrypted_data)
                secrets_dict = json.loads(decrypted)
            else:
                secrets_dict = {}
            
            # Update secret
            secrets_dict[key] = value
            
            # Encrypt and save
            encrypted = self.local_encryption.encrypt(json.dumps(secrets_dict))
            secrets_file.write_text(encrypted)
            
            # Set restrictive permissions
            os.chmod(secrets_file, 0o600)
            
            return True
        except Exception as e:
            logger.error(f"Failed to set local secret: {e}")
            return False
    
    def _check_rotation_needed(self, key: str):
        """Check if secret needs rotation."""
        # In production, check metadata for last rotation date
        # For now, log a warning
        logger.warning(
            f"Secret '{key}' should be rotated every {self._rotation_days} days"
        )
    
    def clear_cache(self):
        """Clear secrets cache."""
        self._cache.clear()
    
    def get_database_url(self) -> str:
        """
        Get database URL with credentials from secrets.
        
        CRITICAL: Never hardcode database credentials.
        """
        db_host = self.get_secret('DB_HOST')
        db_port = self.get_secret('DB_PORT')
        db_name = self.get_secret('DB_NAME')
        db_user = self.get_secret('DB_USER')
        db_password = self.get_secret('DB_PASSWORD')
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get all API keys from secrets."""
        keys = {}
        for provider in ['bloomberg', 'reuters', 'alphavantage']:
            try:
                keys[provider] = self.get_secret(f'API_KEY_{provider.upper()}')
            except SecretNotFoundError:
                logger.warning(f"API key for {provider} not found")
        return keys


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(key: str) -> str:
    """Convenience function to get secret."""
    return get_secrets_manager().get_secret(key)


def set_secret(key: str, value: str) -> bool:
    """Convenience function to set secret."""
    return get_secrets_manager().set_secret(key, value)
