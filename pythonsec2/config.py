# config.py
"""
Simplified Configuration for Image Security Scanner
"""

import os
from pathlib import Path

# ===========================
# BASE PATHS
# ===========================

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR

# ===========================
# DIRECTORIES
# ===========================

YARA_RULES_DIR = BASE_DIR / "yara_rules"
UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
for directory in [YARA_RULES_DIR, UPLOADS_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ===========================
# FILE PATHS
# ===========================

MALWARE_HASH_FILE = BASE_DIR / "full_sha256.txt"

# ===========================
# FLASK SETTINGS
# ===========================

FLASK_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.environ.get('FLASK_PORT', 5001))
FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# ===========================
# SECURITY SETTINGS
# ===========================

# File validation
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MIN_FILE_SIZE = 100  # 100 bytes

ALLOWED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', 
    '.tiff', '.tif', '.webp', '.ico', '.psd'
}

# Hash algorithms to calculate
HASH_ALGORITHMS = ['md5', 'sha1', 'sha256', 'sha512']

# Scan timeouts (seconds)
LIGHT_SCAN_TIMEOUT = 30
FULL_SCAN_TIMEOUT = 120

# Risk levels
RISK_LEVELS = {
    'CLEAN': 0,
    'LOW': 1,
    'MEDIUM': 2,
    'HIGH': 3,
    'CRITICAL': 4
}

# ===========================
# YARA CONFIGURATION
# ===========================

YARA_RULES_FILES = {
    'light_scan': 'light_scan.yar',
    'full_scan': 'full_scan.yar'
}

YARA_SEVERITY_LEVELS = ['low', 'medium', 'high', 'critical']
YARA_SCAN_TYPES = ['light', 'full']

# ===========================
# MIME TYPE MAPPINGS
# ===========================

MIME_TYPE_MAPPING = {
    '.jpg': ['image/jpeg'],
    '.jpeg': ['image/jpeg'],
    '.png': ['image/png'],
    '.gif': ['image/gif'],
    '.bmp': ['image/bmp', 'image/x-ms-bmp'],
    '.tiff': ['image/tiff'],
    '.tif': ['image/tiff'],
    '.webp': ['image/webp'],
    '.ico': ['image/x-icon', 'image/vnd.microsoft.icon'],
    '.psd': ['image/vnd.adobe.photoshop', 'application/octet-stream']
}

# Magic byte signatures
MAGIC_SIGNATURES = {
    'JPEG': [b'\xFF\xD8\xFF'],
    'PNG': [b'\x89PNG\r\n\x1a\n'],
    'GIF87a': [b'GIF87a'],
    'GIF89a': [b'GIF89a'],
    'BMP': [b'BM'],
    'TIFF_LE': [b'II*\x00'],
    'TIFF_BE': [b'MM\x00*'],
    'WEBP': [b'RIFF', b'WEBP'],
    'ICO': [b'\x00\x00\x01\x00'],
    'PSD': [b'8BPS']
}

# ===========================
# ENVIRONMENT SETTINGS
# ===========================

ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')
DEBUG_MODE = ENVIRONMENT == 'development' or FLASK_DEBUG
DETAILED_ERRORS = DEBUG_MODE

# ===========================
# API SETTINGS
# ===========================

# Error codes
ERROR_CODES = {
    'FILE_TOO_LARGE': 'E001',
    'INVALID_FILE_TYPE': 'E002',
    'SCAN_TIMEOUT': 'E003',
    'YARA_COMPILATION_ERROR': 'E004',
    'HASH_LOADING_ERROR': 'E005',
    'INTERNAL_ERROR': 'E999'
}

# Threat categories
THREAT_CATEGORIES = {
    'MALWARE': 'Malware and malicious files',
    'STEGANOGRAPHY': 'Hidden data and steganography',
    'NETWORK_THREATS': 'Network communication threats',
    'PRIVACY': 'Privacy and metadata concerns',
    'FORMAT_MANIPULATION': 'File format manipulation',
    'INJECTION': 'Code and script injection'
}

# ===========================
# UTILITY FUNCTIONS
# ===========================

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        'base_dir': str(BASE_DIR),
        'environment': ENVIRONMENT,
        'flask': {
            'host': FLASK_HOST,
            'port': FLASK_PORT,
            'debug': FLASK_DEBUG
        },
        'security': {
            'allowed_extensions': list(ALLOWED_EXTENSIONS),
            'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024),
            'hash_file_exists': Path(MALWARE_HASH_FILE).exists()
        },
        'directories': {
            'yara_rules': str(YARA_RULES_DIR),
            'uploads': str(UPLOADS_DIR),
            'temp': str(TEMP_DIR)
        }
    }

def validate_configuration():
    """Validate configuration and return any issues"""
    issues = []
    
    # Check required directories exist
    if not YARA_RULES_DIR.exists():
        issues.append(f"YARA rules directory missing: {YARA_RULES_DIR}")
    
    if not UPLOADS_DIR.exists():
        issues.append(f"Uploads directory missing: {UPLOADS_DIR}")
    
    # Check write permissions
    for directory in [UPLOADS_DIR, TEMP_DIR, YARA_RULES_DIR]:
        if not os.access(directory, os.W_OK):
            issues.append(f"No write permission for: {directory}")
    
    # Check optional files (warnings, not errors)
    if not Path(MALWARE_HASH_FILE).exists():
        issues.append(f"Optional: Malware hash file not found: {MALWARE_HASH_FILE}")
    
    return issues

# ===========================
# CONFIGURATION VALIDATION
# ===========================

if __name__ == "__main__":
    print("Image Security Scanner - Configuration")
    print("=" * 40)
    
    # Print configuration summary
    config_summary = get_config_summary()
    for section, details in config_summary.items():
        print(f"\n{section.upper()}:")
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {details}")
    
    # Validate configuration
    print("\nVALIDATION:")
    issues = validate_configuration()
    if issues:
        print("Issues found:")
        for issue in issues:
            if "Optional:" in issue:
                print(f"  ⚠️  {issue}")
            else:
                print(f"  ❌ {issue}")
    else:
        print("  ✅ Configuration is valid!")
    
    print(f"\nConfig file: {__file__}")