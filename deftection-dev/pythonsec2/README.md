# Image Security Scanner

A comprehensive security scanning API for uploaded images using YARA rules and advanced threat detection techniques.

## 🛡️ Features

### Scan Types

#### Light Scan (Fast)

- **Duration**: < 0.5 seconds
- **Purpose**: Quick detection of critical threats
- **Includes**:
  - Known malware hash checking
  - Critical YARA rules (PE executables, web shells)
  - Basic EXIF threat detection
  - File format validation

#### Full Scan (Comprehensive)

- **Duration**: 1-3 seconds
- **Purpose**: Complete security analysis
- **Includes**:
  - All Light Scan features
  - Complete hash analysis (MD5, SHA1, SHA256, SHA512)
  - All YARA rules (malware, steganography, network threats)
  - Complete EXIF analysis with privacy checks
  - Advanced file structure analysis
  - Entropy analysis
  - Format validation and mismatch detection

### Threat Detection Categories

- **Malware**: Executables, web shells, ransomware, APT indicators
- **Steganography**: Hidden data, steganography tools, LSB techniques
- **Network Threats**: C2 communication, phishing, data exfiltration
- **Advanced Threats**: Zero-day indicators, cryptocurrency mining
- **Privacy Concerns**: GPS data, metadata exposure
- **Format Manipulation**: Fake extensions, polyglot files

## 🚀 Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install libmagic1 libyara-dev

# CentOS/RHEL
sudo yum install file-libs yara-devel

# macOS
brew install yara libmagic
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Project Structure

```
image-security-scanner/
├── controllers/
│   └── image_security_controller.py
├── services/
│   └── image_security_service.py
├── yara_rules/                    # Auto-generated
│   ├── light_scan.yar
│   └── full_scan.yar
├── full_sha256.txt                # Your malware hash database
├── requirements.txt
├── app.py                        # Main Flask app
└── README.md
```

### Setup

1. **Clone and setup**:

```bash
git clone <repository>
cd image-security-scanner
pip install -r requirements.txt
```

2. **Add malware hash database**:
   Place your `full_sha256.txt` file in the root directory

3. **Run the scanner**:

```bash
python app.py
```

## 📡 API Endpoints

### POST /scan

Main scanning endpoint with configurable scan depth.

**Parameters**:

- `file`: Image file (form-data upload)
- `image_base64`: Base64 encoded image (JSON)
- `is_full_scan`: boolean (optional, default: false)

**Supported Formats**: JPG, JPEG, PNG, GIF, BMP, TIFF, WebP, ICO, PSD

### GET /health

Health check and service status.

### GET /stats

Scanner statistics and capabilities.

## 🎯 Usage Examples

### Form Data Upload (Light Scan)

```bash
curl -X POST \
  -F 'file=@suspicious.jpg' \
  -F 'is_full_scan=false' \
  http://localhost:5000/scan
```

### Form Data Upload (Full Scan)

```bash
curl -X POST \
  -F 'file=@image.jpg' \
  -F 'is_full_scan=true' \
  http://localhost:5000/scan
```

### JSON Upload (Full Scan)

```bash
curl -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
    "filename": "test.jpg",
    "is_full_scan": true
  }' \
  http://localhost:5000/scan
```

### Health Check

```bash
curl http://localhost:5000/health
```

### Scanner Statistics

```bash
curl http://localhost:5000/stats
```

## 📊 Response Format

All responses follow this structure:

```json
{
  "status": "success|error",
  "scan_result": { ... },
  "timestamp": "2024-01-15T10:30:45.279012"
}
```

### Risk Levels

- **CLEAN**: No threats detected
- **LOW**: Minor privacy concerns
- **MEDIUM**: Suspicious patterns, steganography
- **HIGH**: Network threats, phishing, script injection
- **CRITICAL**: Malware, known malicious hashes, web shells

## 🔧 Configuration

### File Limits

- **Maximum file size**: 50MB
- **Allowed extensions**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.ico`, `.psd`

### Malware Database

Place your malware hash file as `full_sha256.txt` in the format:

```
# Comments start with #
sha256hash1
sha256hash2
sha256hash3
```

### YARA Rules

Rules are automatically generated and stored in `yara_rules/`:

- `light_scan.yar`: Critical threats only
- `full_scan.yar`: Comprehensive detection rules

## 🛠️ Development

### Adding New Threat Detection

1. Update YARA rules in `services/image_security_service.py`
2. Add threat analysis logic in service methods
3. Update response formatting in controller

### Testing

Use the provided test script:

```bash
python test_scanner.py
```

## 🔒 Security Features

### YARA Rule Categories

- **File Format Validation**: Magic bytes, header validation, polyglot detection
- **Malware Detection**: PE/ELF executables, web shells, ransomware
- **Steganography**: Hidden data patterns, tool signatures
- **Network Threats**: C2 communication, data exfiltration
- **Advanced Threats**: Zero-day indicators, living-off-the-land techniques

### EXIF Analysis

- Script injection detection
- Privacy data extraction (GPS coordinates)
- Binary data in text fields
- Buffer overflow indicators

### Hash Reputation

- SHA256 matching against malware databases
- Multi-hash calculation for forensic analysis
- File integrity verification

## 📈 Performance

### Light Scan Performance

- **Average duration**: 0.1-0.5 seconds
- **Memory usage**: ~50MB per request
- **Throughput**: ~100-200 scans/minute

### Full Scan Performance

- **Average duration**: 1-3 seconds
- **Memory usage**: ~100MB per request
- **Throughput**: ~20-60 scans/minute

## 🚨 Error Handling

The API handles various error scenarios:

- File too large (413)
- Invalid file format (400)
- Malformed requests (400)
- Internal processing errors (500)

All errors return structured JSON responses with descriptive messages.

## 📞 Support

For issues or questions:

1. Check the logs for detailed error messages
2. Verify YARA rules compilation
3. Ensure malware hash file format is correct
4. Review file permissions and dependencies
