# services/image_security_service.py
"""
Image Security Service - Updated to use simplified config
"""

import os
import hashlib
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import yara
from PIL import Image
from PIL.ExifTags import TAGS
import magic

# Import simplified configuration
from config import (
    YARA_RULES_DIR,
    MALWARE_HASH_FILE,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    MIN_FILE_SIZE,
    RISK_LEVELS,
    HASH_ALGORITHMS,
    YARA_RULES_FILES,
    MIME_TYPE_MAPPING,
    MAGIC_SIGNATURES,
    LIGHT_SCAN_TIMEOUT,
    FULL_SCAN_TIMEOUT
)

class ImageSecurityService:
    def __init__(self):
        """Initialize service with simplified config"""
        # Configuration from simplified config
        self.YARA_RULES_DIR = str(YARA_RULES_DIR)
        self.MALWARE_HASH_FILE = str(MALWARE_HASH_FILE)
        self.ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
        self.MAX_FILE_SIZE = MAX_FILE_SIZE
        self.RISK_LEVELS = RISK_LEVELS
        
        # Directories are already created by config.py
        # Initialize services
        self.malware_hashes = self._load_malware_hashes()
        self.yara_rules = self._compile_yara_rules()
        self.magic_mime = magic.Magic(mime=True)
        self.magic_desc = magic.Magic()
        
        print(f"ImageSecurityService initialized")
        print(f"  Malware hashes loaded: {len(self.malware_hashes)}")
        print(f"  YARA rules compiled: {self.yara_rules is not None}")
        print(f"  Config: {self.YARA_RULES_DIR}")
    
    def _load_malware_hashes(self) -> set:
        """Load known malware SHA256 hashes"""
        hashes = set()
        
        if os.path.exists(self.MALWARE_HASH_FILE):
            try:
                with open(self.MALWARE_HASH_FILE, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and len(line) == 64:
                            hashes.add(line.lower())
                print(f"Loaded {len(hashes)} malware hashes")
            except Exception as e:
                print(f"Error loading malware hashes: {e}")
        else:
            print(f"Malware hash file not found (optional): {self.MALWARE_HASH_FILE}")
        
        return hashes
    
    def _create_yara_rules(self):
        """Create YARA rules"""
        light_rules = '''
        rule Critical_Malware_Hash
        {
            meta:
                description = "File matches known malware hash"
                severity = "critical"
                scan_type = "light"
                
            condition:
                false  // This will be handled by hash checking
        }

        rule Embedded_PE_Executable_Light
        {
            meta:
                description = "PE executable embedded in image"
                severity = "critical"
                scan_type = "light"
                
            strings:
                $pe_header = { 4D 5A }
                $pe_signature = "This program cannot be run in DOS mode"
                
            condition:
                $pe_header and $pe_signature
        }

        rule Web_Shell_Critical
        {
            meta:
                description = "Critical web shell signatures"
                severity = "critical"
                scan_type = "light"
                
            strings:
                $php_eval = "eval("
                $php_system = "system("
                $php_shell_exec = "shell_exec("
                $backdoor = "backdoor"
                
            condition:
                any of them
        }

        rule Fake_Image_Extension_Light
        {
            meta:
                description = "Non-image content with image extension"
                severity = "high"
                scan_type = "light"
                
            strings:
                $pe_header = { 4D 5A }
                $elf_header = { 7F 45 4C 46 }
                $zip_header = { 50 4B 03 04 }
                
            condition:
                any of them
        }
        '''

        full_rules = light_rules + '''

        rule Advanced_Steganography
        {
            meta:
                description = "Advanced steganography detection"
                severity = "medium"
                scan_type = "full"
                
            strings:
                $steghide = "SteghidE"
                $outguess = "OutGuess"
                $zip_in_image = { 50 4B 03 04 }
                $rar_in_image = { 52 61 72 21 1A 07 00 }
                
            condition:
                any of them
        }

        rule Network_Threats_Full
        {
            meta:
                description = "Network communication threats"
                severity = "high"
                scan_type = "full"
                
            strings:
                $http_post = "POST /"
                $base64_long = /[A-Za-z0-9+\/]{50,}/
                $suspicious_domain = /.+\.(tk|ml|ga|cf)/
                
            condition:
                any of them
        }

        rule JPEG_Header_Mismatch
        {
            meta:
                description = "File without proper JPEG header"
                severity = "medium"
                scan_type = "full"
                
            strings:
                $jpeg_start = { FF D8 FF }
                
            condition:
                not $jpeg_start at 0
        }

        rule PNG_Header_Mismatch
        {
            meta:
                description = "File without proper PNG header"
                severity = "medium" 
                scan_type = "full"
                
            strings:
                $png_header = { 89 50 4E 47 0D 0A 1A 0A }
                
            condition:
                not $png_header at 0
        }

        rule GIF_Header_Mismatch
        {
            meta:
                description = "File without proper GIF header"
                severity = "medium"
                scan_type = "full"
                
            strings:
                $gif87a = { 47 49 46 38 37 61 }
                $gif89a = { 47 49 46 38 39 61 }
                
            condition:
                not ($gif87a at 0 or $gif89a at 0)
        }

        rule Suspicious_Binary_Data
        {
            meta:
                description = "Suspicious binary patterns in image"
                severity = "medium"
                scan_type = "full"
                
            strings:
                $exec_pattern = { 65 78 65 63 28 }  // "exec("
                $system_pattern = { 73 79 73 74 65 6D 28 }  // "system("
                $shell_pattern = { 2F 62 69 6E 2F 73 68 }  // "/bin/sh"
                $cmd_pattern = { 63 6D 64 2E 65 78 65 }  // "cmd.exe"
                
            condition:
                any of them
        }

        rule Hidden_Archive_Content
        {
            meta:
                description = "Hidden archive content in image"
                severity = "high"
                scan_type = "full"
                
            strings:
                $zip_central = { 50 4B 01 02 }
                $zip_end = { 50 4B 05 06 }
                $rar_block = { 52 61 72 21 1A 07 01 00 }
                $7z_header = { 37 7A BC AF 27 1C }
                
            condition:
                any of them
        }
        '''
        try:
            light_file = os.path.join(self.YARA_RULES_DIR, YARA_RULES_FILES['light_scan'])
            full_file = os.path.join(self.YARA_RULES_DIR, YARA_RULES_FILES['full_scan'])
            
            with open(light_file, 'w') as f:
                f.write(light_rules)
            
            with open(full_file, 'w') as f:
                f.write(full_rules)
                
            print(f"YARA rules created successfully")
        except Exception as e:
            print(f"Error creating YARA rules: {e}")
    
    def _compile_yara_rules(self):
        """Compile YARA rules"""
        self._create_yara_rules()
        
        try:
            rule_files = {}
            for rule_file in Path(self.YARA_RULES_DIR).glob('*.yar'):
                rule_files[rule_file.stem] = str(rule_file)
            
            if rule_files:
                compiled_rules = yara.compile(filepaths=rule_files)
                print(f"YARA rules compiled: {list(rule_files.keys())}")
                return compiled_rules
            else:
                print("No YARA rule files found")
                return None
        except Exception as e:
            print(f"Error compiling YARA rules: {e}")
            return None
    
    def validate_file(self, filename: str, file_size: int) -> Dict[str, Any]:
        """Validate file using config parameters"""
        issues = []
        
        # Check file size
        if file_size > MAX_FILE_SIZE:
            issues.append({
                'type': 'file_size',
                'message': f'File too large: {file_size} bytes (max: {MAX_FILE_SIZE})',
                'severity': 'error'
            })
        
        if file_size < MIN_FILE_SIZE:
            issues.append({
                'type': 'file_size',
                'message': f'File too small: {file_size} bytes (min: {MIN_FILE_SIZE})',
                'severity': 'warning'
            })
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            issues.append({
                'type': 'file_extension',
                'message': f'Unsupported file extension: {file_ext}',
                'severity': 'error'
            })
        
        return {
            'valid': len([i for i in issues if i['severity'] == 'error']) == 0,
            'issues': issues,
            'file_extension': file_ext
        }
    
    def calculate_file_hash(self, file_path: str) -> Dict[str, str]:
        """Calculate file hashes"""
        hashes = {}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            for algorithm in HASH_ALGORITHMS:
                if hasattr(hashlib, algorithm):
                    hash_func = getattr(hashlib, algorithm)
                    hashes[algorithm] = hash_func(content).hexdigest()
        except Exception as e:
            print(f"Error calculating hashes: {e}")
        
        return hashes
    
    def check_malware_hash(self, file_hash: str) -> bool:
        """Check if hash matches known malware"""
        return file_hash.lower() in self.malware_hashes
    
    def get_mime_type_validation(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Validate MIME type"""
        try:
            detected_mime = self.magic_mime.from_file(file_path)
            file_extension = Path(filename).suffix.lower()
            
            expected_mimes = MIME_TYPE_MAPPING.get(file_extension, [])
            is_valid = any(expected in detected_mime for expected in expected_mimes)
            
            return {
                'detected_mime': detected_mime,
                'expected_mimes': expected_mimes,
                'is_valid_mime': is_valid,
                'file_description': self.magic_desc.from_file(file_path)
            }
        except Exception as e:
            return {
                'error': f"MIME validation error: {e}",
                'is_valid_mime': False
            }
    
    def determine_risk_level(self, scan_results: Dict[str, Any]) -> str:
        """Determine overall risk level"""
        max_risk = 0
        
        # Check malware hash matches
        if scan_results.get('hash_analysis', {}).get('known_malware', False):
            max_risk = max(max_risk, RISK_LEVELS['CRITICAL'])
        
        # Check YARA matches
        yara_matches = scan_results.get('yara_matches', [])
        for match in yara_matches:
            severity = match.get('severity', 'low')
            if severity == 'critical':
                max_risk = max(max_risk, RISK_LEVELS['CRITICAL'])
            elif severity == 'high':
                max_risk = max(max_risk, RISK_LEVELS['HIGH'])
            elif severity == 'medium':
                max_risk = max(max_risk, RISK_LEVELS['MEDIUM'])
            elif severity == 'low':
                max_risk = max(max_risk, RISK_LEVELS['LOW'])
        
        # Check EXIF threats
        exif_threats = scan_results.get('exif_analysis', {}).get('exif_threats', [])
        for threat in exif_threats:
            severity = threat.get('severity', 'low')
            if severity == 'critical':
                max_risk = max(max_risk, RISK_LEVELS['CRITICAL'])
            elif severity == 'high':
                max_risk = max(max_risk, RISK_LEVELS['HIGH'])
        
        # Convert numeric risk back to string
        risk_map = {v: k for k, v in RISK_LEVELS.items()}
        return risk_map.get(max_risk, 'CLEAN')
    
    def scan_file(self, file_data: bytes, filename: str, is_full_scan: bool = False) -> Dict[str, Any]:
        """Main method to scan uploaded file"""
        start_time = time.time()
        
        # Validate file first
        validation_result = self.validate_file(filename, len(file_data))
        if not validation_result['valid']:
            return {
                'status': 'error',
                'message': 'File validation failed',
                'validation_issues': validation_result['issues'],
                'timestamp': datetime.now().isoformat()
            }
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name
        
        try:
            # Perform scan
            if is_full_scan:
                result = self.perform_full_scan(temp_file_path, filename)
            else:
                result = self.perform_light_scan(temp_file_path, filename)
            
            # Add scan metadata
            result['scan_info'] = {
                'scan_type': 'full' if is_full_scan else 'light',
                'duration': round(time.time() - start_time, 3),
                'file_size': len(file_data),
                'filename': filename
            }
            
            return result
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")
    
    def perform_light_scan(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Perform light scan"""
        start_time = time.time()
        scan_results = {
            'scan_type': 'light',
            'file_info': self.get_file_info(file_path, filename),
            'hash_analysis': {},
            'yara_matches': [],
            'exif_analysis': {},
            'mime_validation': {},
            'threats_detected': 0
        }
        
        try:
            # Hash analysis - only SHA256 for light scan
            hashes = {'sha256': hashlib.sha256(open(file_path, 'rb').read()).hexdigest()}
            scan_results['hash_analysis'] = {
                'hashes': hashes,
                'known_malware': self.check_malware_hash(hashes['sha256'])
            }
            
            # MIME type validation
            scan_results['mime_validation'] = self.get_mime_type_validation(file_path, filename)
            
            # YARA scan - light rules only
            if self.yara_rules:
                yara_matches = self.run_yara_scan(file_path, is_full_scan=False)
                scan_results['yara_matches'] = yara_matches
            
            # Basic EXIF analysis
            exif_data = self.extract_exif_data(file_path)
            exif_threats = self.analyze_exif_threats(exif_data, is_full_scan=False)
            scan_results['exif_analysis'] = {
                'exif_data': exif_data,
                'exif_threats': exif_threats
            }
            
            # Count total threats
            threats_count = 0
            if scan_results['hash_analysis']['known_malware']:
                threats_count += 1
            threats_count += len(scan_results['yara_matches'])
            threats_count += len(exif_threats)
            
            scan_results['threats_detected'] = threats_count
            scan_results['risk_level'] = self.determine_risk_level(scan_results)
            scan_results['scan_duration'] = round(time.time() - start_time, 3)
            scan_results['status'] = 'success'
            
        except Exception as e:
            scan_results['status'] = 'error'
            scan_results['error'] = str(e)
            scan_results['risk_level'] = 'UNKNOWN'
        
        return scan_results
    
    def perform_full_scan(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Perform comprehensive full scan"""
        start_time = time.time()
        scan_results = {
            'scan_type': 'full',
            'file_info': self.get_file_info(file_path, filename),
            'hash_analysis': {},
            'yara_matches': [],
            'exif_analysis': {},
            'mime_validation': {},
            'entropy_analysis': {},
            'format_analysis': {},
            'threats_detected': 0
        }
        
        try:
            # Complete hash analysis
            scan_results['hash_analysis'] = {
                'hashes': self.calculate_file_hash(file_path),
                'known_malware': False
            }
            
            # Check all hashes against malware database
            for hash_type, hash_value in scan_results['hash_analysis']['hashes'].items():
                if hash_type == 'sha256' and self.check_malware_hash(hash_value):
                    scan_results['hash_analysis']['known_malware'] = True
                    break
            
            # Complete MIME validation
            scan_results['mime_validation'] = self.get_mime_type_validation(file_path, filename)
            
            # Full YARA scan
            if self.yara_rules:
                yara_matches = self.run_yara_scan(file_path, is_full_scan=True)
                scan_results['yara_matches'] = yara_matches
            
            # Complete EXIF analysis
            exif_data = self.extract_exif_data(file_path)
            exif_threats = self.analyze_exif_threats(exif_data, is_full_scan=True)
            scan_results['exif_analysis'] = {
                'exif_data': exif_data,
                'exif_threats': exif_threats,
                'privacy_concerns': self._analyze_privacy_data(exif_data)
            }
            
            # Entropy analysis
            scan_results['entropy_analysis'] = self._calculate_entropy(file_path)
            
            # Format analysis
            scan_results['format_analysis'] = self._analyze_file_format(file_path, filename)
            
            # Count total threats
            threats_count = 0
            if scan_results['hash_analysis']['known_malware']:
                threats_count += 1
            threats_count += len(scan_results['yara_matches'])
            threats_count += len(exif_threats)
            
            scan_results['threats_detected'] = threats_count
            scan_results['risk_level'] = self.determine_risk_level(scan_results)
            scan_results['scan_duration'] = round(time.time() - start_time, 3)
            scan_results['status'] = 'success'
            
        except Exception as e:
            scan_results['status'] = 'error'
            scan_results['error'] = str(e)
            scan_results['risk_level'] = 'UNKNOWN'
        
        return scan_results
    
    def get_file_info(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Get basic file information"""
        try:
            file_stat = os.stat(file_path)
            return {
                'filename': filename,
                'size_bytes': file_stat.st_size,
                'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                'mime_type': self.magic_mime.from_file(file_path),
                'file_type': self.magic_desc.from_file(file_path),
                'extension': Path(filename).suffix.lower(),
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            }
        except Exception as e:
            return {
                'filename': filename,
                'error': f"Error getting file info: {str(e)}"
            }
    
    def extract_exif_data(self, file_path: str) -> Dict[str, Any]:
        """Extract EXIF data from image"""
        exif_data = {}
        try:
            with Image.open(file_path) as img:
                exif_dict = img._getexif()
                
                if exif_dict:
                    for tag_id, value in exif_dict.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8')
                            except UnicodeDecodeError:
                                value = value.hex()
                        
                        exif_data[str(tag)] = str(value)
        except Exception as e:
            print(f"Error extracting EXIF: {e}")
        
        return exif_data
    
    def analyze_exif_threats(self, exif_data: Dict[str, Any], is_full_scan: bool) -> List[Dict[str, Any]]:
        """Analyze EXIF data for threats"""
        threats = []
        
        if is_full_scan:
            # Full scan - comprehensive EXIF analysis
            suspicious_patterns = {
                'script_injection': [r'<script', r'javascript:', r'eval\(', r'exec\('],
                'sql_injection': [r'union select', r'drop table', r'insert into'],
                'command_injection': [r'system\(', r'exec\(', r'shell_exec'],
                'path_traversal': [r'\.\./', r'\.\.\\', r'/etc/passwd'],
                'xss_patterns': [r'<.*on\w+\s*=', r'<.*src\s*=\s*["\']javascript:']
            }
            
            for field, value in exif_data.items():
                if isinstance(value, str):
                    value_lower = value.lower()
                    
                    for threat_type, patterns in suspicious_patterns.items():
                        for pattern in patterns:
                            if pattern.lower() in value_lower:
                                threats.append({
                                    'type': 'exif_threat',
                                    'subtype': threat_type,
                                    'field': field,
                                    'value': value[:100],
                                    'severity': 'high' if threat_type in ['script_injection', 'command_injection'] else 'medium',
                                    'description': f'{threat_type.replace("_", " ").title()} detected in EXIF {field}'
                                })
                
                # Check for binary data in text fields
                if isinstance(value, str) and any(ord(c) < 32 or ord(c) > 126 for c in value if c not in '\t\n\r'):
                    threats.append({
                        'type': 'exif_threat',
                        'subtype': 'binary_data',
                        'field': field,
                        'severity': 'medium',
                        'description': f'Binary data in EXIF {field}'
                    })
        else:
            # Light scan - only critical EXIF threats
            critical_patterns = ['<script', 'eval(', 'system(', 'exec(', 'shell_exec(', 'javascript:']
            
            for field, value in exif_data.items():
                if isinstance(value, str):
                    value_lower = value.lower()
                    for pattern in critical_patterns:
                        if pattern in value_lower:
                            threats.append({
                                'type': 'exif_threat',
                                'subtype': 'critical_injection',
                                'field': field,
                                'value': value[:50],
                                'severity': 'critical',
                                'description': f'Critical threat in EXIF {field}'
                            })
        
        return threats
    
    def run_yara_scan(self, file_path: str, is_full_scan: bool) -> List[Dict[str, Any]]:
        """Run YARA rules scan filtered by scan type"""
        matches = []
        
        if not self.yara_rules:
            return matches
        
        try:
            yara_matches = self.yara_rules.match(file_path)
            
            for match in yara_matches:
                # Filter based on scan type
                scan_type = match.meta.get('scan_type', 'full')
                
                if not is_full_scan and scan_type != 'light':
                    continue
                
                matches.append({
                    'rule_name': match.rule,
                    'description': match.meta.get('description', 'No description'),
                    'severity': match.meta.get('severity', 'medium'),
                    'scan_type': scan_type,
                    'tags': list(match.tags),
                    'strings': [{'identifier': s.identifier, 'instances': len(s.instances)} for s in match.strings]
                })
        
        except Exception as e:
            print(f"YARA scan error: {e}")
        
        return matches
    
    def _analyze_privacy_data(self, exif_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze EXIF data for privacy concerns"""
        privacy_issues = []
        
        # Check for GPS data
        gps_fields = ['GPS GPSLatitude', 'GPS GPSLongitude', 'GPS GPSLatitudeRef', 'GPS GPSLongitudeRef', 'GPSInfo']
        for field in gps_fields:
            if field in exif_data and exif_data[field]:
                privacy_issues.append({
                    'type': 'privacy_concern',
                    'subtype': 'gps_location',
                    'field': field,
                    'value': str(exif_data[field])[:50],
                    'severity': 'low',
                    'description': 'GPS location data found'
                })
        
        # Check for device information
        device_fields = ['Make', 'Model', 'Software', 'DateTime']
        for field in device_fields:
            if field in exif_data and exif_data[field]:
                privacy_issues.append({
                    'type': 'privacy_concern',
                    'subtype': 'device_info',
                    'field': field,
                    'value': str(exif_data[field])[:50],
                    'severity': 'low',
                    'description': f'Device {field.lower()} information found'
                })
        
        return privacy_issues
    
    def _calculate_entropy(self, file_path: str) -> Dict[str, Any]:
        """Calculate file entropy for detecting encrypted/packed content"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Calculate Shannon entropy
            if not data:
                return {'entropy': 0, 'analysis': 'empty_file'}
            
            # Count byte frequencies
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
            
            # Calculate entropy
            entropy = 0
            data_len = len(data)
            for count in byte_counts:
                if count > 0:
                    frequency = count / data_len
                    entropy -= frequency * (frequency.bit_length() - 1)
            
            # Normalize to 0-8 range
            entropy = entropy / 8.0 if entropy > 0 else 0
            
            # Analysis
            if entropy > 7.5:
                analysis = 'high_entropy_suspicious'
                risk = 'medium'
            elif entropy > 6.0:
                analysis = 'moderate_entropy'
                risk = 'low'
            else:
                analysis = 'normal_entropy'
                risk = 'none'
            
            return {
                'entropy': round(entropy, 3),
                'analysis': analysis,
                'risk_level': risk,
                'file_size': data_len
            }
        
        except Exception as e:
            return {'error': f'Entropy calculation failed: {e}'}
    
    def _analyze_file_format(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Analyze file format for inconsistencies and polyglot detection"""
        analysis = {
            'format_consistency': True,
            'polyglot_detection': False,
            'header_analysis': {},
            'issues': []
        }
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB
            
            file_extension = Path(filename).suffix.lower()
            
            # Check magic bytes against extension
            magic_checks = []
            for format_name, signatures in MAGIC_SIGNATURES.items():
                for signature in signatures:
                    if header.startswith(signature):
                        magic_checks.append(format_name)
            
            analysis['header_analysis'] = {
                'detected_formats': magic_checks,
                'expected_format': file_extension,
                'header_hex': header[:16].hex()
            }
            
            # Check for format mismatches
            expected_formats = {
                '.jpg': ['JPEG'], '.jpeg': ['JPEG'],
                '.png': ['PNG'],
                '.gif': ['GIF87a', 'GIF89a'],
                '.bmp': ['BMP'],
                '.tiff': ['TIFF_LE', 'TIFF_BE'], '.tif': ['TIFF_LE', 'TIFF_BE']
            }
            
            expected = expected_formats.get(file_extension, [])
            if expected and not any(fmt in magic_checks for fmt in expected):
                analysis['format_consistency'] = False
                analysis['issues'].append({
                    'type': 'format_mismatch',
                    'description': f'Extension {file_extension} does not match detected format',
                    'severity': 'high',
                    'expected': expected,
                    'detected': magic_checks
                })
            
            # Specific header validation for common image formats
            header_validations = self._validate_specific_headers(header, file_extension)
            analysis['issues'].extend(header_validations)
            
            # Check for polyglot (multiple format signatures)
            if len(magic_checks) > 1:
                analysis['polyglot_detection'] = True
                analysis['issues'].append({
                    'type': 'polyglot_file',
                    'description': f'Multiple format signatures detected: {", ".join(magic_checks)}',
                    'severity': 'high'
                })
            
            # Check for embedded executables
            exe_signatures = [
                (b'MZ', 'PE_executable'),
                (b'\x7fELF', 'ELF_executable'), 
                (b'PK\x03\x04', 'ZIP_archive'),
                (b'Rar!', 'RAR_archive'),
                (b'\x50\x4B\x01\x02', 'ZIP_central_directory')
            ]
            
            for sig, desc in exe_signatures:
                offset = header.find(sig)
                if offset >= 0:
                    analysis['issues'].append({
                        'type': 'embedded_executable',
                        'description': f'{desc} signature found at offset {offset}',
                        'severity': 'critical'
                    })
        
        except Exception as e:
            analysis['error'] = f'Format analysis failed: {e}'
        
        return analysis

    def _validate_specific_headers(self, header: bytes, file_extension: str) -> List[Dict[str, Any]]:
        """Validate specific format headers"""
        issues = []
        
        try:
            if file_extension in ['.jpg', '.jpeg']:
                # JPEG should start with FF D8 FF
                if not header.startswith(b'\xFF\xD8\xFF'):
                    issues.append({
                        'type': 'invalid_header',
                        'description': 'Invalid JPEG header - should start with FF D8 FF',
                        'severity': 'medium'
                    })
                
                # Check for JPEG end marker in first 1KB (shouldn't be there normally)
                if b'\xFF\xD9' in header[:500]:
                    issues.append({
                        'type': 'truncated_file',
                        'description': 'JPEG end marker found too early in file',
                        'severity': 'medium'
                    })
            
            elif file_extension == '.png':
                # PNG should start with 89 50 4E 47 0D 0A 1A 0A
                png_signature = b'\x89PNG\r\n\x1a\n'
                if not header.startswith(png_signature):
                    issues.append({
                        'type': 'invalid_header',
                        'description': 'Invalid PNG header signature',
                        'severity': 'medium'
                    })
            
            elif file_extension == '.gif':
                # GIF should start with GIF87a or GIF89a
                if not (header.startswith(b'GIF87a') or header.startswith(b'GIF89a')):
                    issues.append({
                        'type': 'invalid_header',
                        'description': 'Invalid GIF header - should start with GIF87a or GIF89a',
                        'severity': 'medium'
                    })
            
            elif file_extension == '.bmp':
                # BMP should start with BM
                if not header.startswith(b'BM'):
                    issues.append({
                        'type': 'invalid_header',
                        'description': 'Invalid BMP header - should start with BM',
                        'severity': 'medium'
                    })
            
            elif file_extension in ['.tiff', '.tif']:
                # TIFF can be little-endian (II*) or big-endian (MM*)
                if not (header.startswith(b'II*\x00') or header.startswith(b'MM\x00*')):
                    issues.append({
                        'type': 'invalid_header', 
                        'description': 'Invalid TIFF header - should start with II* or MM*',
                        'severity': 'medium'
                    })
        
        except Exception as e:
            issues.append({
                'type': 'validation_error',
                'description': f'Header validation failed: {e}',
                'severity': 'low'
            })
        
        return issues
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and statistics"""
        return {
            'service_name': 'ImageSecurityService',
            'version': '2.0.0',
            'malware_hashes_loaded': len(self.malware_hashes),
            'yara_rules_compiled': self.yara_rules is not None,
            'scan_types': ['light', 'full'],
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024),
            'risk_levels': list(RISK_LEVELS.keys()),
            'hash_algorithms': HASH_ALGORITHMS,
            'configuration': {
                'yara_rules_dir': self.YARA_RULES_DIR,
                'malware_hash_file': self.MALWARE_HASH_FILE,
                'scan_timeouts': {
                    'light_scan': LIGHT_SCAN_TIMEOUT,
                    'full_scan': FULL_SCAN_TIMEOUT
                }
            }
        }