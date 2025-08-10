# controllers/image_security_controller.py
"""
Image Security Controller - Simplified version
"""

import json
import base64
from datetime import datetime
from typing import Dict, Any, Tuple
from flask import jsonify
from werkzeug.datastructures import FileStorage
from pathlib import Path

# Import simplified configuration
from config import (
    MAX_FILE_SIZE,
    MIN_FILE_SIZE,
    ALLOWED_EXTENSIONS,
    RISK_LEVELS,
    ERROR_CODES,
    DEBUG_MODE,
    DETAILED_ERRORS
)

# Import service
from services.image_security_service import ImageSecurityService

class ImageSecurityController:
    def __init__(self):
        """Initialize controller with simplified config"""
        self.service = ImageSecurityService()
        self.MAX_FILE_SIZE = MAX_FILE_SIZE
        self.ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
        self.RISK_LEVELS = RISK_LEVELS
        self.ERROR_CODES = ERROR_CODES
        
        print("ImageSecurityController initialized")
    
    def scan_image(self, request) -> Tuple[Dict[str, Any], int]:
        """Main image scanning endpoint"""
        start_time = datetime.now()
        
        try:
            # Parse request and validate
            file_data, filename, is_full_scan, validation_error = self._parse_scan_request(request)
            
            if validation_error:
                return validation_error
            
            print(f"Starting {'full' if is_full_scan else 'light'} scan for {filename}")
            
            # Perform scan using service
            scan_result = self.service.scan_file(file_data, filename, is_full_scan)
            
            # Format response
            response = self._format_scan_response(scan_result, start_time)
            
            # Log scan completion
            risk_level = scan_result.get('risk_level', 'UNKNOWN')
            threats_count = scan_result.get('threats_detected', 0)
            duration = scan_result.get('scan_duration', 0)
            
            print(f"Scan completed: {filename} - Risk: {risk_level}, Threats: {threats_count}, Duration: {duration}s")
            
            # Return appropriate HTTP status based on risk level
            status_code = self._get_http_status_for_risk(risk_level)
            return response, status_code
        
        except Exception as e:
            print(f"Scan error: {e}")
            return self._error_response(
                message=f"Internal scan error: {str(e) if DETAILED_ERRORS else 'Processing failed'}",
                error_code=self.ERROR_CODES['INTERNAL_ERROR'],
                status_code=500
            )
    
    def _parse_scan_request(self, request) -> Tuple[bytes, str, bool, Dict[str, Any]]:
        """Parse and validate scan request"""
        try:
            file_data = None
            filename = None
            is_full_scan = False
            
            # Handle JSON requests (base64 encoded images)
            if request.is_json:
                json_data = request.get_json()
                
                if not json_data:
                    return None, None, False, self._error_response(
                        "Invalid JSON request",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
                
                # Extract base64 image
                image_base64 = json_data.get('image_base64', '')
                filename = json_data.get('filename', 'uploaded_image.jpg')
                is_full_scan = json_data.get('is_full_scan', False)
                
                if not image_base64:
                    return None, None, False, self._error_response(
                        "Missing image_base64 field",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
                
                # Decode base64 data
                try:
                    # Handle data URI format
                    if ',' in image_base64:
                        image_base64 = image_base64.split(',')[1]
                    
                    file_data = base64.b64decode(image_base64)
                except Exception as e:
                    return None, None, False, self._error_response(
                        f"Invalid base64 image data: {e}",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
            
            # Handle form data requests (file upload)
            else:
                if 'file' not in request.files:
                    return None, None, False, self._error_response(
                        "No file provided in request",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
                
                file_obj: FileStorage = request.files['file']
                
                if file_obj.filename == '':
                    return None, None, False, self._error_response(
                        "No file selected",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
                
                filename = file_obj.filename
                file_data = file_obj.read()
                is_full_scan = request.form.get('is_full_scan', 'false').lower() == 'true'
            
            # Validate file
            validation_result = self._validate_file_request(file_data, filename)
            if validation_result:
                return None, None, False, validation_result
            
            return file_data, filename, is_full_scan, None
        
        except Exception as e:
            print(f"Request parsing error: {e}")
            return None, None, False, self._error_response(
                f"Request parsing failed: {str(e) if DETAILED_ERRORS else 'Invalid request'}",
                self.ERROR_CODES['INVALID_FILE_TYPE'],
                400
            )
    
    def _validate_file_request(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Validate file request"""
        # Check file size
        if len(file_data) > self.MAX_FILE_SIZE:
            max_mb = self.MAX_FILE_SIZE / (1024 * 1024)
            actual_mb = len(file_data) / (1024 * 1024)
            return self._error_response(
                f"File too large: {actual_mb:.1f}MB (max: {max_mb:.1f}MB)",
                self.ERROR_CODES['FILE_TOO_LARGE'],
                413
            )
        
        if len(file_data) < MIN_FILE_SIZE:
            return self._error_response(
                f"File too small: {len(file_data)} bytes (min: {MIN_FILE_SIZE})",
                self.ERROR_CODES['INVALID_FILE_TYPE'],
                400
            )
        
        # Check file extension
        if filename:
            file_extension = Path(filename).suffix.lower()
            
            if file_extension not in self.ALLOWED_EXTENSIONS:
                allowed_list = ', '.join(sorted(self.ALLOWED_EXTENSIONS))
                return self._error_response(
                    f"Unsupported file type: {file_extension}. Allowed: {allowed_list}",
                    self.ERROR_CODES['INVALID_FILE_TYPE'],
                    400
                )
        
        return None
    
    def _format_scan_response(self, scan_result: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """Format scan response"""
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Base response structure
        response = {
            'status': scan_result.get('status', 'success'),
            'timestamp': datetime.now().isoformat(),
            'scan_info': {
                'scan_type': scan_result.get('scan_type', 'unknown'),
                'duration': round(total_duration, 3),
                'risk_level': scan_result.get('risk_level', 'UNKNOWN'),
                'threats_detected': scan_result.get('threats_detected', 0)
            }
        }
        
        # Add scan results
        if scan_result.get('status') == 'success':
            response['scan_result'] = {
                'file_info': scan_result.get('file_info', {}),
                'hash_analysis': scan_result.get('hash_analysis', {}),
                'yara_matches': scan_result.get('yara_matches', []),
                'exif_analysis': scan_result.get('exif_analysis', {}),
                'mime_validation': scan_result.get('mime_validation', {}),
                'risk_assessment': {
                    'level': scan_result.get('risk_level', 'UNKNOWN'),
                    'score': self.RISK_LEVELS.get(scan_result.get('risk_level', 'CLEAN'), 0),
                    'threats_count': scan_result.get('threats_detected', 0)
                }
            }
            
            # Add full scan specific results
            if scan_result.get('scan_type') == 'full':
                response['scan_result'].update({
                    'entropy_analysis': scan_result.get('entropy_analysis', {}),
                    'format_analysis': scan_result.get('format_analysis', {})
                })