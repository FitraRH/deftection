# controllers/detection_controller.py - Stateless Version
"""
Stateless Detection Controller for JSON API
Handles detection-related requests and responses
No database operations, no file saving
"""

from flask import jsonify
import json
import base64
import os
import time
import tempfile
import logging
from datetime import datetime


class DetectionController:
    """
    Stateless Detection Controller
    Handles request processing and response formatting
    No persistent storage, only in-memory operations
    """
    
    def __init__(self, detection_service):
        self.detection_service = detection_service
        self.logger = logging.getLogger(__name__)
        
        # In-memory configuration storage
        self.config = {
            'thresholds': {
                'anomaly_threshold': 0.7,
                'defect_confidence_threshold': 0.85
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def health_check(self):
        """Health check endpoint"""
        try:
            status = self.detection_service.get_health_status()
            return jsonify({
                'status': 'ok',
                'timestamp': datetime.now().isoformat(),
                'services': status,
                'api_version': '1.0.0',
                'mode': 'stateless'
            })
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_system_info(self):
        """Get detailed system information"""
        try:
            info = self.detection_service.get_system_information()
            info['mode'] = 'stateless'
            info['persistent_storage'] = False
            
            return jsonify({
                'status': 'success',
                'data': info,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"System info failed: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_system_status(self):
        """Get current system status"""
        try:
            status = self.detection_service.get_current_status()
            status['mode'] = 'stateless'
            status['storage_mode'] = 'in-memory-only'
            
            return jsonify({
                'status': 'success',
                'data': status,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"System status failed: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def process_image(self, request):
        """Process single image for defect detection - Stateless"""
        start_time = time.time()
        temp_file = None
        
        try:
            self.logger.info(f"Processing image request - Method: {request.method}")
            
            # Handle both form data and JSON properly
            image_data = None
            filename = None
            
            # Try form data first (multipart/form-data)
            if request.files and 'image' in request.files:
                self.logger.info("Processing as form data")
                file = request.files['image']
                if file.filename == '':
                    return jsonify({
                        'status': 'error',
                        'error': 'No file selected',
                        'timestamp': datetime.now().isoformat()
                    }), 400
                
                image_data = file.read()
                filename = file.filename or f"upload_{int(time.time())}.jpg"
                
            # Try JSON data (application/json)
            elif request.json and 'image_base64' in request.json:
                self.logger.info("Processing as JSON base64")
                base64_data = request.json['image_base64']
                if base64_data.startswith('data:image'):
                    base64_data = base64_data.split(',')[1]
                
                try:
                    image_data = base64.b64decode(base64_data)
                except Exception as decode_error:
                    return jsonify({
                        'status': 'error',
                        'error': f'Invalid base64 image data: {str(decode_error)}',
                        'timestamp': datetime.now().isoformat()
                    }), 400
                
                filename = request.json.get('filename', f"upload_{int(time.time())}.jpg")
            
            # If no image data found
            if not image_data:
                return jsonify({
                    'status': 'error',
                    'error': 'No image provided. Use form-data with "image" field or JSON with "image_base64"',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Validate file size (5MB limit)
            if len(image_data) > 5 * 1024 * 1024:
                return jsonify({
                    'status': 'error',
                    'error': 'File too large. Maximum size is 5MB',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            self.logger.info(f"Processing image - filename: {filename}, size: {len(image_data)} bytes")
            
            # Create temporary file for processing
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(image_data)
            temp_file.close()
            
            # Process image
            result = self.detection_service.process_single_image(image_data, filename, temp_file.name)
            
            if not result:
                return jsonify({
                    'status': 'error',
                    'error': 'Image processing failed',
                    'timestamp': datetime.now().isoformat()
                }), 500
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            # Format response (stateless)
            response = self._format_detection_response(result, include_annotation=True)
            
            self.logger.info(f"Image processed successfully - Decision: {result.get('final_decision')}")
            
            return jsonify({
                'status': 'success',
                'data': response,
                'timestamp': datetime.now().isoformat(),
                'mode': 'stateless'
            })
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            
            return jsonify({
                'status': 'error',
                'error': f'Processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file: {e}")
    
    def process_batch(self, request):
        """Process batch of images - Stateless"""
        start_time = time.time()
        temp_files = []
        
        try:
            # Validate batch request
            if not request.json or 'images' not in request.json:
                return jsonify({
                    'status': 'error',
                    'error': 'No images array provided in JSON body',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            images_data = request.json['images']
            if not isinstance(images_data, list) or len(images_data) == 0:
                return jsonify({
                    'status': 'error',
                    'error': 'Images must be a non-empty array',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            self.logger.info(f"Processing batch of {len(images_data)} images")
            
            # Process each image
            results = []
            for i, image_item in enumerate(images_data):
                if 'image_base64' not in image_item:
                    continue
                
                base64_data = image_item['image_base64']
                if base64_data.startswith('data:image'):
                    base64_data = base64_data.split(',')[1]
                
                try:
                    image_data = base64.b64decode(base64_data)
                    filename = image_item.get('filename', f"batch_image_{i+1}.jpg")
                    
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    temp_file.write(image_data)
                    temp_file.close()
                    temp_files.append(temp_file.name)
                    
                    # Process image
                    result = self.detection_service.process_single_image(image_data, filename, temp_file.name)
                    if result:
                        results.append(result)
                    
                except Exception as decode_error:
                    self.logger.warning(f"Error processing image {i+1}: {decode_error}")
                    continue
            
            processing_time = time.time() - start_time
            
            # Format response
            response = self._format_batch_response(results, processing_time)
            
            self.logger.info(f"Batch processed - {len(results)} successful results")
            
            return jsonify({
                'status': 'success',
                'data': response,
                'timestamp': datetime.now().isoformat(),
                'mode': 'stateless'
            })
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
        
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file: {e}")
    
    def get_detection_thresholds(self):
        """Get current detection thresholds from memory"""
        try:
            return jsonify({
                'status': 'success',
                'data': self.config['thresholds'],
                'last_updated': self.config['last_updated'],
                'timestamp': datetime.now().isoformat(),
                'mode': 'stateless'
            })
        except Exception as e:
            self.logger.error(f"Error getting thresholds: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def update_detection_thresholds(self, request):
        """Update detection thresholds in memory"""
        try:
            new_thresholds = request.json
            if not new_thresholds:
                return jsonify({
                    'status': 'error',
                    'error': 'No threshold data provided',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Validate thresholds
            for key, value in new_thresholds.items():
                if key in ['anomaly_threshold', 'defect_confidence_threshold']:
                    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                        return jsonify({
                            'status': 'error',
                            'error': f'{key} must be a number between 0 and 1',
                            'timestamp': datetime.now().isoformat()
                        }), 400
            
            # Update in-memory configuration
            self.config['thresholds'].update(new_thresholds)
            self.config['last_updated'] = datetime.now().isoformat()
            
            # Update detection service thresholds
            self.detection_service.update_thresholds(new_thresholds)
            
            self.logger.info(f"Thresholds updated: {new_thresholds}")
            
            return jsonify({
                'status': 'success',
                'data': {
                    'message': 'Thresholds updated successfully (in-memory)',
                    'new_thresholds': self.config['thresholds']
                },
                'timestamp': datetime.now().isoformat(),
                'mode': 'stateless'
            })
                
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def process_frame(self, request):
        """Process single frame for fast real-time detection - Base64 only"""
        start_time = time.time()
        temp_file = None
        
        try:
            self.logger.info("Processing frame request (fast mode)")
            
            # Only accept JSON with base64 data for frames
            if not request.json or 'frame_base64' not in request.json:
                return jsonify({
                    'status': 'error',
                    'error': 'No frame_base64 data provided in JSON body',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            base64_data = request.json['frame_base64']
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]
            
            try:
                image_data = base64.b64decode(base64_data)
            except Exception as decode_error:
                return jsonify({
                    'status': 'error',
                    'error': f'Invalid base64 frame data: {str(decode_error)}',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            filename = request.json.get('filename', f"frame_{int(time.time())}.jpg")
            
            # Validate file size (5MB limit)
            if len(image_data) > 5 * 1024 * 1024:
                return jsonify({
                    'status': 'error',
                    'error': 'Frame too large. Maximum size is 5MB',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            self.logger.info(f"Processing frame - filename: {filename}, size: {len(image_data)} bytes")
            
            # Create temporary file for processing
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(image_data)
            temp_file.close()
            
            # Fast frame processing (optimized settings)
            fast_mode = request.json.get('fast_mode', True)
            include_annotation = request.json.get('include_annotation', True)
            
            result = self.detection_service.process_frame(
                image_data, filename, temp_file.name, 
                fast_mode=fast_mode, include_annotation=include_annotation
            )
            
            if not result:
                return jsonify({
                    'status': 'error',
                    'error': 'Frame processing failed',
                    'timestamp': datetime.now().isoformat()
                }), 500
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            # Format response for frame processing
            response = self._format_frame_response(result)
            
            self.logger.info(f"Frame processed successfully - Decision: {result.get('final_decision')}")
            
            return jsonify({
                'status': 'success',
                'data': response,
                'timestamp': datetime.now().isoformat(),
                'mode': 'stateless_frame'
            })
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            
            return jsonify({
                'status': 'error',
                'error': f'Frame processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file: {e}")
    
    def _format_detection_response(self, result, include_annotation=True):
        """Format single image detection response"""
        response = {
            'final_decision': result.get('final_decision'),
            'processing_time': result.get('processing_time'),
            'anomaly_detection': {
                'anomaly_score': result.get('anomaly_detection', {}).get('anomaly_score'),
                'decision': result.get('anomaly_detection', {}).get('decision'),
                'threshold_used': result.get('anomaly_detection', {}).get('threshold_used')
            },
            'detected_defects': result.get('detected_defect_types', []),
            'defect_count': len(result.get('detected_defect_types', [])),
            'confidence_level': self._calculate_confidence_level(result),
            'result_summary': {
                'is_defective': result.get('final_decision') == 'DEFECT',
                'defect_types_found': result.get('detected_defect_types', []),
                'processing_status': 'completed'
            },
            'mode': 'stateless'
        }
        
        # Include annotated image if available and requested
        if include_annotation and result.get('annotated_image_base64'):
            response['annotated_image'] = {
                'base64': result['annotated_image_base64'],
                'format': 'jpeg',
                'encoding': 'base64'
            }
        
        return response
    
    def _format_frame_response(self, result):
        """Format frame processing response (optimized for speed)"""
        response = {
            'final_decision': result.get('final_decision'),
            'processing_time': result.get('processing_time'),
            'anomaly_score': result.get('anomaly_detection', {}).get('anomaly_score'),
            'detected_defects': result.get('detected_defect_types', []),
            'defect_count': len(result.get('detected_defect_types', [])),
            'confidence_level': self._calculate_confidence_level(result),
            'is_defective': result.get('final_decision') == 'DEFECT',
            'mode': 'stateless_frame'
        }
        
        # Include annotated image if available
        if result.get('annotated_image_base64'):
            response['annotated_image'] = {
                'base64': result['annotated_image_base64'],
                'format': 'jpeg',
                'encoding': 'base64'
            }
        
        return response
    
    def _format_batch_response(self, results, total_processing_time):
        """Format batch processing response"""
        total_images = len(results)
        defective_count = sum(1 for r in results if r.get('final_decision') == 'DEFECT')
        good_count = sum(1 for r in results if r.get('final_decision') == 'GOOD')
        
        return {
            'summary': {
                'total_images': total_images,
                'good_products': good_count,
                'defective_products': defective_count,
                'defect_rate': (defective_count / total_images * 100) if total_images > 0 else 0,
                'success_rate': 100.0,  # All processed successfully if we got here
                'total_processing_time': total_processing_time,
                'avg_processing_time': total_processing_time / total_images if total_images > 0 else 0
            },
            'results': [self._format_detection_response(r) for r in results],
            'mode': 'stateless'
        }
    
    def _calculate_confidence_level(self, result):
        """Calculate confidence level from detection result"""
        score = result.get('anomaly_detection', {}).get('anomaly_score', 0.0)
        decision = result.get('final_decision', 'UNKNOWN')
        
        if decision == 'GOOD':
            if score < 0.2:
                return "very_high"
            elif score < 0.4:
                return "high"
            elif score < 0.6:
                return "medium"
            else:
                return "low"
        else:  # DEFECT
            if score > 0.9:
                return "very_high"
            elif score > 0.8:
                return "high"
            elif score > 0.7:
                return "medium"
            else:
                return "low"