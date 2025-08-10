# services/detection_service.py - Enhanced with OpenAI
"""
Detection Service with OpenAI integration for enhanced analysis
"""

import os
import cv2
import numpy as np
import tempfile
import time
import logging
from datetime import datetime
from main import create_detector


class DetectionService:
    """Detection Service with OpenAI-enhanced analysis"""
    
    def __init__(self):
        self.detector = None
        self.is_initialized = False
        self.initialization_error = None
        
        # In-memory configuration
        self.config = {
            'anomaly_threshold': 0.7,
            'defect_confidence_threshold': 0.85
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize detection components with OpenAI support"""
        try:
            self.logger.info("Initializing detection service with OpenAI integration...")
            
            # Initialize main detector
            self.detector = create_detector()
            
            if not self.detector or not self.detector.is_ready():
                raise RuntimeError("Main detector initialization failed")
            
            self.is_initialized = True
            self.logger.info("Detection service with OpenAI integration ready")
            
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"Detection service initialization failed: {e}")
            self.is_initialized = False
    
    def get_health_status(self):
        """Get health status including OpenAI availability"""
        from config import OPENAI_API_KEY
        
        return {
            'detector': {
                'available': self.detector is not None,
                'ready': self.detector.is_ready() if self.detector else False,
                'status': 'operational' if self.detector and self.detector.is_ready() else 'not_ready'
            },
            'openai': {
                'available': bool(OPENAI_API_KEY),
                'status': 'operational' if OPENAI_API_KEY else 'not_configured'
            },
            'overall_status': 'healthy' if self.is_initialized else 'degraded',
            'initialization_error': self.initialization_error,
            'mode': 'stateless_with_openai'
        }
    
    def get_system_information(self):
        """Get system information including OpenAI integration status"""
        if not self.detector:
            return {
                'status': 'not_initialized',
                'error': 'Detector not available',
                'mode': 'stateless_with_openai'
            }
        
        try:
            from config import OPENAI_API_KEY, OPENAI_MODEL
            
            system_info = self.detector.get_system_info()
            
            # Add OpenAI integration info
            system_info.update({
                'service_status': 'operational' if self.is_initialized else 'degraded',
                'openai_integration': {
                    'enabled': bool(OPENAI_API_KEY),
                    'model': OPENAI_MODEL if OPENAI_API_KEY else None,
                    'features': ['anomaly_analysis', 'defect_analysis'] if OPENAI_API_KEY else []
                },
                'components': {
                    'main_detector': True,
                    'openai_analyzer': bool(OPENAI_API_KEY),
                    'database': False,
                    'file_storage': False
                },
                'api_version': '1.0.0',
                'mode': 'stateless_with_openai'
            })
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'mode': 'stateless_with_openai'
            }
    
    def get_current_status(self):
        """Get current system status"""
        from config import OPENAI_API_KEY
        
        return {
            'system_ready': self.is_initialized,
            'detector_ready': self.detector.is_ready() if self.detector else False,
            'openai_ready': bool(OPENAI_API_KEY),
            'processing_capabilities': {
                'single_image': self.detector is not None,
                'batch_processing': self.detector is not None,
                'openai_analysis': bool(OPENAI_API_KEY),
                'enhanced_analysis': bool(OPENAI_API_KEY)
            },
            'current_load': self._get_current_load(),
            'memory_usage': self._get_memory_usage(),
            'mode': 'stateless_with_openai',
            'last_check': datetime.now().isoformat()
        }
    
    def process_single_image(self, image_data, filename, temp_file_path=None, include_annotation=True):
        """Process single image with OpenAI-enhanced analysis"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        try:
            self.logger.info(f"Processing image with OpenAI analysis: {filename}")
            
            # Use temporary file path if provided, otherwise create one
            if temp_file_path:
                image_path = temp_file_path
            else:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(image_data)
                temp_file.close()
                image_path = temp_file.name
            
            # Process using existing detector (now with OpenAI integration)
            result = self.detector.process_image(image_path)
            
            # Generate annotated image if requested
            if include_annotation and result:
                annotated_base64 = self._generate_annotated_image(image_path, result)
                if annotated_base64:
                    result['annotated_image_base64'] = annotated_base64
            
            # Clean up temporary file if we created it
            if not temp_file_path and os.path.exists(image_path):
                os.remove(image_path)
            
            if result:
                self.logger.info(f"Image processed with OpenAI analysis - Decision: {result.get('final_decision')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            if not temp_file_path and 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
            raise
    
    def process_frame(self, image_data, filename, temp_file_path, fast_mode=True, include_annotation=True):
        """Process frame with OpenAI analysis (optimized for real-time)"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        try:
            self.logger.info(f"Processing frame with OpenAI (fast: {fast_mode}): {filename}")
            
            # For fast mode, limit OpenAI analysis to critical decisions only
            if fast_mode:
                result = self._process_frame_fast_with_openai(temp_file_path)
            else:
                result = self.detector.process_image(temp_file_path)
            
            # Generate annotated image if requested
            if include_annotation and result:
                annotated_base64 = self._generate_annotated_image(temp_file_path, result, fast_mode=fast_mode)
                if annotated_base64:
                    result['annotated_image_base64'] = annotated_base64
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            raise
    
    def _process_frame_fast_with_openai(self, image_path):
        """Fast frame processing with selective OpenAI analysis"""
        try:
            # Use detector's anomaly detection first
            result = self.detector.detect_anomaly(image_path)
            
            # Only use OpenAI for borderline cases in fast mode
            if result and 'anomaly_detection' in result:
                anomaly_score = result['anomaly_detection'].get('anomaly_score', 0)
                
                # Only analyze with OpenAI if score is borderline (0.6-0.8 range)
                if 0.6 <= anomaly_score <= 0.8:
                    # Quick OpenAI analysis for borderline cases
                    from config import OPENAI_API_KEY
                    if OPENAI_API_KEY and hasattr(self.detector, 'detection_core'):
                        openai_analysis = self.detector.detection_core._analyze_anomaly_with_openai(image_path, result['anomaly_detection'])
                        if openai_analysis:
                            result['openai_analysis'] = openai_analysis
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in fast frame processing: {e}")
            return None
    
    def _generate_annotated_image(self, image_path, result, fast_mode=False):
        """Generate annotated image with OpenAI insights"""
        try:
            from utils.stateless_visualization import create_annotated_image_base64
            return create_annotated_image_base64(image_path, result)
            
        except ImportError:
            import cv2
            import base64
            
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            annotated_image = self._annotate_image_with_openai_insights(image, result, fast_mode)
            
            _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error generating annotated image: {e}")
            return None
    
    def _annotate_image_with_openai_insights(self, image, result, fast_mode=False):
        """Add annotations including OpenAI insights"""
        try:
            import cv2
            from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES
            
            annotated = image.copy()
            height, width = annotated.shape[:2]
            
            # Get decision and score
            decision = result.get('final_decision', 'UNKNOWN')
            anomaly_score = result.get('anomaly_detection', {}).get('anomaly_score', 0.0)
            
            # Add border based on decision
            if decision == 'GOOD':
                annotated = cv2.copyMakeBorder(annotated, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 0))
                cv2.putText(annotated, "GOOD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            elif decision == 'DEFECT':
                annotated = cv2.copyMakeBorder(annotated, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                cv2.putText(annotated, "DEFECT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Draw defect bounding boxes if available
                if result.get('defect_classification'):
                    self._draw_bounding_boxes(annotated, result['defect_classification'])
            
            # Add OpenAI confidence if available
            openai_analysis = result.get('anomaly_detection', {}).get('openai_analysis') or result.get('defect_classification', {}).get('openai_analysis')
            if openai_analysis and 'confidence_percentage' in openai_analysis:
                cv2.putText(annotated, f"AI Confidence: {openai_analysis['confidence_percentage']}%", 
                           (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add processing info
            cv2.putText(annotated, f"Score: {anomaly_score:.3f}", (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            self.logger.error(f"Error annotating image: {e}")
            return image
    
    def _draw_bounding_boxes(self, image, defect_classification):
        """Draw bounding boxes for detected defects"""
        try:
            import cv2
            from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES
            
            defect_analysis = defect_classification.get('defect_analysis', {})
            bounding_boxes = defect_analysis.get('bounding_boxes', {})
            
            if not bounding_boxes:
                bounding_boxes = defect_classification.get('bounding_boxes', {})
            
            for defect_type, boxes in bounding_boxes.items():
                defect_class_id = None
                for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
                    if class_name == defect_type:
                        defect_class_id = class_id
                        break
                
                if defect_class_id is not None and defect_class_id in DEFECT_COLORS:
                    color = DEFECT_COLORS[defect_class_id]
                else:
                    default_colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
                    color = default_colors[hash(defect_type) % len(default_colors)]
                
                for bbox in boxes:
                    x, y = bbox['x'], bbox['y']
                    w, h = bbox['width'], bbox['height']
                    
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, defect_type.upper(), (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        except Exception as e:
            self.logger.error(f"Error drawing bounding boxes: {e}")
    
    def update_thresholds(self, new_thresholds):
        """Update detection thresholds in memory"""
        try:
            for key, value in new_thresholds.items():
                if key in ['anomaly_threshold', 'defect_confidence_threshold']:
                    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                        raise ValueError(f"Invalid value for {key}: must be between 0 and 1")
            
            self.config.update(new_thresholds)
            self.logger.info(f"Thresholds updated: {new_thresholds}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {e}")
            return False
    
    def get_thresholds(self):
        """Get current detection thresholds"""
        return {
            'anomaly_threshold': self.config.get('anomaly_threshold', 0.7),
            'defect_confidence_threshold': self.config.get('defect_confidence_threshold', 0.85),
            'configurable': True,
            'storage': 'in-memory',
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_current_load(self):
        """Get current system load"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'status': 'normal'
            }
        except ImportError:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'status': 'monitoring_unavailable'
            }
    
    def _get_memory_usage(self):
        """Get memory usage information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent_used': memory.percent,
                'mode': 'stateless_with_openai'
            }
        except ImportError:
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percent_used': 0,
                'mode': 'stateless_with_openai'
            }