# services/detection_service.py - Stateless Version
"""
Stateless Detection Service - Business Logic Layer
Handles all detection-related business logic
No database, no file writing, only in-memory processing
"""

import os
import cv2
import numpy as np
import tempfile
import time
import logging
from datetime import datetime

# Import existing modules from your project structure
from main import create_detector


class DetectionService:
    """
    Stateless Detection Service - Core business logic for defect detection
    Utilizes existing project components without modification
    No persistent storage, only in-memory operations
    """
    
    def __init__(self):
        # Initialize existing components
        self.detector = None
        
        # Service state (in-memory only)
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
        """Initialize all detection components using existing code"""
        try:
            self.logger.info("Initializing stateless detection service components...")
            
            # Initialize main detector (from your existing main.py)
            self.detector = create_detector()
            
            if not self.detector or not self.detector.is_ready():
                raise RuntimeError("Main detector initialization failed")
            
            self.is_initialized = True
            self.logger.info("Stateless detection service initialization completed")
            
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"Stateless detection service initialization failed: {e}")
            self.is_initialized = False
    
    def get_health_status(self):
        """Get health status of all components"""
        return {
            'detector': {
                'available': self.detector is not None,
                'ready': self.detector.is_ready() if self.detector else False,
                'status': 'operational' if self.detector and self.detector.is_ready() else 'not_ready'
            },
            'overall_status': 'healthy' if self.is_initialized else 'degraded',
            'initialization_error': self.initialization_error,
            'mode': 'stateless',
            'storage': 'in-memory-only'
        }
    
    def get_system_information(self):
        """Get detailed system information using existing detector methods"""
        if not self.detector:
            return {
                'status': 'not_initialized',
                'error': 'Detector not available',
                'mode': 'stateless'
            }
        
        try:
            # Use existing detector's get_system_info method
            system_info = self.detector.get_system_info()
            
            # Add service-specific information
            system_info.update({
                'service_status': 'operational' if self.is_initialized else 'degraded',
                'components': {
                    'main_detector': True,
                    'database': False,  # Stateless
                    'file_storage': False,  # Stateless
                    'performance_tracker': False  # Stateless
                },
                'api_version': '1.0.0',
                'mode': 'stateless',
                'persistent_storage': False,
                'in_memory_config': True
            })
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'mode': 'stateless',
                'fallback_info': {
                    'device': 'unknown',
                    'models_loaded': False,
                    'system_ready': False
                }
            }
    
    def get_current_status(self):
        """Get current system status"""
        return {
            'system_ready': self.is_initialized,
            'detector_ready': self.detector.is_ready() if self.detector else False,
            'processing_capabilities': {
                'single_image': self.detector is not None,
                'batch_processing': self.detector is not None,
                'video_processing': False,  # Disabled for stateless
                'realtime_processing': False  # Disabled for stateless
            },
            'current_load': self._get_current_load(),
            'memory_usage': self._get_memory_usage(),
            'mode': 'stateless',
            'storage_mode': 'in-memory-only',
            'last_check': datetime.now().isoformat()
        }
    
    def process_single_image(self, image_data, filename, temp_file_path=None, include_annotation=True):
        """Process single image using existing detector"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        try:
            self.logger.info(f"Processing single image: {filename}")
            
            # Use temporary file path if provided, otherwise create one
            if temp_file_path:
                image_path = temp_file_path
            else:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(image_data)
                temp_file.close()
                image_path = temp_file.name
            
            # Process using existing detector
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
                self.logger.info(f"Image processed successfully - Decision: {result.get('final_decision')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            # Cleanup on error
            if not temp_file_path and 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
            raise
    
    def process_frame(self, image_data, filename, temp_file_path, fast_mode=True, include_annotation=True):
        """Process frame with optimizations for speed"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        try:
            self.logger.info(f"Processing frame (fast mode: {fast_mode}): {filename}")
            
            # Fast mode processing - optimized settings
            if fast_mode:
                result = self._process_frame_fast(temp_file_path)
            else:
                result = self.detector.process_image(temp_file_path)
            
            # Generate annotated image if requested
            if include_annotation and result:
                annotated_base64 = self._generate_annotated_image(temp_file_path, result, fast_mode=fast_mode)
                if annotated_base64:
                    result['annotated_image_base64'] = annotated_base64
            
            if result:
                self.logger.info(f"Frame processed successfully - Decision: {result.get('final_decision')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            raise
    
    def _process_frame_fast(self, image_path):
        """Fast frame processing with optimized settings"""
        try:
            # Use simple anomaly detection only for speed
            result = self.detector.detect_anomaly(image_path)
            
            # Skip detailed defect classification for speed if needed
            # Only do basic anomaly detection
            return result
            
        except Exception as e:
            self.logger.error(f"Error in fast frame processing: {e}")
            return None
    
    def _generate_annotated_image(self, image_path, result, fast_mode=False):
        """Generate base64 encoded annotated image using original style"""
        try:
            # Use the stateless visualization utility
            from utils.stateless_visualization import create_annotated_image_base64
            return create_annotated_image_base64(image_path, result)
            
        except ImportError:
            # Fallback to local implementation if utils not available
            import cv2
            import base64
            
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"Could not load image for annotation: {image_path}")
                return None
            
            # Create annotated version using original style
            annotated_image = self._annotate_image(image, result, fast_mode=fast_mode)
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            self.logger.error(f"Error generating annotated image: {e}")
            return None
    
    def _annotate_image(self, image, result, fast_mode=False):
        """Add annotations to image based on detection results - using original style"""
        try:
            import cv2
            from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES
            
            annotated = image.copy()
            height, width = annotated.shape[:2]
            
            # Get decision and score
            decision = result.get('final_decision', 'UNKNOWN')
            anomaly_score = result.get('anomaly_detection', {}).get('anomaly_score', 0.0)
            
            # Add border based on decision (similar to original code)
            if decision == 'GOOD':
                # Add green border for good products
                annotated = cv2.copyMakeBorder(annotated, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 0))
                cv2.putText(annotated, "GOOD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            elif decision == 'DEFECT':
                # Add red border for defective products
                annotated = cv2.copyMakeBorder(annotated, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                cv2.putText(annotated, "DEFECT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Draw defect bounding boxes if available (always draw, no mode check)
                if result.get('defect_classification'):
                    self._draw_bounding_boxes(annotated, result['defect_classification'])
            else:
                # Add yellow border for unknown
                annotated = cv2.copyMakeBorder(annotated, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 255))
                cv2.putText(annotated, "UNKNOWN", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Add processing info (similar to original video processor)
            cv2.putText(annotated, f"Score: {anomaly_score:.3f}", (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(annotated, timestamp, (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return annotated
            
        except Exception as e:
            self.logger.error(f"Error annotating image: {e}")
            return image  # Return original image if annotation fails
    
    def _draw_bounding_boxes(self, image, defect_classification):
        """Draw bounding boxes for detected defects - using original style"""
        try:
            import cv2
            from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES
            
            # Use enhanced detection analysis if available
            defect_analysis = defect_classification.get('defect_analysis', {})
            bounding_boxes = defect_analysis.get('bounding_boxes', {})
            
            if not bounding_boxes:
                # Fallback to basic bounding boxes
                bounding_boxes = defect_classification.get('bounding_boxes', {})
            
            for defect_type, boxes in bounding_boxes.items():
                # Get color from config or use default
                defect_class_id = None
                for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
                    if class_name == defect_type:
                        defect_class_id = class_id
                        break
                
                if defect_class_id is not None and defect_class_id in DEFECT_COLORS:
                    color = DEFECT_COLORS[defect_class_id]
                else:
                    # Default colors if not in config
                    default_colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
                    color = default_colors[hash(defect_type) % len(default_colors)]
                
                for bbox in boxes:
                    x, y = bbox['x'], bbox['y']
                    w, h = bbox['width'], bbox['height']
                    
                    # Draw rectangle (similar to original video processor)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    
                    # Add defect type label (similar to original)
                    cv2.putText(image, defect_type.upper(), (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        except Exception as e:
            self.logger.error(f"Error drawing bounding boxes: {e}")
            # Import error handling - continue without bounding boxes
    
    def update_thresholds(self, new_thresholds):
        """Update detection thresholds in memory"""
        try:
            # Validate thresholds
            for key, value in new_thresholds.items():
                if key in ['anomaly_threshold', 'defect_confidence_threshold']:
                    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                        raise ValueError(f"Invalid value for {key}: must be between 0 and 1")
            
            # Update in-memory configuration
            self.config.update(new_thresholds)
            
            self.logger.info(f"Thresholds updated in memory: {new_thresholds}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {e}")
            return False
    
    def get_thresholds(self):
        """Get current detection thresholds from memory"""
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
                'mode': 'stateless'
            }
        except ImportError:
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percent_used': 0,
                'mode': 'stateless'
            }