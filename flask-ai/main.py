# main.py - Stateless Version
"""
Stateless Main Unified Defect Detector class
No file operations, no database, only in-memory processing
"""

import os
import torch
import tempfile
import logging
from datetime import datetime

# Import your custom config
from config import *

# Import model loader
try:
    from models.model_loader import ModelLoader
except ImportError:
    from model_loader import ModelLoader

# Try to import core detection (simplified for stateless)
try:
    from core.detection import DetectionCore
    DETECTION_CORE_AVAILABLE = True
    print("Detection core available")
except ImportError:
    print("Detection core not available - using simple detection mode")
    DETECTION_CORE_AVAILABLE = False


class StatelessDefectDetector:
    """
    Stateless Unified Defect Detection System
    No file operations, no database, only in-memory processing
    """
    
    def __init__(self, anomalib_model_path=None, hrnet_model_path=None, 
                 anomalib_path=None, hrnet_path=None, device=None, auto_load=True):
        """
        Initialize the stateless detector
        """
        # Handle parameter compatibility
        final_anomalib_path = anomalib_path or anomalib_model_path
        final_hrnet_path = hrnet_path or hrnet_model_path
        
        # Device setup
        self.device = device if device else DEVICE
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model loader
        self.model_loader = ModelLoader(device=self.device)
        self.detection_core = None
        
        print(f"Initializing STATELESS Unified Defect Detection System on {self.device}")
        
        # Load models
        if auto_load:
            success = self.load_models(final_anomalib_path, final_hrnet_path)
            if success:
                self._init_detection_core()
                print("Stateless system ready!")
            else:
                print("Model loading failed")
    
    def load_models(self, anomalib_path=None, hrnet_path=None):
        """Load models"""
        try:
            return self.model_loader.load_models(anomalib_path, hrnet_path)
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def _init_detection_core(self):
        """Initialize detection core if available"""
        if DETECTION_CORE_AVAILABLE:
            try:
                anomalib_model, hrnet_model = self.model_loader.get_models()
                self.detection_core = DetectionCore(anomalib_model, hrnet_model, self.device)
                print("Detection core initialized")
            except Exception as e:
                self.logger.warning(f"Detection core initialization failed: {e}")
    
    def is_ready(self):
        """Check if system is ready"""
        return self.model_loader.is_ready()
    
    def detect_anomaly(self, image_path):
        """
        Simple anomaly detection - stateless
        """
        if not self.is_ready():
            raise RuntimeError("Models not loaded")
        
        self.logger.info(f"Detecting anomaly in: {os.path.basename(image_path)}")
        
        # Get the anomalib model directly
        anomalib_model, _ = self.model_loader.get_models()
        
        # Run prediction
        result = anomalib_model.predict(image=image_path)
        
        # Process result
        if hasattr(result, 'pred_score'):
            if isinstance(result.pred_score, torch.Tensor):
                anomaly_score = float(result.pred_score.cpu().item())
            else:
                anomaly_score = float(result.pred_score)
        else:
            anomaly_score = 0.0
        
        if hasattr(result, 'pred_label'):
            if isinstance(result.pred_label, torch.Tensor):
                is_anomalous = bool(result.pred_label.cpu().item())
            else:
                is_anomalous = bool(result.pred_label)
        else:
            is_anomalous = anomaly_score > ANOMALY_THRESHOLD
        
        # Return stateless result
        return {
            'image_path': image_path,
            'anomaly_score': anomaly_score,
            'is_anomalous': is_anomalous,
            'decision': 'ANOMALY' if is_anomalous else 'NORMAL',
            'threshold_used': ANOMALY_THRESHOLD,
            'final_decision': 'DEFECT' if is_anomalous else 'GOOD',
            'processing_time': 0.1,  # Simple fallback
            'timestamp': datetime.now().isoformat(),
            'anomaly_detection': {
                'anomaly_score': anomaly_score,
                'decision': 'DEFECT' if is_anomalous else 'GOOD',
                'threshold_used': ANOMALY_THRESHOLD
            },
            'detected_defect_types': ['anomaly'] if is_anomalous else [],
            'mode': 'stateless'
        }
    
    def process_image(self, image_path, output_dir=None):
        """
        Process image - stateless version
        Uses advanced processing if available, otherwise simple detection
        """
        import time
        start_time = time.time()
        
        if self.detection_core:
            # Use advanced processing (stateless)
            try:
                # Step 1: Anomaly Detection
                anomaly_result = self.detection_core.detect_anomaly(image_path)
                if not anomaly_result:
                    raise RuntimeError("Anomaly detection failed")
                
                processing_time = time.time() - start_time
                
                # Prepare result
                result = {
                    'image_path': image_path,
                    'final_decision': anomaly_result['decision'],
                    'processing_time': processing_time,
                    'timestamp': datetime.now().isoformat(),
                    'anomaly_detection': anomaly_result,
                    'detected_defect_types': [],
                    'mode': 'stateless'
                }
                
                # Step 2: Defect Classification (if defective)
                if anomaly_result['decision'] == 'DEFECT':
                    defect_result = self.detection_core.classify_defects(
                        image_path, 
                        anomaly_result.get('anomaly_mask')
                    )
                    if defect_result:
                        result['defect_classification'] = defect_result
                        result['detected_defect_types'] = defect_result.get('detected_defects', [])
                
                return result
                
            except Exception as e:
                self.logger.error(f"Advanced processing failed: {e}")
                # Fall back to simple detection
        
        # Use simple detection with enhanced format
        result = self.detect_anomaly(image_path)
        
        # Enhance result for API compatibility
        processing_time = time.time() - start_time
        result.update({
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'mode': 'stateless'
        })
        
        return result
    
    def get_system_info(self):
        """Get system information - stateless version"""
        return {
            'device': self.device,
            'models_loaded': self.is_ready(),
            'system_ready': self.is_ready(),
            'detection_core_available': self.detection_core is not None,
            'anomaly_threshold': ANOMALY_THRESHOLD,
            'defect_threshold': DEFECT_CONFIDENCE_THRESHOLD,
            'supported_classes': SPECIFIC_DEFECT_CLASSES,
            'defect_classes': len(SPECIFIC_DEFECT_CLASSES),
            'version': 'STATELESS_API_COMPATIBLE',
            'detector_ready': self.is_ready(),
            'supported_formats': ['JPG', 'PNG', 'BMP'],
            'max_file_size': '5MB',
            'mode': 'stateless',
            'persistent_storage': False,
            'database_enabled': False,
            'file_writing_enabled': False
        }


# Convenience functions - stateless versions
def create_detector(anomalib_path=None, hrnet_path=None, device=None):
    """
    Quick detector creation - stateless version
    """
    print("Creating STATELESS detector...")
    return StatelessDefectDetector(
        anomalib_path=anomalib_path,
        hrnet_path=hrnet_path, 
        device=device,
        auto_load=True
    )

def quick_detect(image_path, anomalib_path=None, hrnet_path=None):
    """Quick single image detection - stateless version"""
    detector = create_detector(anomalib_path, hrnet_path)
    return detector.detect_anomaly(image_path)


# Simple test function
def test_system():
    """Simple system test - stateless version"""
    print("Testing STATELESS System...")
    print("=" * 50)
    
    try:
        detector = StatelessDefectDetector(auto_load=True)
        
        if detector.is_ready():
            print("System ready!")
            
            system_info = detector.get_system_info()
            print(f"   Device: {system_info['device']}")
            print(f"   Models: {'' if system_info['models_loaded'] else '❌'}")
            print(f"   Detection Core: {'' if system_info['detection_core_available'] else '❌'}")
            print(f"   Mode: {system_info['mode']}")
            print(f"   Persistent Storage: {system_info['persistent_storage']}")
            
            # Test API compatibility
            print("\nTesting API server compatibility...")
            if hasattr(detector, 'process_image') and callable(detector.process_image):
                print("    process_image method available")
            if hasattr(detector, 'get_system_info') and callable(detector.get_system_info):
                print("    get_system_info method available")
            if hasattr(detector, 'is_ready') and callable(detector.is_ready):
                print("    is_ready method available")
            
            # Test create_detector function
            print("\nTesting create_detector function...")
            api_detector = create_detector()
            if api_detector and api_detector.is_ready():
                print("    create_detector() works correctly")
            else:
                print("    create_detector() created detector but not ready")
            
            return True
        else:
            print("System not ready")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Unified Defect Detection System - STATELESS VERSION")
    print("=" * 70)
    print("Features:")
    print("   No database dependencies")
    print("   No file writing operations")
    print("   Only logging to files")
    print("   In-memory processing only")
    print("   Stateless API compatible")
    print("=" * 70)
    
    if test_system():
        print("\nSystem working!")
        print("\nAPI Server Compatible Methods:")
        print("   detector = create_detector()")
        print("   result = detector.process_image(image_path)")
        print("   info = detector.get_system_info()")
        print("   ready = detector.is_ready()")
        print("\nConvenience Functions:")
        print("   result = quick_detect(image_path)")
        print("\nReady for stateless API server integration!")
        print("Storage: In-memory only, no persistent data")
    else:
        print("\nSystem test failed. Check model paths in config.py")
        print("Make sure your model files exist and are accessible")
    
    print("\nThis main.py is STATELESS and compatible with your API server!")
    print("   API server can import: from main import StatelessDefectDetector, create_detector")