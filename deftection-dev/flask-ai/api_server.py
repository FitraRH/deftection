# api_server.py - Stateless API Server
"""
Stateless JSON API Server for Defect Detection System
No database, no file writing, only in-memory processing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import logging
from datetime import datetime

# Import controllers (simplified for stateless operation)
from controllers.detection_controller import DetectionController

# Import services for initialization (simplified)
from services.detection_service import DetectionService

class StatelessAPIServer:
    """
    Stateless JSON API Server
    Contains only endpoint definitions and request routing
    No database, no persistent storage
    """
    
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        
        # Setup logging only
        self._setup_logging()
        
        # Initialize services (stateless)
        self.detection_service = DetectionService()
        
        # Initialize controllers (stateless)
        self.detection_controller = DetectionController(self.detection_service)
        
        # Setup routes
        self._setup_routes()
        
        print("Stateless JSON API Server initialized")
    
    def _setup_logging(self):
        """Setup basic logging to file"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('defect_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """Setup all API endpoints"""
        
        # Health and system endpoints
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return self.detection_controller.health_check()
        
        @self.app.route('/api/system/info', methods=['GET'])
        def system_info():
            return self.detection_controller.get_system_info()
        
        @self.app.route('/api/system/status', methods=['GET'])
        def system_status():
            return self.detection_controller.get_system_status()
        
        # Detection endpoints
        @self.app.route('/api/detection/image', methods=['POST'])
        def detect_image():
            return self.detection_controller.process_image(request)
        
        @self.app.route('/api/detection/frame', methods=['POST'])
        def detect_frame():
            return self.detection_controller.process_frame(request)
        
        @self.app.route('/api/detection/batch', methods=['POST'])
        def detect_batch():
            return self.detection_controller.process_batch(request)
        
        # Configuration endpoints (in-memory only)
        @self.app.route('/api/config/thresholds', methods=['GET'])
        def get_thresholds():
            return self.detection_controller.get_detection_thresholds()
        
        @self.app.route('/api/config/thresholds', methods=['PUT'])
        def update_thresholds():
            return self.detection_controller.update_detection_thresholds(request)
        
        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested API endpoint does not exist',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'error': 'Internal server error',
                'message': 'An internal error occurred while processing the request',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                'error': 'Bad request',
                'message': 'Invalid request data or parameters',
                'timestamp': datetime.now().isoformat()
            }), 400
    
    def run(self, debug=False):
        """Start the Stateless JSON API server"""
        print("Starting Stateless JSON API Server")
        print("=" * 50)
        print(f"Server URL: http://{self.host}:{self.port}")
        print("API Documentation:")
        print(f"  Health Check: GET /api/health")
        print(f"  System Info:  GET /api/system/info")
        print(f"  Detect Image: POST /api/detection/image")
        print(f"  Detect Frame: POST /api/detection/frame (fast mode)")
        print(f"  Batch Detect: POST /api/detection/batch")
        print("=" * 50)
        print("Note: This is a STATELESS server - no database, no persistent storage")
        
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True, use_reloader=False)


def create_api_server(host='0.0.0.0', port=5002):
    """Factory function to create stateless API server instance"""
    return StatelessAPIServer(host=host, port=port)


if __name__ == "__main__":
    # Create and start the stateless API server
    api_server = create_api_server()
    api_server.run(debug=True)