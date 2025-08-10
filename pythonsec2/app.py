# app.py
"""
Image Security Scanner - Main Flask Application
Updated to use simplified configuration
"""

from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# Import simplified configuration
from config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    SECRET_KEY,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS,
    ERROR_CODES,
    DEBUG_MODE,
    DETAILED_ERRORS,
    get_config_summary,
    validate_configuration
)

# Import controller
from controllers.image_security_controller import ImageSecurityController

def create_app():
    """Create Flask application"""
    app = Flask(__name__)
    
    # Apply configuration
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    app.config['DEBUG'] = FLASK_DEBUG
    
    # Enable CORS
    CORS(app)
    
    # Validate configuration on startup
    config_issues = validate_configuration()
    if config_issues:
        print("Configuration issues detected:")
        for issue in config_issues:
            print(f"  - {issue}")
    
    # Initialize controller
    try:
        controller = ImageSecurityController()
        print("✅ ImageSecurityController initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ImageSecurityController: {e}")
        raise
    
    # ===========================
    # ERROR HANDLERS
    # ===========================
    
    @app.errorhandler(413)
    def file_too_large(error):
        """Handle file too large error"""
        max_size_mb = MAX_FILE_SIZE / (1024 * 1024)
        return jsonify({
            'status': 'error',
            'error_code': ERROR_CODES['FILE_TOO_LARGE'],
            'message': f'File too large. Maximum size allowed: {max_size_mb}MB',
            'max_file_size_mb': max_size_mb,
            'timestamp': datetime.now().isoformat()
        }), 413
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors"""
        return jsonify({
            'status': 'error',
            'error_code': ERROR_CODES['INVALID_FILE_TYPE'],
            'message': 'Invalid request format or missing required parameters',
            'timestamp': datetime.now().isoformat()
        }), 400
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server errors"""
        return jsonify({
            'status': 'error',
            'error_code': ERROR_CODES['INTERNAL_ERROR'],
            'message': 'Internal server error occurred during processing',
            'timestamp': datetime.now().isoformat(),
            'debug_info': str(error) if DETAILED_ERRORS else None
        }), 500
    
    # ===========================
    # MAIN ROUTES
    # ===========================
    
    @app.route('/', methods=['GET'])
    def home():
        """Home endpoint with service information"""
        return jsonify({
            'service': 'Image Security Scanner',
            'version': '2.0.0',
            'status': 'running',
            'debug_mode': DEBUG_MODE,
            'endpoints': {
                'scan': '/scan',
                'health': '/health',
                'stats': '/stats',
                'config': '/config'
            },
            'documentation': 'See README.md for usage examples',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/scan', methods=['POST'])
    def scan_image():
        """Main image scanning endpoint"""
        try:
            return controller.scan_image(request)
        except Exception as e:
            print(f"Scan endpoint error: {e}")
            return jsonify({
                'status': 'error',
                'error_code': ERROR_CODES['INTERNAL_ERROR'],
                'message': 'Error occurred during image scanning',
                'details': str(e) if DETAILED_ERRORS else None,
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            return controller.health_check()
        except Exception as e:
            print(f"Health check error: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e) if DETAILED_ERRORS else 'Health check failed',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/stats', methods=['GET'])
    def get_stats():
        """Get scanner statistics and capabilities"""
        try:
            return controller.get_scanner_stats()
        except Exception as e:
            print(f"Stats endpoint error: {e}")
            return jsonify({
                'status': 'error',
                'error_code': ERROR_CODES['INTERNAL_ERROR'],
                'message': 'Error retrieving scanner statistics',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/config', methods=['GET'])
    def get_configuration():
        """Get current configuration summary"""
        try:
            config_summary = get_config_summary()
            
            return jsonify({
                'status': 'success',
                'configuration': config_summary,
                'validation': {
                    'issues': validate_configuration(),
                    'valid': len(validate_configuration()) == 0
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"Config endpoint error: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Error retrieving configuration',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    # ===========================
    # DEBUG ROUTES (Development only)
    # ===========================
    
    if DEBUG_MODE:
        @app.route('/debug/config', methods=['GET'])
        def debug_config():
            """Debug endpoint for detailed configuration"""
            from config import YARA_RULES_DIR, MALWARE_HASH_FILE
            
            return jsonify({
                'flask_config': dict(app.config),
                'security_config': {
                    'allowed_extensions': list(ALLOWED_EXTENSIONS),
                    'max_file_size': MAX_FILE_SIZE,
                    'error_codes': ERROR_CODES
                },
                'paths': {
                    'yara_rules_dir': str(YARA_RULES_DIR),
                    'malware_hash_file': str(MALWARE_HASH_FILE)
                },
                'debug_mode': DEBUG_MODE,
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/debug/test', methods=['POST'])
        def debug_test():
            """Debug endpoint for testing"""
            return jsonify({
                'message': 'Debug test endpoint',
                'request_method': request.method,
                'content_type': request.content_type,
                'content_length': request.content_length,
                'timestamp': datetime.now().isoformat()
            })
    
    return app

def main():
    """Main application entry point"""
    print("=" * 50)
    print("Image Security Scanner - Starting")
    print("=" * 50)
    
    # Create and configure app
    app = create_app()
    
    # Print startup information
    print(f"Host: {FLASK_HOST}")
    print(f"Port: {FLASK_PORT}")
    print(f"Debug: {FLASK_DEBUG}")
    print(f"Max File Size: {MAX_FILE_SIZE / (1024 * 1024):.1f}MB")
    print(f"Allowed Extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    
    # Validate configuration
    config_issues = validate_configuration()
    if config_issues:
        print("\nConfiguration Issues:")
        for issue in config_issues:
            if "Optional:" in issue:
                print(f"  ⚠️  {issue}")
            else:
                print(f"  ❌ {issue}")
    else:
        print("\n✅ Configuration validated successfully!")
    
    print("\n" + "=" * 50)
    print("API Endpoints:")
    print("  POST /scan           - Scan image files")
    print("  GET  /health         - Health check")
    print("  GET  /stats          - Scanner statistics")
    print("  GET  /config         - Configuration summary")
    if DEBUG_MODE:
        print("  GET  /debug/config   - Debug configuration")
        print("  POST /debug/test     - Debug test endpoint")
    print("=" * 50)
    
    # Start the application
    try:
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=FLASK_DEBUG,
            threaded=True
        )
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())