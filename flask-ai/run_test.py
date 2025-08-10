# run_test.py
"""
Easy test runner for OpenAI-enhanced defect detection
"""

import subprocess
import sys
import os
import time

def check_requirements():
    """Check if required packages are installed"""
    required = ['requests', 'opencv-python', 'matplotlib', 'pillow', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("📦 Installing missing packages...")
        for package in missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def check_api_server():
    """Check if API server is running"""
    import requests
    
    try:
        response = requests.get("http://localhost:5002/api/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    
    return False

def main():
    print("🚀 OpenAI Enhanced Defect Detection - Test Runner")
    print("=" * 60)
    
    # Check requirements
    print("1️⃣ Checking requirements...")
    check_requirements()
    print("   ✅ Requirements satisfied")
    
    # Check API server
    print("2️⃣ Checking API server...")
    if not check_api_server():
        print("   ❌ API server not running!")
        print("   Please start the API server first:")
        print("   > python api_server.py")
        return
    
    print("   ✅ API server is running")
    
    # Check test image
    test_image = "1745296632783_jpg.rf.136d6400d4db0fc531a60042da9f37d3.jpg"
    print("3️⃣ Checking test image...")
    
    if not os.path.exists(test_image):
        print(f"   ❌ Test image not found: {test_image}")
        print("   Please make sure the image file is in the current directory")
        return
    
    print("   ✅ Test image found")
    
    # Run test
    print("4️⃣ Running comprehensive test...")
    print("=" * 60)
    
    try:
        from test_openai_detection import OpenAIDetectionTester
        
        tester = OpenAIDetectionTester()
        tester.run_comprehensive_test(test_image)
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()