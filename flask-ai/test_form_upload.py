"""
Test JSON with longer timeout to see if it's just slower
"""

import requests
import base64
import time
import os

def test_json_with_long_timeout():
    """Test JSON upload with extended timeout"""
    
    image_file = "1745296632783_jpg.rf.136d6400d4db0fc531a60042da9f37d3.jpg"
    url = "http://localhost:5002/api/detection/image"
    
    print("🧪 Testing JSON Base64 with Extended Timeout")
    print("=" * 60)
    
    if not os.path.exists(image_file):
        print(f"❌ Image file not found: {image_file}")
        return
    
    try:
        print("📤 Loading and encoding image...")
        start_encode = time.time()
        
        with open(image_file, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        encode_time = time.time() - start_encode
        print(f"✅ Base64 encoding completed in {encode_time:.3f}s")
        print(f"📊 Original size: {len(image_data)} bytes")
        print(f"📊 Base64 size: {len(image_base64)} characters")
        
        payload = {
            "image_base64": image_base64,
            "filename": image_file
        }
        
        print(f"📤 Sending JSON request with 60s timeout...")
        start_request = time.time()
        
        # Extended timeout to 60 seconds
        response = requests.post(url, json=payload, timeout=60)
        
        request_time = time.time() - start_request
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ JSON upload successful in {request_time:.2f}s")
            
            if result['status'] == 'success':
                data = result['data']
                print(f"🎯 Final Decision: {data['final_decision']}")
                print(f"📊 Processing Time: {data['processing_time']:.3f}s")
                print(f"🔍 Anomaly Score: {data['anomaly_detection']['anomaly_score']:.3f}")
                print(f"🐛 Detected Defects: {data['defect_count']}")
                
                # Total time breakdown
                total_time = encode_time + request_time
                print(f"\n⏱️ Time Breakdown:")
                print(f"   Base64 encoding: {encode_time:.3f}s")
                print(f"   HTTP request: {request_time:.3f}s")
                print(f"   Total client time: {total_time:.3f}s")
                print(f"   Server processing: {data['processing_time']:.3f}s")
                
            else:
                print(f"❌ API error: {result}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
    
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 60 seconds")
        print("💡 JSON processing might be significantly slower than form data")
    except Exception as e:
        print(f"❌ JSON test failed: {e}")

def test_small_image_json():
    """Test with a smaller image to see if size matters"""
    
    print("\n🧪 Testing JSON with Smaller Image")
    print("=" * 50)
    
    # Try to create a smaller test image
    try:
        from PIL import Image
        import io
        
        # Create a small test image
        test_img = Image.new('RGB', (100, 100), color='red')
        img_buffer = io.BytesIO()
        test_img.save(img_buffer, format='JPEG', quality=50)
        small_image_data = img_buffer.getvalue()
        
        print(f"📊 Small image size: {len(small_image_data)} bytes")
        
        image_base64 = base64.b64encode(small_image_data).decode('utf-8')
        
        payload = {
            "image_base64": image_base64,
            "filename": "small_test.jpg"
        }
        
        print("📤 Sending small image via JSON...")
        start = time.time()
        
        response = requests.post("http://localhost:5002/api/detection/image", 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            process_time = time.time() - start
            print(f"✅ Small image JSON successful in {process_time:.2f}s")
            
            if result['status'] == 'success':
                data = result['data']
                print(f"🎯 Decision: {data['final_decision']}")
                print(f"🔍 Score: {data['anomaly_detection']['anomaly_score']:.3f}")
        else:
            print(f"❌ Small image test failed: {response.status_code}")
            
    except ImportError:
        print("⚠️ PIL not available, skipping small image test")
    except Exception as e:
        print(f"❌ Small image test error: {e}")

if __name__ == "__main__":
    test_json_with_long_timeout()
    test_small_image_json()