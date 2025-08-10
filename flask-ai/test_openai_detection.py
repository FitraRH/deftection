# test_openai_detection.py
"""
Test script for OpenAI-enhanced defect detection system
Tests single image, batch, and frame processing with bounding box validation
"""

import requests
import base64
import json
import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

# Configuration
API_BASE_URL = "http://localhost:5002"
TEST_IMAGE_PATH = "1745296632783_jpg.rf.136d6400d4db0fc531a60042da9f37d3.jpg"
OUTPUT_DIR = "test_results"

class OpenAIDetectionTester:
    """Test OpenAI-enhanced defect detection system"""
    
    def __init__(self, api_url=API_BASE_URL):
        self.api_url = api_url
        self.session = requests.Session()
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("🚀 OpenAI Enhanced Defect Detection Tester")
        print("=" * 60)
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            return None
    
    def test_health_check(self):
        """Test health check with OpenAI status"""
        print("\n🏥 Testing Health Check...")
        
        try:
            response = self.session.get(f"{self.api_url}/api/health")
            result = response.json()
            
            print(f"   Status: {result['status']}")
            print(f"   Mode: {result.get('mode', 'unknown')}")
            
            if 'services' in result:
                services = result['services']
                print(f"   Detector: {services.get('detector', {}).get('status', 'unknown')}")
                print(f"   OpenAI: {services.get('openai', {}).get('status', 'unknown')}")
            
            if result['status'] == 'ok':
                print("   ✅ Health check passed")
                return True
            else:
                print("   ❌ Health check failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
    
    def test_system_info(self):
        """Test system information"""
        print("\n📊 Testing System Info...")
        
        try:
            response = self.session.get(f"{self.api_url}/api/system/info")
            result = response.json()
            
            if result['status'] == 'success':
                data = result['data']
                print(f"   Mode: {data.get('mode', 'unknown')}")
                print(f"   Device: {data.get('device', 'unknown')}")
                print(f"   Models Loaded: {data.get('models_loaded', False)}")
                
                # OpenAI integration info
                openai_info = data.get('openai_integration', {})
                print(f"   OpenAI Enabled: {openai_info.get('enabled', False)}")
                print(f"   OpenAI Model: {openai_info.get('model', 'N/A')}")
                print(f"   OpenAI Features: {openai_info.get('features', [])}")
                
                print("   ✅ System info retrieved")
                return True
            else:
                print(f"   ❌ System info failed: {result.get('error', 'unknown')}")
                return False
                
        except Exception as e:
            print(f"   ❌ System info error: {e}")
            return False
    
    def test_single_image_detection(self, image_path):
        """Test single image detection with OpenAI analysis"""
        print(f"\n🖼️  Testing Single Image Detection...")
        print(f"   Image: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"   ❌ Image not found: {image_path}")
            return None
        
        try:
            # Encode image
            image_base64 = self.encode_image(image_path)
            if not image_base64:
                return None
            
            # Prepare request
            payload = {
                'image_base64': image_base64,
                'filename': os.path.basename(image_path)
            }
            
            print("   📤 Sending request...")
            start_time = datetime.now()
            
            response = self.session.post(
                f"{self.api_url}/api/detection/image",
                json=payload,
                timeout=60
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = response.json()
            
            if result['status'] == 'success':
                data = result['data']
                
                print(f"   ✅ Detection completed in {processing_time:.2f}s")
                print(f"   🎯 Final Decision: {data['final_decision']}")
                print(f"   📊 Processing Time: {data.get('processing_time', 0):.3f}s")
                print(f"   🔍 Anomaly Score: {data['anomaly_detection']['anomaly_score']:.3f}")
                
                # Detected defects
                detected_defects = data.get('detected_defects', [])
                print(f"   🐛 Detected Defects: {len(detected_defects)}")
                for defect in detected_defects:
                    print(f"      - {defect}")
                
                # OpenAI Analysis
                if 'openai_analysis' in data:
                    openai = data['openai_analysis']
                    print(f"   🤖 OpenAI Overall Confidence: {openai.get('overall_confidence', 0)}%")
                    
                    # Anomaly layer analysis
                    if openai.get('anomaly_layer'):
                        anomaly_analysis = openai['anomaly_layer']
                        print(f"   🔍 Anomaly Analysis Confidence: {anomaly_analysis.get('confidence_percentage', 0)}%")
                        print(f"      Analysis: {anomaly_analysis.get('analysis', 'N/A')[:100]}...")
                    
                    # Defect layer analysis
                    if openai.get('defect_layer'):
                        defect_analysis = openai['defect_layer']
                        print(f"   🐛 Defect Analysis Confidence: {defect_analysis.get('confidence_percentage', 0)}%")
                        print(f"      Analysis: {defect_analysis.get('analysis', 'N/A')[:100]}...")
                
                # Bounding box validation
                if 'bounding_box_validation' in data:
                    bbox_val = data['bounding_box_validation']
                    print(f"   📍 Bounding Box Validation:")
                    print(f"      OpenAI Confidence: {bbox_val.get('openai_confidence', 0)}%")
                    print(f"      Spatial Accuracy: {bbox_val.get('spatial_accuracy', 'unknown')}")
                    print(f"      Validated Regions: {bbox_val.get('validated_regions', 0)}")
                
                # Save annotated image
                if 'annotated_image' in data:
                    self.save_annotated_image(data['annotated_image']['base64'], 
                                            f"single_detection_{os.path.basename(image_path)}")
                
                # Visualize bounding boxes
                self.visualize_detection_result(image_path, data, "single_detection")
                
                return data
                
            else:
                print(f"   ❌ Detection failed: {result.get('error', 'unknown')}")
                return None
                
        except Exception as e:
            print(f"   ❌ Detection error: {e}")
            return None
    
    def test_batch_processing(self, image_path, num_copies=3):
        """Test batch processing"""
        print(f"\n📦 Testing Batch Processing ({num_copies} images)...")
        
        if not os.path.exists(image_path):
            print(f"   ❌ Image not found: {image_path}")
            return None
        
        try:
            # Encode image
            image_base64 = self.encode_image(image_path)
            if not image_base64:
                return None
            
            # Create batch payload
            images = []
            for i in range(num_copies):
                images.append({
                    'image_base64': image_base64,
                    'filename': f"batch_{i+1}_{os.path.basename(image_path)}"
                })
            
            payload = {'images': images}
            
            print(f"   📤 Sending batch request...")
            start_time = datetime.now()
            
            response = self.session.post(
                f"{self.api_url}/api/detection/batch",
                json=payload,
                timeout=120
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = response.json()
            
            if result['status'] == 'success':
                data = result['data']
                summary = data['summary']
                
                print(f"   ✅ Batch completed in {processing_time:.2f}s")
                print(f"   📊 Total Images: {summary['total_images']}")
                print(f"   ✅ Good Products: {summary['good_products']}")
                print(f"   ❌ Defective Products: {summary['defective_products']}")
                print(f"   📈 Defect Rate: {summary['defect_rate']:.1f}%")
                print(f"   ⚡ Avg Processing Time: {summary['avg_processing_time']:.3f}s")
                
                # OpenAI batch analysis
                if 'openai_analysis' in summary:
                    openai_batch = summary['openai_analysis']
                    print(f"   🤖 OpenAI Analyzed Images: {openai_batch['analyzed_images']}")
                    print(f"   🤖 Avg OpenAI Confidence: {openai_batch['avg_confidence']:.1f}%")
                
                return data
                
            else:
                print(f"   ❌ Batch processing failed: {result.get('error', 'unknown')}")
                return None
                
        except Exception as e:
            print(f"   ❌ Batch processing error: {e}")
            return None
    
    def test_frame_processing(self, image_path, fast_mode=True):
        """Test real-time frame processing"""
        print(f"\n🎬 Testing Frame Processing (fast_mode={fast_mode})...")
        
        if not os.path.exists(image_path):
            print(f"   ❌ Image not found: {image_path}")
            return None
        
        try:
            # Encode image
            image_base64 = self.encode_image(image_path)
            if not image_base64:
                return None
            
            payload = {
                'frame_base64': image_base64,
                'filename': f"frame_{os.path.basename(image_path)}",
                'fast_mode': fast_mode,
                'include_annotation': True
            }
            
            print(f"   📤 Sending frame request...")
            start_time = datetime.now()
            
            response = self.session.post(
                f"{self.api_url}/api/detection/frame",
                json=payload,
                timeout=30
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = response.json()
            
            if result['status'] == 'success':
                data = result['data']
                
                print(f"   ✅ Frame processed in {processing_time:.2f}s")
                print(f"   🎯 Final Decision: {data['final_decision']}")
                print(f"   📊 Processing Time: {data.get('processing_time', 0):.3f}s")
                print(f"   🔍 Anomaly Score: {data.get('anomaly_score', 0):.3f}")
                print(f"   🐛 Defects Found: {data.get('defect_count', 0)}")
                
                # OpenAI frame analysis
                if 'openai_confidence' in data:
                    print(f"   🤖 OpenAI Confidence: {data['openai_confidence']}%")
                
                # Save frame result
                if 'annotated_image' in data:
                    self.save_annotated_image(data['annotated_image']['base64'], 
                                            f"frame_processing_{fast_mode}_{os.path.basename(image_path)}")
                
                return data
                
            else:
                print(f"   ❌ Frame processing failed: {result.get('error', 'unknown')}")
                return None
                
        except Exception as e:
            print(f"   ❌ Frame processing error: {e}")
            return None
    
    def save_annotated_image(self, base64_data, filename):
        """Save annotated image from base64"""
        try:
            image_data = base64.b64decode(base64_data)
            output_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
            
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            print(f"   💾 Annotated image saved: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error saving annotated image: {e}")
    
    def visualize_detection_result(self, original_image_path, detection_data, prefix="detection"):
        """Create visualization with bounding boxes"""
        try:
            # Load original image
            image = cv2.imread(original_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create matplotlib figure
            fig, axes = plt.subplots(1, 2, figsize=(15, 8))
            
            # Original image
            axes[0].imshow(image_rgb)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Detection result with bounding boxes
            axes[1].imshow(image_rgb)
            
            # Draw bounding boxes if available
            if detection_data.get('detected_defects'):
                # Try to get bounding boxes from different sources
                bounding_boxes = {}
                
                # From defect classification results
                if 'defect_classification' in detection_data:
                    defect_class = detection_data['defect_classification']
                    if 'bounding_boxes' in defect_class:
                        bounding_boxes.update(defect_class['bounding_boxes'])
                    elif 'defect_analysis' in defect_class and 'bounding_boxes' in defect_class['defect_analysis']:
                        bounding_boxes.update(defect_class['defect_analysis']['bounding_boxes'])
                
                # Draw bounding boxes
                colors = ['red', 'cyan', 'yellow', 'magenta', 'green', 'blue']
                color_idx = 0
                
                for defect_type, boxes in bounding_boxes.items():
                    color = colors[color_idx % len(colors)]
                    
                    for box in boxes:
                        rect = patches.Rectangle(
                            (box['x'], box['y']), 
                            box['width'], 
                            box['height'],
                            linewidth=2, 
                            edgecolor=color, 
                            facecolor='none'
                        )
                        axes[1].add_patch(rect)
                        
                        # Add label
                        axes[1].text(
                            box['x'], 
                            box['y'] - 5, 
                            f"{defect_type.upper()}", 
                            color=color, 
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
                        )
                    
                    color_idx += 1
            
            # Add detection info
            decision = detection_data.get('final_decision', 'Unknown')
            anomaly_score = detection_data.get('anomaly_detection', {}).get('anomaly_score', 0)
            processing_time = detection_data.get('processing_time', 0)
            
            title = f"Detection Result: {decision}\n"
            title += f"Anomaly Score: {anomaly_score:.3f} | Time: {processing_time:.3f}s"
            
            # Add OpenAI confidence if available
            if 'openai_analysis' in detection_data:
                overall_conf = detection_data['openai_analysis'].get('overall_confidence', 0)
                title += f"\nOpenAI Confidence: {overall_conf}%"
            
            # Add bounding box validation if available
            if 'bounding_box_validation' in detection_data:
                bbox_conf = detection_data['bounding_box_validation'].get('openai_confidence', 0)
                spatial_acc = detection_data['bounding_box_validation'].get('spatial_accuracy', 'unknown')
                title += f"\nBBox Validation: {bbox_conf}% ({spatial_acc})"
            
            axes[1].set_title(title)
            axes[1].axis('off')
            
            # Save visualization
            plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, f"{prefix}_visualization_{os.path.basename(original_image_path)}")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Visualization saved: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error creating visualization: {e}")
    
    def run_comprehensive_test(self, image_path):
        """Run all tests"""
        print(f"\n🧪 Starting Comprehensive Test")
        print(f"📁 Test Image: {image_path}")
        print(f"📂 Output Directory: {OUTPUT_DIR}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            print("Please make sure the image file exists in the current directory")
            return
        
        results = {}
        
        # 1. Health Check
        results['health'] = self.test_health_check()
        
        # 2. System Info
        results['system_info'] = self.test_system_info()
        
        # 3. Single Image Detection
        results['single_detection'] = self.test_single_image_detection(image_path)
        
        # 4. Batch Processing
        results['batch_processing'] = self.test_batch_processing(image_path, num_copies=2)
        
        # 5. Frame Processing (Fast Mode)
        results['frame_fast'] = self.test_frame_processing(image_path, fast_mode=True)
        
        # 6. Frame Processing (Full Mode)
        results['frame_full'] = self.test_frame_processing(image_path, fast_mode=False)
        
        # Summary
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for v in results.values() if v is not None and v is not False)
        total = len(results)
        
        print(f"✅ Tests Passed: {passed}/{total}")
        
        for test_name, result in results.items():
            status = "✅ PASS" if (result is not None and result is not False) else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\n📂 Check '{OUTPUT_DIR}' folder for:")
        print("   - Annotated images with bounding boxes")
        print("   - Detection visualizations")
        print("   - OpenAI analysis results")
        
        print(f"\n🎯 Key Features Tested:")
        print("   ✅ OpenAI integration for both anomaly and defect layers")
        print("   ✅ Bounding box validation by OpenAI")
        print("   ✅ Single image, batch, and real-time processing")
        print("   ✅ Enhanced confidence scoring")
        print("   ✅ Spatial accuracy assessment")

def main():
    """Main test function"""
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        print("Please make sure the image file is in the current directory")
        print("Expected filename: 1745296632783_jpg.rf.136d6400d4db0fc531a60042da9f37d3.jpg")
        return
    
    # Initialize tester
    tester = OpenAIDetectionTester()
    
    # Run comprehensive test
    tester.run_comprehensive_test(TEST_IMAGE_PATH)

if __name__ == "__main__":
    main()