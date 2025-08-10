# core/detection.py - Enhanced with OpenAI 1.x
"""
Core detection logic with OpenAI analysis integration (OpenAI 1.x compatible)
"""

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from openai import OpenAI
import base64
import io
from PIL import Image
from config import *


class DetectionCore:
    """Core detection functionality with OpenAI integration"""
    
    def __init__(self, anomalib_model, hrnet_model, device='cuda'):
        self.anomalib_model = anomalib_model
        self.hrnet_model = hrnet_model
        self.device = device
        
        # Setup OpenAI 1.x client
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.openai_enabled = True
            print("OpenAI client initialized (1.x)")
        else:
            self.openai_client = None
            self.openai_enabled = False
            print("Warning: OpenAI API key not found")
        
        # Preprocessing for HRNet
        self.hrnet_transform = A.Compose([
            A.Resize(*IMAGE_SIZE),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2()
        ])
    
    def detect_anomaly(self, image_path):
        """
        Layer 1: Anomaly detection with OpenAI analysis
        """
        if not self.anomalib_model:
            raise ValueError("Anomalib model not loaded")
        
        try:
            # Run Anomalib inference
            result = self.anomalib_model.predict(image=image_path)
            
            # Process anomaly results
            if isinstance(result.pred_score, torch.Tensor):
                anomaly_score = float(result.pred_score.cpu().item())
            else:
                anomaly_score = float(result.pred_score)
                
            if isinstance(result.pred_label, torch.Tensor):
                is_anomalous = bool(result.pred_label.cpu().item())
            else:
                is_anomalous = bool(result.pred_label)
            
            # Get anomaly mask if available
            anomaly_mask = None
            if hasattr(result, 'pred_mask') and result.pred_mask is not None:
                if isinstance(result.pred_mask, torch.Tensor):
                    anomaly_mask = result.pred_mask.cpu().numpy()
                else:
                    anomaly_mask = result.pred_mask
                
                if len(anomaly_mask.shape) > 2:
                    anomaly_mask = anomaly_mask[0]
            
            base_result = {
                'is_anomalous': is_anomalous,
                'anomaly_score': anomaly_score,
                'anomaly_mask': anomaly_mask,
                'threshold_used': ANOMALY_THRESHOLD,
                'decision': 'DEFECT' if (is_anomalous and anomaly_score > ANOMALY_THRESHOLD) else 'GOOD'
            }
            
            # OpenAI Layer 1 Analysis - ALWAYS RUN FOR TESTING
            if self.openai_enabled:
                print(f"Running OpenAI anomaly analysis (score: {anomaly_score:.3f})")
                openai_analysis = self._analyze_anomaly_with_openai(image_path, base_result)
                base_result['openai_analysis'] = openai_analysis
            
            return base_result
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return None
    
    def classify_defects(self, image_path, region_mask=None):
        """
        Layer 2: Defect classification with OpenAI analysis
        """
        if not self.hrnet_model:
            raise ValueError("HRNet model not loaded")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image_rgb.shape[:2]
            
            # Apply region mask if provided
            if region_mask is not None:
                region_mask = cv2.resize(region_mask, (original_size[1], original_size[0]))
                masked_image = image_rgb.copy()
                masked_image[region_mask < 0.5] = 0
                image_rgb = masked_image
            
            # Preprocess for HRNet
            transformed = self.hrnet_transform(image=image_rgb)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # HRNet inference
            with torch.no_grad():
                output = self.hrnet_model(input_tensor)
                predictions = torch.softmax(output, dim=1)
                predicted_mask = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()
                confidence_scores = torch.max(predictions, dim=1)[0].squeeze().cpu().numpy()
            
            # Resize predictions to original size
            predicted_mask = cv2.resize(predicted_mask.astype(np.uint8), 
                                      (original_size[1], original_size[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            confidence_scores = cv2.resize(confidence_scores, 
                                         (original_size[1], original_size[0]), 
                                         interpolation=cv2.INTER_LINEAR)
            
            # Analyze defect predictions
            defect_analysis = self._analyze_defect_predictions(predicted_mask, confidence_scores)
            
            base_result = {
                'predicted_mask': predicted_mask,
                'confidence_scores': confidence_scores,
                'defect_analysis': defect_analysis,
                'detected_defects': defect_analysis['detected_defects'],
                'bounding_boxes': defect_analysis['bounding_boxes'],
                'class_distribution': defect_analysis['class_distribution']
            }
            
            # OpenAI Layer 2 Analysis - ALWAYS RUN IF ENABLED
            if self.openai_enabled:
                print(f"Running OpenAI defect analysis")
                openai_analysis = self._analyze_defects_with_openai(image_path, base_result)
                base_result['openai_analysis'] = openai_analysis
            
            return base_result
            
        except Exception as e:
            print(f"Error in defect classification: {e}")
            return None
    
    def _analyze_anomaly_with_openai(self, image_path, anomaly_result):
        """OpenAI analysis for Layer 1 (Anomaly Detection) - 1.x compatible"""
        try:
            if not self.openai_client:
                return {
                    'analysis': 'OpenAI client not initialized',
                    'confidence_percentage': 0,
                    'error': 'No OpenAI client'
                }
            
            # Encode image to base64
            image_base64 = self._encode_image_to_base64(image_path)
            
            prompt = f"""Analyze this product image for quality control.
            
Model Results:
- Anomaly Score: {anomaly_result['anomaly_score']:.3f}
- Decision: {anomaly_result['decision']}
- Threshold: {anomaly_result['threshold_used']}

Provide analysis on:
1. Visual quality assessment
2. Confidence in model decision (percentage)
3. Key observations
4. Recommendation

Be concise and technical."""

            print("Calling OpenAI API...")
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE
            )
            
            analysis_text = response.choices[0].message.content
            confidence = self._extract_confidence_percentage(analysis_text)
            
            print(f"OpenAI analysis completed - confidence: {confidence}%")
            
            return {
                'analysis': analysis_text,
                'confidence_percentage': confidence,
                'model_used': OPENAI_MODEL,
                'layer': 'anomaly_detection'
            }
            
        except Exception as e:
            print(f"OpenAI anomaly analysis error: {e}")
            return {
                'analysis': f'OpenAI analysis failed: {str(e)}',
                'confidence_percentage': 0,
                'error': str(e)
            }
    
    def _analyze_defects_with_openai(self, image_path, defect_result):
        """OpenAI analysis for Layer 2 (Defect Classification) with bounding box validation - 1.x compatible"""
        try:
            if not self.openai_client:
                return {
                    'analysis': 'OpenAI client not initialized',
                    'confidence_percentage': 0,
                    'error': 'No OpenAI client'
                }
            
            image_base64 = self._encode_image_to_base64(image_path)
            
            detected_defects = defect_result['detected_defects']
            bounding_boxes = defect_result.get('bounding_boxes', {})
            
            # Create visual prompt with bounding box information
            bbox_info = ""
            for defect_type, boxes in bounding_boxes.items():
                bbox_info += f"\n{defect_type}: {len(boxes)} regions detected"
                for i, bbox in enumerate(boxes[:3]):  # Limit to first 3 boxes per type
                    bbox_info += f"\n  - Region {i+1}: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}"
            
            prompt = f"""Analyze this product image with detected defects and their bounding box locations.

DETECTED DEFECTS: {', '.join(detected_defects) if detected_defects else 'None detected'}
BOUNDING BOX LOCATIONS:{bbox_info if bbox_info else ' None'}

DEFECT CLASSES: {list(SPECIFIC_DEFECT_CLASSES.values())}

Provide analysis on:
1. Visual quality assessment
2. Defect severity if any (Minor/Moderate/Significant/Critical)
3. Confidence in detection accuracy (percentage)
4. Spatial accuracy of bounding boxes if present (percentage)
5. Overall product quality recommendation

Be specific and technical."""

            print("Calling OpenAI API for defect analysis...")
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE
            )
            
            analysis_text = response.choices[0].message.content
            confidence = self._extract_confidence_percentage(analysis_text)
            bbox_confidence = self._extract_bbox_confidence(analysis_text)
            
            print(f"OpenAI defect analysis completed - confidence: {confidence}%")
            
            return {
                'analysis': analysis_text,
                'confidence_percentage': confidence,
                'bbox_validation': {
                    'confidence': bbox_confidence,
                    'validated_regions': len(bounding_boxes),
                    'spatial_accuracy': 'high' if bbox_confidence > 80 else 'medium' if bbox_confidence > 60 else 'low'
                },
                'model_used': OPENAI_MODEL,
                'layer': 'defect_classification',
                'defects_analyzed': detected_defects,
                'bounding_boxes_analyzed': {k: len(v) for k, v in bounding_boxes.items()}
            }
            
        except Exception as e:
            print(f"OpenAI defect analysis error: {e}")
            return {
                'analysis': f'OpenAI analysis failed: {str(e)}',
                'confidence_percentage': 0,
                'bbox_validation': {'confidence': 0, 'error': str(e)},
                'error': str(e)
            }
    
    def _encode_image_to_base64(self, image_path):
        """Encode image to base64 for OpenAI"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _extract_confidence_percentage(self, text):
        """Extract general confidence percentage from OpenAI response"""
        import re
        matches = re.findall(r'(\d+)%', text)
        if matches:
            return max([int(match) for match in matches])
        return 75  # Default confidence
    
    def _extract_bbox_confidence(self, text):
        """Extract bounding box confidence from OpenAI response"""
        import re
        
        # Look for bounding box specific confidence patterns
        bbox_patterns = [
            r'bounding box.*?(\d+)%',
            r'spatial.*?(\d+)%',
            r'location.*?(\d+)%',
            r'bbox.*?(\d+)%'
        ]
        
        confidences = []
        for pattern in bbox_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            confidences.extend([int(match) for match in matches])
        
        if confidences:
            return max(confidences)
        
        # Fallback to general confidence if no bbox-specific found
        general_matches = re.findall(r'(\d+)%', text)
        if general_matches:
            return max([int(match) for match in general_matches])
        
        return 75  # Default confidence
    
    def _analyze_defect_predictions(self, predicted_mask, confidence_scores):
        """Analyze HRNet predictions to extract defect information"""
        h, w = predicted_mask.shape
        total_pixels = h * w
        
        analysis = {
            'detected_defects': [],
            'class_distribution': {},
            'bounding_boxes': {},
            'defect_statistics': {}
        }
        
        # Analyze each defect class
        for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
            class_mask = (predicted_mask == class_id)
            pixel_count = np.sum(class_mask)
            percentage = (pixel_count / total_pixels) * 100
            
            analysis['class_distribution'][class_name] = {
                'pixel_count': int(pixel_count),
                'percentage': percentage,
                'class_id': class_id
            }
            
            # Only process actual defects (not background)
            if class_id > 0 and pixel_count > 0:
                # Apply confidence threshold
                confident_mask = class_mask & (confidence_scores > DEFECT_CONFIDENCE_THRESHOLD)
                confident_pixels = np.sum(confident_mask)
                
                # Detection criteria
                min_pixels = max(MIN_DEFECT_PIXELS, total_pixels * MIN_DEFECT_PERCENTAGE)
                
                if confident_pixels > min_pixels or pixel_count > total_pixels * 0.1:
                    analysis['detected_defects'].append(class_name)
                    
                    # Extract bounding boxes
                    bboxes = self._extract_bounding_boxes(
                        confident_mask if confident_pixels > min_pixels else class_mask
                    )
                    analysis['bounding_boxes'][class_name] = bboxes
                    
                    # Calculate statistics
                    analysis['defect_statistics'][class_name] = {
                        'confident_pixels': int(confident_pixels if confident_pixels > 0 else pixel_count),
                        'confidence_ratio': confident_pixels / pixel_count if pixel_count > 0 else 0,
                        'avg_confidence': float(np.mean(confidence_scores[
                            confident_mask if confident_pixels > 0 else class_mask
                        ])),
                        'max_confidence': float(np.max(confidence_scores[
                            confident_mask if confident_pixels > 0 else class_mask
                        ])),
                        'num_regions': len(bboxes)
                    }
        
        return analysis
    
    def _extract_bounding_boxes(self, mask):
        """Extract bounding boxes from binary mask"""
        try:
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bounding_boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= MIN_BBOX_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append({
                        'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                        'area': int(area),
                        'center_x': int(x + w/2), 'center_y': int(y + h/2)
                    })
            
            return bounding_boxes
        except:
            return []
