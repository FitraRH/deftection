# utils/stateless_visualization.py
"""
Stateless visualization utilities for generating annotated images
Based on original code but simplified for stateless operation
"""

import cv2
import numpy as np
import base64
import logging
from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES

logger = logging.getLogger(__name__)

def create_annotated_image_base64(image_path, result):
    """Create base64 encoded annotated image from detection result"""
    try:
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image for annotation: {image_path}")
            return None
        
        # Create annotated version
        annotated_image = annotate_detection_result(image, result)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error creating annotated image: {e}")
        return None

def annotate_detection_result(image, result):
    """Add annotations to image based on detection results - original style"""
    try:
        annotated_frame = image.copy()
        height = annotated_frame.shape[0]
        
        if result:
            if result['final_decision'] == 'GOOD':
                # Add green border for good products (from original video processor)
                annotated_frame = cv2.copyMakeBorder(annotated_frame, 5, 5, 5, 5, 
                                                   cv2.BORDER_CONSTANT, value=(0, 255, 0))
                cv2.putText(annotated_frame, "GOOD", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                           
            elif result['final_decision'] == 'DEFECT':
                # Add red border for defective products (from original video processor)
                annotated_frame = cv2.copyMakeBorder(annotated_frame, 10, 10, 10, 10, 
                                                   cv2.BORDER_CONSTANT, value=(0, 0, 255))
                cv2.putText(annotated_frame, "DEFECT", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Draw defect bounding boxes if available (from original video processor)
                if result.get('defect_classification') and 'bounding_boxes' in result['defect_classification']:
                    draw_defect_bounding_boxes(annotated_frame, result['defect_classification']['bounding_boxes'])
                
                # Also check for enhanced detection bounding boxes
                elif (result.get('defect_classification') and 
                      result['defect_classification'].get('defect_analysis') and
                      'bounding_boxes' in result['defect_classification']['defect_analysis']):
                    draw_defect_bounding_boxes(
                        annotated_frame, 
                        result['defect_classification']['defect_analysis']['bounding_boxes']
                    )
            
            # Add processing info (from original video processor)
            cv2.putText(annotated_frame, f"Score: {result['anomaly_detection']['anomaly_score']:.3f}", 
                       (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
        
    except Exception as e:
        logger.error(f"Error annotating detection result: {e}")
        return image

def draw_defect_bounding_boxes(image, bounding_boxes):
    """Draw bounding boxes for defects - using original style with config colors"""
    try:
        for defect_type, bboxes in bounding_boxes.items():
            # Get color from config based on defect type
            defect_class_id = None
            for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
                if class_name == defect_type:
                    defect_class_id = class_id
                    break
            
            # Use config color if available, otherwise use default
            if defect_class_id is not None and defect_class_id in DEFECT_COLORS:
                color = DEFECT_COLORS[defect_class_id]
            else:
                # Fallback colors similar to original
                default_colors = {
                    'scratch': (0, 255, 255),     # Cyan
                    'stained': (128, 0, 128),     # Purple  
                    'damaged': (255, 0, 0),       # Red
                    'missing_component': (255, 255, 0),  # Yellow
                    'open': (255, 0, 255),        # Magenta
                }
                color = default_colors.get(defect_type, (255, 255, 255))  # White default
            
            for bbox in bboxes:
                x, y = bbox['x'], bbox['y']
                w, h = bbox['width'], bbox['height']
                
                # Draw rectangle (from original video processor)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Add defect type label (from original video processor)
                cv2.putText(image, defect_type.upper(), (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
    except Exception as e:
        logger.error(f"Error drawing defect bounding boxes: {e}")

def create_enhanced_annotation(image, result):
    """Create enhanced annotation similar to original enhanced_detection"""
    try:
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Add decision-based border (from original enhanced detection)
        border_width = 8
        if result['final_decision'] == 'DEFECT':
            # Red gradient border for defects (simplified from original)
            for i in range(border_width):
                alpha = 1.0 - (i / border_width) * 0.7
                color = np.array([255, 0, 0]) * alpha
                cv2.rectangle(annotated, (i, i), (w-i-1, h-i-1), color.astype(int), 2)
        else:
            # Green gradient border for good products (simplified from original)
            for i in range(border_width):
                alpha = 1.0 - (i / border_width) * 0.7
                color = np.array([0, 255, 0]) * alpha
                cv2.rectangle(annotated, (i, i), (w-i-1, h-i-1), color.astype(int), 2)
        
        # Add comprehensive overlay information (from original enhanced detection)
        overlay_text = f"FINAL DECISION: {result['final_decision']}\n"
        overlay_text += f"Processing Time: {result['processing_time']:.3f}s\n"
        overlay_text += f"Anomaly Score: {result['anomaly_detection']['anomaly_score']:.3f}\n"
        
        if result.get('detected_defect_types'):
            overlay_text += f"Defects Found: {len(result['detected_defect_types'])}\n"
            overlay_text += f"Types: {', '.join(result['detected_defect_types'][:3])}"
            if len(result['detected_defect_types']) > 3:
                overlay_text += f" +{len(result['detected_defect_types'])-3} more"
        
        # Styled text box (from original enhanced detection)
        bbox_color = (0, 0, 255) if result['final_decision'] == 'DEFECT' else (0, 255, 0)
        
        # Add text with background
        lines = overlay_text.split('\n')
        y_offset = 30
        for line in lines:
            if line.strip():
                # Add background rectangle for text
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated, (10, y_offset - 20), 
                             (15 + text_size[0], y_offset + 5), bbox_color, -1)
                cv2.putText(annotated, line, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
        
        return annotated
        
    except Exception as e:
        logger.error(f"Error creating enhanced annotation: {e}")
        return image