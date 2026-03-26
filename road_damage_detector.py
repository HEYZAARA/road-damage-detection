# road_damage_detector.py - AI Road Damage Detection
import cv2
import numpy as np

class RoadDamageDetector:
    def __init__(self):
        self.damage_types = {
            'pothole': {'color': (0, 0, 255), 'severity': 'High', 'icon': '🕳️'},
            'crack': {'color': (0, 255, 255), 'severity': 'Medium', 'icon': '⚡'},
            'rutting': {'color': (255, 165, 0), 'severity': 'Medium', 'icon': '〰️'}
        }
        print("✅ Road Damage Detection System Ready!")
    
    def detect_pothole(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potholes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(area / 10000, 0.95)
                potholes.append({
                    'type': 'pothole',
                    'bbox': (x, y, x+w, y+h),
                    'confidence': confidence,
                    'severity': 'High' if area > 10000 else 'Medium' if area > 5000 else 'Low'
                })
        return potholes
    
    def detect_cracks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        
        cracks = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 50:
                    cracks.append({
                        'type': 'crack',
                        'bbox': (x1, y1, x2, y2),
                        'confidence': min(length / 500, 0.9),
                        'severity': 'High' if length > 300 else 'Medium' if length > 150 else 'Low'
                    })
        return cracks
    
    def detect_all_damages(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        original = image.copy()
        all_damages = []
        
        potholes = self.detect_pothole(image)
        cracks = self.detect_cracks(image)
        all_damages.extend(potholes)
        all_damages.extend(cracks)
        
        annotated = original.copy()
        for damage in all_damages:
            d_type = damage['type']
            bbox = damage['bbox']
            conf = damage['confidence']
            color = self.damage_types[d_type]['color']
            
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{d_type.upper()} {conf:.0%}"
            cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return all_damages, annotated
