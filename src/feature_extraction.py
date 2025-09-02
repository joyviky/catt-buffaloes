import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

class FeatureExtractor:
    def __init__(self):
        print("Loading ResNet50 model...")
        self.feature_extractor = ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        print("Model loaded successfully!")
    
    def preprocess_image(self, img_path):
        """Load and preprocess image"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            print(f"Error preprocessing image {img_path}: {e}")
            return None
    
    def extract_deep_features(self, img):
        """Extract deep features using ResNet50"""
        try:
            # Prepare image for ResNet50
            img_array = np.expand_dims(img, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = self.feature_extractor.predict(img_array, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Error extracting deep features: {e}")
            return np.zeros(2048)  # ResNet50 output size
    
    def extract_color_histogram(self, img):
        """Extract color histogram features"""
        try:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # Calculate histograms for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256]) 
            hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
            
            # Normalize and combine
            color_features = np.concatenate([
                hist_h.flatten(),
                hist_s.flatten(), 
                hist_v.flatten()
            ])
            
            # Normalize to unit vector
            norm = np.linalg.norm(color_features)
            if norm > 0:
                color_features = color_features / norm
            
            return color_features
        except Exception as e:
            print(f"Error extracting color features: {e}")
            return np.zeros(150)  # 50*3 histogram bins
    
    def extract_shape_features(self, img):
        """Extract shape-based features"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate shape features
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0
                extent = area / (w * h) if (w * h) > 0 else 0
                
                return np.array([area, perimeter, circularity, aspect_ratio, extent])
            else:
                return np.zeros(5)
                
        except Exception as e:
            print(f"Error extracting shape features: {e}")
            return np.zeros(5)
    
    def extract_all_features(self, img_path):
        """Extract all types of features from an image"""
        img = self.preprocess_image(img_path)
        if img is None:
            return None, None, None
        
        deep_features = self.extract_deep_features(img)
        color_features = self.extract_color_histogram(img)
        shape_features = self.extract_shape_features(img)
        
        return deep_features, color_features, shape_features
