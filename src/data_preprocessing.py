import os
import cv2
import numpy as np
import pandas as pd
from src.feature_extraction import FeatureExtractor
from src.utils import Utils

class DataPreprocessor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.utils = Utils()
    
    def scan_data_directory(self, data_dir="data/raw_images"):
        """Scan directory and collect image paths with labels"""
        image_paths = []
        labels = []
        
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} does not exist!")
            return [], []
        
        for breed_folder in os.listdir(data_dir):
            breed_path = os.path.join(data_dir, breed_folder)
            
            if os.path.isdir(breed_path):
                print(f"Processing breed: {breed_folder}")
                breed_images = []
                
                for img_file in os.listdir(breed_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(breed_path, img_file)
                        
                        # Validate image
                        if self.utils.validate_image(img_path):
                            image_paths.append(img_path)
                            labels.append(breed_folder)
                            breed_images.append(img_file)
                
                print(f"Found {len(breed_images)} valid images for {breed_folder}")
        
        print(f"Total images found: {len(image_paths)}")
        return image_paths, labels
    
    def extract_features_batch(self, image_paths, labels, save_path="data/processed"):
        """Extract features from all images and save"""
        os.makedirs(save_path, exist_ok=True)
        
        deep_features_list = []
        color_features_list = []
        shape_features_list = []
        valid_labels = []
        valid_paths = []
        
        print("Extracting features from all images...")
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            deep_feat, color_feat, shape_feat = self.feature_extractor.extract_all_features(img_path)
            
            if deep_feat is not None:
                deep_features_list.append(deep_feat)
                color_features_list.append(color_feat) 
                shape_features_list.append(shape_feat)
                valid_labels.append(label)
                valid_paths.append(img_path)
        
        # Convert to numpy arrays
        deep_features = np.array(deep_features_list)
        color_features = np.array(color_features_list)
        shape_features = np.array(shape_features_list)
        
        # Save features
        np.save(os.path.join(save_path, 'deep_features.npy'), deep_features)
        np.save(os.path.join(save_path, 'color_features.npy'), color_features)
        np.save(os.path.join(save_path, 'shape_features.npy'), shape_features)
        np.save(os.path.join(save_path, 'labels.npy'), np.array(valid_labels))
        
        # Save paths for reference
        with open(os.path.join(save_path, 'image_paths.txt'), 'w') as f:
            for path in valid_paths:
                f.write(f"{path}\n")
        
        print(f"Features extracted and saved for {len(valid_labels)} images")
        print(f"Deep features shape: {deep_features.shape}")
        print(f"Color features shape: {color_features.shape}")
        print(f"Shape features shape: {shape_features.shape}")
        
        return deep_features, color_features, shape_features, np.array(valid_labels)
    
    def load_processed_features(self, load_path="data/processed"):
        """Load previously processed features"""
        try:
            deep_features = np.load(os.path.join(load_path, 'deep_features.npy'))
            color_features = np.load(os.path.join(load_path, 'color_features.npy'))
            shape_features = np.load(os.path.join(load_path, 'shape_features.npy'))
            labels = np.load(os.path.join(load_path, 'labels.npy'))
            
            print("Loaded processed features successfully")
            print(f"Deep features shape: {deep_features.shape}")
            print(f"Color features shape: {color_features.shape}") 
            print(f"Shape features shape: {shape_features.shape}")
            print(f"Number of labels: {len(labels)}")
            
            return deep_features, color_features, shape_features, labels
        except Exception as e:
            print(f"Error loading processed features: {e}")
            return None, None, None, None
