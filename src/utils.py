import os
import json
import cv2
import numpy as np
from PIL import Image

class Utils:
    @staticmethod
    def create_breed_info():
        """Create breed information database"""
        breed_info = {
            # Buffalo breeds
            "murrah": {
                "type": "buffalo",
                "characteristics": ["jet black color", "tightly curved horns", "compact body"],
                "milk_yield": "1500-2500 kg/lactation",
                "origin": "Punjab, Haryana"
            },
            "jaffarabadi": {
                "type": "buffalo", 
                "characteristics": ["drooping horns", "heaviest Indian breed", "black coat"],
                "milk_yield": "1000-1200 kg/lactation",
                "origin": "Gujarat"
            },
            "surti": {
                "type": "buffalo",
                "characteristics": ["medium size", "curved horns", "fawn colored"],
                "milk_yield": "900-1200 kg/lactation", 
                "origin": "Gujarat"
            },
            # Cattle breeds
            "gir": {
                "type": "cattle",
                "characteristics": ["white with dark red patches", "lyre-shaped horns"],
                "milk_yield": "1200-1800 kg/lactation",
                "origin": "Gujarat"
            },
            "sahiwal": {
                "type": "cattle", 
                "characteristics": ["reddish dun to red color", "loose skin"],
                "milk_yield": "1400-2500 kg/lactation",
                "origin": "Punjab, Pakistan"
            },
            "red_sindhi": {
                "type": "cattle",
                "characteristics": ["red color", "compact body", "small horns"],
                "milk_yield": "1100-1500 kg/lactation",
                "origin": "Sindh"
            }
        }
        
        os.makedirs("data", exist_ok=True)
        with open("data/breeds_info.json", "w") as f:
            json.dump(breed_info, f, indent=4)
        
        return breed_info
    
    @staticmethod
    def setup_data_folders(breed_info):
        """Create folders for each breed"""
        base_path = "data/raw_images"
        os.makedirs(base_path, exist_ok=True)
        
        for breed_name in breed_info.keys():
            breed_path = os.path.join(base_path, breed_name)
            os.makedirs(breed_path, exist_ok=True)
            print(f"Created folder: {breed_path}")
    
    @staticmethod
    def validate_image(image_path):
        """Validate if image is readable"""
        try:
            img = cv2.imread(image_path)
            return img is not None
        except:
            return False
    
    @staticmethod
    def resize_image(image_path, target_size=(224, 224)):
        """Resize image to target size"""
        img = cv2.imread(image_path)
        if img is not None:
            resized = cv2.resize(img, target_size)
            return resized
        return None
