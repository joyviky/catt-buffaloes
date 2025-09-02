import os
import sys
sys.path.append('.')

from src.utils import Utils
from src.data_preprocessing import DataPreprocessor
from src.breed_classifier import BreedClassifier

def main():
    print("=== Cattle & Buffalo Breed Recognition Training ===\n")
    
    # Step 1: Setup breed information and folders
    print("Step 1: Setting up breed information...")
    utils = Utils()
    breed_info = utils.create_breed_info()
    utils.setup_data_folders(breed_info)
    print("✓ Breed information and folders created\n")
    
    # Step 2: Check if we have images
    data_dir = "data/raw_images"
    total_images = 0
    for breed in os.listdir(data_dir):
        breed_path = os.path.join(data_dir, breed)
        if os.path.isdir(breed_path):
            count = len([f for f in os.listdir(breed_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total_images += count
            print(f"{breed}: {count} images")
    
    if total_images == 0:
        print("\n❌ No images found!")
        print("Please add images to the breed folders in data/raw_images/")
        print("Folder structure should be:")
        print("data/raw_images/")
        for breed in breed_info.keys():
            print(f"  ├── {breed}/")
            print(f"  │   ├── image1.jpg")
            print(f"  │   ├── image2.jpg")
            print(f"  │   └── ...")
        return
    
    print(f"\nTotal images found: {total_images}")
    
    if total_images < 30:
        print("⚠️  Warning: Very few images found. For good results, you need:")
        print("- At least 20-30 images per breed")
        print("- Images should be clear and show the animal clearly")
        print("- Different angles and lighting conditions")
    
    # Step 3: Data preprocessing
    print("\nStep 2: Data preprocessing...")
    preprocessor = DataPreprocessor()
    
    # Check if features already exist
    if os.path.exists("data/processed/deep_features.npy"):
        print("Found existing processed features, loading...")
        deep_features, color_features, shape_features, labels = preprocessor.load_processed_features()
    else:
        print("Processing images and extracting features...")
        image_paths, labels = preprocessor.scan_data_directory()
        
        if len(image_paths) == 0:
            print("❌ No valid images found!")
            return
        
        deep_features, color_features, shape_features, labels = preprocessor.extract_features_batch(image_paths, labels)
    
    if deep_features is None:
        print("❌ Feature extraction failed!")
        return
    
    print("✓ Features extracted successfully\n")
    
    # Step 4: Train classifier
    print("Step 3: Training classifier...")
    classifier = BreedClassifier()
    accuracy = classifier.train(deep_features, color_features, shape_features, labels)
    
    # Step 5: Save model
    print("\nStep 4: Saving model...")
    classifier.save_model()
    print("✓ Model saved successfully\n")
    
    print("=== Training Completed ===")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("\nNext steps:")
    print("1. Run: streamlit run app.py")
    print("2. Or test with: python test_model.py")

if __name__ == "__main__":
    main()
