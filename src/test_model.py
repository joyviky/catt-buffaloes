import os
import sys
sys.path.append('.')

from src.breed_classifier import BreedClassifier
from src.feature_extraction import FeatureExtractor

def test_single_image():
    print("=== Testing Single Image ===\n")
    
    # Load trained model
    classifier = BreedClassifier()
    if not classifier.load_model():
        print("❌ Could not load trained model!")
        print("Please run: python train_model.py first")
        return
    
    # Get test image path
    test_image = input("Enter path to test image: ").strip()
    
    if not os.path.exists(test_image):
        print(f"❌ Image not found: {test_image}")
        return
    
    print(f"Testing image: {test_image}")
    print("Analyzing...")
    
    # Make prediction
    breed, confidence = classifier.predict(test_image)
    
    if breed is None:
        print("❌ Could not process image")
        return
    
    print(f"\n✓ Prediction: {breed.title()}")
    print(f"✓ Confidence: {confidence:.2%}")
    
    if confidence > 0.8:
        print("🟢 High confidence - Very reliable prediction")
    elif confidence > 0.6:
        print("🟡 Medium confidence - Fairly reliable prediction")
    else:
        print("🔴 Low confidence - Uncertain prediction")

def test_batch():
    print("=== Testing Batch of Images ===\n")
    
    # Load model
    classifier = BreedClassifier()
    if not classifier.load_model():
        print("❌ Could not load trained model!")
        return
    
    test_dir = input("Enter directory path with test images: ").strip()
    
    if not os.path.exists(test_dir):
        print(f"❌ Directory not found: {test_dir}")
        return
    
    # Find all images in directory
    image_files = []
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(test_dir, file))
    
    if not image_files:
        print("❌ No images found in directory")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    results = []
    for img_path in image_files:
        print(f"Testing: {os.path.basename(img_path)}")
        breed, confidence = classifier.predict(img_path)
        
        if breed is not None:
            results.append({
                'image': os.path.basename(img_path),
                'breed': breed,
                'confidence': confidence
            })
            print(f"  → {breed} ({confidence:.2%})")
        else:
            print(f"  → Failed to process")
        print()
    
    # Summary
    print("=== Batch Results Summary ===")
    for result in results:
        status = "🟢" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.6 else "🔴"
        print(f"{status} {result['image']}: {result['breed']} ({result['confidence']:.2%})")

def main():
    print("Choose testing mode:")
    print("1. Test single image")
    print("2. Test batch of images")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_single_image()
    elif choice == "2":
        test_batch()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
