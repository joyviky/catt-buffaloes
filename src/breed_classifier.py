import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.feature_extraction import FeatureExtractor

class BreedClassifier:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        
        # Scalers for different feature types
        self.deep_scaler = StandardScaler()
        self.color_scaler = StandardScaler()
        self.shape_scaler = StandardScaler()
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        # Individual classifiers
        self.deep_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.color_classifier = SVC(probability=True, random_state=42)
        self.shape_classifier = LogisticRegression(random_state=42)
        
        # Ensemble classifier
        self.ensemble_classifier = VotingClassifier(
            estimators=[
                ('deep', self.deep_classifier),
                ('color', self.color_classifier),
                ('shape', self.shape_classifier)
            ],
            voting='soft'
        )
        
        self.is_trained = False
    
    def train(self, deep_features, color_features, shape_features, labels):
        """Train the multi-modal classifier"""
        print("Training breed classifier...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, 
            stratify=y_encoded
        )
        
        # Prepare training data
        deep_train = deep_features[train_idx]
        color_train = color_features[train_idx]
        shape_train = shape_features[train_idx]
        y_train = y_encoded[train_idx]
        
        # Prepare testing data
        deep_test = deep_features[test_idx]
        color_test = color_features[test_idx]
        shape_test = shape_features[test_idx]
        y_test = y_encoded[test_idx]
        
        # Scale features
        deep_train_scaled = self.deep_scaler.fit_transform(deep_train)
        color_train_scaled = self.color_scaler.fit_transform(color_train)
        shape_train_scaled = self.shape_scaler.fit_transform(shape_train)
        
        deep_test_scaled = self.deep_scaler.transform(deep_test)
        color_test_scaled = self.color_scaler.transform(color_test)
        shape_test_scaled = self.shape_scaler.transform(shape_test)
        
        # Train individual classifiers
        print("Training deep features classifier...")
        self.deep_classifier.fit(deep_train_scaled, y_train)
        
        print("Training color features classifier...")
        self.color_classifier.fit(color_train_scaled, y_train)
        
        print("Training shape features classifier...")
        self.shape_classifier.fit(shape_train_scaled, y_train)
        
        # Train ensemble classifier
        print("Training ensemble classifier...")
        combined_train = np.hstack([deep_train_scaled, color_train_scaled, shape_train_scaled])
        combined_test = np.hstack([deep_test_scaled, color_test_scaled, shape_test_scaled])
        
        self.ensemble_classifier.fit(combined_train, y_train)
        
        # Evaluate
        ensemble_pred = self.ensemble_classifier.predict(combined_test)
        accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"\nTraining completed!")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Available breeds: {list(self.label_encoder.classes_)}")
        
        # Detailed classification report
        target_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, ensemble_pred, target_names=target_names))
        
        self.is_trained = True
        return accuracy
    
    def predict(self, image_path):
        """Predict breed for a single image"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        deep_feat, color_feat, shape_feat = self.feature_extractor.extract_all_features(image_path)
        
        if deep_feat is None:
            return None, 0.0
        
        # Reshape for prediction
        deep_feat = deep_feat.reshape(1, -1)
        color_feat = color_feat.reshape(1, -1)
        shape_feat = shape_feat.reshape(1, -1)
        
        # Scale features
        deep_scaled = self.deep_scaler.transform(deep_feat)
        color_scaled = self.color_scaler.transform(color_feat)
        shape_scaled = self.shape_scaler.transform(shape_feat)
        
        # Combined features for ensemble
        combined = np.hstack([deep_scaled, color_scaled, shape_scaled])
        
        # Get prediction and confidence
        prediction_encoded = self.ensemble_classifier.predict(combined)[0]
        probabilities = self.ensemble_classifier.predict_proba(combined)[0]
        confidence = np.max(probabilities)
        
        # Decode prediction
        breed_name = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        return breed_name, confidence
    
    def save_model(self, save_path="models/breed_classifier.pkl"):
        """Save the trained model"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'deep_scaler': self.deep_scaler,
            'color_scaler': self.color_scaler,
            'shape_scaler': self.shape_scaler,
            'label_encoder': self.label_encoder,
            'deep_classifier': self.deep_classifier,
            'color_classifier': self.color_classifier,
            'shape_classifier': self.shape_classifier,
            'ensemble_classifier': self.ensemble_classifier,
            'is_trained': self.is_trained
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path="models/breed_classifier.pkl"):
        """Load a trained model"""
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.deep_scaler = model_data['deep_scaler']
            self.color_scaler = model_data['color_scaler']
            self.shape_scaler = model_data['shape_scaler']
            self.label_encoder = model_data['label_encoder']
            self.deep_classifier = model_data['deep_classifier']
            self.color_classifier = model_data['color_classifier']
            self.shape_classifier = model_data['shape_classifier']
            self.ensemble_classifier = model_data['ensemble_classifier']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {load_path}")
            print(f"Available breeds: {list(self.label_encoder.classes_)}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
