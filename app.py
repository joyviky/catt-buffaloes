import streamlit as st
import os
import json
from PIL import Image
import sys
sys.path.append('.')

from src.breed_classifier import BreedClassifier

# Page configuration
st.set_page_config(
    page_title="Cattle & Buffalo Breed Recognition",
    page_icon="🐄",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model (cached)"""
    classifier = BreedClassifier()
    if classifier.load_model():
        return classifier
    return None

@st.cache_data
def load_breed_info():
    """Load breed information (cached)"""
    try:
        with open('data/breeds_info.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def main():
    st.title("🐄 Cattle & Buffalo Breed Recognition System")
    st.write("Upload an image to identify the breed of cattle or buffalo")
    
    # Load model and breed info
    classifier = load_model()
    breed_info = load_breed_info()
    
    if classifier is None:
        st.error("❌ Model not found! Please train the model first by running: `python train_model.py`")
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("📋 Supported Breeds")
        
        if breed_info:
            cattle_breeds = [k for k, v in breed_info.items() if v.get('type') == 'cattle']
            buffalo_breeds = [k for k, v in breed_info.items() if v.get('type') == 'buffalo']
            
            if cattle_breeds:
                st.subheader("🐄 Cattle Breeds")
                for breed in cattle_breeds:
                    st.write(f"• {breed.title()}")
            
            if buffalo_breeds:
                st.subheader("🐃 Buffalo Breeds") 
                for breed in buffalo_breeds:
                    st.write(f"• {breed.title()}")
        
        st.subheader("📸 Image Guidelines")
        st.write("For best results:")
        st.write("• Clear, front-facing images")
        st.write("• Good lighting")
        st.write("• Animal should be main subject")
        st.write("• JPG, JPEG, or PNG format")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a cattle or buffalo"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Prediction button
            if st.button("🔍 Identify Breed", type="primary"):
                with st.spinner('Analyzing image...'):
                    try:
                        # Make prediction
                        breed, confidence = classifier.predict(temp_path)
                        
                        if breed is not None:
                            # Display results
                            st.success(f"**Predicted Breed: {breed.title()}**")
                            
                            # Confidence indicator
                            if confidence > 0.8:
                                st.success(f"**Confidence: {confidence:.1%}** 🟢 High")
                            elif confidence > 0.6:
                                st.warning(f"**Confidence: {confidence:.1%}** 🟡 Medium")
                            else:
                                st.error(f"**Confidence: {confidence:.1%}** 🔴 Low")
                            
                        else:
                            st.error("❌ Could not process the image. Please try with a different image.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    with col2:
        if uploaded_file is not None and 'breed' in locals() and breed is not None:
            # Display breed information
            st.subheader("📝 Breed Information")
            
            if breed in breed_info:
                info = breed_info[breed]
                
                st.write(f"**Type:** {info.get('type', 'Unknown').title()}")
                
                if 'characteristics' in info:
                    st.write("**Key Characteristics:**")
                    for char in info['characteristics']:
                        st.write(f"• {char}")
                
                if 'milk_yield' in info:
                    st.write(f"**Milk Yield:** {info['milk_yield']}")
                
                if 'origin' in info:
                    st.write(f"**Origin:** {info['origin']}")
            else:
                st.write("No additional information available for this breed.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ using Streamlit and TensorFlow | "
        "Upload clear images for best results"
    )

if __name__ == "__main__":
    main()
