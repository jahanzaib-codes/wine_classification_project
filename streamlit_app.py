"""
Wine Classification Streamlit Application
Task E: Model Deployment using Streamlit (5 Marks)
Author: Jahanzaib Channa
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="🍷 Wine Classification",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .stCard {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Prediction result styling */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .prediction-box h2 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }
    
    .prediction-box .class-name {
        color: #ffd700;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(78, 205, 196, 0.1);
        border-left: 4px solid #4ecdc4;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Input sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Wine class colors */
    .class-0 { color: #FF6B6B; }
    .class-1 { color: #4ECDC4; }
    .class-2 { color: #45B7D1; }
    
    /* Metric styling */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        flex: 1;
        margin: 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Wine class information
WINE_CLASSES = {
    0: {
        "name": "Class 0",
        "description": "Wine variety with high alcohol content and rich color",
        "color": "#FF6B6B",
        "emoji": "🍷"
    },
    1: {
        "name": "Class 1", 
        "description": "Wine variety with balanced properties and medium intensity",
        "color": "#4ECDC4",
        "emoji": "🥂"
    },
    2: {
        "name": "Class 2",
        "description": "Wine variety with lighter profile and delicate characteristics",
        "color": "#45B7D1",
        "emoji": "🍾"
    }
}

# Feature information with descriptions and ranges
FEATURE_INFO = {
    "alcohol": {"min": 11.0, "max": 15.0, "default": 13.0, "step": 0.1, 
                "desc": "Alcohol content by volume (%)"},
    "malic_acid": {"min": 0.5, "max": 6.0, "default": 2.5, "step": 0.1,
                   "desc": "Malic acid concentration (g/L)"},
    "ash": {"min": 1.0, "max": 4.0, "default": 2.5, "step": 0.1,
            "desc": "Ash content after incineration (%)"},
    "alcalinity_of_ash": {"min": 10.0, "max": 30.0, "default": 20.0, "step": 0.5,
                          "desc": "Alcalinity of ash content"},
    "magnesium": {"min": 70.0, "max": 165.0, "default": 100.0, "step": 1.0,
                  "desc": "Magnesium content (mg/L)"},
    "total_phenols": {"min": 0.5, "max": 4.0, "default": 2.5, "step": 0.1,
                      "desc": "Total phenolic compounds"},
    "flavanoids": {"min": 0.0, "max": 6.0, "default": 2.0, "step": 0.1,
                   "desc": "Flavanoid content"},
    "nonflavanoid_phenols": {"min": 0.0, "max": 1.0, "default": 0.4, "step": 0.05,
                              "desc": "Non-flavanoid phenolic content"},
    "proanthocyanins": {"min": 0.0, "max": 4.0, "default": 1.5, "step": 0.1,
                        "desc": "Proanthocyanin content"},
    "color_intensity": {"min": 1.0, "max": 13.0, "default": 5.0, "step": 0.1,
                        "desc": "Color intensity measurement"},
    "hue": {"min": 0.5, "max": 1.8, "default": 1.0, "step": 0.05,
            "desc": "Hue (color ratio)"},
    "od280/od315_of_diluted_wines": {"min": 1.0, "max": 4.5, "default": 3.0, "step": 0.1,
                                      "desc": "OD280/OD315 ratio of diluted wines"},
    "proline": {"min": 200.0, "max": 1700.0, "default": 750.0, "step": 10.0,
                "desc": "Proline amino acid content (mg/L)"}
}

@st.cache_resource
def load_models():
    """Load the trained model, scaler, and PCA model."""
    try:
        model = joblib.load('best_wine_model.pkl')
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca_model.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, scaler, pca, metadata
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found. Please run 'wine_classification.py' first to train and save the models.")
        st.info("Run this command:\n```\npython wine_classification.py\n```")
        return None, None, None, None

def predict_wine_class(features, model, scaler, pca):
    """Make a prediction using the loaded model."""
    # Convert to numpy array and reshape
    features_array = np.array(features).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features_array)
    
    # Apply PCA transformation
    features_pca = pca.transform(features_scaled)
    
    # Make prediction
    prediction = model.predict(features_pca)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_pca)[0]
    else:
        probabilities = None
    
    return prediction[0], probabilities

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍷 Wine Classification System</h1>
        <p>Data Science Final Lab Exam - Variant 1 | Multiclass Classification with PCA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model, scaler, pca, metadata = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar with model info
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.markdown(f"""
        <div class="info-box">
            <strong>Model:</strong> {metadata['model_name']}<br>
            <strong>Accuracy:</strong> {metadata['accuracy']:.2%}<br>
            <strong>PCA Components:</strong> {metadata['n_pca_components']}<br>
            <strong>Features:</strong> {len(metadata['feature_names'])}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🍷 Wine Classes")
        for class_id, info in WINE_CLASSES.items():
            st.markdown(f"""
            <div style="background: {info['color']}20; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; border-left: 3px solid {info['color']};">
                {info['emoji']} <strong>{info['name']}</strong><br>
                <small>{info['description']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Author")
        st.markdown("**Jahanzaib Channa**")
        st.markdown("Data Science Final Lab Exam")
    
    # Main content area
    st.markdown("## 📝 Enter Wine Chemical Properties")
    st.markdown("Adjust the sliders below to input the chemical properties of the wine sample:")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    features = []
    feature_names = list(FEATURE_INFO.keys())
    
    # Distribute features across columns
    for i, (feature_name, info) in enumerate(FEATURE_INFO.items()):
        col = [col1, col2, col3][i % 3]
        with col:
            st.markdown(f"**{feature_name.replace('_', ' ').title()}**")
            st.caption(info['desc'])
            value = st.slider(
                "",
                min_value=info['min'],
                max_value=info['max'],
                value=info['default'],
                step=info['step'],
                key=feature_name,
                label_visibility="collapsed"
            )
            features.append(value)
    
    st.markdown("---")
    
    # Prediction button with centering
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Wine Class", use_container_width=True)
    
    if predict_button:
        # Make prediction
        predicted_class, probabilities = predict_wine_class(features, model, scaler, pca)
        class_info = WINE_CLASSES[predicted_class]
        
        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")
        
        # Display prediction with animation
        st.markdown(f"""
        <div class="prediction-box" style="background: linear-gradient(135deg, {class_info['color']}CC 0%, {class_info['color']} 100%);">
            <h2>{class_info['emoji']} Predicted Wine Class</h2>
            <div class="class-name">{class_info['name']}</div>
            <p style="color: white; font-size: 1.1rem; margin-top: 0.5rem;">{class_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show probabilities if available
        if probabilities is not None:
            st.markdown("### 📊 Class Probabilities")
            prob_cols = st.columns(3)
            for i, (class_id, info) in enumerate(WINE_CLASSES.items()):
                with prob_cols[i]:
                    prob_percent = probabilities[i] * 100
                    st.markdown(f"""
                    <div style="background: {info['color']}20; padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid {info['color']};">
                        <div style="font-size: 2rem;">{info['emoji']}</div>
                        <div style="font-weight: bold; color: {info['color']};">{info['name']}</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">{prob_percent:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display input summary
        st.markdown("### 📋 Input Summary")
        with st.expander("View Entered Values", expanded=False):
            input_df = pd.DataFrame({
                "Feature": feature_names,
                "Value": features
            })
            st.dataframe(input_df, use_container_width=True)
    
    # Sample Predictions Section
    st.markdown("---")
    st.markdown("## 🧪 Quick Test with Sample Data")
    
    sample_cols = st.columns(3)
    
    sample_data = {
        "Class 0 Sample": [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065],
        "Class 1 Sample": [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520],
        "Class 2 Sample": [13.4, 3.91, 2.48, 23, 102, 1.8, 0.75, 0.43, 1.41, 7.3, 0.7, 1.56, 750]
    }
    
    for i, (sample_name, sample_values) in enumerate(sample_data.items()):
        with sample_cols[i]:
            class_id = i
            info = WINE_CLASSES[class_id]
            if st.button(f"{info['emoji']} {sample_name}", key=f"sample_{i}", use_container_width=True):
                pred_class, _ = predict_wine_class(sample_values, model, scaler, pca)
                st.success(f"Predicted: {WINE_CLASSES[pred_class]['name']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.6); padding: 1rem;">
        <p>🍷 Wine Classification System | Data Science Final Lab Exam - Variant 1</p>
        <p>Built with Streamlit | Author: Jahanzaib Channa</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
