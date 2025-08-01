"""
=================================================================================
STREAMLIT WEB APPLICATION FOR CUSTOMER PURCHASE PREDICTION
=================================================================================
This app provides an interactive interface for business users to:
1. Input customer data
2. Get purchase predictions from multiple models
3. See explanations for predictions
4. Compare model outputs

WHY STREAMLIT:
- No web development knowledge required
- Python-native solution
- Real-time interactive updates
- Professional appearance
- Easy deployment to cloud

HOW TO RUN:
streamlit run app.py
=================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Customer Purchase Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-high {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """
    Load all models and preprocessing components.
    Uses Streamlit caching for performance.
    """
    try:
        # Update this path to match your output directory structure
        assets_dir = './output_latest/streamlit_assets'
        
        # Load models
        lr_model = joblib.load(os.path.join(assets_dir, 'logistic_regression_model.pkl'))
        xgb_model = joblib.load(os.path.join(assets_dir, 'xgboost_model.pkl'))
        scaler = joblib.load(os.path.join(assets_dir, 'feature_scaler.pkl'))
        
        # Load metadata
        with open(os.path.join(assets_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        return lr_model, xgb_model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure you've run the main analysis script first.")
        return None, None, None, None

def create_shap_explanation(model, X, feature_names, model_type):
    """
    Generate SHAP explanation for a single prediction.
    
    WHY: Shows which features contributed to the prediction
    and by how much, building trust in the model.
    """
    if model_type == "Logistic Regression":
        explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(X))
    else:  # XGBoost
        explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X)
    
    return shap_values, explainer

def main():
    """Main application logic"""
    
    # Header
    st.markdown('<h1 class="main-header">🛍️ Customer Purchase Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    lr_model, xgb_model, scaler, metadata = load_models()
    
    if lr_model is None:
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.header("Customer Information")
    st.sidebar.markdown("Enter customer details below:")
    
    # Create input fields based on feature names
    feature_inputs = {}
    feature_names = metadata['feature_names']
    feature_stats = metadata['feature_stats']
    
    # Group features for better organization
    st.sidebar.subheader("Demographics")
    for feature in feature_names[:len(feature_names)//2]:
        # Use appropriate input widget based on feature characteristics
        min_val = float(feature_stats['min'][feature])
        max_val = float(feature_stats['max'][feature])
        mean_val = float(feature_stats['mean'][feature])
        
        if max_val - min_val < 20 and min_val >= 0:  # Likely categorical or small range
            feature_inputs[feature] = st.sidebar.slider(
                feature.replace('_', ' ').title(),
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=1.0
            )
        else:  # Continuous feature
            feature_inputs[feature] = st.sidebar.number_input(
                feature.replace('_', ' ').title(),
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100
            )
    
    st.sidebar.subheader("Behavioral Features")
    for feature in feature_names[len(feature_names)//2:]:
        min_val = float(feature_stats['min'][feature])
        max_val = float(feature_stats['max'][feature])
        mean_val = float(feature_stats['mean'][feature])
        
        if max_val - min_val < 20 and min_val >= 0:
            feature_inputs[feature] = st.sidebar.slider(
                feature.replace('_', ' ').title(),
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=1.0
            )
        else:
            feature_inputs[feature] = st.sidebar.number_input(
                feature.replace('_', ' ').title(),
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100
            )
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    # Prepare data for prediction
    input_df = pd.DataFrame([feature_inputs])
    input_scaled = scaler.transform(input_df)
    
    # Make predictions
    lr_prob = lr_model.predict_proba(input_scaled)[0, 1]
    lr_pred = lr_model.predict(input_scaled)[0]
    
    # XGBoost prediction
    import xgboost as xgb
    dmatrix = xgb.DMatrix(input_scaled)
    xgb_prob = xgb_model.predict(dmatrix)[0]
    xgb_pred = int(xgb_prob > 0.5)
    
    # Ensemble prediction
    ensemble_prob = (lr_prob + xgb_prob) / 2
    ensemble_pred = int(ensemble_prob > 0.5)
    
    # Display predictions
    with col1:
        st.subheader("Logistic Regression")
        st.metric("Purchase Probability", f"{lr_prob:.1%}")
        if lr_pred:
            st.markdown('<p class="prediction-high">✅ Likely to Purchase</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="prediction-low">❌ Unlikely to Purchase</p>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("XGBoost")
        st.metric("Purchase Probability", f"{xgb_prob:.1%}")
        if xgb_pred:
            st.markdown('<p class="prediction-high">✅ Likely to Purchase</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="prediction-low">❌ Unlikely to Purchase</p>', unsafe_allow_html=True)
    
    with col3:
        st.subheader("Ensemble (Average)")
        st.metric("Purchase Probability", f"{ensemble_prob:.1%}")
        if ensemble_pred:
            st.markdown('<p class="prediction-high">✅ Likely to Purchase</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="prediction-low">❌ Unlikely to Purchase</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model explanations
    st.header("📊 Prediction Explanations")
    
    explanation_tab1, explanation_tab2 = st.tabs(["Logistic Regression", "XGBoost"])
    
    with explanation_tab1:
        st.subheader("Feature Contributions - Logistic Regression")
        
        # Calculate feature contributions
        feature_contributions = input_scaled[0] * lr_model.coef_[0]
        contrib_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_df.iloc[0].values,
            'Contribution': feature_contributions
        }).sort_values('Contribution', key=abs, ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in contrib_df['Contribution']]
        ax.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors)
        ax.set_xlabel('Contribution to Purchase Probability')
        ax.set_title('Feature Contributions (Logistic Regression)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature importance table
        st.dataframe(contrib_df.style.background_gradient(subset=['Contribution'], cmap='RdYlGn'))
    
    with explanation_tab2:
        st.subheader("Feature Importance - XGBoost")
        st.info("XGBoost uses complex tree-based decisions. The importance shown is based on how often each feature is used in the trees.")
        
        # Get feature importance from XGBoost
        importance_dict = xgb_model.get_score(importance_type='gain')
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
        }).sort_values('Importance', ascending=False)
        
        # Normalize importance
        importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        ax.set_xlabel('Relative Importance')
        ax.set_title('Feature Importance (XGBoost)')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(importance_df.style.background_gradient(subset=['Importance'], cmap='Blues'))
    
    st.markdown("---")
    
    # Business recommendations
    st.header("💡 Business Recommendations")
    
    if ensemble_prob > 0.7:
        st.success("""
        **High Purchase Probability Customer**
        - Priority target for marketing campaigns
        - Consider premium product offerings
        - Personalized communication recommended
        - Expected ROI: High
        """)
    elif ensemble_prob > 0.3:
        st.warning("""
        **Medium Purchase Probability Customer**
        - Targeted incentives may convert
        - A/B test different marketing messages
        - Monitor engagement closely
        - Expected ROI: Moderate
        """)
    else:
        st.error("""
        **Low Purchase Probability Customer**
        - Minimal marketing investment recommended
        - Include in broad, low-cost campaigns only
        - Focus resources on higher probability segments
        - Expected ROI: Low
        """)
    
    # Additional features
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        ### Model Information
        
        **Logistic Regression**
        - Linear model that provides probability estimates
        - Highly interpretable with clear feature contributions
        - Best for: Regulatory compliance, explainable decisions
        
        **XGBoost**
        - Advanced tree-based ensemble model
        - Captures non-linear relationships and interactions
        - Best for: Maximum predictive performance
        
        **Ensemble**
        - Averages predictions from both models
        - Balances interpretability and performance
        - Recommended for most business applications
        
        ### Data Information
        - Models trained on: {}
        - Features used: {}
        """.format(metadata['model_date'], len(feature_names)))
    
    # Footer
    st.markdown("---")
    st.caption("Customer Purchase Prediction Model | FINAN 6520-090 Final Project")
    st.caption("Built with Streamlit, Scikit-learn, XGBoost, and SHAP")

if __name__ == "__main__":
    main()

"""
=================================================================================
STREAMLIT APP EDUCATIONAL NOTES:

WHY WE CHOSE STREAMLIT:
1. Rapid Prototyping: Create web apps with pure Python
2. Interactive Widgets: Automatic UI generation from Python code
3. Caching: Built-in performance optimization
4. Deployment: Easy hosting on Streamlit Cloud, Heroku, or AWS
5. Professional: Looks good without CSS/JavaScript knowledge

KEY FEATURES IMPLEMENTED:
1. Real-time Predictions: Instant results as users adjust inputs
2. Model Comparison: Side-by-side evaluation of different approaches
3. Explainability: Visual explanations of predictions
4. Business Context: Actionable recommendations based on probabilities
5. Professional UI: Clean, intuitive interface for non-technical users

DEPLOYMENT CONSIDERATIONS:
1. Authentication: Add user login for production
2. Logging: Track predictions for model monitoring
3. API Integration: Connect to live customer databases
4. Scalability: Use cloud deployment for multiple users
5. Updates: Implement model versioning and A/B testing

This app demonstrates how machine learning can be made accessible
to business users through thoughtful interface design.
=================================================================================
"""