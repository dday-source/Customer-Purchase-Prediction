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
    Load pre-trained models and associated preprocessing components.
    
    We utilize Streamlit's caching decorator to prevent redundant loading
    operations, significantly improving application responsiveness. The
    cache persists across user sessions, reducing server load.
    """
    try:
        # Path configuration based on our project structure
        assets_dir = './output_latest/streamlit_assets'
        
        # Load serialized models
        lr_model = joblib.load(os.path.join(assets_dir, 'logistic_regression_model.pkl'))
        xgb_model = joblib.load(os.path.join(assets_dir, 'xgboost_model.pkl'))
        scaler = joblib.load(os.path.join(assets_dir, 'feature_scaler.pkl'))
        
        # Load metadata for feature configuration
        with open(os.path.join(assets_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        return lr_model, xgb_model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure you've run the main analysis script first.")
        return None, None, None, None

def create_shap_explanation(model, X, feature_names, model_type):
    """
    Generate SHAP explanations for model predictions.
    
    SHAP (SHapley Additive exPlanations) provides game-theoretic explanations
    for individual predictions. This transparency is essential for:
    - Regulatory compliance in financial services
    - Building stakeholder trust in model decisions
    - Identifying potential biases or unexpected patterns
    
    We learned about SHAP's importance through industry guest lectures
    emphasizing explainable AI in finance.
    """
    if model_type == "Logistic Regression":
        explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(X))
    else:  # XGBoost
        explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X)
    
    return shap_values, explainer

def main():
    """
    Main application logic orchestrating the user interface and model interactions.
    """
    
    # Application header
    st.markdown('<h1 class="main-header">🛍️ Customer Purchase Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    lr_model, xgb_model, scaler, metadata = load_models()
    
    if lr_model is None:
        st.stop()
    
    # Sidebar configuration for user inputs
    st.sidebar.header("Customer Information")
    st.sidebar.markdown("Enter customer details below:")
    
    # Dynamic input generation based on feature metadata
    feature_inputs = {}
    feature_names = metadata['feature_names']
    feature_stats = metadata['feature_stats']
    
    # Organize features into logical groups for better UX
    st.sidebar.subheader("Demographics")
    for feature in feature_names[:len(feature_names)//2]:
        # Select appropriate input widget based on feature characteristics
        min_val = float(feature_stats['min'][feature])
        max_val = float(feature_stats['max'][feature])
        mean_val = float(feature_stats['mean'][feature])
        
        if max_val - min_val < 20 and min_val >= 0:  # Likely discrete values
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
    
    # Main content area with model predictions
    col1, col2, col3 = st.columns(3)
    
    # Prepare data for prediction
    input_df = pd.DataFrame([feature_inputs])
    input_scaled = scaler.transform(input_df)
    
    # Generate predictions from both models
    lr_prob = lr_model.predict_proba(input_scaled)[0, 1]
    lr_pred = lr_model.predict(input_scaled)[0]
    
    # XGBoost prediction requires DMatrix format
    import xgboost as xgb
    dmatrix = xgb.DMatrix(input_scaled)
    xgb_prob = xgb_model.predict(dmatrix)[0]
    xgb_pred = int(xgb_prob > 0.5)
    
    # Ensemble prediction combines both models
    ensemble_prob = (lr_prob + xgb_prob) / 2
    ensemble_pred = int(ensemble_prob > 0.5)
    
    # Display predictions in organized columns
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
    
    # Model explanation section
    st.header("📊 Prediction Explanations")
    
    explanation_tab1, explanation_tab2 = st.tabs(["Logistic Regression", "XGBoost"])
    
    with explanation_tab1:
        st.subheader("Feature Contributions - Logistic Regression")
        
        # Calculate linear feature contributions
        feature_contributions = input_scaled[0] * lr_model.coef_[0]
        contrib_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_df.iloc[0].values,
            'Contribution': feature_contributions
        }).sort_values('Contribution', key=abs, ascending=False)
        
        # Visualization of feature impacts
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in contrib_df['Contribution']]
        ax.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors)
        ax.set_xlabel('Contribution to Purchase Probability')
        ax.set_title('Feature Contributions (Logistic Regression)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabular representation with color coding
        st.dataframe(contrib_df.style.background_gradient(subset=['Contribution'], cmap='RdYlGn'))
    
    with explanation_tab2:
        st.subheader("Feature Importance - XGBoost")
        st.info("XGBoost uses ensemble tree methods. The importance shown reflects how frequently each feature is used for splitting decisions across all trees.")
        
        # Extract feature importance from XGBoost model
        importance_dict = xgb_model.get_score(importance_type='gain')
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
        }).sort_values('Importance', ascending=False)
        
        # Normalize for interpretation
        importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        ax.set_xlabel('Relative Importance')
        ax.set_title('Feature Importance (XGBoost)')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(importance_df.style.background_gradient(subset=['Importance'], cmap='Blues'))
    
    st.markdown("---")
    
    # Business recommendations section
    st.header("💡 Business Recommendations")
    
    # Threshold-based recommendations align with marketing strategy
    if ensemble_prob > 0.7:
        st.success("""
        **High Purchase Probability Customer**
        - Priority segment for targeted marketing campaigns
        - Recommend premium product offerings and upselling
        - Personalized communication likely to yield high ROI
        - Consider exclusive offers to maximize conversion value
        """)
    elif ensemble_prob > 0.3:
        st.warning("""
        **Medium Purchase Probability Customer**
        - Potential for conversion with appropriate incentives
        - Recommend A/B testing different marketing messages
        - Monitor engagement metrics for optimization opportunities
        - Consider time-limited offers to create urgency
        """)
    else:
        st.error("""
        **Low Purchase Probability Customer**
        - Minimize direct marketing investment
        - Include only in broad, low-cost digital campaigns
        - Focus resources on higher probability segments
        - Consider long-term nurturing strategies
        """)
    
    # Additional information and context
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        ### Model Information
        
        **Logistic Regression**
        - Traditional statistical approach widely accepted in finance
        - Provides clear probability estimates with confidence intervals
        - Highly interpretable with direct feature impact coefficients
        - Optimal for: Regulatory compliance, transparent decision-making
        
        **XGBoost**
        - State-of-the-art ensemble method using gradient boosting
        - Captures complex non-linear relationships and interactions
        - Generally achieves superior predictive performance
        - Optimal for: Maximum accuracy, complex pattern detection
        
        **Ensemble Approach**
        - Combines strengths of both methodologies
        - Reduces individual model bias and variance
        - Provides robust predictions for business decisions
        - Recommended for production deployment
        
        ### Technical Details
        - Training Date: {}
        - Number of Features: {}
        - Cross-validation performed with stratified sampling
        - Models optimized for business profit metrics
        """.format(metadata['model_date'], len(feature_names)))
    
    # Footer with attribution
    st.markdown("---")
    st.caption("Customer Purchase Prediction Model | FINAN 6520-090 Final Project")
    st.caption("Developed using Python, Scikit-learn, XGBoost, and Streamlit")

if __name__ == "__main__":
    main()