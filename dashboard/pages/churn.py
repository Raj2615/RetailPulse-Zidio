import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)

from src.data_processing import load_raw_data, generate_rfm_features, preprocess_for_churn
from src.churn import train_churn_model

st.set_page_config(page_title="Churn Prediction", page_icon="🚨", layout="wide")
st.title("🚨 Churn Prediction Risk Assessment")

@st.cache_resource
def build_churn_model():
    sales, customers, _ = load_raw_data(base_dir)
    if sales.empty or customers.empty:
        return None, None
    
    rfm = generate_rfm_features(sales, customers)
    X, y = preprocess_for_churn(rfm)
    
    model, metrics, X_test, y_test = train_churn_model(X, y)
    
    # Get feature importances
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return metrics, importance

metrics, importance = build_churn_model()

if not metrics:
    st.warning("Data not generated. Run `python scripts/generate_data.py`")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC Score", f"{metrics['roc_auc']:.2f}")
    col2.metric("Precision", f"{metrics['precision']:.2f}")
    col3.metric("Recall", f"{metrics['recall']:.2f}")
    
    st.markdown("---")
    st.subheader("Feature Importance (XGBoost)")
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title="What drives customer churn?")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Integration**: In production, these predictions feed into marketing workflows (e.g., sending targeted discounts).")
