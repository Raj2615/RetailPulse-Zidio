import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys

# Add src to path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)

from src.data_processing import load_raw_data, generate_rfm_features
from src.segmentation import perform_kmeans_segmentation, analyze_segments

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
st.title("👥 Customer Segmentation (RFM + K-Means)")

@st.cache_data
def get_segmentation_data():
    sales, customers, _ = load_raw_data(base_dir)
    if sales.empty or customers.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    rfm = generate_rfm_features(sales, customers)
    rfm_segmented, _, _ = perform_kmeans_segmentation(rfm, n_clusters=6)
    summary = analyze_segments(rfm_segmented)
    
    return rfm_segmented, summary

data, summary = get_segmentation_data()

if data.empty:
    st.warning("Data not generated. Run `python scripts/generate_data.py`")
else:
    st.subheader("Segment Analysis")
    st.dataframe(summary)
    
    st.subheader("3D Cluster Visualization")
    fig = px.scatter_3d(data, x='Recency', y='Frequency', z='Monetary',
                        color='KMeans_Cluster', opacity=0.7,
                        title="3D Scatter of RFM Clusters")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Business Interpretation
    * **Cluster 0**: High value, frequent buyers (Champions)
    * **Cluster 1**: Recent but low frequency (New Customers)
    * **Cluster X**: High recency, low frequency (At Risk)
    """)
