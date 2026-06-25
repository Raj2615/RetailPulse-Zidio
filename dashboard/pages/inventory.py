import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)

from src.data_processing import load_raw_data
from src.inventory import calculate_reorder_points

st.set_page_config(page_title="Inventory Optimization", page_icon="📦", layout="wide")
st.title("📦 Inventory Optimization & Replenishment")

@st.cache_data
def get_inventory_recommendations():
    _, _, inventory = load_raw_data(base_dir)
    if inventory.empty:
        return pd.DataFrame()
    
    optimized = calculate_reorder_points(inventory)
    return optimized

inventory_df = get_inventory_recommendations()

if inventory_df.empty:
    st.warning("Data not generated. Run `python scripts/generate_data.py`")
else:
    status_counts = inventory_df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Inventory Status")
        fig = px.pie(status_counts, values='Count', names='Status', hole=0.4,
                     color='Status', color_discrete_map={
                         'Optimal': 'green', 
                         'Understocked (Reorder Needed)': 'red',
                         'Overstocked': 'orange'
                     })
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Urgent Reorder Recommendations")
        urgent = inventory_df[inventory_df['Status'] == 'Understocked (Reorder Needed)']
        urgent = urgent.sort_values(by='RecommendedReorderQty', ascending=False).head(10)
        
        st.dataframe(urgent[['StoreID', 'ProductID', 'CurrentStock', 'OptimalReorderPoint', 'RecommendedReorderQty']], use_container_width=True)
        
    st.markdown("### Cost Impact Simulation")
    st.write(f"Implementing these recommendations will directly mitigate stockouts for **{len(inventory_df[inventory_df['Status'] == 'Understocked (Reorder Needed)'])}** product lines across stores.")
