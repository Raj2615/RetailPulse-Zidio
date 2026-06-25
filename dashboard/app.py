import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 RetailPulse – AI-Powered Analytics Platform")
st.markdown("### Customer Analytics & Demand Forecasting Platform")

st.sidebar.title("Navigation")
st.sidebar.markdown("Select a module to view analytics.")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    sales_path = os.path.join(base_dir, 'data', 'raw', 'sales_data.csv')
    customer_path = os.path.join(base_dir, 'data', 'raw', 'customer_data.csv')
    inventory_path = os.path.join(base_dir, 'data', 'raw', 'inventory_data.csv')
    
    sales_df = pd.read_csv(sales_path) if os.path.exists(sales_path) else pd.DataFrame()
    customers_df = pd.read_csv(customer_path) if os.path.exists(customer_path) else pd.DataFrame()
    inventory_df = pd.read_csv(inventory_path) if os.path.exists(inventory_path) else pd.DataFrame()
    
    return sales_df, customers_df, inventory_df

sales_df, customers_df, inventory_df = load_data()

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", f"{len(customers_df):,}" if not customers_df.empty else 0)
with col2:
    st.metric("Total Transactions", f"{len(sales_df):,}" if not sales_df.empty else 0)
with col3:
    st.metric("Inventory Records", f"{len(inventory_df):,}" if not inventory_df.empty else 0)

st.markdown("---")
st.subheader("Platform Overview")
st.write("""
Welcome to **RetailPulse**. Navigate to different modules using the sidebar (pages coming soon):
* **Demand Forecasting**: Predictive modeling to reduce stockouts and optimize inventory.
* **Customer Segmentation**: RFM analysis to group customers and target appropriately.
* **Churn Prediction**: Identify at-risk customers early with Machine Learning.
* **Inventory Optimization**: Data-driven recommendations for reorder points.

*Generated dataset snapshot:*
""")

if not sales_df.empty:
    st.dataframe(sales_df.head(5))
else:
    st.warning("Data not generated yet. Please run `python scripts/generate_data.py`")

st.info("💡 **Tip**: Build out your specific functionalities in `pages/demand_forecast.py` and others as per the 28-day roadmap.")
