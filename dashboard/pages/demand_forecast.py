import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")
st.title("📈 Demand Forecasting")

@st.cache_data
def load_sales_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sales_path = os.path.join(base_dir, 'data', 'raw', 'sales_data.csv')
    return pd.read_csv(sales_path) if os.path.exists(sales_path) else pd.DataFrame()

sales_df = load_sales_data()

if sales_df.empty:
    st.warning("Data not generated yet. Please run `python scripts/generate_data.py`")
else:
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    
    st.subheader("Daily Sales Volume Overview")
    
    # Import prophet logic
    sys.path.append(base_dir)
    from src.forecasting import prepare_prophet_data, train_prophet_model
    
    # Let user select a store to forecast (for speed, just forecast overall or first store)
    store_options = ['All'] + list(sales_df['StoreID'].unique())
    selected_store = st.selectbox("Select Store for Forecast", store_options)
    
    if selected_store == 'All':
        prophet_df = prepare_prophet_data(sales_df)
    else:
        prophet_df = prepare_prophet_data(sales_df, store_id=selected_store)
        
    st.write("Training Prophet model on historical data...")
    model, forecast = train_prophet_model(prophet_df, periods=30)
    
    if forecast is not None:
        fig = px.line(forecast, x='ds', y='yhat', title=f'30-Day Demand Forecast ({selected_store})')
        fig.add_scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Historical', opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Next Steps (MLOps Integration)")
        st.write("- **LSTM Ensemble**: Integrate PyTorch predictions to reduce MAPE.")
        st.write("- **MLflow**: Track model versions and hyperparameter experiments.")
        st.write("- **Evidently AI**: Monitor data drift in production.")
    else:
        st.error("Not enough data to train the model.")
