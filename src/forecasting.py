import pandas as pd
import numpy as np
from prophet import Prophet
import torch
import torch.nn as nn

def prepare_prophet_data(sales_df, product_id=None, store_id=None):
    """Prepares time-series data for Prophet forecasting."""
    df = sales_df.copy()
    if product_id:
        df = df[df['ProductID'] == product_id]
    if store_id:
        df = df[df['StoreID'] == store_id]
        
    daily_sales = df.groupby('Date')['Quantity'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    return daily_sales

def train_prophet_model(df, periods=30):
    """Trains a Prophet model and predicts future demand."""
    if len(df) < 20:
        return None, None # Not enough data
        
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df)
    
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    
    return m, forecast

# Basic LSTM implementation for hybrid model requirement
class DemandLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def train_lstm_model(data, epochs=50):
    """Skeleton for LSTM training."""
    # Data scaling and sequencing goes here.
    # We provide this structure for the 28-day roadmap milestone.
    model = DemandLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Mock training loop
    # In a real scenario, we loop through sequences of past N days to predict N+1
    return model
