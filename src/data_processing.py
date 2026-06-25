import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_raw_data(base_dir):
    """Loads raw CSV data."""
    sales = pd.read_csv(os.path.join(base_dir, 'data', 'raw', 'sales_data.csv'))
    customers = pd.read_csv(os.path.join(base_dir, 'data', 'raw', 'customer_data.csv'))
    inventory = pd.read_csv(os.path.join(base_dir, 'data', 'raw', 'inventory_data.csv'))
    
    sales['Date'] = pd.to_datetime(sales['Date'])
    customers['JoinDate'] = pd.to_datetime(customers['JoinDate'])
    
    return sales, customers, inventory

def generate_rfm_features(sales_df, customers_df):
    """Generates RFM (Recency, Frequency, Monetary) features for customers."""
    max_date = sales_df['Date'].max()
    
    rfm = sales_df.groupby('CustomerID').agg({
        'Date': lambda x: (max_date - x.max()).days, # Recency
        'TransactionID': 'count',                     # Frequency
        'TotalAmount': 'sum'                          # Monetary
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Merge with customer data
    customer_features = pd.merge(customers_df, rfm, on='CustomerID', how='left')
    customer_features.fillna(0, inplace=True)
    
    return customer_features

def preprocess_for_churn(customer_features):
    """Creates a synthetic churn label and prepares data for XGBoost."""
    # Define churn synthetically based on Recency
    # Since this is synthetic data and max recency is low, we use the 80th percentile
    df = customer_features.copy()
    churn_threshold = df['Recency'].quantile(0.80)
    df['Churn'] = (df['Recency'] > churn_threshold).astype(int)
    
    # Encode categorical features
    df['Gender'] = df['Gender'].map({'M': 0, 'F': 1, 'Other': 2})
    df = pd.get_dummies(df, columns=['Location'], drop_first=True)
    
    # Drop IDs and Dates
    X = df.drop(['CustomerID', 'JoinDate', 'Churn'], axis=1)
    y = df['Churn']
    
    return X, y

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    sales, customers, inventory = load_raw_data(base_dir)
    print("Data loaded successfully.")
    rfm = generate_rfm_features(sales, customers)
    print("RFM Features generated. Sample:")
    print(rfm.head())
