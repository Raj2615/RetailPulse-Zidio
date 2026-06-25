import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configurations
NUM_CUSTOMERS = 5000
NUM_PRODUCTS = 200
NUM_STORES = 10
DAYS_OF_DATA = 365 * 2 # 2 years of data
START_DATE = datetime(2024, 1, 1)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

print("Generating synthetic datasets for RetailPulse...")

# 1. Generate Customer Data
print("Generating customer data...")
customer_ids = [f'CUST_{str(i).zfill(5)}' for i in range(1, NUM_CUSTOMERS + 1)]
ages = np.random.normal(loc=35, scale=12, size=NUM_CUSTOMERS).astype(int)
ages = np.clip(ages, 18, 80)
genders = np.random.choice(['M', 'F', 'Other'], size=NUM_CUSTOMERS, p=[0.48, 0.48, 0.04])
locations = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], size=NUM_CUSTOMERS)
join_dates = [START_DATE - timedelta(days=random.randint(0, 1000)) for _ in range(NUM_CUSTOMERS)]
loyalty_scores = np.random.randint(1, 100, size=NUM_CUSTOMERS)

customers_df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Age': ages,
    'Gender': genders,
    'Location': locations,
    'JoinDate': [d.strftime('%Y-%m-%d') for d in join_dates],
    'LoyaltyScore': loyalty_scores
})
customers_df.to_csv(os.path.join(DATA_DIR, 'customer_data.csv'), index=False)

# 2. Generate Product / Inventory Data
print("Generating inventory data...")
product_ids = [f'PROD_{str(i).zfill(4)}' for i in range(1, NUM_PRODUCTS + 1)]
store_ids = [f'STORE_{str(i).zfill(2)}' for i in range(1, NUM_STORES + 1)]

inventory_records = []
product_prices = {pid: round(random.uniform(5.0, 500.0), 2) for pid in product_ids}

for store in store_ids:
    for prod in product_ids:
        inventory_records.append({
            'StoreID': store,
            'ProductID': prod,
            'CurrentStock': random.randint(10, 500),
            'ReorderLevel': random.randint(20, 100),
            'LeadTimeDays': random.randint(2, 14)
        })

inventory_df = pd.DataFrame(inventory_records)
inventory_df.to_csv(os.path.join(DATA_DIR, 'inventory_data.csv'), index=False)

# 3. Generate Sales Data
print("Generating sales data (this might take a minute)...")
date_range = [START_DATE + timedelta(days=i) for i in range(DAYS_OF_DATA)]
sales_records = []

# To make data realistic, add seasonality and trends
for date in date_range:
    # Base number of transactions per day
    base_transactions = 500
    
    # Weekly seasonality (higher on weekends)
    if date.weekday() >= 5:
        base_transactions += 200
        
    # Yearly seasonality (higher in Nov/Dec)
    if date.month in [11, 12]:
        base_transactions += 300
        
    num_daily_tx = int(np.random.normal(base_transactions, 50))
    num_daily_tx = max(50, num_daily_tx)
    
    # Sample daily customers and products
    daily_customers = np.random.choice(customer_ids, size=num_daily_tx)
    daily_products = np.random.choice(product_ids, size=num_daily_tx)
    daily_stores = np.random.choice(store_ids, size=num_daily_tx)
    
    for i in range(num_daily_tx):
        qty = max(1, int(np.random.exponential(scale=2.0)))
        price = product_prices[daily_products[i]]
        
        sales_records.append({
            'TransactionID': f'TXN_{date.strftime("%Y%m%d")}_{str(i).zfill(4)}',
            'Date': date.strftime('%Y-%m-%d'),
            'CustomerID': daily_customers[i],
            'ProductID': daily_products[i],
            'StoreID': daily_stores[i],
            'Quantity': qty,
            'Price': price,
            'TotalAmount': round(qty * price, 2)
        })

sales_df = pd.DataFrame(sales_records)
sales_df.to_csv(os.path.join(DATA_DIR, 'sales_data.csv'), index=False)

print(f"Data generation complete! Files saved in {DATA_DIR}")
print(f"Generated {len(customers_df)} customers, {len(inventory_df)} inventory records, and {len(sales_df)} sales transactions.")
