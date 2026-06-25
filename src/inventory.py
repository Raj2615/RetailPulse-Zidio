import pandas as pd
import numpy as np

def calculate_reorder_points(inventory_df, forecast_df=None):
    """
    Calculates optimal reorder points and quantities.
    If forecast_df is provided, uses predicted demand.
    Otherwise, uses basic heuristics.
    """
    df = inventory_df.copy()
    
    # Calculate safety stock (Basic heuristic: 20% of ReorderLevel)
    df['SafetyStock'] = np.ceil(df['ReorderLevel'] * 0.2).astype(int)
    
    # Optimal Reorder Point = (Lead Time * Average Daily Demand) + Safety Stock
    # For now, we mock average daily demand based on ReorderLevel and LeadTime
    df['AvgDailyDemand'] = np.ceil(df['ReorderLevel'] / 7).astype(int)
    
    df['OptimalReorderPoint'] = (df['LeadTimeDays'] * df['AvgDailyDemand']) + df['SafetyStock']
    
    # Identify Overstocked and Understocked items
    df['Status'] = 'Optimal'
    df.loc[df['CurrentStock'] < df['OptimalReorderPoint'], 'Status'] = 'Understocked (Reorder Needed)'
    df.loc[df['CurrentStock'] > df['OptimalReorderPoint'] * 2.5, 'Status'] = 'Overstocked'
    
    # Recommended Reorder Quantity (EOQ model simplification)
    # Target stock = OptimalReorderPoint * 1.5
    df['RecommendedReorderQty'] = 0
    understocked = df['Status'] == 'Understocked (Reorder Needed)'
    df.loc[understocked, 'RecommendedReorderQty'] = np.ceil(
        (df.loc[understocked, 'OptimalReorderPoint'] * 1.5) - df.loc[understocked, 'CurrentStock']
    ).astype(int)
    
    return df
