import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_drift_report(base_dir):
    """Generates an Evidently Data Drift Report for the sales dataset."""
    sales_path = os.path.join(base_dir, 'data', 'raw', 'sales_data.csv')
    if not os.path.exists(sales_path):
        return None
    
    df = pd.read_csv(sales_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values(by='Date')
    
    # Split into reference (first 50%) and current (last 50%) to simulate drift check
    split_idx = len(df) // 2
    reference_data = df.iloc[:split_idx]
    current_data = df.iloc[split_idx:]
    
    # Select subset of columns to check for drift (numerical & categorical)
    cols_to_check = ['Quantity', 'Price', 'TotalAmount']
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data[cols_to_check], current_data=current_data[cols_to_check])
    
    # Save the report
    report_path = os.path.join(base_dir, 'dashboard', 'drift_report.html')
    report.save_html(report_path)
    
    return report_path

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = generate_drift_report(base_dir)
    print(f"Drift report generated at: {path}")
