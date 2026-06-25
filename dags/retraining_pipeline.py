from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Append project root to path
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

from src.data_processing import load_raw_data, generate_rfm_features, preprocess_for_churn
from src.churn import train_churn_model

default_args = {
    'owner': 'retailpulse',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retailpulse_retraining_pipeline',
    default_args=default_args,
    description='Automated weekly retraining for Churn Prediction',
    schedule_interval=timedelta(weeks=1),
    catchup=False
)

def retrain_churn():
    print("Loading data for retraining...")
    sales, customers, _ = load_raw_data(base_dir)
    rfm = generate_rfm_features(sales, customers)
    X, y = preprocess_for_churn(rfm)
    
    print("Retraining XGBoost Churn Model...")
    model, metrics, _, _ = train_churn_model(X, y)
    print(f"Retraining complete. New ROC-AUC: {metrics['roc_auc']:.4f}")
    return metrics['roc_auc']

retrain_task = PythonOperator(
    task_id='retrain_churn_model',
    python_callable=retrain_churn,
    dag=dag,
)

retrain_task
