from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np

def perform_kmeans_segmentation(customer_features, n_clusters=6):
    """Performs K-Means clustering on RFM features."""
    features = customer_features[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_features['KMeans_Cluster'] = kmeans.fit_predict(scaled_features)
    
    return customer_features, kmeans, scaler

def perform_dbscan_segmentation(customer_features, eps=0.5, min_samples=10):
    """Performs DBSCAN clustering on RFM features."""
    features = customer_features[['Recency', 'Frequency', 'Monetary']].copy()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    customer_features['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_features)
    
    return customer_features, dbscan

def analyze_segments(customer_features, cluster_col='KMeans_Cluster'):
    """Returns average RFM values per segment."""
    summary = customer_features.groupby(cluster_col).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'}).reset_index()
    
    return summary
