from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import logging
import io

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate the dataframe for clustering"""
    if df.empty:
        return False
    if df.isnull().values.any():
        return False
    return True

def preprocess_data(df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
    """Preprocess data for clustering"""
    scaler = StandardScaler()
    data = df[numeric_cols].values
    return scaler.fit_transform(data)

@router.post("/kmeans", tags=["Clustering"])
async def kmeans_clustering(
    file: UploadFile = File(...),
    n_clusters: int = 3,
    numeric_columns: Optional[str] = None
):
    """
    Perform KMeans clustering on uploaded data.
    
    Parameters:
    - file: CSV file containing data
    - n_clusters: Number of clusters to form
    - numeric_columns: Comma-separated string of numeric columns to use
    
    Returns:
    - JSON with cluster labels and silhouette score
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate dataframe
        if not validate_dataframe(df):
            raise HTTPException(status_code=400, detail="Invalid data: empty or contains null values")
        
        # Determine numeric columns
        if numeric_columns:
            numeric_cols = [col.strip() for col in numeric_columns.split(',')]
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            raise HTTPException(status_code=400, detail="No numeric columns found in data")
        
        # Preprocess data
        processed_data = preprocess_data(df, numeric_cols)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(processed_data)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(processed_data, labels)
        
        # Prepare response
        response = {
            "status": "success",
            "algorithm": "KMeans",
            "n_clusters": n_clusters,
            "cluster_labels": labels.tolist(),
            "silhouette_score": silhouette_avg,
            "columns_used": numeric_cols,
            "sample_size": len(df)
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error in KMeans clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dbscan", tags=["Clustering"])
async def dbscan_clustering(
    file: UploadFile = File(...),
    eps: float = 0.5,
    min_samples: int = 5,
    numeric_columns: Optional[str] = None
):
    """
    Perform DBSCAN clustering on uploaded data.
    
    Parameters:
    - file: CSV file containing data
    - eps: Maximum distance between two samples for one to be in the neighborhood of the other
    - min_samples: Number of samples in a neighborhood for a point to be a core point
    - numeric_columns: Comma-separated string of numeric columns to use
    
    Returns:
    - JSON with cluster labels and silhouette score (if more than one cluster)
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate dataframe
        if not validate_dataframe(df):
            raise HTTPException(status_code=400, detail="Invalid data: empty or contains null values")
        
        # Determine numeric columns
        if numeric_columns:
            numeric_cols = [col.strip() for col in numeric_columns.split(',')]
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            raise HTTPException(status_code=400, detail="No numeric columns found in data")
        
        # Preprocess data
        processed_data = preprocess_data(df, numeric_cols)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(processed_data)
        
        # Calculate silhouette score if more than one cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        silhouette_avg = None
        if n_clusters > 1:
            silhouette_avg = silhouette_score(processed_data, labels)
        
        # Prepare response
        response = {
            "status": "success",
            "algorithm": "DBSCAN",
            "eps": eps,
            "min_samples": min_samples,
            "cluster_labels": labels.tolist(),
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_avg,
            "columns_used": numeric_cols,
            "sample_size": len(df)
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error in DBSCAN clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hierarchical", tags=["Clustering"])
async def hierarchical_clustering(
    file: UploadFile = File(...),
    n_clusters: int = 3,
    linkage: str = "ward",
    numeric_columns: Optional[str] = None
):
    """
    Perform Hierarchical clustering on uploaded data.
    
    Parameters:
    - file: CSV file containing data
    - n_clusters: Number of clusters to form
    - linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
    - numeric_columns: Comma-separated string of numeric columns to use
    
    Returns:
    - JSON with cluster labels and silhouette score
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate dataframe
        if not validate_dataframe(df):
            raise HTTPException(status_code=400, detail="Invalid data: empty or contains null values")
        
        # Determine numeric columns
        if numeric_columns:
            numeric_cols = [col.strip() for col in numeric_columns.split(',')]
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            raise HTTPException(status_code=400, detail="No numeric columns found in data")
        
        # Preprocess data
        processed_data = preprocess_data(df, numeric_cols)
        
        # Perform Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hierarchical.fit_predict(processed_data)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(processed_data, labels)
        
        # Prepare response
        response = {
            "status": "success",
            "algorithm": "AgglomerativeClustering",
            "n_clusters": n_clusters,
            "linkage": linkage,
            "cluster_labels": labels.tolist(),
            "silhouette_score": silhouette_avg,
            "columns_used": numeric_cols,
            "sample_size": len(df)
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error in Hierarchical clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))