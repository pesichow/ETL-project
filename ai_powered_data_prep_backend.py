from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, date
from dotenv import load_dotenv
import re
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/data-prep", tags=["AI-powered Data Prep (Auto-suggestions)"])

# Import the fetch_data function
from utils.fetch_data import fetch_data

# Pydantic models
class DataQualityIssue(BaseModel):
    column: str
    issue_type: str
    severity: str
    description: str
    affected_rows: int
    suggested_fix: str
    code_snippet: str

class DatasetInfoRequest(BaseModel):
    job_id: str
    table_names: List[str]
    limit: int = 10000

class CleaningSuggestion(BaseModel):
    suggestion_id: str
    priority: str
    category: str
    description: str
    affected_columns: List[str]
    code_snippet: str
    preview_sample: Dict[str, Any]

# Global storage for sessions
data_prep_sessions = {}

def convert_to_serializable(obj):
    """Convert various types to JSON-serializable formats"""
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif pd.isna(obj):
        return None
    return obj

def detect_data_quality_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect various data quality issues in the dataset"""
    issues = []
    
    for column in df.columns:
        col_data = df[column]
        
        # Missing values
        missing_count = col_data.isnull().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / len(df)) * 100
            severity = "high" if missing_percentage > 50 else "medium" if missing_percentage > 20 else "low"
            issues.append({
                "column": column,
                "issue_type": "missing_values",
                "severity": severity,
                "description": f"{int(missing_count)} missing values ({missing_percentage:.1f}%)",
                "affected_rows": int(missing_count),
                "suggested_fix": f"Fill with median/mode or drop rows" if missing_percentage < 50 else "Consider dropping column",
                "code_snippet": f"df['{column}'].fillna(df['{column}'].median())" if col_data.dtype in ['int64', 'float64'] else f"df['{column}'].fillna(df['{column}'].mode()[0])"
            })
        
        # Duplicate values (for non-numeric columns)
        if col_data.dtype == 'object':
            # Check for inconsistent text casing
            unique_values = col_data.dropna().unique()
            if len(unique_values) > 1:
                lower_values = [str(v).lower() for v in unique_values]
                if len(set(lower_values)) < len(unique_values):
                    issues.append({
                        "column": column,
                        "issue_type": "inconsistent_casing",
                        "severity": "medium",
                        "description": f"Inconsistent text casing detected",
                        "affected_rows": len(df),
                        "suggested_fix": "Standardize to lowercase",
                        "code_snippet": f"df['{column}'] = df['{column}'].str.lower()"
                    })
            
            # Check for potential date strings
            sample_values = col_data.dropna().head(10).astype(str)
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            ]
            
            for pattern in date_patterns:
                if any(re.match(pattern, str(val)) for val in sample_values):
                    issues.append({
                        "column": column,
                        "issue_type": "date_format",
                        "severity": "medium",
                        "description": f"Column appears to contain dates as strings",
                        "affected_rows": len(df),
                        "suggested_fix": "Convert to datetime format",
                        "code_snippet": f"df['{column}'] = pd.to_datetime(df['{column}'], errors='coerce')"
                    })
                    break
        
        # Numeric columns with string formatting
        if col_data.dtype == 'object':
            # Check for numeric strings with commas
            sample_values = col_data.dropna().head(10).astype(str)
            if any(re.match(r'^\d{1,3}(,\d{3})*(\.\d+)?$', str(val)) for val in sample_values):
                issues.append({
                    "column": column,
                    "issue_type": "numeric_formatting",
                    "severity": "medium",
                    "description": f"Numeric values stored as formatted strings",
                    "affected_rows": len(df),
                    "suggested_fix": "Convert to numeric format",
                    "code_snippet": f"df['{column}'] = df['{column}'].str.replace(',', '').astype(float)"
                })
        
        # Outliers in numeric columns
        if col_data.dtype in ['int64', 'float64']:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(df)) * 100
                if outlier_percentage > 5:  # Only flag if >5% are outliers
                    issues.append({
                        "column": column,
                        "issue_type": "outliers",
                        "severity": "low",
                        "description": f"{len(outliers)} potential outliers detected ({outlier_percentage:.1f}%)",
                        "affected_rows": int(len(outliers)),
                        "suggested_fix": "Review and potentially cap or remove outliers",
                        "code_snippet": f"# Remove outliers\nQ1 = df['{column}'].quantile(0.25)\nQ3 = df['{column}'].quantile(0.75)\nIQR = Q3 - Q1\ndf = df[~((df['{column}'] < (Q1 - 1.5 * IQR)) | (df['{column}'] > (Q3 + 1.5 * IQR)))]"
                    })
    
    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        issues.append({
            "column": "all_columns",
            "issue_type": "duplicate_rows",
            "severity": "medium",
            "description": f"{int(duplicate_rows)} duplicate rows found",
            "affected_rows": int(duplicate_rows),
            "suggested_fix": "Remove duplicate rows",
            "code_snippet": "df = df.drop_duplicates()"
        })
    
    return issues

@router.post("/upload-dataset")
async def upload_dataset(request: DatasetInfoRequest):
    """📁 Upload Dataset from Iceberg - Step 1: Fetch dataset from Iceberg for AI-powered cleaning"""
    try:
        # Generate session ID
        session_id = f"dataprep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.job_id) % 10000}"
        
        # Fetch data from Iceberg
        logger.info(f"Fetching data from Iceberg tables: {request.table_names}")
        iceberg_result = fetch_data(request.job_id, request.table_names, request.limit)
        
        if "results" not in iceberg_result or not iceberg_result["results"]:
            raise HTTPException(status_code=400, detail="No data returned from Iceberg")
        
        # Extract table-wise row data
        data = {
            item["table_name"]: item["row_data"]
            for item in iceberg_result["results"]
            if item["row_data"]
        }

        if not data:
            raise HTTPException(status_code=400, detail="No row data returned from Iceberg")

        # Combine data from all tables
        combined_df = None
        for table_name, table_data in data.items():
            if table_data:
                df = pd.DataFrame(table_data)
                df['_source_table'] = table_name
                combined_df = df if combined_df is None else pd.concat([combined_df, df], ignore_index=True)
        
        if combined_df is None or combined_df.empty:
            raise HTTPException(status_code=400, detail="No valid data found for data preparation")

        # Convert date columns to strings for serialization
        for col in combined_df.select_dtypes(include=['datetime64']).columns:
            combined_df[col] = combined_df[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

        # Store dataset in session
        data_prep_sessions[session_id] = {
            "original_dataset": combined_df.copy(),
            "current_dataset": combined_df.copy(),
            "job_id": request.job_id,
            "table_names": request.table_names,
            "upload_time": datetime.now().isoformat(),
            "quality_issues": None,
            "cleaning_suggestions": None
        }
        
        # Prepare sample data with proper serialization
        sample_data = combined_df.head(5).applymap(convert_to_serializable).to_dict(orient='records')
        
        # Basic dataset info
        dataset_info = {
            "session_id": session_id,
            "job_id": request.job_id,
            "table_names": request.table_names,
            "shape": list(combined_df.shape),
            "columns": combined_df.columns.tolist(),
            "dtypes": {k: str(v) for k, v in combined_df.dtypes.to_dict().items()},
            "missing_values": convert_to_serializable(combined_df.isnull().sum().to_dict()),
            "memory_usage": f"{combined_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "sample_data": sample_data,
            "basic_stats": {
                "numeric_columns": combined_df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": combined_df.select_dtypes(include=['object']).columns.tolist(),
                "datetime_columns": combined_df.select_dtypes(include=['datetime64']).columns.tolist(),
                "total_missing_values": int(combined_df.isnull().sum().sum()),
                "duplicate_rows": int(combined_df.duplicated().sum())
            }
        }
        
        return JSONResponse(content=convert_to_serializable({
            "status": "success",
            "message": "Dataset fetched from Iceberg successfully",
            "session_id": session_id,
            "dataset_info": dataset_info,
            "next_step": f"Scan data quality using: /api/v1/data-prep/scan-quality/{session_id}"
        }))
        
    except Exception as e:
        logger.error(f"Error fetching dataset from Iceberg: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching dataset: {str(e)}")

@router.get("/scan-quality/{session_id}")
async def scan_data_quality(session_id: str):
    """🔍 AI Scans the Data - Step 2: AI analyzes your data for quality issues"""
    try:
        if session_id not in data_prep_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = data_prep_sessions[session_id]["current_dataset"]
        
        # Detect data quality issues
        issues = detect_data_quality_issues(df)
        
        # Store issues in session
        data_prep_sessions[session_id]["quality_issues"] = issues
        
        # Categorize issues by severity and type
        issue_summary = {
            "total_issues": len(issues),
            "by_severity": {
                "high": len([i for i in issues if i["severity"] == "high"]),
                "medium": len([i for i in issues if i["severity"] == "medium"]),
                "low": len([i for i in issues if i["severity"] == "low"])
            },
            "by_type": {}
        }
        
        for issue in issues:
            issue_type = issue["issue_type"]
            if issue_type not in issue_summary["by_type"]:
                issue_summary["by_type"][issue_type] = 0
            issue_summary["by_type"][issue_type] += 1
        
        return JSONResponse(content=convert_to_serializable({
            "status": "success",
            "message": "Data quality scan completed",
            "session_id": session_id,
            "scan_results": {
                "summary": issue_summary,
                "issues": issues,
                "recommendations": [
                    "Address high-severity issues first",
                    "Review missing value patterns",
                    "Check data type consistency"
                ]
            },
            "next_step": f"Get cleaning suggestions: /api/v1/data-prep/cleaning-suggestions/{session_id}"
        }))
        
    except Exception as e:
        logger.error(f"Error scanning data quality: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error scanning data quality: {str(e)}")

@router.get("/cleaning-suggestions/{session_id}")
async def get_cleaning_suggestions(session_id: str):
    """🧹 Auto-Generated Cleaning Suggestions - Step 3: AI provides specific cleaning recommendations"""
    try:
        if session_id not in data_prep_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = data_prep_sessions[session_id]
        df = session_data["current_dataset"]
        issues = session_data["quality_issues"]
        
        if not issues:
            raise HTTPException(status_code=400, detail="No quality issues found. Please scan data quality first.")
        
        # Create suggestions from detected issues
        suggestions = []
        for i, issue in enumerate(issues):
            suggestions.append({
                "suggestion_id": f"suggestion_{i+1}",
                "priority": issue["severity"],
                "category": issue["issue_type"],
                "description": f"Fix {issue['issue_type']} in {issue['column']}",
                "affected_columns": [issue["column"]] if issue["column"] != "all_columns" else [],
                "code_snippet": issue["code_snippet"],
                "preview_sample": {
                    "before": convert_to_serializable(df.head(3).to_dict(orient='records')),
                    "after": "Preview not generated yet"
                }
            })
        
        # Store suggestions in session
        data_prep_sessions[session_id]["cleaning_suggestions"] = suggestions
        
        return JSONResponse(content=convert_to_serializable({
            "status": "success",
            "message": "Cleaning suggestions generated",
            "session_id": session_id,
            "suggestions": suggestions,
            "summary": {
                "total_suggestions": len(suggestions),
                "high_priority": len([s for s in suggestions if s["priority"] == "high"]),
                "medium_priority": len([s for s in suggestions if s["priority"] == "medium"]),
                "low_priority": len([s for s in suggestions if s["priority"] == "low"])
            },
            "next_steps": [
                "Review and apply suggestions to clean data",
                "Download cleaned dataset when ready"
            ]
        }))
        
    except Exception as e:
        logger.error(f"Error generating cleaning suggestions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating cleaning suggestions: {str(e)}")

@router.get("/health")
async def data_prep_health():
    """Health check for Data Preparation service"""
    return JSONResponse(content={
        "status": "healthy",
        "service": "AI-powered Data Preparation with Iceberg Integration",
        "timestamp": datetime.now().isoformat(),
        "data_source": "Apache Iceberg",
        "active_sessions": len(data_prep_sessions)
    })

logger.info("✅ AI-powered Data Preparation FastAPI router with Iceberg integration created successfully")
