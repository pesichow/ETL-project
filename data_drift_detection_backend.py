"""
Data Drift Detection Backend - FastAPI APIRouter Module with Iceberg Integration
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import uuid
import time
import traceback
import logging
import tempfile
from datetime import datetime
from scipy.stats import ks_2samp, chi2_contingency, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create APIRouter
router = APIRouter(prefix="/api/v1/data-drift", tags=["Data Drift Detection"])

# Import the fetch_data function
from utils.fetch_data import fetch_data

# Initialize Azure OpenAI client
client = None
try:
    from openai import AzureOpenAI
    import os
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-04-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    logger.info("✅ Azure OpenAI client initialized successfully")
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize Azure OpenAI client: {str(e)}")
    client = None

# In-memory data store (in production, use Redis or database)
drift_analysis_store = {}
monitoring_sessions = {}

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models for request/response
class DriftAnalysisRequest(BaseModel):
    job_id: str
    table_names: List[str]
    selected_columns: List[str]
    split_method: Optional[str] = "temporal"  # temporal, random
    split_ratio: Optional[float] = 0.7
    drift_threshold: Optional[float] = 0.05
    enable_realtime: Optional[bool] = True
    advanced_metrics: Optional[bool] = True
    limit: Optional[int] = 10000

class ColumnDriftResult(BaseModel):
    column_name: str
    drift_score: float
    p_value: float
    drift_detected: bool
    test_method: str
    severity: str
    reference_stats: Dict[str, Any]
    current_stats: Dict[str, Any]

class DriftAnalysisResponse(BaseModel):
    success: bool
    drift_id: str
    job_id: str
    table_names: List[str]
    overall_drift_score: float
    drift_detected: bool
    column_results: List[ColumnDriftResult]
    visualizations: List[Dict[str, Any]]
    ai_insights: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    summary_stats: Dict[str, Any]
    processing_time: float
    reference_period: Dict[str, Any]
    current_period: Dict[str, Any]
    advanced_metrics: Optional[Dict[str, Any]]

class DatasetInfoRequest(BaseModel):
    job_id: str
    table_names: List[str]
    limit: int = 1000

class ColumnInfo(BaseModel):
    name: str
    type: str
    missing: int
    missing_pct: str
    unique_count: int
    drift_suitable: bool
    drift_type: str
    sample_values: List[str]

class DatasetInfoResponse(BaseModel):
    success: bool
    job_id: str
    table_names: List[str]
    rows: int
    columns: int
    columns_info: List[ColumnInfo]
    drift_readiness: str

class MonitoringRequest(BaseModel):
    job_id: str
    table_names: List[str]
    monitoring_columns: List[str]
    alert_threshold: Optional[float] = 0.1
    check_interval: Optional[int] = 60  # seconds
    window_size: Optional[int] = 1000  # number of samples
    limit: Optional[int] = 10000

class DriftAlert(BaseModel):
    alert_id: str
    job_id: str
    table_names: List[str]
    timestamp: str
    severity: str
    message: str
    affected_columns: List[str]
    drift_scores: Dict[str, float]

class DataDriftEngine:
    """Advanced Data Drift Detection Engine with Iceberg Integration"""
    
    def __init__(self):
        self.statistical_tests = {
            'numerical': self._ks_test,
            'categorical': self._chi_square_test,
            'mixed': self._psi_test
        }
        
        self.severity_thresholds = {
            'low': 0.05,
            'medium': 0.1,
            'high': 0.2,
            'critical': 0.3
        }
    
    def assess_drift_readiness(self, df: pd.DataFrame) -> str:
        """Assess dataset readiness for drift detection"""
        try:
            # Check data quality factors
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
            
            readiness_score = 100 - (missing_ratio * 30) - (constant_cols / len(df.columns) * 40)
            
            if readiness_score >= 90:
                return "Excellent"
            elif readiness_score >= 75:
                return "Good"
            elif readiness_score >= 60:
                return "Fair"
            else:
                return "Poor"
                
        except Exception:
            return "Unknown"
    
    def split_data(self, df: pd.DataFrame, method: str = "temporal", ratio: float = 0.7) -> tuple:
        """Split data into reference and current periods"""
        try:
            if method == "temporal":
                # Assume data is ordered chronologically
                split_idx = int(len(df) * ratio)
                reference_df = df.iloc[:split_idx].copy()
                current_df = df.iloc[split_idx:].copy()
            elif method == "random":
                # Random split
                reference_df = df.sample(frac=ratio, random_state=42)
                current_df = df.drop(reference_df.index)
            else:
                raise ValueError(f"Unknown split method: {method}")
            
            return reference_df, current_df
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            # Fallback to simple split
            split_idx = int(len(df) * ratio)
            return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    
    def _ks_test(self, ref_data: pd.Series, curr_data: pd.Series) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for numerical data"""
        try:
            statistic, p_value = ks_2samp(ref_data.dropna(), curr_data.dropna())
            
            return {
                'test_method': 'Kolmogorov-Smirnov',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'drift_score': float(statistic)
            }
        except Exception as e:
            logger.error(f"Error in KS test: {str(e)}")
            return {
                'test_method': 'Kolmogorov-Smirnov',
                'statistic': 0.0,
                'p_value': 1.0,
                'drift_score': 0.0,
                'error': str(e)
            }
    
    def _chi_square_test(self, ref_data: pd.Series, curr_data: pd.Series) -> Dict[str, Any]:
        """Perform Chi-square test for categorical data"""
        try:
            # Get value counts for both datasets
            ref_counts = ref_data.value_counts()
            curr_counts = curr_data.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
            
            # Create contingency table
            contingency_table = np.array([ref_aligned, curr_aligned])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate drift score (normalized chi-square)
            drift_score = chi2 / (len(ref_data) + len(curr_data))
            
            return {
                'test_method': 'Chi-square',
                'statistic': float(chi2),
                'p_value': float(p_value),
                'drift_score': float(drift_score),
                'degrees_of_freedom': int(dof)
            }
        except Exception as e:
            logger.error(f"Error in Chi-square test: {str(e)}")
            return {
                'test_method': 'Chi-square',
                'statistic': 0.0,
                'p_value': 1.0,
                'drift_score': 0.0,
                'error': str(e)
            }
    
    def _psi_test(self, ref_data: pd.Series, curr_data: pd.Series, bins: int = 10) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI)"""
        try:
            if pd.api.types.is_numeric_dtype(ref_data):
                # For numerical data, create bins
                _, bin_edges = np.histogram(ref_data.dropna(), bins=bins)
                ref_binned = pd.cut(ref_data, bins=bin_edges, include_lowest=True)
                curr_binned = pd.cut(curr_data, bins=bin_edges, include_lowest=True)
            else:
                # For categorical data, use categories as bins
                ref_binned = ref_data
                curr_binned = curr_data
            
            # Calculate proportions
            ref_props = ref_binned.value_counts(normalize=True, dropna=False)
            curr_props = curr_binned.value_counts(normalize=True, dropna=False)
            
            # Align categories
            all_categories = set(ref_props.index) | set(curr_props.index)
            
            psi = 0
            for category in all_categories:
                ref_prop = ref_props.get(category, 1e-6)  # Small value to avoid log(0)
                curr_prop = curr_props.get(category, 1e-6)
                
                if ref_prop > 0 and curr_prop > 0:
                    psi += (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
            
            return {
                'test_method': 'Population Stability Index',
                'statistic': float(psi),
                'p_value': None,  # PSI doesn't have a p-value
                'drift_score': float(psi)
            }
        except Exception as e:
            logger.error(f"Error in PSI calculation: {str(e)}")
            return {
                'test_method': 'Population Stability Index',
                'statistic': 0.0,
                'p_value': None,
                'drift_score': 0.0,
                'error': str(e)
            }
    
    def detect_column_drift(self, ref_data: pd.Series, curr_data: pd.Series, 
                          column_name: str, threshold: float = 0.05) -> ColumnDriftResult:
        """Detect drift for a single column"""
        try:
            # Choose appropriate test based on data type
            if pd.api.types.is_numeric_dtype(ref_data):
                test_result = self._ks_test(ref_data, curr_data)
            else:
                test_result = self._chi_square_test(ref_data, curr_data)
            
            # Determine drift detection
            drift_score = test_result['drift_score']
            p_value = test_result.get('p_value', 0.0)
            
            if p_value is not None:
                drift_detected = p_value < threshold
            else:
                # For PSI, use direct threshold comparison
                drift_detected = drift_score > threshold
            
            # Determine severity
            severity = self._determine_severity(drift_score)
            
            # Calculate statistics
            ref_stats = self._calculate_column_stats(ref_data)
            curr_stats = self._calculate_column_stats(curr_data)
            
            return ColumnDriftResult(
                column_name=column_name,
                drift_score=drift_score,
                p_value=p_value if p_value is not None else 0.0,
                drift_detected=drift_detected,
                test_method=test_result['test_method'],
                severity=severity,
                reference_stats=ref_stats,
                current_stats=curr_stats
            )
            
        except Exception as e:
            logger.error(f"Error detecting drift for column {column_name}: {str(e)}")
            return ColumnDriftResult(
                column_name=column_name,
                drift_score=0.0,
                p_value=1.0,
                drift_detected=False,
                test_method="Error",
                severity="unknown",
                reference_stats={},
                current_stats={}
            )
    
    def _determine_severity(self, drift_score: float) -> str:
        """Determine drift severity based on score"""
        if drift_score >= self.severity_thresholds['critical']:
            return "critical"
        elif drift_score >= self.severity_thresholds['high']:
            return "high"
        elif drift_score >= self.severity_thresholds['medium']:
            return "medium"
        elif drift_score >= self.severity_thresholds['low']:
            return "low"
        else:
            return "none"
    
    def _calculate_column_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive column statistics"""
        try:
            if pd.api.types.is_numeric_dtype(series):
                return {
                    'count': int(len(series)),
                    'missing': int(series.isnull().sum()),
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'q25': float(series.quantile(0.25)),
                    'q75': float(series.quantile(0.75)),
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis())
                }
            else:
                value_counts = series.value_counts().head(10)
                return {
                    'count': int(len(series)),
                    'missing': int(series.isnull().sum()),
                    'unique_count': int(series.nunique()),
                    'most_common': {str(k): int(v) for k, v in value_counts.items()},
                    'entropy': float(entropy(series.value_counts(normalize=True)))
                }
        except Exception as e:
            logger.error(f"Error calculating column stats: {str(e)}")
            return {'count': int(len(series)), 'error': str(e)}
    
    def create_drift_visualizations(self, ref_df: pd.DataFrame, curr_df: pd.DataFrame, 
                                  columns: List[str]) -> List[Dict[str, Any]]:
        """Create drift visualization plots"""
        visualizations = []
        
        try:
            for col in columns[:6]:  # Limit to 6 visualizations
                if col not in ref_df.columns or col not in curr_df.columns:
                    continue
                
                plt.figure(figsize=(12, 6))
                
                if pd.api.types.is_numeric_dtype(ref_df[col]):
                    # Histogram comparison for numerical data
                    plt.subplot(1, 2, 1)
                    plt.hist(ref_df[col].dropna(), alpha=0.7, label='Reference', bins=30, density=True)
                    plt.hist(curr_df[col].dropna(), alpha=0.7, label='Current', bins=30, density=True)
                    plt.title(f'{col} - Distribution Comparison')
                    plt.xlabel(col)
                    plt.ylabel('Density')
                    plt.legend()
                    
                    # Box plot comparison
                    plt.subplot(1, 2, 2)
                    data_to_plot = [ref_df[col].dropna(), curr_df[col].dropna()]
                    plt.boxplot(data_to_plot, labels=['Reference', 'Current'])
                    plt.title(f'{col} - Box Plot Comparison')
                    plt.ylabel(col)
                    
                else:
                    # Bar chart comparison for categorical data
                    ref_counts = ref_df[col].value_counts().head(10)
                    curr_counts = curr_df[col].value_counts().head(10)
                    
                    categories = list(set(ref_counts.index) | set(curr_counts.index))
                    ref_values = [ref_counts.get(cat, 0) for cat in categories]
                    curr_values = [curr_counts.get(cat, 0) for cat in categories]
                    
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    plt.bar(x - width/2, ref_values, width, label='Reference', alpha=0.7)
                    plt.bar(x + width/2, curr_values, width, label='Current', alpha=0.7)
                    plt.title(f'{col} - Category Distribution Comparison')
                    plt.xlabel('Categories')
                    plt.ylabel('Count')
                    plt.xticks(x, categories, rotation=45)
                    plt.legend()
                
                plt.tight_layout()
                
                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                visualizations.append({
                    'column': col,
                    'type': 'distribution_comparison',
                    'plot': plot_data,
                    'description': f'Distribution comparison for {col}'
                })
                
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
        
        return visualizations

# Initialize the drift detection engine
drift_engine = DataDriftEngine()

def generate_ai_insights(drift_results: List[ColumnDriftResult], overall_score: float, 
                        job_id: str, table_names: List[str]) -> List[Dict[str, Any]]:
    """Generate AI-powered insights for drift analysis"""
    if not client:
        return generate_fallback_insights(drift_results, overall_score, job_id, table_names)
    
    try:
        # Prepare drift summary for AI
        drift_summary = {
            'overall_drift_score': overall_score,
            'total_columns': len(drift_results),
            'drifted_columns': len([r for r in drift_results if r.drift_detected]),
            'high_severity_columns': len([r for r in drift_results if r.severity in ['high', 'critical']]),
            'column_details': [
                {
                    'column': r.column_name,
                    'drift_score': r.drift_score,
                    'severity': r.severity,
                    'test_method': r.test_method
                }
                for r in drift_results if r.drift_detected
            ]
        }
        
        prompt = f"""
        You are an expert in data drift detection and ML model monitoring. Analyze the drift detection results and provide strategic insights.
        
        Drift Analysis Summary:
        - Job ID: {job_id}
        - Source Tables: {', '.join(table_names)}
        - Overall Drift Score: {overall_score:.3f}
        - Total Columns Analyzed: {drift_summary['total_columns']}
        - Columns with Drift: {drift_summary['drifted_columns']}
        - High Severity Drifts: {drift_summary['high_severity_columns']}
        
        Detailed Results: {json.dumps(drift_summary['column_details'], indent=2)}
        
        Provide 5-6 strategic insights covering:
        1. Drift severity assessment and implications
        2. Root cause analysis and potential triggers
        3. Model performance impact predictions
        4. Immediate action recommendations
        5. Long-term monitoring strategy
        6. Business impact and risk assessment
        
        Format as JSON with insights array containing objects with title, description, category, priority, and actionable fields.
        """
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are an expert in data drift detection and ML monitoring. Provide strategic insights in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        return ai_response.get('insights', [])
        
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        return generate_fallback_insights(drift_results, overall_score, job_id, table_names)

def generate_fallback_insights(drift_results: List[ColumnDriftResult], overall_score: float,
                              job_id: str, table_names: List[str]) -> List[Dict[str, Any]]:
    """Generate fallback insights when AI is not available"""
    insights = []
    
    drifted_columns = [r for r in drift_results if r.drift_detected]
    high_severity = [r for r in drift_results if r.severity in ['high', 'critical']]
    
    # Overall assessment
    if overall_score > 0.2:
        insights.append({
            'title': 'Significant Data Drift Detected',
            'description': f'Overall drift score of {overall_score:.3f} indicates significant changes in data distribution across Iceberg tables.',
            'category': 'Assessment',
            'priority': 'High',
            'actionable': True
        })
    elif overall_score > 0.1:
        insights.append({
            'title': 'Moderate Data Drift Detected',
            'description': f'Moderate drift detected with score {overall_score:.3f}. Monitor closely and consider model retraining.',
            'category': 'Assessment',
            'priority': 'Medium',
            'actionable': True
        })
    
    # Column-specific insights
    if high_severity:
        insights.append({
            'title': 'Critical Columns Require Immediate Attention',
            'description': f'High severity drift detected in {len(high_severity)} columns: {", ".join([r.column_name for r in high_severity[:3]])}',
            'category': 'Critical',
            'priority': 'High',
            'actionable': True
        })
    
    # Model impact
    if len(drifted_columns) > len(drift_results) * 0.3:
        insights.append({
            'title': 'Model Performance at Risk',
            'description': f'{len(drifted_columns)} out of {len(drift_results)} features show drift. Model retraining recommended.',
            'category': 'Model Impact',
            'priority': 'High',
            'actionable': True
        })
    
    # Iceberg-specific insight
    insights.append({
        'title': 'Iceberg Data Lake Monitoring Active',
        'description': f'Drift detection successfully integrated with Iceberg tables {table_names}. Consider setting up automated monitoring.',
        'category': 'Infrastructure',
        'priority': 'Medium',
        'actionable': True
    })
    
    # Recommendations
    insights.append({
        'title': 'Implement Continuous Monitoring',
        'description': 'Set up automated drift detection with alerts to catch data quality issues early in your ML pipeline.',
        'category': 'Recommendation',
        'priority': 'Medium',
        'actionable': True
    })
    
    return insights

def perform_data_drift_analysis(df: pd.DataFrame, selected_columns: List[str], 
                               split_method: str, split_ratio: float, threshold: float,
                               data_source: str, advanced_metrics: bool = True) -> Dict[str, Any]:
    """Perform comprehensive data drift analysis"""
    try:
        logger.info(f"🔍 Starting drift analysis on {len(selected_columns)} columns")
        
        # Split data into reference and current periods
        ref_df, curr_df = drift_engine.split_data(df, split_method, split_ratio)
        
        logger.info(f"📊 Reference period: {len(ref_df)} rows, Current period: {len(curr_df)} rows")
        
        # Analyze each column for drift
        column_results = []
        drift_scores = []
        
        for col in selected_columns:
            if col in ref_df.columns and col in curr_df.columns:
                result = drift_engine.detect_column_drift(
                    ref_df[col], curr_df[col], col, threshold
                )
                column_results.append(result)
                drift_scores.append(result.drift_score)
        
        # Calculate overall drift score
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        drift_detected = any(r.drift_detected for r in column_results)
        
        # Create visualizations
        visualizations = drift_engine.create_drift_visualizations(ref_df, curr_df, selected_columns)
        
        # Generate AI insights
        ai_insights = generate_ai_insights(column_results, overall_drift_score, "iceberg_analysis", ["combined_tables"])
        
        # Generate recommendations
        recommendations = generate_drift_recommendations(column_results, overall_drift_score)
        
        # Calculate summary statistics
        summary_stats = {
            'total_columns_analyzed': len(selected_columns),
            'columns_with_drift': len([r for r in column_results if r.drift_detected]),
            'average_drift_score': overall_drift_score,
            'max_drift_score': max(drift_scores) if drift_scores else 0.0,
            'min_drift_score': min(drift_scores) if drift_scores else 0.0,
            'reference_period_size': len(ref_df),
            'current_period_size': len(curr_df),
            'split_method': split_method,
            'split_ratio': split_ratio
        }
        
        # Advanced metrics
        advanced_metrics_result = None
        if advanced_metrics:
            advanced_metrics_result = calculate_advanced_metrics(ref_df, curr_df, selected_columns)
        
        return {
            'overall_drift_score': overall_drift_score,
            'drift_detected': drift_detected,
            'column_results': [r.dict() for r in column_results],
            'visualizations': visualizations,
            'ai_insights': ai_insights,
            'recommendations': recommendations,
            'summary_stats': summary_stats,
            'reference_period': {
                'size': len(ref_df),
                'start_index': 0,
                'end_index': len(ref_df) - 1
            },
            'current_period': {
                'size': len(curr_df),
                'start_index': len(ref_df),
                'end_index': len(df) - 1
            },
            'advanced_metrics': advanced_metrics_result
        }
        
    except Exception as e:
        logger.error(f"Error in drift analysis: {str(e)}")
        raise

def generate_drift_recommendations(column_results: List[ColumnDriftResult], 
                                 overall_score: float) -> List[Dict[str, Any]]:
    """Generate actionable recommendations based on drift analysis"""
    recommendations = []
    
    try:
        high_drift_columns = [r for r in column_results if r.severity in ['high', 'critical']]
        medium_drift_columns = [r for r in column_results if r.severity == 'medium']
        
        # High priority recommendations
        if high_drift_columns:
            recommendations.append({
                'priority': 'High',
                'category': 'Immediate Action',
                'title': 'Address Critical Drift',
                'description': f'Investigate and address high drift in columns: {", ".join([r.column_name for r in high_drift_columns[:3]])}',
                'actionable': True
            })
        
        if overall_score > 0.15:
            recommendations.append({
                'priority': 'High',
                'category': 'Model Management',
                'title': 'Model Retraining Required',
                'description': f'Overall drift score of {overall_score:.3f} suggests model retraining is necessary',
                'actionable': True
            })
        
        # Medium priority recommendations
        if medium_drift_columns:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Monitoring',
                'title': 'Enhanced Monitoring',
                'description': f'Set up enhanced monitoring for {len(medium_drift_columns)} columns with medium drift',
                'actionable': True
            })
        
        # General recommendations
        recommendations.extend([
            {
                'priority': 'Medium',
                'category': 'Infrastructure',
                'title': 'Automated Drift Detection',
                'description': 'Implement automated drift detection in your ML pipeline',
                'actionable': True
            },
            {
                'priority': 'Low',
                'category': 'Documentation',
                'title': 'Document Drift Patterns',
                'description': 'Maintain documentation of drift patterns for future reference',
                'actionable': True
            }
        ])
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
    
    return recommendations

def calculate_advanced_metrics(ref_df: pd.DataFrame, curr_df: pd.DataFrame, 
                             columns: List[str]) -> Dict[str, Any]:
    """Calculate advanced drift metrics"""
    try:
        metrics = {}
        
        # Multivariate drift using PCA
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(ref_df[col])]
        if len(numeric_cols) >= 2:
            try:
                # Combine and standardize data
                ref_numeric = ref_df[numeric_cols].fillna(0)
                curr_numeric = curr_df[numeric_cols].fillna(0)
                
                scaler = StandardScaler()
                ref_scaled = scaler.fit_transform(ref_numeric)
                curr_scaled = scaler.transform(curr_numeric)
                
                # Apply PCA
                pca = PCA(n_components=min(3, len(numeric_cols)))
                ref_pca = pca.fit_transform(ref_scaled)
                curr_pca = pca.transform(curr_scaled)
                
                # Calculate multivariate drift
                multivariate_drift = 0
                for i in range(pca.n_components_):
                    stat, _ = ks_2samp(ref_pca[:, i], curr_pca[:, i])
                    multivariate_drift += stat
                
                multivariate_drift /= pca.n_components_
                
                metrics['multivariate_drift'] = {
                    'score': float(multivariate_drift),
                    'components_analyzed': int(pca.n_components_),
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
                }
                
            except Exception as e:
                logger.warning(f"Error calculating multivariate drift: {str(e)}")
        
        # Data quality metrics
        ref_quality = {
            'missing_ratio': ref_df[columns].isnull().sum().sum() / (len(ref_df) * len(columns)),
            'duplicate_ratio': ref_df.duplicated().sum() / len(ref_df)
        }
        
        curr_quality = {
            'missing_ratio': curr_df[columns].isnull().sum().sum() / (len(curr_df) * len(columns)),
            'duplicate_ratio': curr_df.duplicated().sum() / len(curr_df)
        }
        
        metrics['data_quality_drift'] = {
            'missing_ratio_change': curr_quality['missing_ratio'] - ref_quality['missing_ratio'],
            'duplicate_ratio_change': curr_quality['duplicate_ratio'] - ref_quality['duplicate_ratio'],
            'reference_quality': ref_quality,
            'current_quality': curr_quality
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating advanced metrics: {str(e)}")
        return {}

# API Routes
@router.get("/")
async def data_drift_info():
    """Get Data Drift Detection service information"""
    return {
        "service": "Data Drift Detection & Monitoring with Iceberg Integration",
        "version": "2.0.0",
        "description": "Advanced data drift detection and monitoring system for ML pipelines with Iceberg data sources",
        "endpoints": {
            "dataset_info": "/dataset-info",
            "analyze": "/analyze",
            "start_monitoring": "/start-monitoring",
            "download_results": "/download-results",
            "test_iceberg": "/test-iceberg"
        },
        "capabilities": [
            "Statistical Drift Detection (KS-test, Chi-square, PSI)",
            "Real-time Monitoring & Alerting",
            "Advanced Drift Metrics (Multivariate, Data Quality)",
            "AI-powered Insights & Recommendations",
            "Comprehensive Visualizations",
            "Iceberg Data Source Integration",
            "Automated Model Performance Protection"
        ],
        "statistical_tests": ["Kolmogorov-Smirnov", "Chi-square", "Population Stability Index"],
        "data_sources": ["Apache Iceberg"],
        "ai_enabled": client is not None
    }

@router.get("/test-iceberg")
async def test_iceberg_drift(job_id: str, table_names: List[str] = Query(...), limit: int = 10):
    """
    Test API to verify Iceberg data fetching for drift detection.
    Example: /test-iceberg?job_id=my_job&table_names=users&table_names=transactions
    """
    try:
        result = fetch_data(job_id, table_names, limit)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Iceberg test failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/dataset-info", response_model=DatasetInfoResponse)
async def get_dataset_info(request: DatasetInfoRequest):
    """Get dataset information for drift analysis from Iceberg data"""
    try:
        logger.info(f"🔍 Dataset info requested: job_id={request.job_id}, tables={request.table_names}")
        
        # Fetch data from Iceberg
        iceberg_result = fetch_data(request.job_id, request.table_names, request.limit)
        
        if "results" not in iceberg_result or not iceberg_result["results"]:
            raise HTTPException(status_code=400, detail="No data returned from Iceberg")
        
        # Combine data from all tables
        combined_df = None
        for item in iceberg_result["results"]:
            if item["row_data"]:
                df = pd.DataFrame(item["row_data"])
                df['_source_table'] = item["table_name"]
                
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        if combined_df is None or combined_df.empty:
            raise HTTPException(status_code=400, detail="No valid data found for analysis")
        
        logger.info(f"📊 Combined dataset shape: {combined_df.shape}")
        
        # Analyze columns for drift detection suitability
        columns_info = []
        for col in combined_df.columns:
            if col == '_source_table':
                continue
                
            col_type = str(combined_df[col].dtype)
            missing = combined_df[col].isna().sum()
            missing_pct = (missing / len(combined_df)) * 100
            unique_count = combined_df[col].nunique()
            
            # Determine drift detection suitability
            drift_suitable = True
            drift_type = "Statistical"
            
            if pd.api.types.is_numeric_dtype(combined_df[col]):
                if unique_count < 2:
                    drift_suitable = False
                    drift_type = "Constant"
                elif missing_pct > 90:
                    drift_suitable = False
                    drift_type = "Too many missing"
                else:
                    drift_type = "Numerical Distribution"
            elif pd.api.types.is_object_dtype(combined_df[col]):
                if unique_count > len(combined_df) * 0.9:
                    drift_suitable = False
                    drift_type = "High cardinality"
                else:
                    drift_type = "Categorical Distribution"
            
            # Get sample values
            sample_values = []
            try:
                non_null_values = combined_df[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(5, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception:
                sample_values = ["N/A"]
            
            column_info = ColumnInfo(
                name=col,
                type=col_type,
                missing=int(missing),
                missing_pct=f"{missing_pct:.2f}%",
                unique_count=int(unique_count),
                drift_suitable=drift_suitable,
                drift_type=drift_type,
                sample_values=sample_values
            )
            
            columns_info.append(column_info)
        
        # Assess drift readiness
        drift_readiness = drift_engine.assess_drift_readiness(combined_df)
        
        return DatasetInfoResponse(
            success=True,
            job_id=request.job_id,
            table_names=request.table_names,
            rows=len(combined_df),
            columns=len(combined_df.columns) - 1,  # Exclude _source_table
            columns_info=columns_info,
            drift_readiness=drift_readiness
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset info: {str(e)}")

@router.post("/analyze", response_model=DriftAnalysisResponse)
async def analyze_data_drift(request: DriftAnalysisRequest, background_tasks: BackgroundTasks):
    """Perform comprehensive data drift analysis on Iceberg data"""
    try:
        logger.info(f"🔍 Drift analysis requested: job_id={request.job_id}, tables={request.table_names}")
        
        if not request.selected_columns:
            raise HTTPException(status_code=400, detail="No columns selected for drift analysis")
        
        start_time = time.time()
        
        # Fetch data from Iceberg
        iceberg_result = fetch_data(request.job_id, request.table_names, request.limit)
        
        if "results" not in iceberg_result or not iceberg_result["results"]:
            raise HTTPException(status_code=400, detail="No data returned from Iceberg")
        
        # Combine data from all tables
        combined_df = None
        for item in iceberg_result["results"]:
            if item["row_data"]:
                df = pd.DataFrame(item["row_data"])
                df['_source_table'] = item["table_name"]
                
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        if combined_df is None or combined_df.empty:
            raise HTTPException(status_code=400, detail="No valid data found for analysis")
        
        # Validate selected columns
        missing_cols = [col for col in request.selected_columns if col not in combined_df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Columns not found: {missing_cols}")
        
        logger.info(f"📊 Starting drift analysis on {combined_df.shape} dataset")
        
        # Run CPU-intensive drift analysis in thread pool
        loop = asyncio.get_event_loop()
        drift_result = await loop.run_in_executor(
            executor,
            perform_data_drift_analysis,
            combined_df,
            request.selected_columns,
            request.split_method,
            request.split_ratio,
            request.drift_threshold,
            f"Iceberg Tables: {', '.join(request.table_names)}",
            request.advanced_metrics
        )
        
        processing_time = round(time.time() - start_time, 2)
        
        # Store drift analysis result
        drift_id = str(uuid.uuid4())
        drift_analysis_store[drift_id] = {
            'result': drift_result,
            'original_df': combined_df,
            'job_id': request.job_id,
            'table_names': request.table_names,
            'timestamp': datetime.now().isoformat(),
            'parameters': request.dict(),
            'processing_time': processing_time
        }
        
        # Convert column results to ColumnDriftResult objects
        column_results = [
            ColumnDriftResult(**result) for result in drift_result['column_results']
        ]
        
        logger.info(f"✅ Drift analysis completed in {processing_time}s")
        
        return DriftAnalysisResponse(
            success=True,
            drift_id=drift_id,
            job_id=request.job_id,
            table_names=request.table_names,
            overall_drift_score=drift_result['overall_drift_score'],
            drift_detected=drift_result['drift_detected'],
            column_results=column_results,
            visualizations=drift_result['visualizations'],
            ai_insights=drift_result['ai_insights'],
            recommendations=drift_result['recommendations'],
            summary_stats=drift_result['summary_stats'],
            processing_time=processing_time,
            reference_period=drift_result['reference_period'],
            current_period=drift_result['current_period'],
            advanced_metrics=drift_result.get('advanced_metrics')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in drift analysis: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")

@router.post("/start-monitoring")
async def start_realtime_monitoring(request: MonitoringRequest, background_tasks: BackgroundTasks):
    """Start real-time drift monitoring on Iceberg data"""
    try:
        logger.info(f"🔄 Starting monitoring: job_id={request.job_id}, tables={request.table_names}")
        
        # Fetch initial data from Iceberg
        iceberg_result = fetch_data(request.job_id, request.table_names, request.limit)
        
        if "results" not in iceberg_result or not iceberg_result["results"]:
            raise HTTPException(status_code=400, detail="No data returned from Iceberg")
        
        # Combine data from all tables
        combined_df = None
        for item in iceberg_result["results"]:
            if item["row_data"]:
                df = pd.DataFrame(item["row_data"])
                df['_source_table'] = item["table_name"]
                combined_df = df if combined_df is None else pd.concat([combined_df, df], ignore_index=True)
        
        if combined_df is None or combined_df.empty:
            raise HTTPException(status_code=400, detail="No valid data found for monitoring")
        
        # Create monitoring session
        monitoring_id = str(uuid.uuid4())
        monitoring_sessions[monitoring_id] = {
            'job_id': request.job_id,
            'table_names': request.table_names,
            'monitoring_columns': request.monitoring_columns,
            'alert_threshold': request.alert_threshold,
            'check_interval': request.check_interval,
            'window_size': request.window_size,
            'baseline_df': combined_df,
            'active': True,
            'alerts': [],
            'last_check': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Monitoring session {monitoring_id} created successfully")
        
        return {
            'success': True,
            'monitoring_id': monitoring_id,
            'job_id': request.job_id,
            'table_names': request.table_names,
            'monitoring_columns': request.monitoring_columns,
            'alert_threshold': request.alert_threshold,
            'check_interval': request.check_interval,
            'message': 'Real-time drift monitoring started on Iceberg data'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.post("/download-results")
async def download_drift_results(drift_id: str):
    """Download drift analysis results"""
    try:
        if drift_id not in drift_analysis_store:
            raise HTTPException(status_code=404, detail="Drift analysis not found")
        
        analysis_data = drift_analysis_store[drift_id]
        result = analysis_data['result']
        
        # Create comprehensive report
        report_data = {
            'analysis_id': drift_id,
            'job_id': analysis_data['job_id'],
            'table_names': analysis_data['table_names'],
            'timestamp': analysis_data['timestamp'],
            'processing_time': analysis_data['processing_time'],
            'overall_drift_score': result['overall_drift_score'],
            'drift_detected': result['drift_detected'],
            'summary_stats': result['summary_stats'],
            'column_results': result['column_results'],
            'ai_insights': result['ai_insights'],
            'recommendations': result['recommendations'],
            'advanced_metrics': result.get('advanced_metrics', {})
        }
        
        # Create temporary file
        temp_filename = f"drift_analysis_{drift_id[:8]}_{int(time.time())}.json"
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save report to JSON
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return FileResponse(
            temp_path,
            media_type='application/json',
            filename=temp_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.get("/health")
async def drift_detection_health():
    """Health check for Data Drift Detection service"""
    return {
        "status": "healthy",
        "service": "Data Drift Detection & Monitoring with Iceberg Integration",
        "timestamp": datetime.now().isoformat(),
        "data_source": "Apache Iceberg",
        "features": [
            "Statistical Drift Detection",
            "Real-time Monitoring",
            "AI-powered Insights",
            "Advanced Metrics",
            "Comprehensive Visualizations",
            "Iceberg Integration"
        ],
        "active_analyses": len(drift_analysis_store),
        "active_monitoring_sessions": len(monitoring_sessions),
        "ai_enabled": client is not None
    }

logger.info("✅ Data Drift Detection FastAPI router with Iceberg integration created successfully")
