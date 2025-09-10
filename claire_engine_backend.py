from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
import io
import json
import os
import uuid
import base64
import re
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import networkx as nx
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/claire", tags=["CLAIRE Engine (AI-powered metadata intelligence)"])

# Configure Azure OpenAI
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-04-01-preview")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Global storage for CLAIRE intelligence
claire_sessions = {}
metadata_registry = {}
lineage_graph = nx.DiGraph()
feature_recommendations = {}
user_patterns = defaultdict(list)
governance_rules = {}

# Pydantic models
class DatasetIngestion(BaseModel):
    dataset_name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = []
    source_type: str = "upload"  # upload, database, api
    business_context: Optional[str] = None

class ColumnInsightRequest(BaseModel):
    session_id: str
    column_name: str
    context: Optional[str] = None

class DataPrepSuggestionRequest(BaseModel):
    session_id: str
    target_column: Optional[str] = None
    use_case: str = "general"  # general, classification, regression, clustering

class LineageQuery(BaseModel):
    dataset_id: str
    direction: str = "both"  # upstream, downstream, both
    depth: int = 3

class FeatureRecommendationRequest(BaseModel):
    session_id: str
    target_column: str
    problem_type: str  # classification, regression
    max_features: int = 20

class GovernanceRule(BaseModel):
    rule_name: str
    rule_type: str  # data_quality, privacy, compliance
    conditions: Dict[str, Any]
    actions: List[str]

class UserAction(BaseModel):
    session_id: str
    action_type: str
    action_details: Dict[str, Any]
    timestamp: Optional[datetime] = None

# CLAIRE Intelligence Engine
class CLAIREEngine:
    def __init__(self):
        self.semantic_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[1-9]?[0-9]{7,15}$',
            'zip_code': r'^\d{5}(-\d{4})?$',
            'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'ip_address': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
            'date': r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$',
            'currency': r'^\$?[\d,]+\.?\d{0,2}$'
        }
        
        self.column_role_keywords = {
            'id': ['id', 'key', 'index', 'identifier', 'uuid'],
            'target': ['target', 'label', 'outcome', 'result', 'class'],
            'feature': ['feature', 'attribute', 'variable', 'predictor'],
            'timestamp': ['time', 'date', 'timestamp', 'created', 'updated'],
            'categorical': ['category', 'type', 'group', 'class', 'status'],
            'numerical': ['amount', 'count', 'value', 'score', 'rate', 'price']
        }
    
    def ingest_dataset_metadata(self, df: pd.DataFrame, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive metadata extraction and analysis"""
        metadata = {
            "basic_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                "creation_time": datetime.now().isoformat()
            },
            "data_quality": self._analyze_data_quality(df),
            "column_profiles": self._profile_columns(df),
            "relationships": self._detect_relationships(df),
            "semantic_types": self._detect_semantic_types(df),
            "statistical_summary": self._generate_statistical_summary(df),
            "data_lineage": {
                "source": dataset_info.get("source_type", "unknown"),
                "upstream_datasets": [],
                "downstream_datasets": [],
                "transformations": []
            },
            "business_context": dataset_info.get("business_context", ""),
            "tags": dataset_info.get("tags", []),
            "governance": self._apply_governance_rules(df)
        }
        
        return metadata
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        quality_metrics = {
            "completeness": {
                "overall": (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                "by_column": ((1 - df.isnull().sum() / len(df)) * 100).to_dict()
            },
            "uniqueness": {
                "duplicate_rows": df.duplicated().sum(),
                "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100,
                "unique_values_per_column": df.nunique().to_dict()
            },
            "consistency": self._check_consistency(df),
            "validity": self._check_validity(df),
            "accuracy": self._estimate_accuracy(df)
        }
        
        return quality_metrics
    
    def _profile_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed column profiles"""
        profiles = {}
        
        for column in df.columns:
            col_data = df[column]
            
            profile = {
                "name": column,
                "dtype": str(col_data.dtype),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(df)) * 100,
                "unique_count": col_data.nunique(),
                "unique_percentage": (col_data.nunique() / len(df)) * 100,
                "suggested_role": self._suggest_column_role(column, col_data),
                "data_quality_issues": self._identify_column_issues(col_data),
                "transformations_suggested": self._suggest_column_transformations(col_data)
            }
            
            # Type-specific analysis
            if col_data.dtype in ['int64', 'float64']:
                profile.update(self._analyze_numeric_column(col_data))
            elif col_data.dtype == 'object':
                profile.update(self._analyze_text_column(col_data))
            elif col_data.dtype == 'datetime64[ns]':
                profile.update(self._analyze_datetime_column(col_data))
            
            profiles[column] = profile
        
        return profiles
    
    def _detect_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect relationships between columns"""
        relationships = {
            "correlations": {},
            "dependencies": {},
            "hierarchies": [],
            "foreign_keys": []
        }
        
        # Numeric correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            relationships["correlations"] = {
                "matrix": corr_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(corr_matrix)
            }
        
        # Categorical dependencies
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 1:
            relationships["dependencies"] = self._analyze_categorical_dependencies(df[categorical_cols])
        
        # Detect hierarchical relationships
        relationships["hierarchies"] = self._detect_hierarchies(df)
        
        # Potential foreign key relationships
        relationships["foreign_keys"] = self._detect_foreign_keys(df)
        
        return relationships
    
    def _detect_semantic_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect semantic types using pattern matching"""
        semantic_types = {}
        
        for column in df.columns:
            col_data = df[column].dropna().astype(str)
            
            if len(col_data) == 0:
                semantic_types[column] = "unknown"
                continue
            
            # Sample a subset for pattern matching
            sample_data = col_data.sample(min(100, len(col_data)))
            
            detected_type = "generic"
            max_matches = 0
            
            for semantic_type, pattern in self.semantic_patterns.items():
                matches = sum(1 for value in sample_data if re.match(pattern, str(value)))
                match_percentage = matches / len(sample_data)
                
                if match_percentage > 0.8 and matches > max_matches:
                    detected_type = semantic_type
                    max_matches = matches
            
            semantic_types[column] = detected_type
        
        return semantic_types
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        summary = {
            "numeric_summary": {},
            "categorical_summary": {},
            "distribution_analysis": {},
            "outlier_analysis": {}
        }
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            # Distribution analysis
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    summary["distribution_analysis"][col] = {
                        "skewness": stats.skew(col_data),
                        "kurtosis": stats.kurtosis(col_data),
                        "normality_test": stats.normaltest(col_data)[1] if len(col_data) > 8 else None,
                        "distribution_type": self._identify_distribution(col_data)
                    }
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                summary["categorical_summary"][col] = {
                    "unique_values": len(value_counts),
                    "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                    "frequency": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    "cardinality": "high" if len(value_counts) > len(df) * 0.5 else "low",
                    "top_values": value_counts.head(10).to_dict()
                }
        
        # Outlier analysis
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                
                summary["outlier_analysis"][col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": (len(outliers) / len(col_data)) * 100,
                    "outlier_bounds": {"lower": Q1 - 1.5 * IQR, "upper": Q3 + 1.5 * IQR}
                }
        
        return summary
    
    def _suggest_column_role(self, column_name: str, col_data: pd.Series) -> str:
        """Suggest the role of a column based on name and data characteristics"""
        column_lower = column_name.lower()
        
        # Check for keyword matches
        for role, keywords in self.column_role_keywords.items():
            if any(keyword in column_lower for keyword in keywords):
                return role
        
        # Data-based suggestions
        unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
        
        if unique_ratio > 0.95:
            return "id"
        elif col_data.dtype in ['int64', 'float64']:
            return "numerical"
        elif col_data.dtype == 'object':
            if unique_ratio < 0.1:
                return "categorical"
            else:
                return "text"
        elif col_data.dtype == 'datetime64[ns]':
            return "timestamp"
        
        return "feature"
    
    def _identify_column_issues(self, col_data: pd.Series) -> List[Dict[str, Any]]:
        """Identify data quality issues in a column"""
        issues = []
        
        # Missing values
        null_percentage = (col_data.isnull().sum() / len(col_data)) * 100
        if null_percentage > 5:
            issues.append({
                "type": "missing_values",
                "severity": "high" if null_percentage > 50 else "medium",
                "description": f"{null_percentage:.1f}% missing values",
                "suggestion": "Consider imputation or removal"
            })
        
        # High cardinality for categorical
        if col_data.dtype == 'object':
            unique_ratio = col_data.nunique() / len(col_data)
            if unique_ratio > 0.5:
                issues.append({
                    "type": "high_cardinality",
                    "severity": "medium",
                    "description": f"High cardinality: {col_data.nunique()} unique values",
                    "suggestion": "Consider grouping or encoding strategies"
                })
        
        # Outliers for numeric
        if col_data.dtype in ['int64', 'float64']:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
            outlier_percentage = (len(outliers) / len(col_data)) * 100
            
            if outlier_percentage > 5:
                issues.append({
                    "type": "outliers",
                    "severity": "low",
                    "description": f"{outlier_percentage:.1f}% outliers detected",
                    "suggestion": "Review and consider capping or transformation"
                })
        
        return issues
    
    def _suggest_column_transformations(self, col_data: pd.Series) -> List[Dict[str, str]]:
        """Suggest transformations for a column"""
        suggestions = []
        
        if col_data.dtype in ['int64', 'float64']:
            # Numeric transformations
            if col_data.min() >= 0 and col_data.max() / col_data.min() > 10:
                suggestions.append({
                    "transformation": "log_transform",
                    "reason": "High variance, log transformation may help",
                    "code": f"np.log1p({col_data.name})"
                })
            
            if abs(stats.skew(col_data.dropna())) > 1:
                suggestions.append({
                    "transformation": "normalization",
                    "reason": "Skewed distribution detected",
                    "code": f"StandardScaler().fit_transform({col_data.name}.values.reshape(-1, 1))"
                })
        
        elif col_data.dtype == 'object':
            # Text transformations
            if col_data.nunique() / len(col_data) > 0.5:
                suggestions.append({
                    "transformation": "text_encoding",
                    "reason": "High cardinality text column",
                    "code": f"TfidfVectorizer().fit_transform({col_data.name})"
                })
            else:
                suggestions.append({
                    "transformation": "label_encoding",
                    "reason": "Categorical column with moderate cardinality",
                    "code": f"LabelEncoder().fit_transform({col_data.name})"
                })
        
        return suggestions
    
    def generate_ai_insights(self, metadata: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Generate AI-powered insights using Azure OpenAI"""
        try:
            prompt = f"""
            Analyze this dataset metadata and provide intelligent insights:
            
            Dataset Overview:
            - Shape: {metadata['basic_info']['shape']}
            - Columns: {len(metadata['basic_info']['columns'])}
            - Data Quality Score: {metadata['data_quality']['completeness']['overall']:.1f}%
            
            Column Profiles:
            {json.dumps(metadata['column_profiles'], indent=2)[:2000]}...
            
            Context: {context}
            
            Provide insights in JSON format:
            {{
                "overall_assessment": "Brief assessment of dataset quality and potential",
                "key_insights": ["insight1", "insight2", "insight3"],
                "recommended_actions": [
                    {{"action": "action_name", "priority": "high/medium/low", "reason": "explanation"}}
                ],
                "feature_engineering_suggestions": ["suggestion1", "suggestion2"],
                "potential_issues": ["issue1", "issue2"],
                "modeling_recommendations": {{
                    "suitable_algorithms": ["algorithm1", "algorithm2"],
                    "target_suggestions": ["column1", "column2"],
                    "feature_importance_hints": {{"column": "importance_reason"}}
                }}
            }}
            """
            
            response = openai.ChatCompletion.create(
                engine=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[
                    {"role": "system", "content": "You are CLAIRE, an AI metadata intelligence engine. Provide actionable insights for data scientists. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            ai_insights = json.loads(response.choices[0].message.content.strip())
            return ai_insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return self._generate_fallback_insights(metadata)
    
    def _generate_fallback_insights(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback insights when AI is not available"""
        return {
            "overall_assessment": "Dataset analysis completed using rule-based insights",
            "key_insights": [
                f"Dataset contains {metadata['basic_info']['shape'][0]} rows and {metadata['basic_info']['shape'][1]} columns",
                f"Overall data completeness: {metadata['data_quality']['completeness']['overall']:.1f}%",
                f"Found {len([col for col, profile in metadata['column_profiles'].items() if profile['suggested_role'] == 'numerical'])} numerical columns"
            ],
            "recommended_actions": [
                {"action": "address_missing_values", "priority": "high", "reason": "Improve data completeness"},
                {"action": "feature_engineering", "priority": "medium", "reason": "Enhance predictive power"}
            ],
            "feature_engineering_suggestions": [
                "Create interaction features between highly correlated variables",
                "Apply appropriate transformations to skewed distributions"
            ],
            "potential_issues": [
                "Check for data leakage in high-cardinality columns",
                "Validate outliers in numerical columns"
            ],
            "modeling_recommendations": {
                "suitable_algorithms": ["Random Forest", "Gradient Boosting", "Linear Models"],
                "target_suggestions": [col for col, profile in metadata['column_profiles'].items() if profile['suggested_role'] == 'target'],
                "feature_importance_hints": {"numerical_features": "Often provide good signal for ML models"}
            }
        }
    
    def create_visualization_charts(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Create visualization charts for metadata insights"""
        charts = {}
        
        # Data Quality Overview Chart
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Missing Values Heatmap
        plt.subplot(2, 3, 1)
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            plt.bar(range(len(missing_data)), missing_data.values)
            plt.xticks(range(len(missing_data)), missing_data.index, rotation=45)
            plt.title('Missing Values by Column')
            plt.ylabel('Count')
        else:
            plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Missing Values by Column')
        
        # Subplot 2: Data Types Distribution
        plt.subplot(2, 3, 2)
        dtype_counts = df.dtypes.value_counts()
        plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        plt.title('Data Types Distribution')
        
        # Subplot 3: Correlation Heatmap (for numeric columns)
        plt.subplot(2, 3, 3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
            plt.title('Correlation Matrix')
        else:
            plt.text(0.5, 0.5, 'Insufficient Numeric\nColumns for Correlation', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Correlation Matrix')
        
        # Subplot 4: Unique Values Distribution
        plt.subplot(2, 3, 4)
        unique_counts = df.nunique()
        plt.hist(unique_counts, bins=20, alpha=0.7)
        plt.xlabel('Unique Values Count')
        plt.ylabel('Number of Columns')
        plt.title('Unique Values Distribution')
        
        # Subplot 5: Data Quality Score
        plt.subplot(2, 3, 5)
        quality_score = metadata['data_quality']['completeness']['overall']
        colors = ['red' if quality_score < 70 else 'orange' if quality_score < 90 else 'green']
        plt.bar(['Data Quality'], [quality_score], color=colors)
        plt.ylim(0, 100)
        plt.ylabel('Completeness %')
        plt.title('Overall Data Quality')
        
        # Subplot 6: Column Roles Distribution
        plt.subplot(2, 3, 6)
        roles = [profile['suggested_role'] for profile in metadata['column_profiles'].values()]
        role_counts = Counter(roles)
        plt.bar(role_counts.keys(), role_counts.values())
        plt.xticks(rotation=45)
        plt.title('Suggested Column Roles')
        plt.ylabel('Count')
        
        plt.tight_layout()
        
        # Save chart as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        charts['metadata_overview'] = chart_data
        
        # Create lineage visualization if relationships exist
        if metadata['relationships']['correlations']:
            charts['lineage_diagram'] = self._create_lineage_diagram(metadata)
        
        return charts
    
    def _create_lineage_diagram(self, metadata: Dict[str, Any]) -> str:
        """Create data lineage diagram"""
        plt.figure(figsize=(10, 8))
        
        # Create a simple network graph showing column relationships
        G = nx.Graph()
        
        # Add nodes for each column
        columns = list(metadata['column_profiles'].keys())
        G.add_nodes_from(columns)
        
        # Add edges for high correlations
        correlations = metadata['relationships']['correlations'].get('high_correlations', [])
        for corr in correlations:
            if len(corr) >= 2:
                G.add_edge(corr[0], corr[1])
        
        # Draw the graph
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Color nodes by suggested role
        node_colors = []
        for node in G.nodes():
            role = metadata['column_profiles'][node]['suggested_role']
            if role == 'target':
                node_colors.append('red')
            elif role == 'id':
                node_colors.append('gray')
            elif role == 'numerical':
                node_colors.append('blue')
            elif role == 'categorical':
                node_colors.append('green')
            else:
                node_colors.append('orange')
        
        nx.draw(G, pos, node_color=node_colors, with_labels=True, 
                node_size=1000, font_size=8, font_weight='bold')
        
        plt.title('Column Relationship Network')
        plt.axis('off')
        
        # Save as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return chart_data
    
    # Helper methods for analysis
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency"""
        consistency_issues = []
        
        for column in df.select_dtypes(include=['object']).columns:
            # Check for inconsistent casing
            values = df[column].dropna().astype(str)
            if len(values) > 0:
                unique_values = values.unique()
                lower_values = [v.lower() for v in unique_values]
                if len(set(lower_values)) < len(unique_values):
                    consistency_issues.append({
                        "column": column,
                        "issue": "inconsistent_casing",
                        "severity": "medium"
                    })
        
        return {"issues": consistency_issues, "score": max(0, 100 - len(consistency_issues) * 10)}
    
    def _check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity"""
        validity_issues = []
        
        for column in df.columns:
            col_data = df[column].dropna()
            
            # Check for negative values in potentially positive-only columns
            if column.lower() in ['age', 'price', 'amount', 'count', 'quantity']:
                if col_data.dtype in ['int64', 'float64'] and (col_data < 0).any():
                    validity_issues.append({
                        "column": column,
                        "issue": "negative_values_in_positive_column",
                        "severity": "high"
                    })
        
        return {"issues": validity_issues, "score": max(0, 100 - len(validity_issues) * 15)}
    
    def _estimate_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate data accuracy"""
        # This is a simplified accuracy estimation
        # In practice, this would require domain knowledge or reference data
        
        accuracy_score = 85  # Default assumption
        accuracy_issues = []
        
        # Check for obvious accuracy issues
        for column in df.columns:
            if column.lower() in ['age'] and df[column].dtype in ['int64', 'float64']:
                if (df[column] > 150).any() or (df[column] < 0).any():
                    accuracy_issues.append({
                        "column": column,
                        "issue": "unrealistic_age_values",
                        "severity": "high"
                    })
                    accuracy_score -= 10
        
        return {"score": max(0, accuracy_score), "issues": accuracy_issues}
    
    def _analyze_numeric_column(self, col_data: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column specifics"""
        clean_data = col_data.dropna()
        
        if len(clean_data) == 0:
            return {"analysis": "no_data"}
        
        return {
            "min": float(clean_data.min()),
            "max": float(clean_data.max()),
            "mean": float(clean_data.mean()),
            "median": float(clean_data.median()),
            "std": float(clean_data.std()),
            "skewness": float(stats.skew(clean_data)),
            "kurtosis": float(stats.kurtosis(clean_data)),
            "zeros_count": int((clean_data == 0).sum()),
            "negative_count": int((clean_data < 0).sum())
        }
    
    def _analyze_text_column(self, col_data: pd.Series) -> Dict[str, Any]:
        """Analyze text column specifics"""
        clean_data = col_data.dropna().astype(str)
        
        if len(clean_data) == 0:
            return {"analysis": "no_data"}
        
        return {
            "avg_length": float(clean_data.str.len().mean()),
            "max_length": int(clean_data.str.len().max()),
            "min_length": int(clean_data.str.len().min()),
            "empty_strings": int((clean_data == "").sum()),
            "most_common": clean_data.value_counts().head(5).to_dict(),
            "contains_numbers": int(clean_data.str.contains(r'\d').sum()),
            "contains_special_chars": int(clean_data.str.contains(r'[^a-zA-Z0-9\s]').sum())
        }
    
    def _analyze_datetime_column(self, col_data: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column specifics"""
        clean_data = col_data.dropna()
        
        if len(clean_data) == 0:
            return {"analysis": "no_data"}
        
        return {
            "min_date": clean_data.min().isoformat(),
            "max_date": clean_data.max().isoformat(),
            "date_range_days": (clean_data.max() - clean_data.min()).days,
            "most_common_year": clean_data.dt.year.mode().iloc[0] if len(clean_data.dt.year.mode()) > 0 else None,
            "most_common_month": clean_data.dt.month.mode().iloc[0] if len(clean_data.dt.month.mode()) > 0 else None,
            "weekday_distribution": clean_data.dt.dayofweek.value_counts().to_dict()
        }
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[tuple]:
        """Find highly correlated column pairs"""
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr_value)
                    ))
        
        return high_corr
    
    def _analyze_categorical_dependencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dependencies between categorical columns"""
        dependencies = {}
        columns = df.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Calculate Cramér's V
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2 = stats.chi2_contingency(contingency_table)[0]
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                
                if cramers_v > 0.3:  # Threshold for significant dependency
                    dependencies[f"{col1}__{col2}"] = {
                        "cramers_v": float(cramers_v),
                        "dependency_strength": "strong" if cramers_v > 0.6 else "moderate"
                    }
        
        return dependencies
    
    def _detect_hierarchies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect hierarchical relationships in data"""
        hierarchies = []
        
        # Look for potential hierarchical patterns
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            values = df[col].dropna().astype(str)
            
            # Check if values contain hierarchical separators
            if values.str.contains(r'[/\\>-]').any():
                hierarchies.append({
                    "column": col,
                    "type": "path_hierarchy",
                    "separator_detected": True,
                    "example": values[values.str.contains(r'[/\\>-]')].iloc[0] if len(values[values.str.contains(r'[/\\>-]')]) > 0 else None
                })
        
        return hierarchies
    
    def _detect_foreign_keys(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential foreign key relationships"""
        foreign_keys = []
        
        # Look for columns that might be foreign keys
        for col in df.columns:
            if col.lower().endswith('_id') or col.lower().startswith('id_'):
                unique_ratio = df[col].nunique() / len(df)
                if 0.1 < unique_ratio < 0.9:  # Not too unique, not too repetitive
                    foreign_keys.append({
                        "column": col,
                        "confidence": "medium",
                        "unique_ratio": float(unique_ratio),
                        "suggested_reference": "external_table"
                    })
        
        return foreign_keys
    
    def _identify_distribution(self, data: pd.Series) -> str:
        """Identify the likely distribution of numeric data"""
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            return "insufficient_data"
        
        # Test for normal distribution
        _, p_normal = stats.normaltest(clean_data)
        if p_normal > 0.05:
            return "normal"
        
        # Test for uniform distribution
        _, p_uniform = stats.kstest(clean_data, 'uniform')
        if p_uniform > 0.05:
            return "uniform"
        
        # Check skewness
        skewness = stats.skew(clean_data)
        if abs(skewness) > 1:
            return "skewed"
        
        return "unknown"
    
    def _apply_governance_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply governance rules to the dataset"""
        governance_results = {
            "privacy_check": self._check_privacy_columns(df),
            "compliance_check": self._check_compliance_requirements(df),
            "data_retention": self._check_data_retention_requirements(df),
            "access_control": "standard"  # Placeholder
        }
        
        return governance_results
    
    def _check_privacy_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for columns that might contain PII"""
        pii_columns = []
        
        for column in df.columns:
            column_lower = column.lower()
            
            # Check column names for PII indicators
            pii_indicators = ['email', 'phone', 'ssn', 'social', 'address', 'name', 'credit', 'card']
            if any(indicator in column_lower for indicator in pii_indicators):
                pii_columns.append({
                    "column": column,
                    "pii_type": "potential_pii",
                    "confidence": "high",
                    "recommendation": "Apply privacy protection"
                })
            
            # Check data patterns for PII
            if df[column].dtype == 'object':
                sample_data = df[column].dropna().astype(str).head(100)
                
                # Email pattern
                if sample_data.str.match(self.semantic_patterns['email']).any():
                    pii_columns.append({
                        "column": column,
                        "pii_type": "email",
                        "confidence": "high",
                        "recommendation": "Hash or mask email addresses"
                    })
                
                # Phone pattern
                elif sample_data.str.match(self.semantic_patterns['phone']).any():
                    pii_columns.append({
                        "column": column,
                        "pii_type": "phone",
                        "confidence": "high",
                        "recommendation": "Mask phone numbers"
                    })
        
        return {
            "pii_columns_detected": len(pii_columns),
            "details": pii_columns,
            "compliance_status": "review_required" if pii_columns else "compliant"
        }
    
    def _check_compliance_requirements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check compliance requirements"""
        return {
            "gdpr_compliance": "requires_review",
            "data_classification": "internal",
            "retention_policy": "standard_7_years"
        }
    
    def _check_data_retention_requirements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data retention requirements"""
        return {
            "retention_period": "7_years",
            "archival_required": True,
            "deletion_schedule": "automatic"
        }

# Initialize CLAIRE Engine
claire_engine = CLAIREEngine()

# API Endpoints

@router.get("/")
async def get_claire_info():
    """Get information about the CLAIRE Engine service"""
    return {
        "service": "CLAIRE Engine (AI-powered metadata intelligence)",
        "description": "Intelligent metadata management and AI-driven data insights",
        "version": "1.0.0",
        "capabilities": [
            "🔍 Automated Metadata Capture",
            "🧠 AI-powered Column Insights",
            "🔧 Smart Data Prep Suggestions",
            "🌐 Data Lineage Tracking",
            "💡 Feature Recommendations",
            "📝 Auto-Documentation",
            "🛡️ Governance & Compliance",
            "📊 Interactive Visualizations",
            "🤖 Continuous Learning"
        ],
        "workflow": [
            "1. Ingest dataset and capture comprehensive metadata",
            "2. AI analyzes columns and provides intelligent insights",
            "3. Generate data preparation recommendations",
            "4. Track data lineage and relationships",
            "5. Provide feature engineering suggestions",
            "6. Apply governance rules and compliance checks",
            "7. Create interactive visualizations",
            "8. Learn from user actions and improve"
        ],
        "endpoints": {
            "ingest": "/api/v1/claire/ingest-dataset",
            "insights": "/api/v1/claire/column-insights/{session_id}",
            "suggestions": "/api/v1/claire/data-prep-suggestions",
            "lineage": "/api/v1/claire/data-lineage",
            "features": "/api/v1/claire/feature-recommendations",
            "governance": "/api/v1/claire/governance-check/{session_id}",
            "visualizations": "/api/v1/claire/visualizations/{session_id}",
            "learn": "/api/v1/claire/learn-from-action",
            "download": "/api/v1/claire/download-report/{session_id}"
        }
    }

@router.post("/ingest-dataset")
async def ingest_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    description: str = Form(None),
    business_context: str = Form(None),
    tags: str = Form("[]")
):
    """🔍 Ingest Dataset and Capture Metadata - Step 1: Upload dataset and extract comprehensive metadata"""
    try:
        # Generate session ID
        session_id = f"claire_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Parse tags
        try:
            tags_list = json.loads(tags) if tags else []
        except:
            tags_list = []
        
        # Read file content
        content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel files.")
        
        # Prepare dataset info
        dataset_info = {
            "dataset_name": dataset_name,
            "description": description,
            "business_context": business_context,
            "tags": tags_list,
            "source_type": "upload",
            "filename": file.filename
        }
        
        # Extract comprehensive metadata using CLAIRE
        metadata = claire_engine.ingest_dataset_metadata(df, dataset_info)
        
        # Generate AI insights
        ai_insights = claire_engine.generate_ai_insights(metadata, business_context or "")
        
        # Store in CLAIRE sessions
        claire_sessions[session_id] = {
            "session_id": session_id,
            "dataset": df,
            "metadata": metadata,
            "ai_insights": ai_insights,
            "dataset_info": dataset_info,
            "created_at": datetime.now().isoformat(),
            "user_actions": []
        }
        
        # Register in metadata registry
        metadata_registry[session_id] = {
            "name": dataset_name,
            "session_id": session_id,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        
        # Add to lineage graph
        lineage_graph.add_node(session_id, **dataset_info)
        
        return {
            "status": "success",
            "message": f"Dataset '{dataset_name}' ingested and analyzed by CLAIRE",
            "session_id": session_id,
            "metadata_summary": {
                "shape": metadata["basic_info"]["shape"],
                "columns": len(metadata["basic_info"]["columns"]),
                "data_quality_score": metadata["data_quality"]["completeness"]["overall"],
                "semantic_types_detected": len(set(metadata["semantic_types"].values())),
                "relationships_found": len(metadata["relationships"]["correlations"].get("high_correlations", [])),
                "governance_issues": len(metadata["governance"]["privacy_check"]["details"])
            },
            "ai_insights": ai_insights,
            "next_steps": [
                f"Get detailed column insights: /api/v1/claire/column-insights/{session_id}",
                f"View data prep suggestions: /api/v1/claire/data-prep-suggestions",
                f"Explore visualizations: /api/v1/claire/visualizations/{session_id}"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error ingesting dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting dataset: {str(e)}")

@router.get("/column-insights/{session_id}")
async def get_column_insights(session_id: str, column_name: str = None):
    """🧠 Auto-Profiling and Column Insights - Step 2: Get AI-driven insights for columns"""
    try:
        if session_id not in claire_sessions:
            raise HTTPException(status_code=404, detail="CLAIRE session not found")
        
        session_data = claire_sessions[session_id]
        metadata = session_data["metadata"]
        
        if column_name:
            # Get insights for specific column
            if column_name not in metadata["column_profiles"]:
                raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found")
            
            column_profile = metadata["column_profiles"][column_name]
            
            # Generate AI-specific insights for this column
            column_insights = {
                "column_name": column_name,
                "profile": column_profile,
                "semantic_type": metadata["semantic_types"].get(column_name, "unknown"),
                "relationships": {
                    "correlations": [
                        corr for corr in metadata["relationships"]["correlations"].get("high_correlations", [])
                        if column_name in corr[:2]
                    ]
                },
                "recommendations": {
                    "role_suggestion": column_profile["suggested_role"],
                    "transformations": column_profile["transformations_suggested"],
                    "quality_issues": column_profile["data_quality_issues"]
                }
            }
            
            return {
                "status": "success",
                "session_id": session_id,
                "column_insights": column_insights
            }
        else:
            # Get insights for all columns
            all_insights = {}
            
            for col_name, profile in metadata["column_profiles"].items():
                all_insights[col_name] = {
                    "suggested_role": profile["suggested_role"],
                    "data_quality_score": 100 - len(profile["data_quality_issues"]) * 10,
                    "semantic_type": metadata["semantic_types"].get(col_name, "unknown"),
                    "transformation_count": len(profile["transformations_suggested"]),
                    "issues_count": len(profile["data_quality_issues"])
                }
            
            return {
                "status": "success",
                "session_id": session_id,
                "all_column_insights": all_insights,
                "summary": {
                    "total_columns": len(all_insights),
                    "columns_with_issues": len([col for col, insights in all_insights.items() if insights["issues_count"] > 0]),
                    "high_quality_columns": len([col for col, insights in all_insights.items() if insights["data_quality_score"] > 80]),
                    "semantic_types_distribution": dict(Counter(metadata["semantic_types"].values()))
                }
            }
        
    except Exception as e:
        logger.error(f"Error getting column insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting column insights: {str(e)}")

@router.post("/data-prep-suggestions")
async def get_data_prep_suggestions(request: DataPrepSuggestionRequest):
    """🔧 Suggest Data Prep Steps Using Metadata - Step 3: Get intelligent data preparation recommendations"""
    try:
        if request.session_id not in claire_sessions:
            raise HTTPException(status_code=404, detail="CLAIRE session not found")
        
        session_data = claire_sessions[request.session_id]
        metadata = session_data["metadata"]
        df = session_data["dataset"]
        
        # Generate comprehensive data prep suggestions
        suggestions = {
            "data_quality_improvements": [],
            "feature_engineering": [],
            "transformations": [],
            "encoding_strategies": [],
            "scaling_recommendations": []
        }
        
        # Data quality improvements
        for col_name, profile in metadata["column_profiles"].items():
            for issue in profile["data_quality_issues"]:
                suggestions["data_quality_improvements"].append({
                    "column": col_name,
                    "issue": issue["type"],
                    "severity": issue["severity"],
                    "suggestion": issue["suggestion"],
                    "priority": "high" if issue["severity"] == "high" else "medium"
                })
        
        # Feature engineering suggestions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) > 1:
            suggestions["feature_engineering"].extend([
                {
                    "type": "interaction_features",
                    "description": "Create interaction features between highly correlated numeric columns",
                    "columns": numeric_cols[:3],
                    "code": f"df['{numeric_cols[0]}_x_{numeric_cols[1]}'] = df['{numeric_cols[0]}'] * df['{numeric_cols[1]}']"
                },
                {
                    "type": "polynomial_features",
                    "description": "Create polynomial features for non-linear relationships",
                    "columns": numeric_cols[:2],
                    "code": f"df['{numeric_cols[0]}_squared'] = df['{numeric_cols[0]}'] ** 2"
                }
            ])
        
        # Transformation suggestions
        for col_name, profile in metadata["column_profiles"].items():
            suggestions["transformations"].extend(profile["transformations_suggested"])
        
        # Encoding strategies
        for col in categorical_cols:
            cardinality = df[col].nunique()
            if cardinality < 10:
                suggestions["encoding_strategies"].append({
                    "column": col,
                    "strategy": "one_hot_encoding",
                    "reason": f"Low cardinality ({cardinality} unique values)",
                    "code": f"pd.get_dummies(df['{col}'], prefix='{col}')"
                })
            else:
                suggestions["encoding_strategies"].append({
                    "column": col,
                    "strategy": "target_encoding",
                    "reason": f"High cardinality ({cardinality} unique values)",
                    "code": f"# Target encoding for {col} (requires target column)"
                })
        
        # Scaling recommendations
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                if col_data.std() > col_data.mean():
                    suggestions["scaling_recommendations"].append({
                        "column": col,
                        "strategy": "standard_scaling",
                        "reason": "High variance detected",
                        "code": f"StandardScaler().fit_transform(df[['{col}']])"
                    })
                elif col_data.min() >= 0:
                    suggestions["scaling_recommendations"].append({
                        "column": col,
                        "strategy": "min_max_scaling",
                        "reason": "Non-negative values, suitable for min-max scaling",
                        "code": f"MinMaxScaler().fit_transform(df[['{col}']])"
                    })
        
        # Prioritize suggestions
        all_suggestions = []
        
        # Add all suggestions with priorities
        for category, items in suggestions.items():
            for item in items:
                all_suggestions.append({
                    "category": category,
                    "priority": item.get("priority", "medium"),
                    "suggestion": item,
                    "estimated_impact": "high" if category == "data_quality_improvements" else "medium"
                })
        
        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        all_suggestions.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "use_case": request.use_case,
            "target_column": request.target_column,
            "suggestions_by_category": suggestions,
            "prioritized_suggestions": all_suggestions[:20],  # Top 20 suggestions
            "summary": {
                "total_suggestions": len(all_suggestions),
                "high_priority": len([s for s in all_suggestions if s["priority"] == "high"]),
                "data_quality_issues": len(suggestions["data_quality_improvements"]),
                "feature_engineering_opportunities": len(suggestions["feature_engineering"])
            },
            "implementation_order": [
                "1. Address data quality improvements first",
                "2. Apply necessary transformations",
                "3. Implement encoding strategies",
                "4. Apply scaling if needed",
                "5. Create engineered features"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating data prep suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating data prep suggestions: {str(e)}")

@router.post("/data-lineage")
async def get_data_lineage(query: LineageQuery):
    """🌐 Understand Data Lineage & Impact - Step 4: Trace dataset relationships and dependencies"""
    try:
        if query.dataset_id not in metadata_registry:
            raise HTTPException(status_code=404, detail="Dataset not found in lineage graph")
        
        # Get lineage information
        lineage_info = {
            "dataset_id": query.dataset_id,
            "upstream_dependencies": [],
            "downstream_dependencies": [],
            "transformation_history": [],
            "impact_analysis": {}
        }
        
        # Get upstream dependencies
        if query.direction in ["upstream", "both"]:
            upstream_nodes = []
            try:
                # Get predecessors in the lineage graph
                predecessors = list(lineage_graph.predecessors(query.dataset_id))
                for pred in predecessors[:query.depth]:
                    if pred in metadata_registry:
                        upstream_nodes.append({
                            "dataset_id": pred,
                            "name": metadata_registry[pred]["name"],
                            "created_at": metadata_registry[pred]["created_at"],
                            "relationship_type": "source"
                        })
            except:
                pass
            
            lineage_info["upstream_dependencies"] = upstream_nodes
        
        # Get downstream dependencies
        if query.direction in ["downstream", "both"]:
            downstream_nodes = []
            try:
                # Get successors in the lineage graph
                successors = list(lineage_graph.successors(query.dataset_id))
                for succ in successors[:query.depth]:
                    if succ in metadata_registry:
                        downstream_nodes.append({
                            "dataset_id": succ,
                            "name": metadata_registry[succ]["name"],
                            "created_at": metadata_registry[succ]["created_at"],
                            "relationship_type": "derived"
                        })
            except:
                pass
            
            lineage_info["downstream_dependencies"] = downstream_nodes
        
        # Get transformation history
        if query.dataset_id in claire_sessions:
            session_data = claire_sessions[query.dataset_id]
            lineage_info["transformation_history"] = session_data.get("user_actions", [])
        
        # Impact analysis
        total_dependencies = len(lineage_info["upstream_dependencies"]) + len(lineage_info["downstream_dependencies"])
        lineage_info["impact_analysis"] = {
            "total_connected_datasets": total_dependencies,
            "impact_score": min(100, total_dependencies * 10),
            "change_risk": "high" if total_dependencies > 5 else "medium" if total_dependencies > 2 else "low",
            "recommendations": [
                "Test changes in development environment first",
                "Notify downstream consumers before making changes",
                "Implement gradual rollout for high-impact changes"
            ] if total_dependencies > 3 else [
                "Changes have limited impact",
                "Standard testing procedures apply"
            ]
        }
        
        # Create lineage visualization data
        lineage_graph_data = {
            "nodes": [
                {
                    "id": query.dataset_id,
                    "label": metadata_registry[query.dataset_id]["name"],
                    "type": "current",
                    "metadata": metadata_registry[query.dataset_id]
                }
            ],
            "edges": []
        }
        
        # Add upstream nodes and edges
        for upstream in lineage_info["upstream_dependencies"]:
            lineage_graph_data["nodes"].append({
                "id": upstream["dataset_id"],
                "label": upstream["name"],
                "type": "upstream"
            })
            lineage_graph_data["edges"].append({
                "source": upstream["dataset_id"],
                "target": query.dataset_id,
                "type": "data_flow"
            })
        
        # Add downstream nodes and edges
        for downstream in lineage_info["downstream_dependencies"]:
            lineage_graph_data["nodes"].append({
                "id": downstream["dataset_id"],
                "label": downstream["name"],
                "type": "downstream"
            })
            lineage_graph_data["edges"].append({
                "source": query.dataset_id,
                "target": downstream["dataset_id"],
                "type": "data_flow"
            })
        
        return {
            "status": "success",
            "lineage_info": lineage_info,
            "lineage_graph": lineage_graph_data,
            "query_parameters": {
                "dataset_id": query.dataset_id,
                "direction": query.direction,
                "depth": query.depth
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting data lineage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting data lineage: {str(e)}")

@router.post("/feature-recommendations")
async def get_feature_recommendations(request: FeatureRecommendationRequest):
    """💡 Provide Feature Recommendations for Modeling - Step 5: AI-driven feature selection and engineering"""
    try:
        if request.session_id not in claire_sessions:
            raise HTTPException(status_code=404, detail="CLAIRE session not found")
        
        session_data = claire_sessions[request.session_id]
        df = session_data["dataset"]
        metadata = session_data["metadata"]
        
        if request.target_column not in df.columns:
            raise HTTPException(status_code=404, detail=f"Target column '{request.target_column}' not found")
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != request.target_column]
        X = df[feature_columns]
        y = df[request.target_column]
        
        # Handle missing values for analysis
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].dtype in ['int64', 'float64']:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                X_clean[col] = X_clean[col].fillna(X_clean[col].mode().iloc[0] if len(X_clean[col].mode()) > 0 else 'Unknown')
        
        # Encode categorical variables for analysis
        le = LabelEncoder()
        for col in X_clean.select_dtypes(include=['object']).columns:
            X_clean[col] = le.fit_transform(X_clean[col].astype(str))
        
        # Encode target if categorical
        y_clean = y.copy()
        if y.dtype == 'object':
            y_clean = le.fit_transform(y.astype(str))
        
        # Calculate feature importance using mutual information
        if request.problem_type == "classification":
            mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)
        else:
            mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
        
        # Create feature recommendations
        feature_scores = list(zip(feature_columns, mi_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = {
            "high_signal_features": [],
            "redundant_features": [],
            "engineered_features": [],
            "feature_interactions": [],
            "feature_selection_strategy": {}
        }
        
        # High signal features (top features by mutual information)
        top_features = feature_scores[:min(request.max_features, len(feature_scores))]
        for feature, score in top_features:
            if score > 0.1:  # Threshold for significant features
                feature_profile = metadata["column_profiles"][feature]
                recommendations["high_signal_features"].append({
                    "feature": feature,
                    "importance_score": float(score),
                    "suggested_role": feature_profile["suggested_role"],
                    "data_quality": 100 - len(feature_profile["data_quality_issues"]) * 10,
                    "reasoning": f"High mutual information score ({score:.3f}) with target variable"
                })
        
        # Identify redundant features (highly correlated)
        correlations = metadata["relationships"]["correlations"].get("high_correlations", [])
        for corr in correlations:
            if len(corr) >= 3 and abs(corr[2]) > 0.9:
                recommendations["redundant_features"].append({
                    "feature_pair": [corr[0], corr[1]],
                    "correlation": float(corr[2]),
                    "recommendation": f"Consider removing one of these highly correlated features",
                    "suggested_action": f"Keep {corr[0]} (higher importance)" if corr[0] in [f[0] for f in top_features[:10]] else f"Keep {corr[1]}"
                })
        
        # Feature engineering suggestions
        numeric_features = [f for f in feature_columns if df[f].dtype in ['int64', 'float64']]
        categorical_features = [f for f in feature_columns if df[f].dtype == 'object']
        
        if len(numeric_features) >= 2:
            # Suggest ratio features
            for i, feat1 in enumerate(numeric_features[:3]):
                for feat2 in numeric_features[i+1:4]:
                    recommendations["engineered_features"].append({
                        "type": "ratio_feature",
                        "features": [feat1, feat2],
                        "new_feature": f"{feat1}_to_{feat2}_ratio",
                        "code": f"df['{feat1}_to_{feat2}_ratio'] = df['{feat1}'] / (df['{feat2}'] + 1e-8)",
                        "reasoning": "Ratio features can capture relative relationships"
                    })
        
        if len(categorical_features) >= 1 and len(numeric_features) >= 1:
            # Suggest aggregation features
            recommendations["engineered_features"].append({
                "type": "aggregation_feature",
                "features": [categorical_features[0], numeric_features[0]],
                "new_feature": f"{numeric_features[0]}_mean_by_{categorical_features[0]}",
                "code": f"df['{numeric_features[0]}_mean_by_{categorical_features[0]}'] = df.groupby('{categorical_features[0]}')['{numeric_features[0]}'].transform('mean')",
                "reasoning": "Group-based statistics can reveal patterns"
            })
        
        # Feature interactions
        top_feature_names = [f[0] for f in top_features[:5]]
        for i, feat1 in enumerate(top_feature_names):
            for feat2 in top_feature_names[i+1:]:
                if df[feat1].dtype in ['int64', 'float64'] and df[feat2].dtype in ['int64', 'float64']:
                    recommendations["feature_interactions"].append({
                        "features": [feat1, feat2],
                        "interaction_type": "multiplicative",
                        "new_feature": f"{feat1}_x_{feat2}",
                        "code": f"df['{feat1}_x_{feat2}'] = df['{feat1}'] * df['{feat2}']",
                        "expected_benefit": "medium"
                    })
        
        # Feature selection strategy
        recommendations["feature_selection_strategy"] = {
            "recommended_method": "mutual_information" if request.problem_type == "classification" else "f_regression",
            "target_feature_count": min(20, len([f for f in top_features if f[1] > 0.05])),
            "selection_criteria": [
                "Remove features with mutual information < 0.05",
                "Remove one feature from highly correlated pairs (r > 0.9)",
                "Prioritize features with high data quality scores",
                "Consider domain knowledge for final selection"
            ],
            "validation_approach": "cross_validation_with_feature_importance"
        }
        
        # Store recommendations for future reference
        feature_recommendations[request.session_id] = recommendations
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "target_column": request.target_column,
            "problem_type": request.problem_type,
            "feature_recommendations": recommendations,
            "summary": {
                "total_features_analyzed": len(feature_columns),
                "high_signal_features": len(recommendations["high_signal_features"]),
                "redundant_features": len(recommendations["redundant_features"]),
                "engineering_opportunities": len(recommendations["engineered_features"]),
                "interaction_suggestions": len(recommendations["feature_interactions"])
            },
            "implementation_priority": [
                "1. Remove or address redundant features",
                "2. Focus on high-signal features first",
                "3. Create engineered features with high expected benefit",
                "4. Test feature interactions",
                "5. Validate with cross-validation"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating feature recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating feature recommendations: {str(e)}")

@router.get("/governance-check/{session_id}")
async def get_governance_check(session_id: str):
    """🛡️ Drives Explainability and Governance - Step 6: Apply governance rules and compliance checks"""
    try:
        if session_id not in claire_sessions:
            raise HTTPException(status_code=404, detail="CLAIRE session not found")
        
        session_data = claire_sessions[session_id]
        metadata = session_data["metadata"]
        df = session_data["dataset"]
        
        governance_report = {
            "privacy_compliance": metadata["governance"]["privacy_check"],
            "data_quality_governance": {
                "overall_score": metadata["data_quality"]["completeness"]["overall"],
                "quality_gates": {
                    "completeness_threshold": 90,
                    "completeness_actual": metadata["data_quality"]["completeness"]["overall"],
                    "completeness_passed": metadata["data_quality"]["completeness"]["overall"] >= 90,
                    "uniqueness_check": metadata["data_quality"]["uniqueness"]["duplicate_percentage"] < 5,
                    "consistency_check": metadata["data_quality"]["consistency"]["score"] > 80
                }
            },
            "compliance_status": {
                "gdpr_compliance": metadata["governance"]["compliance_check"]["gdpr_compliance"],
                "data_classification": metadata["governance"]["compliance_check"]["data_classification"],
                "retention_policy": metadata["governance"]["data_retention"]["retention_period"]
            },
            "explainability_features": {
                "feature_interpretability": self._assess_feature_interpretability(df, metadata),
                "model_explainability_readiness": self._assess_explainability_readiness(metadata),
                "bias_detection": self._detect_potential_bias(df, metadata)
            },
            "governance_recommendations": []
        }
        
        # Generate governance recommendations
        if governance_report["privacy_compliance"]["pii_columns_detected"] > 0:
            governance_report["governance_recommendations"].append({
                "type": "privacy",
                "priority": "high",
                "recommendation": "Apply data masking or anonymization to PII columns",
                "affected_columns": [col["column"] for col in governance_report["privacy_compliance"]["details"]]
            })
        
        if governance_report["data_quality_governance"]["overall_score"] < 80:
            governance_report["governance_recommendations"].append({
                "type": "data_quality",
                "priority": "high",
                "recommendation": "Improve data quality before using in production models",
                "details": "Address missing values and data consistency issues"
            })
        
        if not governance_report["data_quality_governance"]["quality_gates"]["uniqueness_check"]:
            governance_report["governance_recommendations"].append({
                "type": "data_integrity",
                "priority": "medium",
                "recommendation": "Remove or investigate duplicate records",
                "details": f"Found {metadata['data_quality']['uniqueness']['duplicate_rows']} duplicate rows"
            })
        
        return {
            "status": "success",
            "session_id": session_id,
            "governance_report": governance_report,
            "compliance_summary": {
                "overall_compliance_score": self._calculate_compliance_score(governance_report),
                "critical_issues": len([r for r in governance_report["governance_recommendations"] if r["priority"] == "high"]),
                "recommendations_count": len(governance_report["governance_recommendations"]),
                "ready_for_production": governance_report["data_quality_governance"]["overall_score"] >= 80 and governance_report["privacy_compliance"]["pii_columns_detected"] == 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error performing governance check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing governance check: {str(e)}")

@router.get("/visualizations/{session_id}")
async def get_visualizations(session_id: str):
    """📊 Interactive Visualizations - Step 7: Create comprehensive data visualizations"""
    try:
        if session_id not in claire_sessions:
            raise HTTPException(status_code=404, detail="CLAIRE session not found")
        
        session_data = claire_sessions[session_id]
        df = session_data["dataset"]
        metadata = session_data["metadata"]
        
        # Generate visualizations using CLAIRE
        charts = claire_engine.create_visualization_charts(df, metadata)
        
        # Create additional specialized charts
        additional_charts = {}
        
        # Feature importance chart (if recommendations exist)
        if session_id in feature_recommendations:
            recommendations = feature_recommendations[session_id]
            if recommendations["high_signal_features"]:
                plt.figure(figsize=(10, 6))
                features = [f["feature"] for f in recommendations["high_signal_features"][:10]]
                scores = [f["importance_score"] for f in recommendations["high_signal_features"][:10]]
                
                plt.barh(features, scores)
                plt.xlabel('Importance Score')
                plt.title('Top 10 Feature Importance Scores')
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                additional_charts['feature_importance'] = chart_data
        
        # Data quality dashboard
        plt.figure(figsize=(12, 8))
        
        # Quality metrics
        quality_metrics = {
            'Completeness': metadata['data_quality']['completeness']['overall'],
            'Consistency': metadata['data_quality']['consistency']['score'],
            'Validity': metadata['data_quality']['validity']['score'],
            'Accuracy': metadata['data_quality']['accuracy']['score']
        }
        
        plt.subplot(2, 2, 1)
        metrics_names = list(quality_metrics.keys())
        metrics_values = list(quality_metrics.values())
        colors = ['green' if v >= 80 else 'orange' if v >= 60 else 'red' for v in metrics_values]
        plt.bar(metrics_names, metrics_values, color=colors)
        plt.ylim(0, 100)
        plt.title('Data Quality Metrics')
        plt.ylabel('Score (%)')
        
        # Semantic types distribution
        plt.subplot(2, 2, 2)
        semantic_types = list(metadata['semantic_types'].values())
        type_counts = Counter(semantic_types)
        plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        plt.title('Semantic Types Distribution')
        
        # Column roles distribution
        plt.subplot(2, 2, 3)
        roles = [profile['suggested_role'] for profile in metadata['column_profiles'].values()]
        role_counts = Counter(roles)
        plt.bar(role_counts.keys(), role_counts.values())
        plt.xticks(rotation=45)
        plt.title('Suggested Column Roles')
        plt.ylabel('Count')
        
        # Issues by severity
        plt.subplot(2, 2, 4)
        all_issues = []
        for profile in metadata['column_profiles'].values():
            all_issues.extend([issue['severity'] for issue in profile['data_quality_issues']])
        
        if all_issues:
            issue_counts = Counter(all_issues)
            colors = {'high': 'red', 'medium': 'orange', 'low': 'yellow'}
            plt.bar(issue_counts.keys(), issue_counts.values(), 
                   color=[colors.get(k, 'gray') for k in issue_counts.keys()])
            plt.title('Data Quality Issues by Severity')
            plt.ylabel('Count')
        else:
            plt.text(0.5, 0.5, 'No Issues Detected', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title('Data Quality Issues by Severity')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        additional_charts['data_quality_dashboard'] = chart_data
        
        # Combine all charts
        all_charts = {**charts, **additional_charts}
        
        return {
            "status": "success",
            "session_id": session_id,
            "visualizations": all_charts,
            "chart_descriptions": {
                "metadata_overview": "Comprehensive overview of dataset metadata including missing values, data types, correlations, and quality metrics",
                "lineage_diagram": "Network diagram showing relationships between columns based on correlations",
                "feature_importance": "Bar chart showing the most important features for modeling",
                "data_quality_dashboard": "Dashboard showing data quality metrics, semantic types, column roles, and issues"
            },
            "interactive_features": {
                "downloadable": True,
                "formats": ["png", "svg", "pdf"],
                "customizable": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")

@router.post("/learn-from-action")
async def learn_from_user_action(action: UserAction):
    """🤖 Continuously Learns from User Actions - Step 8: Learn and adapt from user behavior"""
    try:
        if action.session_id not in claire_sessions:
            raise HTTPException(status_code=404, detail="CLAIRE session not found")
        
        # Record user action
        action_record = {
            "action_type": action.action_type,
            "action_details": action.action_details,
            "timestamp": action.timestamp or datetime.now(),
            "session_id": action.session_id
        }
        
        # Store in session
        claire_sessions[action.session_id]["user_actions"].append(action_record)
        
        # Store in global user patterns
        user_patterns[action.action_type].append(action_record)
        
        # Analyze patterns and update recommendations
        learning_insights = {
            "action_recorded": True,
            "pattern_analysis": self._analyze_user_patterns(action.action_type),
            "updated_recommendations": self._update_recommendations_based_on_learning(action),
            "adaptation_summary": {}
        }
        
        # Generate adaptation insights
        if len(user_patterns[action.action_type]) >= 5:
            learning_insights["adaptation_summary"] = {
                "pattern_detected": True,
                "frequency": len(user_patterns[action.action_type]),
                "common_parameters": self._extract_common_parameters(action.action_type),
                "suggested_defaults": self._suggest_new_defaults(action.action_type)
            }
        
        return {
            "status": "success",
            "message": "CLAIRE has learned from your action",
            "action_recorded": action_record,
            "learning_insights": learning_insights,
            "claire_adaptation": {
                "improved_suggestions": True,
                "personalized_recommendations": True,
                "pattern_recognition": len(user_patterns[action.action_type]) >= 3
            }
        }
        
    except Exception as e:
        logger.error(f"Error learning from user action: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error learning from user action: {str(e)}")

@router.get("/download-report/{session_id}")
async def download_comprehensive_report(session_id: str, format: str = "json"):
    """📥 Download Comprehensive Report - Get complete CLAIRE analysis report"""
    try:
        if session_id not in claire_sessions:
            raise HTTPException(status_code=404, detail="CLAIRE session not found")
        
        session_data = claire_sessions[session_id]
        
        # Compile comprehensive report
        comprehensive_report = {
            "report_metadata": {
                "session_id": session_id,
                "dataset_name": session_data["dataset_info"]["dataset_name"],
                "generated_at": datetime.now().isoformat(),
                "claire_version": "1.0.0",
                "report_type": "comprehensive_analysis"
            },
            "dataset_overview": {
                "basic_info": session_data["metadata"]["basic_info"],
                "business_context": session_data["dataset_info"]["business_context"],
                "tags": session_data["dataset_info"]["tags"]
            },
            "data_quality_analysis": session_data["metadata"]["data_quality"],
            "column_profiles": session_data["metadata"]["column_profiles"],
            "relationships_analysis": session_data["metadata"]["relationships"],
            "semantic_types": session_data["metadata"]["semantic_types"],
            "statistical_summary": session_data["metadata"]["statistical_summary"],
            "ai_insights": session_data["ai_insights"],
            "governance_analysis": session_data["metadata"]["governance"],
            "feature_recommendations": feature_recommendations.get(session_id, {}),
            "user_actions_log": session_data["user_actions"],
            "recommendations_summary": {
                "data_quality_improvements": self._count_recommendations_by_type(session_data, "data_quality"),
                "feature_engineering_opportunities": self._count_recommendations_by_type(session_data, "feature_engineering"),
                "governance_actions_required": self._count_recommendations_by_type(session_data, "governance")
            }
        }
        
        if format.lower() == "json":
            return {
                "status": "success",
                "report": comprehensive_report,
                "download_info": {
                    "format": "json",
                    "size_estimate": f"{len(json.dumps(comprehensive_report)) / 1024:.1f} KB",
                    "sections": list(comprehensive_report.keys())
                }
            }
        else:
            # For other formats, return download instructions
            return {
                "status": "success",
                "message": f"Report prepared for download in {format} format",
                "download_url": f"/api/v1/claire/download-report/{session_id}?format=json",
                "available_formats": ["json", "csv", "excel"],
                "report_summary": {
                    "total_sections": len(comprehensive_report),
                    "data_quality_score": session_data["metadata"]["data_quality"]["completeness"]["overall"],
                    "recommendations_count": len(session_data.get("user_actions", []))
                }
            }
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating comprehensive report: {str(e)}")

@router.get("/sessions")
async def list_claire_sessions():
    """📋 List all CLAIRE analysis sessions"""
    session_list = []
    for session_id, data in claire_sessions.items():
        session_info = {
            "session_id": session_id,
            "dataset_name": data["dataset_info"]["dataset_name"],
            "created_at": data["created_at"],
            "data_quality_score": data["metadata"]["data_quality"]["completeness"]["overall"],
            "columns_count": len(data["metadata"]["column_profiles"]),
            "governance_issues": len(data["metadata"]["governance"]["privacy_check"]["details"]),
            "user_actions_count": len(data["user_actions"])
        }
        session_list.append(session_info)
    
    return {
        "status": "success",
        "claire_sessions": session_list,
        "total_sessions": len(session_list),
        "summary": {
            "avg_data_quality": sum([s["data_quality_score"] for s in session_list]) / len(session_list) if session_list else 0,
            "total_datasets_analyzed": len(session_list),
            "total_governance_issues": sum([s["governance_issues"] for s in session_list])
        }
    }

@router.delete("/session/{session_id}")
async def delete_claire_session(session_id: str):
    """🗑️ Delete CLAIRE analysis session"""
    if session_id not in claire_sessions:
        raise HTTPException(status_code=404, detail="CLAIRE session not found")
    
    # Remove from all registries
    session_data = claire_sessions.pop(session_id)
    
    if session_id in metadata_registry:
        del metadata_registry[session_id]
    
    if session_id in feature_recommendations:
        del feature_recommendations[session_id]
    
    # Remove from lineage graph
    if lineage_graph.has_node(session_id):
        lineage_graph.remove_node(session_id)
    
    return {
        "status": "success",
        "message": f"CLAIRE session '{session_data['dataset_info']['dataset_name']}' deleted successfully",
        "session_id": session_id
    }

# Helper methods for CLAIRE Engine
def _assess_feature_interpretability(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Assess how interpretable the features are"""
    interpretability_scores = {}
    
    for col_name, profile in metadata["column_profiles"].items():
        score = 100  # Start with perfect score
        
        # Reduce score for high cardinality categorical variables
        if profile["suggested_role"] == "categorical" and profile["unique_percentage"] > 50:
            score -= 30
        
        # Reduce score for complex transformations needed
        if len(profile["transformations_suggested"]) > 2:
            score -= 20
        
        # Reduce score for data quality issues
        score -= len(profile["data_quality_issues"]) * 10
        
        interpretability_scores[col_name] = max(0, score)
    
    return {
        "feature_scores": interpretability_scores,
        "overall_interpretability": sum(interpretability_scores.values()) / len(interpretability_scores) if interpretability_scores else 0,
        "highly_interpretable_features": [col for col, score in interpretability_scores.items() if score > 80]
    }

def _assess_explainability_readiness(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Assess how ready the dataset is for model explainability"""
    readiness_factors = {
        "feature_interpretability": True,
        "data_quality_sufficient": metadata["data_quality"]["completeness"]["overall"] > 80,
        "reasonable_feature_count": len(metadata["column_profiles"]) < 100,
        "no_high_cardinality_issues": len([
            col for col, profile in metadata["column_profiles"].items()
            if profile["unique_percentage"] > 90
        ]) < 3
    }
    
    readiness_score = sum(readiness_factors.values()) / len(readiness_factors) * 100
    
    return {
        "readiness_score": readiness_score,
        "readiness_factors": readiness_factors,
        "explainability_methods": [
            "SHAP values",
            "Permutation importance",
            "Partial dependence plots"
        ] if readiness_score > 75 else [
            "Feature importance only",
            "Simple correlation analysis"
        ]
    }

def _detect_potential_bias(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Detect potential sources of bias in the dataset"""
    bias_indicators = []
    
    # Check for imbalanced categorical variables
    for col_name, profile in metadata["column_profiles"].items():
        if profile["suggested_role"] == "categorical":
            col_data = df[col_name].value_counts()
            if len(col_data) > 1:
                imbalance_ratio = col_data.iloc[0] / col_data.iloc[-1]
                if imbalance_ratio > 10:
                    bias_indicators.append({
                        "type": "class_imbalance",
                        "column": col_name,
                        "severity": "high" if imbalance_ratio > 50 else "medium",
                        "description": f"Highly imbalanced categorical variable (ratio: {imbalance_ratio:.1f}:1)"
                    })
    
    # Check for missing data patterns that might indicate bias
    missing_patterns = {}
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 20:
            missing_patterns[col] = missing_pct
    
    if missing_patterns:
        bias_indicators.append({
            "type": "missing_data_bias",
            "columns": list(missing_patterns.keys()),
            "severity": "medium",
            "description": "High missing data rates may indicate systematic bias"
        })
    
    return {
        "bias_indicators": bias_indicators,
        "bias_risk_score": min(100, len(bias_indicators) * 25),
        "mitigation_strategies": [
            "Apply stratified sampling",
            "Use bias-aware algorithms",
            "Implement fairness constraints"
        ] if bias_indicators else ["No specific bias mitigation needed"]
    }

def _calculate_compliance_score(self, governance_report: Dict[str, Any]) -> float:
    """Calculate overall compliance score"""
    scores = []
    
    # Data quality score
    scores.append(governance_report["data_quality_governance"]["overall_score"])
    
    # Privacy compliance score
    privacy_score = 100 if governance_report["privacy_compliance"]["pii_columns_detected"] == 0 else 60
    scores.append(privacy_score)
    
    # Quality gates score
    quality_gates = governance_report["data_quality_governance"]["quality_gates"]
    gates_passed = sum([
        quality_gates["completeness_passed"],
        quality_gates["uniqueness_check"],
        quality_gates["consistency_check"]
    ])
    gates_score = (gates_passed / 3) * 100
    scores.append(gates_score)
    
    return sum(scores) / len(scores)

def _analyze_user_patterns(self, action_type: str) -> Dict[str, Any]:
    """Analyze patterns in user actions"""
    actions = user_patterns[action_type]
    
    if len(actions) < 3:
        return {"pattern_detected": False, "message": "Insufficient data for pattern analysis"}
    
    # Analyze common parameters
    common_params = {}
    for action in actions:
        for key, value in action["action_details"].items():
            if key not in common_params:
                common_params[key] = []
            common_params[key].append(value)
    
    # Find most common values
    patterns = {}
    for key, values in common_params.items():
        if isinstance(values[0], str):
            most_common = Counter(values).most_common(1)[0]
            if most_common[1] / len(values) > 0.6:  # 60% threshold
                patterns[key] = most_common[0]
    
    return {
        "pattern_detected": len(patterns) > 0,
        "common_patterns": patterns,
        "frequency": len(actions),
        "confidence": len(patterns) / len(common_params) if common_params else 0
    }

def _update_recommendations_based_on_learning(self, action: UserAction) -> Dict[str, Any]:
    """Update recommendations based on user learning"""
    # This is a simplified implementation
    # In practice, this would use more sophisticated ML techniques
    
    updates = {
        "recommendation_adjustments": [],
        "new_suggestions": [],
        "priority_changes": []
    }
    
    # Example: If user frequently chooses certain transformations, prioritize them
    if action.action_type == "apply_transformation":
        transformation_type = action.action_details.get("transformation_type")
        if transformation_type:
            updates["recommendation_adjustments"].append({
                "type": "priority_boost",
                "target": transformation_type,
                "reason": "User frequently applies this transformation"
            })
    
    return updates

def _extract_common_parameters(self, action_type: str) -> Dict[str, Any]:
    """Extract common parameters from user actions"""
    actions = user_patterns[action_type]
    common_params = {}
    
    for action in actions:
        for key, value in action["action_details"].items():
            if key not in common_params:
                common_params[key] = []
            common_params[key].append(value)
    
    # Return most common values
    result = {}
    for key, values in common_params.items():
        if values:
            if isinstance(values[0], (str, int, float)):
                most_common = Counter(values).most_common(1)[0][0]
                result[key] = most_common
    
    return result

def _suggest_new_defaults(self, action_type: str) -> Dict[str, Any]:
    """Suggest new default values based on user patterns"""
    common_params = self._extract_common_parameters(action_type)
    
    suggestions = {}
    for key, value in common_params.items():
        suggestions[f"default_{key}"] = value
    
    return suggestions

def _count_recommendations_by_type(self, session_data: Dict[str, Any], rec_type: str) -> int:
    """Count recommendations by type"""
    count = 0
    
    # Count from column profiles
    for profile in session_data["metadata"]["column_profiles"].values():
        if rec_type == "data_quality":
            count += len(profile["data_quality_issues"])
        elif rec_type == "feature_engineering":
            count += len(profile["transformations_suggested"])
    
    # Count from governance
    if rec_type == "governance":
        count += len(session_data["metadata"]["governance"]["privacy_check"]["details"])
    
    return count

# Bind helper methods to CLAIREEngine class
CLAIREEngine._assess_feature_interpretability = _assess_feature_interpretability
CLAIREEngine._assess_explainability_readiness = _assess_explainability_readiness
CLAIREEngine._detect_potential_bias = _detect_potential_bias
CLAIREEngine._calculate_compliance_score = _calculate_compliance_score
CLAIREEngine._analyze_user_patterns = _analyze_user_patterns
CLAIREEngine._update_recommendations_based_on_learning = _update_recommendations_based_on_learning
CLAIREEngine._extract_common_parameters = _extract_common_parameters
CLAIREEngine._suggest_new_defaults = _suggest_new_defaults
CLAIREEngine._count_recommendations_by_type = _count_recommendations_by_type
