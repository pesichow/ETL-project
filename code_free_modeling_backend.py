"""
Complete Code-Free Modeling Backend - FastAPI APIRouter Module with Iceberg Integration
Fixed Prediction Endpoint + All Previous Functionality
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
import os
import tempfile
import pickle
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
import traceback

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

# Create APIRouter
router = APIRouter(prefix="/api/v1/code-free-modeling", tags=["Code-Free Modeling"])

# Import the fetch_data function
from utils.fetch_data import fetch_data

# Pydantic models for request/response
class DatasetInfoRequest(BaseModel):
    job_id: str
    table_names: List[str]
    limit: int = 10000

class DatasetInfoResponse(BaseModel):
    job_id: str
    table_names: List[str]
    rows: int
    columns: List[Dict[str, Any]]
    size: int
    data_quality: Dict[str, Any]
    modeling_recommendations: List[Dict[str, str]]
    target_suggestions: List[Dict[str, Any]]
    feature_analysis: Dict[str, Any]
    success: bool

class ModelTrainingRequest(BaseModel):
    job_id: str
    table_names: List[str]
    target_column: str
    feature_columns: List[str]
    problem_type: str  # "classification" or "regression"
    model_type: str = "auto"  # "auto", "random_forest", "logistic_regression", etc.
    test_size: float = 0.2
    random_state: int = 42
    limit: int = 10000

class ModelTrainingResponse(BaseModel):
    model_id: str
    job_id: str
    table_names: List[str]
    problem_type: str
    model_type: str
    target_column: str
    feature_columns: List[str]
    training_metrics: Dict[str, Any]
    model_performance: Dict[str, Any]
    feature_importance: List[Dict[str, Any]]
    training_time: float
    dataset_info: Dict[str, Any]
    insights: List[Dict[str, Any]]
    success: bool

class PredictionRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]

class PredictionResponse(BaseModel):
    model_id: str
    prediction: Union[float, int, str, List]
    prediction_proba: Optional[List[float]] = None
    confidence: float
    feature_contributions: Dict[str, float] = {}
    input_processed: Dict[str, Any] = {}
    success: bool

# Global model store (in production, use Redis or database)
trained_models_store = {}
model_metadata_store = {}

class CodeFreeModelingEngine:
    """Advanced Code-Free Modeling Engine with Iceberg Integration"""
    
    def __init__(self):
        self.classification_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        
        self.regression_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'svm': SVR(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'knn': KNeighborsRegressor(n_neighbors=5)
        }
    
    def analyze_dataset_for_modeling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataset analysis for modeling"""
        try:
            analysis = {
                'basic_info': self._get_basic_info(df),
                'column_analysis': self._analyze_columns_for_modeling(df),
                'data_quality': self._assess_data_quality(df),
                'target_suggestions': self._suggest_target_columns(df),
                'modeling_recommendations': self._generate_modeling_recommendations(df),
                'feature_analysis': self._analyze_features(df)
            }
            return analysis
        except Exception as e:
            logger.error(f"Error in dataset analysis for modeling: {str(e)}")
            raise
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
    
    def _analyze_columns_for_modeling(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze each column for modeling potential"""
        columns_info = []
        
        for col in df.columns:
            try:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'missing_count': int(df[col].isnull().sum()),
                    'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                    'unique_count': int(df[col].nunique()),
                    'unique_percentage': float(df[col].nunique() / len(df) * 100),
                    'modeling_role': self._determine_modeling_role(df[col]),
                    'target_suitability': self._assess_target_suitability(df[col]),
                    'feature_quality': self._assess_feature_quality(df[col])
                }
                
                # Add type-specific analysis
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info.update(self._analyze_numeric_for_modeling(df[col]))
                elif pd.api.types.is_object_dtype(df[col]):
                    col_info.update(self._analyze_categorical_for_modeling(df[col]))
                
                columns_info.append(col_info)
                
            except Exception as e:
                logger.warning(f"Error analyzing column {col}: {str(e)}")
                continue
        
        return columns_info
    
    def _determine_modeling_role(self, series: pd.Series) -> str:
        """Determine the potential modeling role of a column"""
        try:
            missing_pct = series.isnull().sum() / len(series) * 100
            unique_count = series.nunique()
            unique_ratio = unique_count / len(series)
            
            if missing_pct > 50:
                return "poor_feature"
            
            if pd.api.types.is_numeric_dtype(series):
                if unique_count == 2:
                    return "binary_target_or_feature"
                elif unique_count < 10 and unique_ratio < 0.1:
                    return "categorical_target_or_feature"
                elif unique_ratio > 0.95:
                    return "identifier"
                else:
                    return "numeric_feature_or_target"
            
            elif pd.api.types.is_object_dtype(series):
                if unique_ratio > 0.95:
                    return "identifier"
                elif unique_count < 20:
                    return "categorical_feature_or_target"
                else:
                    return "text_feature"
            
            return "feature"
        except:
            return "unknown"
    
    def _assess_target_suitability(self, series: pd.Series) -> str:
        """Assess how suitable a column is as a target variable"""
        try:
            missing_pct = series.isnull().sum() / len(series) * 100
            unique_count = series.nunique()
            unique_ratio = unique_count / len(series)
            
            if missing_pct > 10:
                return "poor"
            
            if pd.api.types.is_numeric_dtype(series):
                if unique_count == 2:
                    return "excellent_binary_classification"
                elif 2 < unique_count <= 10 and unique_ratio < 0.1:
                    return "good_multiclass_classification"
                elif unique_count > 10 and unique_ratio > 0.1:
                    return "good_regression"
                else:
                    return "fair"
            
            elif pd.api.types.is_object_dtype(series):
                if 2 <= unique_count <= 20:
                    return "good_classification"
                else:
                    return "poor"
            
            return "fair"
        except:
            return "unknown"
    
    def _assess_feature_quality(self, series: pd.Series) -> str:
        """Assess the quality of a column as a feature"""
        try:
            missing_pct = series.isnull().sum() / len(series) * 100
            unique_ratio = series.nunique() / len(series)
            
            if missing_pct > 70:
                return "poor"
            elif missing_pct > 30:
                return "fair"
            
            if pd.api.types.is_numeric_dtype(series):
                if series.std() == 0:
                    return "poor"
                elif unique_ratio < 0.01:
                    return "fair"
                else:
                    return "good"
            
            elif pd.api.types.is_object_dtype(series):
                if unique_ratio > 0.95:
                    return "poor"
                elif unique_ratio < 0.01:
                    return "fair"
                else:
                    return "good"
            
            return "good"
        except:
            return "unknown"
    
    def _analyze_numeric_for_modeling(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column for modeling"""
        try:
            return {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'skewness': float(series.skew()),
                'outliers_count': int(self._count_outliers(series)),
                'zero_count': int((series == 0).sum()),
                'distribution_type': self._identify_distribution(series)
            }
        except:
            return {}
    
    def _analyze_categorical_for_modeling(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column for modeling"""
        try:
            value_counts = series.value_counts()
            return {
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                'cardinality': int(series.nunique()),
                'cardinality_ratio': float(series.nunique() / len(series)),
                'top_categories': value_counts.head(5).to_dict()
            }
        except:
            return {}
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((series < lower_bound) | (series > upper_bound)).sum()
        except:
            return 0
    
    def _identify_distribution(self, series: pd.Series) -> str:
        """Identify distribution type"""
        try:
            skewness = series.skew()
            if abs(skewness) < 0.5:
                return "normal"
            elif skewness > 1:
                return "right_skewed"
            elif skewness < -1:
                return "left_skewed"
            else:
                return "slightly_skewed"
        except:
            return "unknown"
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality for modeling"""
        try:
            return {
                'completeness_score': float((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
                'consistency_score': self._calculate_consistency_score(df),
                'modeling_readiness': self._assess_modeling_readiness(df),
                'data_balance': self._assess_data_balance(df)
            }
        except Exception as e:
            logger.warning(f"Error assessing data quality: {str(e)}")
            return {}
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate consistency score"""
        try:
            consistency_issues = 0
            total_checks = 0
            
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]):
                    sample = df[col].dropna().head(100)
                    if len(sample) > 0:
                        types = set(type(x).__name__ for x in sample)
                        if len(types) > 1:
                            consistency_issues += 1
                        total_checks += 1
            
            if total_checks == 0:
                return 100.0
            
            return max(0, (1 - consistency_issues / total_checks) * 100)
        except:
            return 85.0
    
    def _assess_modeling_readiness(self, df: pd.DataFrame) -> str:
        """Assess how ready the data is for modeling"""
        try:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            
            if missing_pct > 30:
                return "needs_preprocessing"
            elif missing_pct > 10:
                return "minor_preprocessing"
            elif numeric_cols == 0 and categorical_cols > 0:
                return "needs_encoding"
            else:
                return "ready"
        except:
            return "unknown"
    
    def _assess_data_balance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data balance for potential target columns"""
        try:
            balance_info = {}
            
            for col in df.columns:
                if df[col].nunique() <= 20 and df[col].nunique() >= 2:
                    value_counts = df[col].value_counts()
                    balance_ratio = value_counts.min() / value_counts.max()
                    
                    if balance_ratio > 0.8:
                        balance_status = "well_balanced"
                    elif balance_ratio > 0.5:
                        balance_status = "moderately_balanced"
                    elif balance_ratio > 0.1:
                        balance_status = "imbalanced"
                    else:
                        balance_status = "highly_imbalanced"
                    
                    balance_info[col] = {
                        'balance_ratio': float(balance_ratio),
                        'balance_status': balance_status,
                        'class_distribution': value_counts.to_dict()
                    }
            
            return balance_info
        except:
            return {}
    
    def _suggest_target_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest potential target columns"""
        suggestions = []
        
        try:
            for col in df.columns:
                suitability = self._assess_target_suitability(df[col])
                
                if suitability not in ['poor', 'unknown']:
                    suggestion = {
                        'column': col,
                        'suitability': suitability,
                        'problem_type': self._infer_problem_type(df[col]),
                        'unique_values': int(df[col].nunique()),
                        'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                        'data_type': str(df[col].dtype)
                    }
                    
                    if pd.api.types.is_object_dtype(df[col]) and df[col].nunique() <= 10:
                        suggestion['sample_values'] = df[col].value_counts().head(5).index.tolist()
                    
                    suggestions.append(suggestion)
            
            # Sort by suitability
            suitability_order = {
                'excellent_binary_classification': 5,
                'good_multiclass_classification': 4,
                'good_regression': 4,
                'good_classification': 3,
                'fair': 2
            }
            
            suggestions.sort(key=lambda x: suitability_order.get(x['suitability'], 0), reverse=True)
            
        except Exception as e:
            logger.warning(f"Error suggesting target columns: {str(e)}")
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def _infer_problem_type(self, series: pd.Series) -> str:
        """Infer the problem type based on target column"""
        try:
            unique_count = series.nunique()
            
            if pd.api.types.is_numeric_dtype(series):
                if unique_count == 2:
                    return "binary_classification"
                elif unique_count <= 10 and (series.nunique() / len(series)) < 0.1:
                    return "multiclass_classification"
                else:
                    return "regression"
            
            elif pd.api.types.is_object_dtype(series):
                if unique_count <= 20:
                    return "multiclass_classification" if unique_count > 2 else "binary_classification"
                else:
                    return "text_classification"
            
            return "regression"
        except:
            return "classification"
    
    def _generate_modeling_recommendations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate modeling recommendations"""
        recommendations = []
        
        try:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            if missing_pct > 20:
                recommendations.append({
                    'type': 'preprocessing',
                    'title': 'Handle Missing Values',
                    'description': f'{missing_pct:.1f}% of data is missing. Consider imputation or removal strategies.',
                    'priority': 'high'
                })
            
            if len(categorical_cols) > 0:
                high_cardinality = [col for col in categorical_cols if df[col].nunique() > 20]
                if high_cardinality:
                    recommendations.append({
                        'type': 'encoding',
                        'title': 'Encode Categorical Variables',
                        'description': f'Encode {len(high_cardinality)} high-cardinality categorical columns before modeling.',
                        'priority': 'high'
                    })
            
            if len(numeric_cols) > 1:
                recommendations.append({
                    'type': 'scaling',
                    'title': 'Feature Scaling Recommended',
                    'description': 'Scale numeric features for algorithms like SVM and Neural Networks.',
                    'priority': 'medium'
                })
            
            if len(df) < 1000:
                recommendations.append({
                    'type': 'data_size',
                    'title': 'Small Dataset Warning',
                    'description': 'Dataset is small. Consider simpler models and cross-validation.',
                    'priority': 'medium'
                })
            
            outlier_cols = []
            for col in numeric_cols:
                if self._count_outliers(df[col]) > len(df) * 0.05:
                    outlier_cols.append(col)
            
            if outlier_cols:
                recommendations.append({
                    'type': 'outliers',
                    'title': 'Outlier Detection',
                    'description': f'Consider outlier treatment for {len(outlier_cols)} columns with significant outliers.',
                    'priority': 'medium'
                })
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def _analyze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze features for modeling"""
        try:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()
            
            return {
                'total_features': len(df.columns),
                'numeric_features': len(numeric_features),
                'categorical_features': len(categorical_features),
                'feature_types': {
                    'numeric': numeric_features,
                    'categorical': categorical_features
                },
                'feature_quality_summary': self._summarize_feature_quality(df)
            }
        except:
            return {}
    
    def _summarize_feature_quality(self, df: pd.DataFrame) -> Dict[str, int]:
        """Summarize feature quality"""
        try:
            quality_counts = {'good': 0, 'fair': 0, 'poor': 0}
            
            for col in df.columns:
                quality = self._assess_feature_quality(df[col])
                if quality in quality_counts:
                    quality_counts[quality] += 1
            
            return quality_counts
        except:
            return {'good': 0, 'fair': 0, 'poor': 0}
    
    def train_model(self, df: pd.DataFrame, target_column: str, feature_columns: List[str],
                   problem_type: str, model_type: str = "auto", test_size: float = 0.2,
                   random_state: int = 42) -> Dict[str, Any]:
        """Train a machine learning model"""
        try:
            logger.info(f"🚀 Starting model training...")
            logger.info(f"📊 Dataset shape: {df.shape}")
            logger.info(f"🎯 Target: {target_column}")
            logger.info(f"🔧 Features: {len(feature_columns)} columns")
            
            # Prepare data
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            logger.info(f"📋 Original X shape: {X.shape}, y shape: {y.shape}")
            
            # Handle missing values in features
            X = self._handle_missing_values(X)
            
            # Remove rows where target is missing
            valid_indices = y.notna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            logger.info(f"📋 After cleaning - X shape: {X.shape}, y shape: {y.shape}")
            
            if len(X) == 0:
                raise ValueError("No valid data remaining after cleaning")
            
            # Encode categorical variables in features
            X_processed, encoders = self._encode_features(X)
            logger.info(f"📋 After encoding - X shape: {X_processed.shape}")
            logger.info(f"🔧 Encoders created: {list(encoders.keys())}")
            
            # Encode target if classification
            target_encoder = None
            if problem_type == "classification" and pd.api.types.is_object_dtype(y):
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y.astype(str))
                logger.info(f"🎯 Target encoded: {len(target_encoder.classes_)} classes")
            else:
                y_encoded = y.values
            
            # Split data
            stratify_param = y_encoded if problem_type == "classification" and len(np.unique(y_encoded)) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_encoded, test_size=test_size, random_state=random_state,
                stratify=stratify_param
            )
            
            logger.info(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Scale features if needed
            scaler = None
            if model_type in ['svm', 'knn'] or model_type == 'auto':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                logger.info(f"📏 Features scaled")
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Select and train model
            if model_type == "auto":
                model = self._select_best_model(X_train_scaled, y_train, problem_type)
            else:
                models = self.classification_models if problem_type == "classification" else self.regression_models
                model = models.get(model_type, list(models.values())[0])
            
            logger.info(f"🤖 Selected model: {type(model).__name__}")
            
            # Train model
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            logger.info(f"⏱️ Training completed in {training_time:.2f} seconds")
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, problem_type, model, X_test_scaled)
            logger.info(f"📈 Metrics calculated: {list(metrics.keys())}")
            
            # Feature importance
            feature_importance = self._get_feature_importance(model, X_processed.columns.tolist())
            
            # Store model with all necessary components
            model_id = str(uuid.uuid4())
            model_data = {
                'model': model,
                'encoders': encoders,
                'scaler': scaler,
                'target_encoder': target_encoder,
                'feature_columns': feature_columns,  # Original feature columns
                'processed_feature_columns': X_processed.columns.tolist(),  # After encoding
                'problem_type': problem_type,
                'model_type': type(model).__name__
            }
            
            trained_models_store[model_id] = model_data
            logger.info(f"💾 Model stored with ID: {model_id}")
            
            return {
                'model_id': model_id,
                'training_time': training_time,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'model_type': type(model).__name__,
                'dataset_info': {
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'features_used': len(feature_columns),
                    'processed_features': len(X_processed.columns),
                    'target_classes': len(np.unique(y_encoded)) if problem_type == "classification" else None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in model training: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        X_processed = X.copy()
        
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    median_val = X_processed[col].median()
                    X_processed[col] = X_processed[col].fillna(median_val)
                    logger.info(f"🔧 Filled {col} missing values with median: {median_val}")
                else:
                    mode_val = X_processed[col].mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else 'missing'
                    X_processed[col] = X_processed[col].fillna(fill_val)
                    logger.info(f"🔧 Filled {col} missing values with: {fill_val}")
        
        return X_processed
    
    def _encode_features(self, X: pd.DataFrame) -> tuple:
        """Encode categorical features"""
        X_processed = X.copy()
        encoders = {}
        
        for col in X_processed.columns:
            if pd.api.types.is_object_dtype(X_processed[col]):
                unique_count = X_processed[col].nunique()
                
                if unique_count <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(X_processed[col], prefix=col, dummy_na=True)
                    X_processed = pd.concat([X_processed.drop(col, axis=1), dummies], axis=1)
                    encoders[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
                    logger.info(f"🔄 One-hot encoded {col}: {unique_count} categories -> {len(dummies.columns)} columns")
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                    encoders[col] = {'type': 'label', 'encoder': le}
                    logger.info(f"🔄 Label encoded {col}: {unique_count} categories")
        
        return X_processed, encoders
    
    def _select_best_model(self, X_train, y_train, problem_type: str):
        """Select the best model using simple heuristics"""
        if problem_type == "classification":
            if len(np.unique(y_train)) == 2:
                return LogisticRegression(random_state=42, max_iter=1000)
            else:
                return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _calculate_metrics(self, y_true, y_pred, problem_type: str, model, X_test) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {}
        
        try:
            if problem_type == "classification":
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                
                if len(np.unique(y_true)) == 2:
                    metrics['precision'] = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
                    metrics['recall'] = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
                    metrics['f1_score'] = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
                    
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_proba = model.predict_proba(X_test)[:, 1]
                            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
                        except:
                            pass
                else:
                    metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                    metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                    metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            
            else:  # regression
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
                metrics['r2_score'] = float(r2_score(y_true, y_pred))
        
        except Exception as e:
            logger.warning(f"⚠️ Error calculating metrics: {str(e)}")
        
        return metrics
    
    def _get_feature_importance(self, model, feature_columns: List[str]) -> List[Dict[str, Any]]:
        """Get feature importance from the model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = [
                    {
                        'feature': feature_columns[i] if i < len(feature_columns) else f'feature_{i}',
                        'importance': float(importances[i])
                    }
                    for i in range(len(importances))
                ]
                return sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
            
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]  # For multiclass, take first class
                
                feature_importance = [
                    {
                        'feature': feature_columns[i] if i < len(feature_columns) else f'feature_{i}',
                        'importance': float(abs(coef[i]))
                    }
                    for i in range(len(coef))
                ]
                return sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
            
            else:
                return []
        
        except Exception as e:
            logger.warning(f"⚠️ Error getting feature importance: {str(e)}")
            return []

def generate_modeling_insights(training_result: Dict[str, Any], job_id: str, table_names: List[str]) -> List[Dict]:
    """Generate comprehensive modeling insights"""
    try:
        insights = []
        
        metrics = training_result.get('metrics', {})
        model_type = training_result.get('model_type', 'Unknown')
        dataset_info = training_result.get('dataset_info', {})
        
        insights = [
            {
                "title": "Model Training Complete",
                "description": f"Successfully trained {model_type} model on Iceberg data from {table_names} (job: {job_id}). Training completed in {training_result.get('training_time', 0):.2f} seconds.",
                "priority": "high",
                "category": "success"
            },
            {
                "title": "Model Performance",
                "description": f"Model achieved {list(metrics.values())[0]:.3f} on primary metric. Review all metrics to assess model quality.",
                "priority": "high",
                "category": "performance"
            },
            {
                "title": "Feature Utilization",
                "description": f"Model trained on {dataset_info.get('features_used', 0)} features from {len(table_names)} Iceberg table(s). Check feature importance for insights.",
                "priority": "medium",
                "category": "features"
            },
            {
                "title": "Data Split Information",
                "description": f"Used {dataset_info.get('train_size', 0)} samples for training and {dataset_info.get('test_size', 0)} for testing. Ensure sufficient data for reliable evaluation.",
                "priority": "medium",
                "category": "data"
            },
            {
                "title": "Model Deployment Ready",
                "description": "Model is ready for predictions. Use the model ID to make predictions on new data with the same feature structure.",
                "priority": "high",
                "category": "deployment"
            }
        ]
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return [
            {
                "title": "Training Complete",
                "description": f"Model training completed for Iceberg data (job: {job_id}). Review results and proceed with evaluation.",
                "priority": "medium",
                "category": "success"
            }
        ]

# Initialize the modeling engine
modeling_engine = CodeFreeModelingEngine()

# API Routes
@router.get("/")
async def modeling_info():
    """Get Code-Free Modeling service information"""
    return {
        "service": "Code-Free Modeling Backend with Iceberg Integration",
        "version": "2.0.0",
        "description": "Advanced automated ML modeling API with Iceberg data fetching and intelligent model selection",
        "endpoints": {
            "dataset_info": "/dataset-info",
            "train": "/train",
            "predict": "/predict",
            "models": "/models",
            "debug": "/debug/model/{model_id}",
            "test_predict": "/test-predict/{model_id}"
        },
        "supported_algorithms": {
            "classification": ["random_forest", "logistic_regression", "svm", "decision_tree", "naive_bayes", "knn"],
            "regression": ["random_forest", "linear_regression", "svm", "decision_tree", "knn"]
        }
    }

@router.post("/dataset-info", response_model=DatasetInfoResponse)
async def get_dataset_info(request: DatasetInfoRequest):
    """Get comprehensive dataset information and modeling analysis from Iceberg data"""
    try:
        logger.info(f"🔍 Dataset analysis for modeling: job_id={request.job_id}, tables={request.table_names}")
        
        # Fetch data from Iceberg
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

        logger.info(f"📦 Received data from {len(data)} tables")

        # Combine data from all tables
        combined_df = None
        for table_name, row_data in data.items():
            if row_data:
                logger.info(f"📊 Processing table '{table_name}' with {len(row_data)} rows")
                try:
                    df = pd.DataFrame(row_data)
                    df['_source_table'] = table_name

                    combined_df = df if combined_df is None else pd.concat([combined_df, df], ignore_index=True)

                    logger.info(f"✅ Added {len(df)} rows from '{table_name}'")
                except Exception as e:
                    logger.error(f"❌ Error processing table '{table_name}': {str(e)}")
                    continue

        if combined_df is None or combined_df.empty:
            logger.error("🚫 No valid data found after processing all tables")
            raise HTTPException(status_code=400, detail="No valid data found for analysis")

        logger.info(f"🧩 Combined dataset shape: {combined_df.shape}")
        
        # Analyze dataset for modeling
        analysis = modeling_engine.analyze_dataset_for_modeling(combined_df)

        return DatasetInfoResponse(
            job_id=request.job_id,
            table_names=request.table_names,
            rows=len(combined_df),
            columns=analysis['column_analysis'],
            size=int(analysis['basic_info']['memory_usage']),
            data_quality=analysis['data_quality'],
            modeling_recommendations=analysis['modeling_recommendations'],
            target_suggestions=analysis['target_suggestions'],
            feature_analysis=analysis['feature_analysis'],
            success=True
        )

    except HTTPException as he:
        logger.error(f"🚨 HTTPException: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"🔥 Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest):
    """Train a machine learning model using Iceberg data"""
    try:
        logger.info(f"🚀 Model training requested: job_id={request.job_id}, tables={request.table_names}")
        
        if not request.feature_columns:
            raise HTTPException(status_code=400, detail="No feature columns selected")
        
        if not request.target_column:
            raise HTTPException(status_code=400, detail="No target column specified")
        
        # Fetch data from Iceberg
        iceberg_result = fetch_data(request.job_id, request.table_names, request.limit)
        
        if "results" not in iceberg_result or not iceberg_result["results"]:
            raise HTTPException(status_code=400, detail="No data returned from Iceberg")
        
        # Extract and combine data
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
                if combined_df is None:
                    combined_df = df
                else:
                    df['_source_table'] = table_name
                    if '_source_table' not in combined_df.columns:
                        combined_df['_source_table'] = list(data.keys())[0]
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        if combined_df is None or combined_df.empty:
            raise HTTPException(status_code=400, detail="No valid data found for model training")
        
        # Validate columns exist
        all_columns = request.feature_columns + [request.target_column]
        missing_columns = [col for col in all_columns if col not in combined_df.columns]
        if missing_columns:
            available_columns = combined_df.columns.tolist()
            raise HTTPException(
                status_code=400, 
                detail=f"Columns not found: {missing_columns}. Available columns: {available_columns}"
            )
        
        logger.info(f"📊 Starting model training on {combined_df.shape} dataset")
        
        # Train model
        training_result = modeling_engine.train_model(
            combined_df,
            request.target_column,
            request.feature_columns,
            request.problem_type,
            request.model_type,
            request.test_size,
            request.random_state
        )
        
        # Store metadata
        model_metadata_store[training_result['model_id']] = {
            'job_id': request.job_id,
            'table_names': request.table_names,
            'target_column': request.target_column,
            'feature_columns': request.feature_columns,
            'problem_type': request.problem_type,
            'created_at': datetime.now().isoformat(),
            'dataset_shape': combined_df.shape
        }
        
        # Generate insights
        insights = generate_modeling_insights(training_result, request.job_id, request.table_names)
        
        logger.info(f"✅ Model training completed: {training_result['model_id']}")
        
        return ModelTrainingResponse(
            model_id=training_result['model_id'],
            job_id=request.job_id,
            table_names=request.table_names,
            problem_type=request.problem_type,
            model_type=training_result['model_type'],
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            training_metrics=training_result['metrics'],
            model_performance=training_result['metrics'],
            feature_importance=training_result['feature_importance'],
            training_time=training_result['training_time'],
            dataset_info=training_result['dataset_info'],
            insights=insights,
            success=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Make predictions using a trained model with improved error handling"""
    try:
        logger.info(f"🔮 Prediction requested for model: {request.model_id}")
        logger.info(f"📥 Input data: {request.input_data}")
        
        # Check if model exists
        if request.model_id not in trained_models_store:
            logger.error(f"❌ Model {request.model_id} not found in store")
            logger.info(f"Available models: {list(trained_models_store.keys())}")
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        model_data = trained_models_store[request.model_id]
        model = model_data['model']
        encoders = model_data.get('encoders', {})
        scaler = model_data.get('scaler')
        target_encoder = model_data.get('target_encoder')
        feature_columns = model_data['feature_columns']  # Original feature columns
        problem_type = model_data['problem_type']
        
        logger.info(f"📋 Model info - Type: {type(model).__name__}, Features: {len(feature_columns)}")
        logger.info(f"🔧 Required features: {feature_columns}")
        
        # Validate input features
        missing_features = [col for col in feature_columns if col not in request.input_data]
        if missing_features:
            logger.error(f"❌ Missing features: {missing_features}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}. Required features: {feature_columns}"
            )
        
        # Create DataFrame from input
        input_df = pd.DataFrame([request.input_data])
        logger.info(f"📊 Input DataFrame shape: {input_df.shape}")
        logger.info(f"📊 Input DataFrame columns: {input_df.columns.tolist()}")
        
        # Select only required features in correct order
        try:
            X = input_df[feature_columns].copy()
            logger.info(f"✅ Selected features successfully: {X.shape}")
        except KeyError as e:
            logger.error(f"❌ Error selecting features: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error selecting features: {str(e)}"
            )
        
        # Handle missing values in input (same as training)
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(0)  # Use 0 as fallback for prediction
                    logger.info(f"🔧 Filled missing numeric values in {col} with 0")
                else:
                    X[col] = X[col].fillna('missing')
                    logger.info(f"🔧 Filled missing categorical values in {col} with 'missing'")
        
        # Store processed input for response
        input_processed = X.iloc[0].to_dict()
        
        # Apply encodings step by step (same as training)
        logger.info(f"🔄 Applying {len(encoders)} encoders...")
        
        for col, encoder_info in encoders.items():
            if col in X.columns:
                logger.info(f"🔧 Encoding column {col} with {encoder_info['type']} encoding")
                
                try:
                    if encoder_info['type'] == 'onehot':
                        # Handle one-hot encoding
                        dummies = pd.get_dummies(X[col], prefix=col, dummy_na=True)
                        
                        # Ensure all expected columns are present
                        expected_columns = encoder_info['columns']
                        for expected_col in expected_columns:
                            if expected_col not in dummies.columns:
                                dummies[expected_col] = 0
                                logger.info(f"➕ Added missing dummy column: {expected_col}")
                        
                        # Select only the expected columns in the right order
                        dummies_selected = dummies[expected_columns].fillna(0)
                        
                        # Remove original column and add dummy columns
                        X = X.drop(col, axis=1)
                        X = pd.concat([X, dummies_selected], axis=1)
                        
                        logger.info(f"✅ One-hot encoded {col} -> {len(expected_columns)} columns")
                        
                    elif encoder_info['type'] == 'label':
                        # Handle label encoding with unseen categories
                        le = encoder_info['encoder']
                        original_values = X[col].astype(str).values
                        
                        # Handle unseen categories
                        encoded_values = []
                        for val in original_values:
                            try:
                                encoded_val = le.transform([val])[0]
                                encoded_values.append(encoded_val)
                            except ValueError:
                                # Assign 0 for unseen categories
                                encoded_values.append(0)
                                logger.warning(f"⚠️ Unseen category '{val}' in {col}, assigned 0")
                        
                        X[col] = encoded_values
                        logger.info(f"✅ Label encoded {col}")
                        
                except Exception as e:
                    logger.error(f"❌ Error encoding {col}: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error encoding column {col}: {str(e)}"
                    )
        
        logger.info(f"📊 After encoding - DataFrame shape: {X.shape}")
        logger.info(f"📊 After encoding - Columns: {X.columns.tolist()}")
        
        # Convert to numpy array for model prediction
        try:
            X_array = X.values.astype(float)
            logger.info(f"✅ Converted to numpy array: {X_array.shape}")
        except Exception as e:
            logger.error(f"❌ Error converting to numpy array: {str(e)}")
            logger.error(f"Data types: {X.dtypes.to_dict()}")
            logger.error(f"Sample data: {X.head().to_dict()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error converting data to numeric format: {str(e)}"
            )
        
        # Apply scaling if needed
        if scaler is not None:
            try:
                X_array = scaler.transform(X_array)
                logger.info(f"✅ Applied scaling")
            except Exception as e:
                logger.error(f"❌ Error applying scaling: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error applying scaling: {str(e)}"
                )
        
        # Make prediction
        try:
            prediction = model.predict(X_array)[0]
            logger.info(f"🎯 Raw prediction: {prediction}")
        except Exception as e:
            logger.error(f"❌ Error making prediction: {str(e)}")
            logger.error(f"Model type: {type(model)}")
            logger.error(f"Input shape: {X_array.shape}")
            raise HTTPException(
                status_code=500,
                detail=f"Error making prediction: {str(e)}"
            )
        
        # Get prediction probabilities and confidence
        prediction_proba = None
        confidence = 0.5
        
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_array)[0]
                prediction_proba = proba.tolist()
                confidence = float(max(proba))
                logger.info(f"📊 Prediction probabilities: {prediction_proba}")
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(X_array)[0]
                confidence = float(min(abs(decision), 1.0))  # Cap at 1.0
                logger.info(f"📊 Decision function: {decision}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get prediction probabilities: {str(e)}")
        
        # Decode prediction if needed
        final_prediction = prediction
        if target_encoder is not None:
            try:
                final_prediction = target_encoder.inverse_transform([prediction])[0]
                logger.info(f"🔄 Decoded prediction: {prediction} -> {final_prediction}")
            except Exception as e:
                logger.warning(f"⚠️ Could not decode prediction: {str(e)}")
                final_prediction = prediction
        
        # Calculate feature contributions (simplified)
        feature_contributions = {}
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Map importances back to original feature names
                current_features = X.columns.tolist()
                for i, importance in enumerate(importances):
                    if i < len(current_features):
                        feature_name = current_features[i]
                        # Try to map back to original feature name
                        original_feature = feature_name
                        for orig_col in feature_columns:
                            if feature_name.startswith(f"{orig_col}_"):
                                original_feature = orig_col
                                break
                        
                        if original_feature not in feature_contributions:
                            feature_contributions[original_feature] = 0
                        feature_contributions[original_feature] += float(importance)
                
                logger.info(f"📈 Feature contributions calculated: {len(feature_contributions)} features")
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate feature contributions: {str(e)}")
        
        logger.info(f"✅ Prediction completed successfully")
        
        return PredictionResponse(
            model_id=request.model_id,
            prediction=final_prediction,
            prediction_proba=prediction_proba,
            confidence=confidence,
            feature_contributions=feature_contributions,
            input_processed=input_processed,
            success=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔥 Unexpected error in prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/models")
async def list_models():
    """List all trained models"""
    try:
        models_list = []
        
        for model_id, metadata in model_metadata_store.items():
            model_info = {
                'model_id': model_id,
                'job_id': metadata.get('job_id'),
                'table_names': metadata.get('table_names'),
                'target_column': metadata.get('target_column'),
                'problem_type': metadata.get('problem_type'),
                'feature_count': len(metadata.get('feature_columns', [])),
                'created_at': metadata.get('created_at'),
                'dataset_shape': metadata.get('dataset_shape')
            }
            
            if model_id in trained_models_store:
                model_data = trained_models_store[model_id]
                model_info['model_type'] = model_data['model_type']
                model_info['status'] = 'ready'
            else:
                model_info['status'] = 'metadata_only'
            
            models_list.append(model_info)
        
        return {
            'success': True,
            'total_models': len(models_list),
            'models': models_list
        }
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/models/{model_id}")
async def get_model_details(model_id: str):
    """Get detailed information about a specific model"""
    try:
        if model_id not in model_metadata_store:
            raise HTTPException(status_code=404, detail="Model not found")
        
        metadata = model_metadata_store[model_id]
        model_details = {
            'model_id': model_id,
            'metadata': metadata,
            'status': 'ready' if model_id in trained_models_store else 'metadata_only'
        }
        
        if model_id in trained_models_store:
            model_data = trained_models_store[model_id]
            model_details['model_info'] = {
                'model_type': model_data['model_type'],
                'feature_columns': model_data['feature_columns'],
                'problem_type': model_data['problem_type'],
                'has_scaler': model_data['scaler'] is not None,
                'has_target_encoder': model_data['target_encoder'] is not None,
                'encoders_count': len(model_data['encoders'])
            }
        
        return {
            'success': True,
            'model': model_details
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model details: {str(e)}")

# Debug endpoints
@router.get("/debug/model/{model_id}")
async def debug_model(model_id: str):
    """Debug endpoint to check model details"""
    try:
        if model_id not in trained_models_store:
            return {
                "error": "Model not found",
                "available_models": list(trained_models_store.keys())
            }
        
        model_data = trained_models_store[model_id]
        
        debug_info = {
            "model_id": model_id,
            "model_type": type(model_data['model']).__name__,
            "feature_columns": model_data['feature_columns'],
            "processed_feature_columns": model_data.get('processed_feature_columns', []),
            "problem_type": model_data['problem_type'],
            "has_scaler": model_data.get('scaler') is not None,
            "has_target_encoder": model_data.get('target_encoder') is not None,
            "encoders": {}
        }
        
        # Add encoder information
        for col, encoder_info in model_data.get('encoders', {}).items():
            debug_info["encoders"][col] = {
                "type": encoder_info['type'],
                "details": str(encoder_info)[:200] + "..." if len(str(encoder_info)) > 200 else str(encoder_info)
            }
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}

@router.post("/test-predict/{model_id}")
async def test_prediction(model_id: str):
    """Test prediction with sample data"""
    try:
        if model_id not in trained_models_store:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_data = trained_models_store[model_id]
        feature_columns = model_data['feature_columns']
        
        # Create sample input data
        sample_input = {}
        for col in feature_columns:
            # Generate sample values based on column name patterns
            if any(keyword in col.lower() for keyword in ['age', 'year', 'count', 'number']):
                sample_input[col] = 25
            elif any(keyword in col.lower() for keyword in ['price', 'amount', 'cost', 'salary']):
                sample_input[col] = 50000.0
            elif any(keyword in col.lower() for keyword in ['rate', 'ratio', 'percent']):
                sample_input[col] = 0.5
            else:
                sample_input[col] = "test_value"

        # Wrap in Pydantic request model
        test_request = PredictionRequest(
            model_id=model_id,
            input_data={"sample_1": sample_input}
        )

        return await make_prediction(test_request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    
