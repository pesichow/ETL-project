from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import uuid
import os
import time
import psutil
import traceback
from datetime import datetime, timedelta
import asyncio
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-04-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Create router
router = APIRouter(prefix="/api/v1/pipeline-optimization", tags=["Context-aware Pipeline Optimization"])

# Pydantic models
class PipelineStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_name: str
    step_type: str  # "data_source", "filter", "join", "transform", "ml", "output"
    engine: str = "pandas"  # "pandas", "spark", "sql"
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    input_rows: Optional[int] = None
    output_rows: Optional[int] = None
    parameters: Dict[str, Any] = {}
    dependencies: List[str] = []  # List of step_ids this step depends on

class DataSource(BaseModel):
    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_name: str
    source_type: str  # "csv", "database", "api", "stream"
    size_mb: Optional[float] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    data_quality_score: Optional[float] = None
    schema: Dict[str, str] = {}  # column_name: data_type

class PipelineDefinition(BaseModel):
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_name: str
    description: Optional[str] = None
    data_sources: List[DataSource]
    steps: List[PipelineStep]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class PerformanceMetrics(BaseModel):
    total_execution_time: float
    memory_peak_usage: float
    cpu_usage: float
    data_throughput: float  # rows per second
    bottleneck_steps: List[str] = []
    resource_utilization: Dict[str, float] = {}

class DataQualityMetrics(BaseModel):
    missing_values_percentage: float
    duplicate_rows_percentage: float
    data_skewness: Dict[str, float] = {}
    outliers_percentage: float
    data_consistency_score: float
    schema_violations: List[str] = []

class OptimizationSuggestion(BaseModel):
    suggestion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: str  # "performance", "memory", "data_quality", "architecture"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    explanation: str
    impact_estimate: str  # "20% faster execution", "30% memory reduction"
    affected_steps: List[str] = []
    implementation_code: Optional[str] = None
    one_click_applicable: bool = False
    estimated_effort: str  # "low", "medium", "high"
    confidence_score: float = 0.0

class PipelineAnalysisRequest(BaseModel):
    session_id: str
    pipeline_definition: PipelineDefinition
    include_performance_analysis: bool = True
    include_data_quality_analysis: bool = True
    optimization_focus: List[str] = ["performance", "memory", "data_quality"]

class ApplySuggestionRequest(BaseModel):
    session_id: str
    suggestion_id: str
    auto_apply: bool = False
    custom_parameters: Dict[str, Any] = {}

class PipelineOptimizationSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_definition: PipelineDefinition
    performance_metrics: Optional[PerformanceMetrics] = None
    data_quality_metrics: Optional[DataQualityMetrics] = None
    suggestions: List[OptimizationSuggestion] = []
    applied_suggestions: List[str] = []
    optimization_history: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# In-memory storage
optimization_sessions = {}
datasets = {}
pipeline_templates = {}

def get_azure_openai_response(prompt: str, system_message: str = None) -> str:
    """Get response from Azure OpenAI"""
    try:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            messages=messages,
            max_tokens=2000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Azure OpenAI API error: {str(e)}")
        return f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}"

def analyze_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
    """Analyze data quality metrics"""
    try:
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
        
        # Calculate skewness for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewness = {}
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                skewness[col] = float(df[col].skew())
        
        # Estimate outliers using IQR method
        outliers_count = 0
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                outliers_count += len(outliers)
        
        outliers_percentage = (outliers_count / len(df)) * 100 if len(df) > 0 else 0
        
        # Simple data consistency score (inverse of missing + duplicates + outliers)
        consistency_score = max(0, 100 - missing_percentage - duplicate_percentage - (outliers_percentage / 10))
        
        return DataQualityMetrics(
            missing_values_percentage=missing_percentage,
            duplicate_rows_percentage=duplicate_percentage,
            data_skewness=skewness,
            outliers_percentage=outliers_percentage,
            data_consistency_score=consistency_score,
            schema_violations=[]
        )
    
    except Exception as e:
        logger.error(f"Data quality analysis error: {str(e)}")
        return DataQualityMetrics(
            missing_values_percentage=0,
            duplicate_rows_percentage=0,
            outliers_percentage=0,
            data_consistency_score=50
        )

def simulate_performance_metrics(pipeline: PipelineDefinition, df: pd.DataFrame) -> PerformanceMetrics:
    """Simulate performance metrics based on pipeline complexity and data size"""
    try:
        # Simulate execution time based on data size and pipeline complexity
        base_time = len(df) * len(df.columns) / 100000  # Base time in seconds
        complexity_multiplier = len(pipeline.steps) * 0.1
        total_execution_time = base_time * (1 + complexity_multiplier)
        
        # Simulate memory usage
        data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        memory_multiplier = 1.5 + (len(pipeline.steps) * 0.2)
        memory_peak_usage = data_size_mb * memory_multiplier
        
        # Simulate CPU usage
        cpu_usage = min(95, 30 + (len(pipeline.steps) * 5))
        
        # Calculate data throughput
        data_throughput = len(df) / total_execution_time if total_execution_time > 0 else 0
        
        # Identify potential bottleneck steps
        bottleneck_steps = []
        for step in pipeline.steps:
            if step.step_type in ["join", "ml"] or step.engine == "pandas":
                bottleneck_steps.append(step.step_id)
        
        return PerformanceMetrics(
            total_execution_time=total_execution_time,
            memory_peak_usage=memory_peak_usage,
            cpu_usage=cpu_usage,
            data_throughput=data_throughput,
            bottleneck_steps=bottleneck_steps,
            resource_utilization={
                "cpu": cpu_usage,
                "memory": min(95, memory_peak_usage / 100),
                "disk_io": min(90, len(pipeline.steps) * 10)
            }
        )
    
    except Exception as e:
        logger.error(f"Performance metrics simulation error: {str(e)}")
        return PerformanceMetrics(
            total_execution_time=10.0,
            memory_peak_usage=100.0,
            cpu_usage=50.0,
            data_throughput=1000.0
        )

def generate_optimization_suggestions(
    pipeline: PipelineDefinition, 
    performance_metrics: PerformanceMetrics,
    data_quality_metrics: DataQualityMetrics,
    df: pd.DataFrame
) -> List[OptimizationSuggestion]:
    """Generate AI-powered optimization suggestions"""
    try:
        suggestions = []
        
        # Create context for LLM
        pipeline_context = f"""
        Pipeline Analysis Context:
        - Pipeline Name: {pipeline.pipeline_name}
        - Total Steps: {len(pipeline.steps)}
        - Data Sources: {len(pipeline.data_sources)}
        - Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Execution Time: {performance_metrics.total_execution_time:.2f} seconds
        - Memory Usage: {performance_metrics.memory_peak_usage:.2f} MB
        - CPU Usage: {performance_metrics.cpu_usage:.1f}%
        - Data Quality Score: {data_quality_metrics.data_consistency_score:.1f}%
        - Missing Values: {data_quality_metrics.missing_values_percentage:.1f}%
        - Duplicate Rows: {data_quality_metrics.duplicate_rows_percentage:.1f}%
        
        Pipeline Steps:
        """
        
        for i, step in enumerate(pipeline.steps):
            pipeline_context += f"\n{i+1}. {step.step_name} ({step.step_type}) - Engine: {step.engine}"
        
        # Performance optimization suggestions
        if performance_metrics.total_execution_time > 5:
            system_message = """
            You are an expert data pipeline optimization consultant. Analyze the pipeline and provide specific, actionable optimization suggestions.
            Focus on performance improvements, memory optimization, and architectural enhancements.
            Each suggestion should include a clear title, description, explanation, and impact estimate.
            """
            
            prompt = f"""
            Based on the following pipeline analysis, provide 3-5 specific optimization suggestions:
            
            {pipeline_context}
            
            Focus on:
            1. Performance bottlenecks
            2. Memory optimization
            3. Engine selection (pandas vs Spark)
            4. Step ordering and dependencies
            5. Data processing efficiency
            
            Format each suggestion as:
            TITLE: [Clear, actionable title]
            DESCRIPTION: [What to do]
            EXPLANATION: [Why this helps]
            IMPACT: [Expected improvement]
            CATEGORY: [performance/memory/architecture]
            PRIORITY: [high/medium/low]
            """
            
            response = get_azure_openai_response(prompt, system_message)
            
            # Parse LLM response into suggestions
            suggestion_blocks = response.split("TITLE:")
            for block in suggestion_blocks[1:]:  # Skip first empty block
                try:
                    lines = block.strip().split("\n")
                    title = lines[0].strip()
                    
                    description = ""
                    explanation = ""
                    impact = ""
                    category = "performance"
                    priority = "medium"
                    
                    current_section = ""
                    for line in lines[1:]:
                        line = line.strip()
                        if line.startswith("DESCRIPTION:"):
                            current_section = "description"
                            description = line.replace("DESCRIPTION:", "").strip()
                        elif line.startswith("EXPLANATION:"):
                            current_section = "explanation"
                            explanation = line.replace("EXPLANATION:", "").strip()
                        elif line.startswith("IMPACT:"):
                            current_section = "impact"
                            impact = line.replace("IMPACT:", "").strip()
                        elif line.startswith("CATEGORY:"):
                            category = line.replace("CATEGORY:", "").strip().lower()
                        elif line.startswith("PRIORITY:"):
                            priority = line.replace("PRIORITY:", "").strip().lower()
                        elif current_section and line:
                            if current_section == "description":
                                description += " " + line
                            elif current_section == "explanation":
                                explanation += " " + line
                            elif current_section == "impact":
                                impact += " " + line
                    
                    if title and description:
                        suggestion = OptimizationSuggestion(
                            category=category,
                            priority=priority,
                            title=title,
                            description=description,
                            explanation=explanation,
                            impact_estimate=impact,
                            confidence_score=0.8,
                            estimated_effort="medium"
                        )
                        suggestions.append(suggestion)
                
                except Exception as e:
                    logger.error(f"Error parsing suggestion: {str(e)}")
                    continue
        
        # Add rule-based suggestions
        
        # Memory optimization
        if performance_metrics.memory_peak_usage > 500:  # > 500MB
            suggestions.append(OptimizationSuggestion(
                category="memory",
                priority="high",
                title="Optimize Memory Usage with Column Selection",
                description="Drop unused columns early in the pipeline to reduce memory footprint",
                explanation="Removing unnecessary columns at the beginning of the pipeline reduces memory usage throughout all subsequent operations",
                impact_estimate="30-50% memory reduction",
                confidence_score=0.9,
                estimated_effort="low",
                one_click_applicable=True,
                implementation_code="df = df[['col1', 'col2', 'col3']]  # Keep only necessary columns"
            ))
        
        # Data quality suggestions
        if data_quality_metrics.missing_values_percentage > 10:
            suggestions.append(OptimizationSuggestion(
                category="data_quality",
                priority="high",
                title="Handle Missing Values Early",
                description="Address missing values at the beginning of the pipeline to improve downstream processing",
                explanation="Handling missing values early prevents propagation of null values and improves model accuracy",
                impact_estimate="Improved data quality and model performance",
                confidence_score=0.85,
                estimated_effort="medium",
                implementation_code="df = df.fillna(method='forward')  # or df.dropna()"
            ))
        
        # Engine optimization
        if len(df) > 100000 and any(step.engine == "pandas" for step in pipeline.steps):
            suggestions.append(OptimizationSuggestion(
                category="performance",
                priority="medium",
                title="Consider Spark for Large Dataset Processing",
                description="Switch to Spark engine for operations on large datasets (>100K rows)",
                explanation="Spark provides distributed processing capabilities that can significantly improve performance on large datasets",
                impact_estimate="2-5x faster execution for large datasets",
                confidence_score=0.75,
                estimated_effort="high"
            ))
        
        # Join optimization
        join_steps = [step for step in pipeline.steps if step.step_type == "join"]
        if join_steps:
            suggestions.append(OptimizationSuggestion(
                category="performance",
                priority="medium",
                title="Optimize Join Operations",
                description="Consider using broadcast joins for small tables and ensure proper indexing",
                explanation="Broadcast joins are more efficient when one table is significantly smaller than the other",
                impact_estimate="20-40% faster join operations",
                confidence_score=0.8,
                estimated_effort="medium",
                implementation_code="# Use broadcast join for small tables\ndf_result = df_large.join(broadcast(df_small), on='key')"
            ))
        
        return suggestions
    
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return []

def create_optimization_visualization(session: PipelineOptimizationSession) -> str:
    """Create visualization of optimization metrics"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics
        if session.performance_metrics:
            metrics = ['Execution Time', 'Memory Usage', 'CPU Usage', 'Data Throughput']
            values = [
                session.performance_metrics.total_execution_time,
                session.performance_metrics.memory_peak_usage / 100,  # Normalize
                session.performance_metrics.cpu_usage,
                session.performance_metrics.data_throughput / 1000  # Normalize
            ]
            
            ax1.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax1.set_title('Performance Metrics')
            ax1.set_ylabel('Normalized Values')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Data quality metrics
        if session.data_quality_metrics:
            quality_metrics = ['Missing Values %', 'Duplicates %', 'Outliers %', 'Consistency Score']
            quality_values = [
                session.data_quality_metrics.missing_values_percentage,
                session.data_quality_metrics.duplicate_rows_percentage,
                session.data_quality_metrics.outliers_percentage,
                session.data_quality_metrics.data_consistency_score
            ]
            
            ax2.bar(quality_metrics, quality_values, color=['#FF8A80', '#FFB74D', '#FFF176', '#81C784'])
            ax2.set_title('Data Quality Metrics')
            ax2.set_ylabel('Percentage')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Suggestions by category
        if session.suggestions:
            categories = {}
            for suggestion in session.suggestions:
                categories[suggestion.category] = categories.get(suggestion.category, 0) + 1
            
            ax3.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%', startangle=90)
            ax3.set_title('Optimization Suggestions by Category')
        
        # Priority distribution
        if session.suggestions:
            priorities = {'high': 0, 'medium': 0, 'low': 0}
            for suggestion in session.suggestions:
                priorities[suggestion.priority] = priorities.get(suggestion.priority, 0) + 1
            
            colors = {'high': '#FF6B6B', 'medium': '#FFB74D', 'low': '#81C784'}
            ax4.bar(priorities.keys(), priorities.values(), 
                   color=[colors[p] for p in priorities.keys()])
            ax4.set_title('Suggestions by Priority')
            ax4.set_ylabel('Count')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return None

@router.get("/")
async def get_info():
    """Get information about the Pipeline Optimization API"""
    return {
        "service": "Context-aware Suggestions for Pipeline Optimization",
        "description": "AI-powered pipeline analysis and optimization recommendations",
        "version": "1.0.0",
        "capabilities": [
            "Pipeline performance analysis",
            "Data quality assessment",
            "AI-powered optimization suggestions",
            "One-click suggestion application",
            "Continuous pipeline monitoring",
            "Visualization and reporting"
        ],
        "endpoints": {
            "upload_dataset": "/upload-dataset",
            "analyze_pipeline": "/analyze-pipeline",
            "get_suggestions": "/suggestions/{session_id}",
            "apply_suggestion": "/apply-suggestion",
            "get_session": "/session/{session_id}",
            "optimize_pipeline": "/optimize/{session_id}",
            "export_report": "/export-report/{session_id}",
            "sessions": "/sessions"
        },
        "optimization_categories": [
            "performance", "memory", "data_quality", "architecture"
        ],
        "azure_openai_configured": bool(os.getenv("AZURE_OPENAI_API_KEY"))
    }

@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dataset for pipeline optimization analysis"""
    try:
        session_id = str(uuid.uuid4())
        
        # Read file based on extension
        file_extension = file.filename.split('.')[-1].lower()
        content = await file.read()
        
        if file_extension == 'csv':
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(content))
        elif file_extension == 'json':
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Store dataset
        datasets[session_id] = df
        
        # Create basic pipeline definition
        data_source = DataSource(
            source_name=file.filename,
            source_type=file_extension,
            size_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            row_count=len(df),
            column_count=len(df.columns),
            schema={col: str(df[col].dtype) for col in df.columns}
        )
        
        # Create default pipeline steps
        steps = [
            PipelineStep(
                step_name="Data Loading",
                step_type="data_source",
                engine="pandas",
                input_rows=0,
                output_rows=len(df)
            ),
            PipelineStep(
                step_name="Data Validation",
                step_type="transform",
                engine="pandas",
                input_rows=len(df),
                output_rows=len(df)
            )
        ]
        
        pipeline = PipelineDefinition(
            pipeline_name=f"Pipeline for {file.filename}",
            description="Auto-generated pipeline from uploaded dataset",
            data_sources=[data_source],
            steps=steps
        )
        
        # Analyze data quality
        data_quality = analyze_data_quality(df)
        
        # Simulate performance metrics
        performance = simulate_performance_metrics(pipeline, df)
        
        # Create optimization session
        session = PipelineOptimizationSession(
            session_id=session_id,
            pipeline_definition=pipeline,
            performance_metrics=performance,
            data_quality_metrics=data_quality
        )
        
        optimization_sessions[session_id] = session
        
        return JSONResponse({
            "session_id": session_id,
            "message": "Dataset uploaded and pipeline initialized",
            "filename": file.filename,
            "dataset_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(df[col].dtype) for col in df.columns}
            },
            "pipeline_summary": {
                "total_steps": len(pipeline.steps),
                "data_sources": len(pipeline.data_sources),
                "estimated_execution_time": performance.total_execution_time,
                "estimated_memory_usage": performance.memory_peak_usage
            },
            "data_quality_summary": {
                "consistency_score": data_quality.data_consistency_score,
                "missing_values_percentage": data_quality.missing_values_percentage,
                "duplicate_rows_percentage": data_quality.duplicate_rows_percentage
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/analyze-pipeline")
async def analyze_pipeline(request: PipelineAnalysisRequest):
    """Analyze pipeline and generate optimization suggestions"""
    try:
        session_id = request.session_id
        
        if session_id not in datasets:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a dataset first.")
        
        df = datasets[session_id]
        
        # Update session with new pipeline definition
        if session_id in optimization_sessions:
            session = optimization_sessions[session_id]
            session.pipeline_definition = request.pipeline_definition
            session.updated_at = datetime.now()
        else:
            session = PipelineOptimizationSession(
                session_id=session_id,
                pipeline_definition=request.pipeline_definition
            )
        
        # Perform analysis
        if request.include_performance_analysis:
            session.performance_metrics = simulate_performance_metrics(request.pipeline_definition, df)
        
        if request.include_data_quality_analysis:
            session.data_quality_metrics = analyze_data_quality(df)
        
        # Generate optimization suggestions
        suggestions = generate_optimization_suggestions(
            request.pipeline_definition,
            session.performance_metrics,
            session.data_quality_metrics,
            df
        )
        
        session.suggestions = suggestions
        optimization_sessions[session_id] = session
        
        return JSONResponse({
            "session_id": session_id,
            "analysis_complete": True,
            "pipeline_name": request.pipeline_definition.pipeline_name,
            "performance_metrics": session.performance_metrics.dict() if session.performance_metrics else None,
            "data_quality_metrics": session.data_quality_metrics.dict() if session.data_quality_metrics else None,
            "suggestions_count": len(suggestions),
            "high_priority_suggestions": len([s for s in suggestions if s.priority == "high"]),
            "one_click_applicable": len([s for s in suggestions if s.one_click_applicable]),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Pipeline analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline analysis failed: {str(e)}")

@router.get("/suggestions/{session_id}")
async def get_suggestions(session_id: str):
    """Get optimization suggestions for a session"""
    try:
        if session_id not in optimization_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = optimization_sessions[session_id]
        
        # Group suggestions by category
        suggestions_by_category = {}
        for suggestion in session.suggestions:
            if suggestion.category not in suggestions_by_category:
                suggestions_by_category[suggestion.category] = []
            suggestions_by_category[suggestion.category].append(suggestion.dict())
        
        return JSONResponse({
            "session_id": session_id,
            "pipeline_name": session.pipeline_definition.pipeline_name,
            "total_suggestions": len(session.suggestions),
            "suggestions_by_category": suggestions_by_category,
            "applied_suggestions": session.applied_suggestions,
            "summary": {
                "high_priority": len([s for s in session.suggestions if s.priority == "high"]),
                "medium_priority": len([s for s in session.suggestions if s.priority == "medium"]),
                "low_priority": len([s for s in session.suggestions if s.priority == "low"]),
                "one_click_applicable": len([s for s in session.suggestions if s.one_click_applicable])
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get suggestions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

@router.post("/apply-suggestion")
async def apply_suggestion(request: ApplySuggestionRequest):
    """Apply an optimization suggestion"""
    try:
        session_id = request.session_id
        
        if session_id not in optimization_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = optimization_sessions[session_id]
        
        # Find the suggestion
        suggestion = None
        for s in session.suggestions:
            if s.suggestion_id == request.suggestion_id:
                suggestion = s
                break
        
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        # Apply the suggestion
        if request.auto_apply and suggestion.one_click_applicable and suggestion.implementation_code:
            try:
                # In a real implementation, you would apply the code to the pipeline
                # For now, we'll simulate the application
                session.applied_suggestions.append(request.suggestion_id)
                
                # Add to optimization history
                session.optimization_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "suggestion_id": request.suggestion_id,
                    "suggestion_title": suggestion.title,
                    "action": "applied",
                    "auto_applied": True,
                    "custom_parameters": request.custom_parameters
                })
                
                # Simulate improved metrics
                if session.performance_metrics and suggestion.category == "performance":
                    session.performance_metrics.total_execution_time *= 0.8  # 20% improvement
                
                optimization_sessions[session_id] = session
                
                return JSONResponse({
                    "session_id": session_id,
                    "suggestion_id": request.suggestion_id,
                    "applied": True,
                    "auto_applied": True,
                    "message": f"Successfully applied: {suggestion.title}",
                    "estimated_impact": suggestion.impact_estimate,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({
                    "session_id": session_id,
                    "suggestion_id": request.suggestion_id,
                    "applied": False,
                    "error": str(e),
                    "message": "Failed to auto-apply suggestion",
                    "timestamp": datetime.now().isoformat()
                })
        else:
            # Manual application - just mark as applied
            session.applied_suggestions.append(request.suggestion_id)
            session.optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "suggestion_id": request.suggestion_id,
                "suggestion_title": suggestion.title,
                "action": "applied",
                "auto_applied": False,
                "custom_parameters": request.custom_parameters
            })
            
            optimization_sessions[session_id] = session
            
            return JSONResponse({
                "session_id": session_id,
                "suggestion_id": request.suggestion_id,
                "applied": True,
                "auto_applied": False,
                "message": f"Marked as applied: {suggestion.title}",
                "implementation_code": suggestion.implementation_code,
                "timestamp": datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Apply suggestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to apply suggestion: {str(e)}")

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get complete session information"""
    try:
        if session_id not in optimization_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = optimization_sessions[session_id]
        
        return JSONResponse({
            "session": session.dict(),
            "dataset_info": {
                "shape": datasets[session_id].shape if session_id in datasets else None,
                "columns": datasets[session_id].columns.tolist() if session_id in datasets else None
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get session error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@router.post("/optimize/{session_id}")
async def optimize_pipeline(session_id: str):
    """Run comprehensive pipeline optimization"""
    try:
        if session_id not in optimization_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = optimization_sessions[session_id]
        df = datasets[session_id]
        
        # Re-analyze with current pipeline state
        session.performance_metrics = simulate_performance_metrics(session.pipeline_definition, df)
        session.data_quality_metrics = analyze_data_quality(df)
        
        # Generate fresh suggestions
        suggestions = generate_optimization_suggestions(
            session.pipeline_definition,
            session.performance_metrics,
            session.data_quality_metrics,
            df
        )
        
        # Filter out already applied suggestions
        new_suggestions = [s for s in suggestions if s.suggestion_id not in session.applied_suggestions]
        session.suggestions = new_suggestions
        
        # Create optimization visualization
        visualization = create_optimization_visualization(session)
        
        optimization_sessions[session_id] = session
        
        return JSONResponse({
            "session_id": session_id,
            "optimization_complete": True,
            "new_suggestions_count": len(new_suggestions),
            "performance_metrics": session.performance_metrics.dict(),
            "data_quality_metrics": session.data_quality_metrics.dict(),
            "visualization": visualization,
            "recommendations": {
                "immediate_actions": [s.dict() for s in new_suggestions if s.priority == "high"],
                "quick_wins": [s.dict() for s in new_suggestions if s.one_click_applicable],
                "long_term_improvements": [s.dict() for s in new_suggestions if s.estimated_effort == "high"]
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Optimize pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline optimization failed: {str(e)}")

@router.get("/export-report/{session_id}")
async def export_optimization_report(session_id: str):
    """Export comprehensive optimization report"""
    try:
        if session_id not in optimization_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = optimization_sessions[session_id]
        
        # Create comprehensive report
        report = {
            "report_metadata": {
                "session_id": session_id,
                "pipeline_name": session.pipeline_definition.pipeline_name,
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0"
            },
            "executive_summary": {
                "total_suggestions": len(session.suggestions),
                "applied_suggestions": len(session.applied_suggestions),
                "potential_improvements": {
                    "performance": len([s for s in session.suggestions if s.category == "performance"]),
                    "memory": len([s for s in session.suggestions if s.category == "memory"]),
                    "data_quality": len([s for s in session.suggestions if s.category == "data_quality"])
                }
            },
            "pipeline_analysis": {
                "pipeline_definition": session.pipeline_definition.dict(),
                "performance_metrics": session.performance_metrics.dict() if session.performance_metrics else None,
                "data_quality_metrics": session.data_quality_metrics.dict() if session.data_quality_metrics else None
            },
            "optimization_suggestions": [s.dict() for s in session.suggestions],
            "optimization_history": session.optimization_history,
            "visualization": create_optimization_visualization(session)
        }
        
        return JSONResponse({
            "session_id": session_id,
            "report": report,
            "export_format": "json",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Export report error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export report: {str(e)}")

@router.get("/sessions")
async def get_all_sessions():
    """Get all optimization sessions"""
    try:
        sessions_info = []
        for session_id, session in optimization_sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "pipeline_name": session.pipeline_definition.pipeline_name,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "total_suggestions": len(session.suggestions),
                "applied_suggestions": len(session.applied_suggestions),
                "data_quality_score": session.data_quality_metrics.data_consistency_score if session.data_quality_metrics else None,
                "execution_time": session.performance_metrics.total_execution_time if session.performance_metrics else None
            })
        
        return JSONResponse({
            "total_sessions": len(sessions_info),
            "sessions": sessions_info,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get sessions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete an optimization session"""
    try:
        if session_id not in optimization_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Clean up session data
        if session_id in optimization_sessions:
            del optimization_sessions[session_id]
        if session_id in datasets:
            del datasets[session_id]
        
        return JSONResponse({
            "message": "Session deleted successfully",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Delete session error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@router.get("/templates")
async def get_pipeline_templates():
    """Get available pipeline templates"""
    try:
        templates = {
            "data_ingestion": {
                "name": "Data Ingestion Pipeline",
                "description": "Template for data ingestion and basic cleaning",
                "steps": ["data_source", "validation", "cleaning", "output"]
            },
            "ml_training": {
                "name": "ML Training Pipeline",
                "description": "Template for machine learning model training",
                "steps": ["data_source", "preprocessing", "feature_engineering", "model_training", "evaluation"]
            },
            "etl_batch": {
                "name": "Batch ETL Pipeline",
                "description": "Template for batch ETL processing",
                "steps": ["extract", "transform", "validate", "load"]
            }
        }
        
        return JSONResponse({
            "templates": templates,
            "total_templates": len(templates),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get templates error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")
