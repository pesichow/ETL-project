from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
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
from datetime import datetime
import asyncio
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
from pathlib import Path
import tempfile
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
router = APIRouter(prefix="/api/v1/chatbot", tags=["Chatbot Data Understanding"])

# Pydantic models
class ChatMessage(BaseModel):
    session_id: str
    message: str
    timestamp: Optional[datetime] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    data_insights: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[Dict[str, str]]] = None
    code_generated: Optional[str] = None
    transformation_applied: Optional[bool] = False
    timestamp: datetime

class DataTransformRequest(BaseModel):
    session_id: str
    transformation_query: str
    preview_only: bool = True

class ExportRequest(BaseModel):
    session_id: str
    export_format: str = "csv"  # csv, excel, json

# In-memory storage for sessions (in production, use Redis or database)
chat_sessions = {}
dataset_storage = {}
chat_history = {}

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
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Azure OpenAI API error: {str(e)}")
        return f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}"

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze dataset and return comprehensive insights"""
    try:
        analysis = {
            "basic_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            },
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": {},
            "unique_values": {},
            "data_quality": {
                "duplicate_rows": df.duplicated().sum(),
                "completeness_score": (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            }
        }
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["summary_stats"] = df[numeric_cols].describe().to_dict()
        
        # Unique values for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            analysis["unique_values"][col] = {
                "count": df[col].nunique(),
                "top_values": df[col].value_counts().head(10).to_dict()
            }
        
        return analysis
    except Exception as e:
        logger.error(f"Dataset analysis error: {str(e)}")
        return {"error": str(e)}

def create_visualization(df: pd.DataFrame, viz_type: str, columns: List[str] = None) -> str:
    """Create visualization and return base64 encoded image"""
    try:
        plt.figure(figsize=(10, 6))
        
        if viz_type == "correlation_heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
                plt.title("Correlation Heatmap")
            else:
                plt.text(0.5, 0.5, "Not enough numeric columns for correlation", 
                        ha='center', va='center', transform=plt.gca().transAxes)
        
        elif viz_type == "distribution" and columns:
            col = columns[0]
            if df[col].dtype in ['int64', 'float64']:
                plt.hist(df[col].dropna(), bins=30, alpha=0.7)
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
            else:
                df[col].value_counts().head(10).plot(kind='bar')
                plt.title(f"Top 10 values in {col}")
                plt.xticks(rotation=45)
        
        elif viz_type == "missing_values":
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            if len(missing_data) > 0:
                missing_data.plot(kind='bar')
                plt.title("Missing Values by Column")
                plt.ylabel("Number of Missing Values")
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, "No missing values found", 
                        ha='center', va='center', transform=plt.gca().transAxes)
        
        elif viz_type == "scatter" and len(columns) >= 2:
            plt.scatter(df[columns[0]], df[columns[1]], alpha=0.6)
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            plt.title(f"Scatter plot: {columns[0]} vs {columns[1]}")
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return None

def process_natural_language_query(query: str, df: pd.DataFrame, session_id: str) -> Dict[str, Any]:
    """Process natural language query and return appropriate response"""
    try:
        # Create context about the dataset
        dataset_context = f"""
        Dataset Information:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Columns: {', '.join(df.columns.tolist())}
        - Data types: {df.dtypes.to_dict()}
        - Missing values: {df.isnull().sum().to_dict()}
        """
        
        system_message = f"""
        You are a data analysis assistant. You have access to a dataset with the following information:
        {dataset_context}
        
        Based on the user's query, provide insights, suggest visualizations, or recommend data transformations.
        If the user asks for a specific analysis, provide the results and suggest relevant visualizations.
        If they ask for transformations, provide the pandas code to accomplish it.
        
        Always be specific and actionable in your responses.
        """
        
        response = get_azure_openai_response(query, system_message)
        
        # Determine if visualizations should be created based on query
        visualizations = []
        
        query_lower = query.lower()
        if any(word in query_lower for word in ['correlation', 'heatmap', 'relationship']):
            viz = create_visualization(df, "correlation_heatmap")
            if viz:
                visualizations.append({
                    "type": "correlation_heatmap",
                    "title": "Correlation Heatmap",
                    "image": viz
                })
        
        if any(word in query_lower for word in ['distribution', 'histogram', 'spread']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                viz = create_visualization(df, "distribution", [numeric_cols[0]])
                if viz:
                    visualizations.append({
                        "type": "distribution",
                        "title": f"Distribution of {numeric_cols[0]}",
                        "image": viz
                    })
        
        if any(word in query_lower for word in ['missing', 'null', 'empty']):
            viz = create_visualization(df, "missing_values")
            if viz:
                visualizations.append({
                    "type": "missing_values",
                    "title": "Missing Values Analysis",
                    "image": viz
                })
        
        # Generate data insights
        insights = analyze_dataset(df)
        
        return {
            "response": response,
            "visualizations": visualizations,
            "data_insights": insights,
            "code_generated": None
        }
    
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        return {
            "response": f"I encountered an error processing your query: {str(e)}",
            "visualizations": [],
            "data_insights": None,
            "code_generated": None
        }

@router.get("/")
async def chatbot_info():
    """Get chatbot interface information"""
    return {
        "service": "Chatbot Interface for Data Understanding",
        "description": "Natural language interface for data analysis and insights",
        "version": "1.0.0",
        "capabilities": [
            "Upload and analyze datasets",
            "Natural language data queries",
            "Automatic visualization generation",
            "Data transformation suggestions",
            "Export processed data and insights",
            "Chat history management"
        ],
        "endpoints": {
            "upload_dataset": "/upload-dataset",
            "chat": "/chat",
            "get_dataset_info": "/dataset-info/{session_id}",
            "apply_transformation": "/transform",
            "export_data": "/export",
            "chat_history": "/history/{session_id}",
            "download_visualization": "/download-viz/{session_id}",
            "sessions": "/sessions"
        },
        "supported_formats": ["CSV", "Excel", "JSON"],
        "azure_openai_configured": bool(os.getenv("AZURE_OPENAI_API_KEY"))
    }

@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dataset for chatbot analysis"""
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
        
        # Store dataset and initialize session
        dataset_storage[session_id] = df
        chat_sessions[session_id] = {
            "created_at": datetime.now(),
            "filename": file.filename,
            "dataset_shape": df.shape,
            "columns": df.columns.tolist()
        }
        chat_history[session_id] = []
        
        # Initial analysis
        initial_analysis = analyze_dataset(df)
        
        # Generate welcome message
        welcome_message = f"""
        Dataset uploaded successfully! Here's what I found:
        
        📊 **Dataset Overview:**
        - **Rows:** {df.shape[0]:,}
        - **Columns:** {df.shape[1]}
        - **Size:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
        
        📋 **Columns:** {', '.join(df.columns.tolist())}
        
        🔍 **Data Quality:**
        - **Completeness:** {initial_analysis['data_quality']['completeness_score']:.1f}%
        - **Duplicate Rows:** {initial_analysis['data_quality']['duplicate_rows']}
        
        💬 **What would you like to explore?**
        You can ask me questions like:
        - "Show me summary statistics"
        - "What columns have missing values?"
        - "Create a correlation heatmap"
        - "Show distribution of [column name]"
        - "Filter data where [condition]"
        """
        
        return JSONResponse({
            "session_id": session_id,
            "message": "Dataset uploaded successfully",
            "filename": file.filename,
            "dataset_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict()
            },
            "initial_analysis": initial_analysis,
            "welcome_message": welcome_message,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/chat")
async def chat_with_data(chat_request: ChatMessage):
    """Chat interface for data understanding"""
    try:
        session_id = chat_request.session_id
        
        if session_id not in dataset_storage:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a dataset first.")
        
        df = dataset_storage[session_id]
        
        # Process the natural language query
        result = process_natural_language_query(chat_request.message, df, session_id)
        
        # Create response
        response = ChatResponse(
            session_id=session_id,
            response=result["response"],
            data_insights=result["data_insights"],
            visualizations=result["visualizations"],
            code_generated=result["code_generated"],
            timestamp=datetime.now()
        )
        
        # Store in chat history
        if session_id not in chat_history:
            chat_history[session_id] = []
        
        chat_history[session_id].append({
            "user_message": chat_request.message,
            "bot_response": result["response"],
            "timestamp": datetime.now().isoformat(),
            "visualizations_count": len(result["visualizations"]) if result["visualizations"] else 0
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.get("/dataset-info/{session_id}")
async def get_dataset_info(session_id: str):
    """Get detailed dataset information"""
    try:
        if session_id not in dataset_storage:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = dataset_storage[session_id]
        analysis = analyze_dataset(df)
        
        return JSONResponse({
            "session_id": session_id,
            "dataset_analysis": analysis,
            "session_info": chat_sessions.get(session_id, {}),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Dataset info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transform")
async def apply_transformation(transform_request: DataTransformRequest):
    """Apply data transformation based on natural language request"""
    try:
        session_id = transform_request.session_id
        
        if session_id not in dataset_storage:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = dataset_storage[session_id]
        
        # Generate transformation code using LLM
        system_message = f"""
        You are a data transformation expert. Given a dataset with columns {df.columns.tolist()}, 
        generate pandas code to perform the requested transformation.
        
        Return only the pandas code that can be executed directly.
        Use 'df' as the dataframe variable name.
        
        Dataset info:
        - Shape: {df.shape}
        - Columns: {df.columns.tolist()}
        - Data types: {df.dtypes.to_dict()}
        """
        
        code_response = get_azure_openai_response(
            f"Generate pandas code for: {transform_request.transformation_query}",
            system_message
        )
        
        # Extract code from response (basic extraction)
        code_lines = []
        for line in code_response.split('\n'):
            if line.strip().startswith('df') or 'pd.' in line:
                code_lines.append(line.strip())
        
        generated_code = '\n'.join(code_lines) if code_lines else code_response
        
        if transform_request.preview_only:
            return JSONResponse({
                "session_id": session_id,
                "transformation_query": transform_request.transformation_query,
                "generated_code": generated_code,
                "preview_only": True,
                "message": "Transformation code generated. Set preview_only=false to apply.",
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Apply transformation (be careful with eval in production)
            try:
                # Create a safe execution environment
                exec_globals = {'df': df.copy(), 'pd': pd, 'np': np}
                exec(generated_code, exec_globals)
                transformed_df = exec_globals['df']
                
                # Update stored dataset
                dataset_storage[session_id] = transformed_df
                
                return JSONResponse({
                    "session_id": session_id,
                    "transformation_applied": True,
                    "generated_code": generated_code,
                    "new_shape": transformed_df.shape,
                    "message": "Transformation applied successfully",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as exec_error:
                return JSONResponse({
                    "session_id": session_id,
                    "transformation_applied": False,
                    "generated_code": generated_code,
                    "error": str(exec_error),
                    "message": "Failed to apply transformation",
                    "timestamp": datetime.now().isoformat()
                })
        
    except Exception as e:
        logger.error(f"Transformation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
async def export_data(export_request: ExportRequest):
    """Export processed dataset"""
    try:
        session_id = export_request.session_id
        
        if session_id not in dataset_storage:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = dataset_storage[session_id]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{export_request.export_format}') as tmp_file:
            if export_request.export_format == 'csv':
                df.to_csv(tmp_file.name, index=False)
            elif export_request.export_format == 'excel':
                df.to_excel(tmp_file.name, index=False)
            elif export_request.export_format == 'json':
                df.to_json(tmp_file.name, orient='records', indent=2)
            
            # Read file content and encode as base64
            with open(tmp_file.name, 'rb') as f:
                file_content = base64.b64encode(f.read()).decode()
            
            # Clean up temp file
            os.unlink(tmp_file.name)
        
        return JSONResponse({
            "session_id": session_id,
            "export_format": export_request.export_format,
            "file_content": file_content,
            "filename": f"processed_data_{session_id[:8]}.{export_request.export_format}",
            "dataset_shape": df.shape,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        if session_id not in chat_history:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return JSONResponse({
            "session_id": session_id,
            "chat_history": chat_history[session_id],
            "total_messages": len(chat_history[session_id]),
            "session_info": chat_sessions.get(session_id, {}),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-viz/{session_id}")
async def download_visualizations(session_id: str):
    """Download all visualizations for a session"""
    try:
        if session_id not in dataset_storage:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = dataset_storage[session_id]
        
        # Generate comprehensive visualizations
        visualizations = []
        
        # Correlation heatmap
        viz = create_visualization(df, "correlation_heatmap")
        if viz:
            visualizations.append({
                "type": "correlation_heatmap",
                "title": "Correlation Heatmap",
                "image": viz
            })
        
        # Missing values
        viz = create_visualization(df, "missing_values")
        if viz:
            visualizations.append({
                "type": "missing_values",
                "title": "Missing Values Analysis",
                "image": viz
            })
        
        # Distribution for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            viz = create_visualization(df, "distribution", [col])
            if viz:
                visualizations.append({
                    "type": "distribution",
                    "title": f"Distribution of {col}",
                    "image": viz
                })
        
        return JSONResponse({
            "session_id": session_id,
            "visualizations": visualizations,
            "total_visualizations": len(visualizations),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Visualization download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def get_all_sessions():
    """Get all active chat sessions"""
    try:
        sessions_info = []
        for session_id, session_data in chat_sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "created_at": session_data["created_at"].isoformat(),
                "filename": session_data["filename"],
                "dataset_shape": session_data["dataset_shape"],
                "columns_count": len(session_data["columns"]),
                "chat_messages": len(chat_history.get(session_id, []))
            })
        
        return JSONResponse({
            "total_sessions": len(sessions_info),
            "sessions": sessions_info,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Sessions error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and its data"""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Clean up session data
        if session_id in dataset_storage:
            del dataset_storage[session_id]
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        if session_id in chat_history:
            del chat_history[session_id]
        
        return JSONResponse({
            "message": "Session deleted successfully",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Delete session error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
