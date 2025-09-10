from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from dotenv import load_dotenv
from upload_handler import router as upload_router
from data_profiling_backend import router as data_profiling_router
from automl_backend import router as automl_backend_router
from genai_docs_backend import router as genai_docs_backend_router
from embedded_ml_clustering_backend import router as embedded_ml_clustering_backend_router
from deploy_to_alteryx_promote import router as deploy_to_alteryx_promote_router
from llm_code_generation_backend import router as llm_code_generation_backend_router
from llm_mesh_backend import router as llm_mesh_backend_router
from model_governance_backend import router as model_governance_backend_router
from automated_feature_engineering_backend import router as automated_feature_engineering_backend_router
from prompt_engineering_interface import router as prompt_engineering_interface_router
from nlp_tools_backend import router as nlp_tools_backend_router
from object_detection import router as object_detection_router
from time_series_forecasting_backend import router as time_series_forecasting_backend_router
from text_classification_summarization_backend import router as text_classification_summarization_backend_router
from anomaly_detection_backend import router as anomaly_detection_backend_router
from language_converter_backend import router as language_converter_backend_router
from geospatial_analytics_backend import router as geospatial_analytics_backend_router
from predictive_modeling_backend import router as predictive_modeling_backend_router
from text_mining_backend import router as text_mining_backend_router
from out_of_box_ml_backend import router as out_of_box_ml_backend_router
from integration_notebooks_backend import router as integration_notebooks_backend_router
from ai_copilot_backend import router as ai_copilot_backend_router
from data_drift_detection_backend import router as data_drift_detection_backend_router
from assisted_modeling_backend import router as assisted_modeling_backend_router
from ai_powered_data_prep_backend import router as ai_powered_data_prep_backend_router
from notebooks_backend import router as notebooks_backend_router
from claire_engine_backend import router as claire_engine_backend_router
from model_versioning_monitoring_backend import router as model_versioning_monitoring_backend_router
from chatbot_interface_backend import router as chatbot_interface_backend_router
from visual_sql_spark_backend import router as visual_sql_spark_backend_router
from context_aware_pipeline_optimization_backend import router as context_aware_pipeline_optimization_backend_router
from prompt_driven_data_transformation_backend import router as prompt_driven_data_transformation_backend_router  # NEW IMPORT
from code_free_modeling_backend import router as code_free_modeling_backend_router
from data_storage import data_store
# Add this with your other import statements
from fuzzy_matching_backend import router as fuzzy_matching_router
 # NEW IMPORT FUZZY
# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GenAI ETL Backend - Full-stack GenAI-powered Data Analysis Platform",
    description="A comprehensive FastAPI backend for ETL operations with AutoML and GenAI document processing capabilities, modeled like Dataiku's AI/ML tools",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload_router, prefix="/api/v1", tags=["File Upload"])
app.include_router(data_profiling_router, prefix="/api/v1/profiling", tags=["Data Profiling"])
app.include_router(automated_feature_engineering_backend_router)
app.include_router(automl_backend_router)
app.include_router(genai_docs_backend_router)
app.include_router(embedded_ml_clustering_backend_router, prefix="/api/v1/embedded-ml", tags=["Embedded ML Clustering"])
app.include_router(deploy_to_alteryx_promote_router)
app.include_router(llm_code_generation_backend_router)
app.include_router(llm_mesh_backend_router)
app.include_router(model_governance_backend_router)
app.include_router(prompt_engineering_interface_router)
app.include_router(object_detection_router)
app.include_router(time_series_forecasting_backend_router)
app.include_router(text_classification_summarization_backend_router)
app.include_router(anomaly_detection_backend_router)
app.include_router(language_converter_backend_router)
app.include_router(geospatial_analytics_backend_router)
app.include_router(predictive_modeling_backend_router)
app.include_router(text_mining_backend_router)
app.include_router(out_of_box_ml_backend_router)
app.include_router(integration_notebooks_backend_router, prefix="/api/v1/integration-notebooks", tags=["Integration Notebooks"])
app.include_router(ai_copilot_backend_router)
app.include_router(data_drift_detection_backend_router)
app.include_router(assisted_modeling_backend_router)
app.include_router(ai_powered_data_prep_backend_router)
app.include_router(notebooks_backend_router)
app.include_router(claire_engine_backend_router)
app.include_router(model_versioning_monitoring_backend_router)
app.include_router(chatbot_interface_backend_router)
app.include_router(visual_sql_spark_backend_router)
app.include_router(context_aware_pipeline_optimization_backend_router)
app.include_router(prompt_driven_data_transformation_backend_router)  # NEW ROUTER REGISTRATION
app.include_router(nlp_tools_backend_router)
app.include_router(code_free_modeling_backend_router)
app.include_router(fuzzy_matching_router)
# Add this line with your other app.include_router() calls

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    logger.info(f"WebSocket connected for session: {session_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected for session: {session_id}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "GenAI ETL Backend"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "GenAI ETL Backend API - Full-stack GenAI-powered Data Analysis Platform",
        "description": "Comprehensive AI/ML platform modeled like Dataiku's tools",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/{session_id}",
        "azure_openai_configured": bool(os.getenv("AZURE_OPENAI_API_KEY")),
        "services": {
            "prompt_transformation": {  # NEW SERVICE
                "info": "/api/v1/prompt-transformation/",
                "upload_dataset": "/api/v1/prompt-transformation/upload-dataset",
                "transform": "/api/v1/prompt-transformation/transform",
                "preview": "/api/v1/prompt-transformation/preview-transformation",
                "visualize": "/api/v1/prompt-transformation/visualize",
                "history": "/api/v1/prompt-transformation/history/{session_id}",
                "export": "/api/v1/prompt-transformation/export/{session_id}",
                "dataset_info": "/api/v1/prompt-transformation/dataset-info/{session_id}",
                "sessions": "/api/v1/prompt-transformation/sessions",
                "delete_session": "/api/v1/prompt-transformation/session/{session_id}"
            },
            #FUZZY ADD
            "fuzzy_matching": {
                "info": "/api/v1/fuzzy/",
                "upload_dataset": "/api/v1/fuzzy/upload-dataset",
                "match": "/api/v1/fuzzy/match",
                "bulk_match": "/api/v1/fuzzy/bulk-match",
                "session_info": "/api/v1/fuzzy/session-info/{session_id}"
            },
            "pipeline_optimization": {
                "info": "/api/v1/pipeline-optimization/",
                "upload_dataset": "/api/v1/pipeline-optimization/upload-dataset",
                "analyze_pipeline": "/api/v1/pipeline-optimization/analyze-pipeline",
                "suggestions": "/api/v1/pipeline-optimization/suggestions/{session_id}",
                "apply_suggestion": "/api/v1/pipeline-optimization/apply-suggestion",
                "optimize": "/api/v1/pipeline-optimization/optimize/{session_id}",
                "export_report": "/api/v1/pipeline-optimization/export-report/{session_id}",
                "sessions": "/api/v1/pipeline-optimization/sessions",
                "templates": "/api/v1/pipeline-optimization/templates"
            },
            "visual_sql_spark": {
                "info": "/api/v1/visual-sql-spark/",
                "upload_dataset": "/api/v1/visual-sql-spark/upload-dataset",
                "execute_sql": "/api/v1/visual-sql-spark/execute-sql",
                "execute_spark": "/api/v1/visual-sql-spark/execute-spark",
                "nl_query": "/api/v1/visual-sql-spark/nl-query",
                "visualize": "/api/v1/visual-sql-spark/visualize",
                "recipes": "/api/v1/visual-sql-spark/recipes/{session_id}",
                "export": "/api/v1/visual-sql-spark/export",
                "sessions": "/api/v1/visual-sql-spark/sessions"
            },
            "chatbot_interface": {
                "info": "/api/v1/chatbot/",
                "upload_dataset": "/api/v1/chatbot/upload-dataset",
                "chat": "/api/v1/chatbot/chat",
                "dataset_info": "/api/v1/chatbot/dataset-info/{session_id}",
                "transform": "/api/v1/chatbot/transform",
                "export": "/api/v1/chatbot/export",
                "history": "/api/v1/chatbot/history/{session_id}",
                "download_viz": "/api/v1/chatbot/download-viz/{session_id}",
                "sessions": "/api/v1/chatbot/sessions",
                "delete_session": "/api/v1/chatbot/session/{session_id}"
            },
            "claire_engine": {
                "info": "/api/v1/claire/",
                "ingest": "/api/v1/claire/ingest-dataset",
                "insights": "/api/v1/claire/column-insights/{session_id}",
                "suggestions": "/api/v1/claire/data-prep-suggestions",
                "lineage": "/api/v1/claire/data-lineage",
                "features": "/api/v1/claire/feature-recommendations",
                "governance": "/api/v1/claire/governance-check/{session_id}",
                "visualizations": "/api/v1/claire/visualizations/{session_id}",
                "learn": "/api/v1/claire/learn-from-action",
                "download": "/api/v1/claire/download-report/{session_id}",
                "sessions": "/api/v1/claire/sessions"
            },
            "notebooks": {
                "info": "/api/v1/notebooks/",
                "create": "/api/v1/notebooks/create",
                "list": "/api/v1/notebooks/list",
                "get": "/api/v1/notebooks/{notebook_id}",
                "update": "/api/v1/notebooks/update",
                "execute": "/api/v1/notebooks/execute",
                "kernel": "/api/v1/notebooks/kernel",
                "dataset": "/api/v1/notebooks/dataset",
                "export": "/api/v1/notebooks/export",
                "schedule": "/api/v1/notebooks/schedule",
                "websocket": "/api/v1/notebooks/ws/{notebook_id}"
            },
            "ai_data_prep": {
                "info": "/api/v1/data-prep/",
                "upload": "/api/v1/data-prep/upload-dataset",
                "scan": "/api/v1/data-prep/scan-quality/{session_id}",
                "suggestions": "/api/v1/data-prep/cleaning-suggestions/{session_id}",
                "natural_language": "/api/v1/data-prep/natural-language-transform",
                "preview": "/api/v1/data-prep/preview-transformation",
                "apply": "/api/v1/data-prep/apply-transformations",
                "download": "/api/v1/data-prep/download-clean-data/{session_id}",
                "sessions": "/api/v1/data-prep/sessions"
            },
            "assisted_modeling": {
                "info": "/api/v1/assisted-modeling/",
                "upload": "/api/v1/assisted-modeling/upload-dataset",
                "analyze": "/api/v1/assisted-modeling/analyze-dataset/{session_id}",
                "train": "/api/v1/assisted-modeling/train-model",
                "generate_pipeline": "/api/v1/assisted-modeling/generate-pipeline",
                "export": "/api/v1/assisted-modeling/export-model/{session_id}",
                "results": "/api/v1/assisted-modeling/results/{session_id}",
                "sessions": "/api/v1/assisted-modeling/sessions"
            },
            "data_drift_detection": {
                "info": "/api/v1/data-drift/",
                "upload": "/api/v1/data-drift/upload-dataset",
                "dataset_info": "/api/v1/data-drift/dataset-info/{session_id}",
                "analyze": "/api/v1/data-drift/analyze",
                "start_monitoring": "/api/v1/data-drift/start-monitoring",
                "alerts": "/api/v1/data-drift/alerts/{session_id}",
                "download": "/api/v1/data-drift/download-results"
            },
            "automl": {
                "info": "/api/v1/automl/",
                "upload": "/api/v1/automl/upload-dataset",
                "train": "/api/v1/automl/train",
                "predict": "/api/v1/automl/predict",
                "models": "/api/v1/automl/models"
            },
            "genai_docs": {
                "info": "/api/v1/genai-docs/",
                "upload": "/api/v1/genai-docs/upload",
                "analyze": "/api/v1/genai-docs/analyze",
                "question": "/api/v1/genai-docs/question",
                "search": "/api/v1/genai-docs/search",
                "documents": "/api/v1/genai-docs/documents"
            },
            "data_profiling": "/api/v1/profiling/",
            "file_upload": "/api/v1/upload",
            "model_versioning": {
                "info": "/api/v1/model-versioning/",
                "upload": "/api/v1/model-versioning/upload-dataset",
                "train": "/api/v1/model-versioning/train-model",
                "register": "/api/v1/model-versioning/register-version",
                "compare": "/api/v1/model-versioning/compare-versions/{model_id}",
                "deploy": "/api/v1/model-versioning/deploy-model",
                "predict": "/api/v1/model-versioning/predict",
                "monitor": "/api/v1/model-versioning/monitor/{model_id}",
                "drift": "/api/v1/model-versioning/drift-analysis",
                "feedback": "/api/v1/model-versioning/feedback",
                "audit": "/api/v1/model-versioning/audit-trail/{model_id}",
                "visualizations": "/api/v1/model-versioning/visualizations/{model_id}",
                "download": "/api/v1/model-versioning/download-report/{model_id}",
                "sessions": "/api/v1/model-versioning/sessions",
                "models": "/api/v1/model-versioning/models"
            },
        },
        "environment_variables": {
            "AZURE_OPENAI_API_KEY": "✅ Set" if os.getenv("AZURE_OPENAI_API_KEY") else "❌ Missing",
            "AZURE_OPENAI_ENDPOINT": "✅ Set" if os.getenv("AZURE_OPENAI_ENDPOINT") else "❌ Missing",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "✅ Set" if os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") else "❌ Missing"
        }
    }

if __name__ == "__main__":
    import subprocess
    import sys
    
    # Load environment variables again to ensure they're available
    load_dotenv()
    
    # Install requirements if needed
    try:
        import fastapi
        import uvicorn
        import pandas
        import aiofiles
        import openai
        import matplotlib
        import seaborn
        import sklearn
        import joblib
        import PyPDF2
        import docx
        import tiktoken
        import networkx
        import psutil
        from dotenv import load_dotenv
        from scipy import stats
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "fastapi==0.104.1", 
            "uvicorn[standard]==0.24.0",
            "pandas==2.1.3",
            "openpyxl==3.1.2",
            "xlrd==2.0.1",
            "python-multipart==0.0.6",
            "aiofiles==23.2.1",
            "websockets==12.0",
            "openai==1.3.0",
            "matplotlib==3.7.2",
            "seaborn==0.12.2",
            "python-dotenv==1.0.0",
            "scikit-learn==1.3.2",
            "joblib==1.3.2",
            "numpy==1.24.3",
            "pydantic==2.5.0",
            "PyPDF2==3.0.1",
            "python-docx==0.8.11",
            "tiktoken==0.5.1",
            "scipy==1.11.4",
            "networkx==3.2.1",
            "sqlparse==0.4.4",
            "duckdb==0.9.2",
            "psutil==5.9.6"  # Added for system monitoring
            "fuzzywuzzy==0.18.0",  # Added for fuzzy matching
            "python-Levenshtein==0.21.1"  # Optional, for faster performance
        ])
        print("Packages installed successfully!")
    
    # Try to install PySpark if not available
    try:
        import pyspark
    except ImportError:
        print("Installing PySpark (optional)...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "pyspark==3.5.0"
            ])
            print("PySpark installed successfully!")
        except:
            print("PySpark installation failed. Visual SQL & Spark will run with limited functionality.")
    
    # Check for notebook-specific dependencies
    try:
        import ipykernel
    except ImportError:
        print("Installing notebook-specific packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "ipykernel==6.25.2",
            "jupyter-client==8.3.1",
            "nbformat==5.9.2"
        ])
        print("Notebook packages installed successfully!")
    
    print("=" * 60)
    print("🚀 Starting GenAI ETL FastAPI Server")
    print("📊 Full-stack GenAI-powered Data Analysis Platform")
    print("🤖 Modeled like Dataiku's AI/ML Tools")
    print("=" * 60)
    print("📍 Server: http://localhost:8000")
    print("📚 Swagger UI: http://localhost:8000/docs")
    print("📖 ReDoc: http://localhost:8000/redoc")
    print("🔌 WebSocket: ws://localhost:8000/ws/{session_id}")
    print("=" * 60)
    print("🔧 Available Services:")
    print("   🎯 NEW: Prompt-driven Data Transformation: http://localhost:8000/api/v1/prompt-transformation/")
    print("   🎯 Pipeline Optimization: http://localhost:8000/api/v1/pipeline-optimization/")
    print("   🔍 Visual SQL & Spark: http://localhost:8000/api/v1/visual-sql-spark/")
    print("   💬 Chatbot Interface: http://localhost:8000/api/v1/chatbot/")
    print("   🧠 CLAIRE Engine: http://localhost:8000/api/v1/claire/")
    print("   📓 Notebooks: http://localhost:8000/api/v1/notebooks/")
    print("   🧹 AI Data Prep: http://localhost:8000/api/v1/data-prep/")
    print("   🧠 Assisted Modeling: http://localhost:8000/api/v1/assisted-modeling/")
    print("   📊 Data Drift Detection: http://localhost:8000/api/v1/data-drift/")
    print("   🤖 AutoML API: http://localhost:8000/api/v1/automl/")
    print("   📄 GenAI Docs: http://localhost:8000/api/v1/genai-docs/")
    print("   📊 Data Profiling: http://localhost:8000/api/v1/profiling/")
    print("   📁 File Upload: http://localhost:8000/api/v1/upload")
    print("   🔄 Model Versioning: http://localhost:8000/api/v1/model-versioning/")
    print("=" * 60)
    print("🎯 NEW: Prompt-driven Data Transformation Features:")
    print("   📁 Upload Dataset → 💬 Natural Language Prompts → 🤖 LLM Code Generation")
    print("   🔒 Safe Code Execution → 📊 Real-time Preview → 🔄 Iterative Transformations")
    print("   📈 Impact Analysis → 📊 Automated Visualizations → 📋 Export & History")
    print("=" * 60)
    print("🔧 Environment Variables:")
    print(f"   AZURE_OPENAI_API_KEY: {'✅ Set' if os.getenv('AZURE_OPENAI_API_KEY') else '❌ Missing'}")
    print(f"   AZURE_OPENAI_ENDPOINT: {'✅ Set' if os.getenv('AZURE_OPENAI_ENDPOINT') else '❌ Missing'}")
    print(f"   AZURE_OPENAI_DEPLOYMENT_NAME: {'✅ Set' if os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME') else '❌ Missing'}")
    print("=" * 60)
    print("🎯 Prompt Transformation Workflow:")
    print("   1. 📂 Upload CSV/Excel dataset")
    print("   2. 💬 Enter natural language transformation prompts")
    print("   3. 🤖 LLM generates safe pandas code")
    print("   4. 🔒 Execute in sandboxed environment")
    print("   5. 📊 Preview results and apply changes")
    print("   6. 🔄 Chain multiple transformations")
    print("   7. 📈 Visualize and export results")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print()
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )