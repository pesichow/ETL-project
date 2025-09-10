from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from typing import List, Optional, Dict, Any, Union
import logging
import pandas as pd
import os
import uuid
import tempfile
from utils.fetch_data import fetch_data
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader
)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain import hub
import matplotlib.pyplot as plt
import io
import base64
import json
from sqlalchemy import create_engine, inspect
from datetime import datetime
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Hugging Face Embeddings
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("✅ Hugging Face embeddings initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Hugging Face embeddings: {str(e)}")
    raise

# Configuration - Update these based on your Azure OpenAI deployment
AZURE_OPENAI_CONFIG = {
    "api_key": "",
    "api_version": "2024-12-01-preview",
    "azure_endpoint": "https://genaideployment.openai.azure.com",
    "deployment_name": "gpt-4o"
}

# Initialize Azure OpenAI client and LLM
try:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_CONFIG["api_key"],
        api_version=AZURE_OPENAI_CONFIG["api_version"],
        azure_endpoint=AZURE_OPENAI_CONFIG["azure_endpoint"]
    )
    logger.info("✅ Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Azure OpenAI client: {str(e)}")
    client = None

try:
    llm = AzureChatOpenAI(
        deployment_name=AZURE_OPENAI_CONFIG["deployment_name"],
        api_key=AZURE_OPENAI_CONFIG["api_key"],
        api_version=AZURE_OPENAI_CONFIG["api_version"],
        azure_endpoint=AZURE_OPENAI_CONFIG["azure_endpoint"],
        temperature=0
    )
    test_response = llm.predict("Hello")
    logger.info(f"✅ Azure OpenAI LLM initialized and tested successfully: {test_response[:50]}...")
except Exception as e:
    logger.error(f"❌ Failed to initialize Azure OpenAI LLM: {str(e)}")
    raise

router = APIRouter(
    prefix="/api/v1/rag-chatbot",
    tags=["Multi-Source RAG Chatbot"],
)

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Global storage
user_sessions = {}
document_stores = {}
dataframe_stores = {}
file_storage = {}
class AssetTagger:
    """Class for automatic tagging and categorization of assets"""
    
    @staticmethod
    def generate_tags(content: str, metadata: dict = None) -> List[str]:
        """Generate tags based on content and metadata"""
        prompt = f"""
        Analyze the following content and generate relevant tags:
        Content: {content[:5000]}  # Limit content size
        Metadata: {metadata}
        
        Generate 3-5 most relevant tags that categorize this content.
        Return ONLY a JSON array of tag strings.
        """
        try:
            response = llm.predict(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Tag generation failed: {str(e)}")
            return []
class DataSource:
    """Enhanced class to represent different data sources with tagging"""
    def __init__(self, source_type: str, name: str, data: Any, metadata: Dict = None):
        self.source_type = source_type
        self.name = name
        self.data = data
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.tags = AssetTagger.generate_tags(str(data), self.metadata)  # Auto-tagging

class SessionManager:
    """Enhanced session management class"""
    
    @staticmethod
    def create_new_session() -> str:
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "data_sources": {},
            "conversation_history": [],
            "session_type": "active"
        }
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    @staticmethod
    def validate_session(session_id: str) -> bool:
        return session_id in user_sessions
    
    @staticmethod
    def update_session_timestamp(session_id: str):
        if session_id in user_sessions:
            user_sessions[session_id]["last_modified"] = datetime.now().isoformat()
    
    @staticmethod
    def get_or_create_session(session_id: Optional[str] = None) -> str:
        if session_id and SessionManager.validate_session(session_id):
            SessionManager.update_session_timestamp(session_id)
            return session_id
        else:
            return SessionManager.create_new_session()

class QueryIntentClassifier:
    """Class to classify user queries and determine appropriate response strategy"""
    
    @staticmethod
    def classify_query_intent(question: str, available_data_sources: Dict) -> Dict[str, Any]:
        """
        Classify the user's query intent to determine the best response strategy
        
        Returns:
        {
            'intent': 'sql_generation' | 'data_analysis' | 'document_search' | 'general_question',
            'confidence': float,
            'suggested_action': str,
            'reasoning': str
        }
        """
        question_lower = question.lower().strip()
        
        # Define intent patterns
        sql_generation_patterns = [
            r'\b(generate|create|write|show me|give me)\s+sql\b',
            r'\bsql\s+(query|for|to)\b',
            r'\bconvert\s+to\s+sql\b',
            r'\bhow\s+to\s+write\s+sql\b'
        ]
        sql_generation_patterns = [
            r'generate\s+sql',
            r'create\s+sql',
            r'write\s+sql',
            r'sql\s+query\s+for',
            r'give\s+me\s+sql',
            r'show\s+me\s+sql',
            r'sql\s+to\s+',
            r'convert\s+to\s+sql',
            r'sql\s+statement'
        ]
        
        data_analysis_patterns = [
            r'what\s+is\s+the\s+(average|mean|sum|total|count|maximum|minimum)',
            r'how\s+many\s+',
            r'show\s+me\s+(top|bottom|highest|lowest)',
            r'(analyze|analysis)\s+',
            r'what\s+are\s+the\s+(trends|patterns)',
            r'compare\s+',
            r'find\s+(all|records|data)\s+where',
            r'list\s+(all|the)\s+',
            r'get\s+(me\s+)?(all|the)\s+',
            r'which\s+.*\s+has\s+the\s+(highest|lowest|most|least)',
            r'summarize\s+',
            r'group\s+by',
            r'calculate\s+',
            r'percentage\s+of',
            r'distribution\s+of',
            r'list\s+.*\s+(customers|users|records|data|entries)',
            r'show\s+.*\s+(customers|users|records|data|entries)',
            r'find\s+.*\s+(customers|users|records|data|entries)',
            r'get\s+.*\s+(customers|users|records|data|entries)',
            r'display\s+.*\s+(customers|users|records|data|entries)',
            r'retrieve\s+.*\s+(customers|users|records|data|entries)',
            r'(list|show|find|get|display)\s+.*\s+(from|in|where|who|that)',
            r'.*\s+who\s+(are|have|meet)',
            r'.*\s+that\s+(are|have|meet)',
            r'.*\s+with\s+',
            r'.*\s+where\s+'
        ]
        
        document_search_patterns = [
            r'what\s+does\s+the\s+document\s+say',
            r'according\s+to\s+the\s+(document|pdf|file)',
            r'find\s+information\s+about',
            r'search\s+for\s+',
            r'what\s+is\s+mentioned\s+about',
            r'explain\s+.*\s+from\s+the\s+document'
        ]
        
        # Check for explicit SQL generation request
        sql_score = 0
        for pattern in sql_generation_patterns:
            if re.search(pattern, question_lower):
                sql_score += 1
        
        # Check for data analysis intent
        analysis_score = 0
        for pattern in data_analysis_patterns:
            if re.search(pattern, question_lower):
                analysis_score += 1
        
        # Check for document search intent
        doc_score = 0
        for pattern in document_search_patterns:
            if re.search(pattern, question_lower):
                doc_score += 1
        
        # Additional scoring based on question words and structure
        question_words = ['what', 'how', 'when', 'where', 'which', 'who']
        analysis_indicators = ['total', 'average', 'count', 'sum', 'maximum', 'minimum', 'highest', 'lowest']
        retrieval_words = ['list', 'show', 'find', 'get', 'display', 'retrieve', 'give', 'provide']
        
        # Boost analysis score for data-oriented questions
        if any(word in question_lower for word in question_words):
            if any(indicator in question_lower for indicator in analysis_indicators):
                analysis_score += 2
            elif any(word in question_lower for word in retrieval_words):
                analysis_score += 2  # Increased boost for retrieval words
        
        # Extra boost for listing/showing data
        if any(word in question_lower for word in retrieval_words):
            analysis_score += 1
        
        # Penalty for SQL generation if it's clearly a data retrieval request
        if any(word in question_lower for word in retrieval_words) and not any(pattern in question_lower for pattern in ['generate sql', 'create sql', 'write sql', 'sql query', 'sql for']):
            sql_score = max(0, sql_score - 2)  # Reduce SQL score for retrieval requests
        
        # Consider available data sources
        has_documents = any(source.source_type in ['document', 'pdf'] for source in available_data_sources.values())
        has_structured_data = any(source.source_type in ['postgresql', 'iceberg', 'file_data', 'csv', 'excel'] 
                                 for source in available_data_sources.values())
        
        # Determine intent based on scores and context
        if sql_score > 0 and sql_score >= analysis_score:
            return {
                'intent': 'sql_generation',
                'confidence': min(0.9, 0.6 + (sql_score * 0.1)),
                'suggested_action': 'Generate SQL query for the user\'s request',
                'reasoning': f'Explicit SQL generation keywords detected (score: {sql_score})'
            }
        
        elif analysis_score > 0 and has_structured_data:
            return {
                'intent': 'data_analysis',
                'confidence': min(0.9, 0.7 + (analysis_score * 0.05)),
                'suggested_action': 'Perform direct data analysis and provide results',
                'reasoning': f'Data analysis patterns detected (score: {analysis_score}) with structured data available'
            }
        
        elif doc_score > 0 and has_documents:
            return {
                'intent': 'document_search',
                'confidence': min(0.9, 0.8 + (doc_score * 0.05)),
                'suggested_action': 'Search documents for relevant information',
                'reasoning': f'Document search patterns detected (score: {doc_score}) with documents available'
            }
        
        elif has_structured_data and any(word in question_lower for word in question_words):
            return {
                'intent': 'data_analysis',
                'confidence': 0.6,
                'suggested_action': 'Analyze data to answer the question',
                'reasoning': 'Question word detected with structured data available, likely needs data analysis'
            }
        
        else:
            return {
                'intent': 'general_question',
                'confidence': 0.5,
                'suggested_action': 'Use available tools to attempt answering the question',
                'reasoning': 'No clear intent pattern detected, will use general approach'
            }
class FuzzyMatcher:
    """Class for fuzzy matching records using various algorithms"""
    
    @staticmethod
    def levenshtein_match(query: str, choices: List[str], threshold: float = 0.8) -> List[dict]:
        """Fuzzy match using Levenshtein distance"""
        from thefuzz import fuzz
        results = []
        for choice in choices:
            score = fuzz.ratio(query.lower(), choice.lower()) / 100
            if score >= threshold:
                results.append({"match": choice, "score": score, "algorithm": "levenshtein"})
        return sorted(results, key=lambda x: x["score"], reverse=True)
    
    @staticmethod
    def jaccard_match(query: str, choices: List[str], threshold: float = 0.5) -> List[dict]:
        """Fuzzy match using Jaccard similarity"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        tfidf = vectorizer.fit_transform([query] + choices)
        similarities = cosine_similarity(tfidf[0:1], tfidf[1:])
        
        results = []
        for i, score in enumerate(similarities[0]):
            if score >= threshold:
                results.append({
                    "match": choices[i],
                    "score": float(score),
                    "algorithm": "jaccard"
                })
        return sorted(results, key=lambda x: x["score"], reverse=True)

class SQLQueryGenerator:
    """Enhanced SQL query generation class with clean response"""
    
    @staticmethod
    def generate_sql_query(question: str, df: pd.DataFrame, table_name: str) -> dict:
        """Generate SQL query from natural language question"""
        try:
            # Get DataFrame schema information
            columns_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                sample_values = df[col].dropna().head(3).astype(str).tolist()
                unique_count = int(df[col].nunique())
                
                columns_info.append({
                    'name': col,
                    'type': col_type,
                    'unique_count': unique_count,
                    'sample_values': sample_values
                })
            
            # Create a comprehensive prompt for SQL generation
            sql_prompt = f"""
You are an expert SQL query generator. Based on the following table schema and user question, generate ONLY the SQL query syntax. Do not include any explanations, comments, or descriptive text.

Table Name: {table_name}
Table Schema:
{json.dumps(columns_info, indent=2)}

Table has {len(df)} rows and {len(df.columns)} columns.

Sample data (first 3 rows):
{df.head(3).to_string()}

User Question: {question}

IMPORTANT: Return ONLY the SQL query without any explanations, descriptions, or additional text. No introductory phrases like "The SQL query is:" or "Here's the SQL:". Just the pure SQL syntax.

Requirements:
1. Generate ONLY SQL syntax - no explanations or comments
2. Use standard SQL syntax that works with most databases
3. Use the exact table name: {table_name}
4. Use the exact column names from the schema
5. Handle data types appropriately (strings in quotes, numbers without quotes)
6. If the question is unclear, generate the most reasonable SQL query
7. For aggregations, use appropriate GROUP BY clauses
8. For filtering, use appropriate WHERE clauses
9. For sorting, use ORDER BY clauses

SQL Query:
"""
            
            # Use LLM to generate SQL
            sql_response = llm.predict(sql_prompt)
            
            # Enhanced cleaning of the response
            sql_query = SQLQueryGenerator._extract_clean_sql(sql_response)
            
            return {
                "success": True,
                "sql_query": sql_query,
                "table_name": table_name,
                "question": question,
                "columns_available": [col['name'] for col in columns_info]
            }
            
        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "table_name": table_name
            }
    
    @staticmethod
    def generate_human_readable_sql_explanation(question: str, sql_query: str, table_name: str, df: pd.DataFrame) -> str:
        """Generate human-readable explanation of the SQL query"""
        try:
            # Create prompt for human-readable explanation
            explanation_prompt = f"""
You are explaining SQL queries to users in a conversational, human-friendly way.

User's Original Question: {question}
Generated SQL Query: {sql_query}
Table Name: {table_name}
Table has {len(df)} rows and {len(df.columns)} columns.
Available Columns: {', '.join(df.columns.tolist())}

Create a natural, conversational response that:
1. Acknowledges the user's question
2. Explains what the SQL query does in simple terms
3. Mentions the relevant columns and table being used
4. Explains the logic behind the query
5. Uses friendly, professional language

Do NOT:
- Show the actual SQL code in your response
- Use technical SQL terminology excessively
- Make it sound robotic or formal

Make it sound like you're having a conversation with the user about their data.

Example style: "Based on your question about finding customers, I've created a query that will search through your customer data table to find all records where..."
"""
            
            explanation = llm.predict(explanation_prompt)
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate SQL explanation: {str(e)}")
            # Fallback explanation
            return f"I've generated a SQL query to answer your question '{question}' using the {table_name} table. The query will analyze your data with {len(df)} rows and {len(df.columns)} columns to provide the information you're looking for."
    
    @staticmethod
    def _extract_clean_sql(sql_response: str) -> str:
        """Extract clean SQL query from LLM response"""
        try:
            # First, replace literal \n with actual newlines if they exist
            sql_response = sql_response.replace('\\n', '\n')
            
            # Remove leading/trailing whitespace
            sql_response = sql_response.strip()
            
            # Handle markdown code blocks
            if '```sql' in sql_response:
                # Extract content between ```sql and ```
                start_marker = sql_response.find('```sql') + 6
                end_marker = sql_response.find('```', start_marker)
                if end_marker != -1:
                    sql_response = sql_response[start_marker:end_marker].strip()
                else:
                    sql_response = sql_response[start_marker:].strip()
            elif '```' in sql_response:
                # Extract content between ``` and ```
                parts = sql_response.split('```')
                if len(parts) >= 3:
                    sql_response = parts[1].strip()
                elif len(parts) == 2:
                    sql_response = parts[1].strip()
            
            # Remove any remaining "sql" prefix that might be left from markdown
            if sql_response.lower().startswith('sql'):
                sql_response = sql_response[3:].strip()
            
            # Find the first SQL keyword and start from there
            sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE', 'DROP', 'ALTER']
            sql_upper = sql_response.upper()
            
            earliest_keyword_pos = len(sql_response)
            for keyword in sql_keywords:
                keyword_pos = sql_upper.find(keyword)
                if keyword_pos != -1 and keyword_pos < earliest_keyword_pos:
                    earliest_keyword_pos = keyword_pos
            
            if earliest_keyword_pos < len(sql_response):
                sql_response = sql_response[earliest_keyword_pos:].strip()
            
            # Split by semicolon and take only the first SQL statement
            if ';' in sql_response:
                semicolon_pos = sql_response.find(';')
                sql_part = sql_response[:semicolon_pos + 1].strip()
                
                # Check what comes after the semicolon
                after_semicolon = sql_response[semicolon_pos + 1:].strip()
                
                # If there's text after semicolon, check if it's explanatory
                if after_semicolon:
                    # Common explanatory phrases that indicate we should stop
                    explanatory_phrases = [
                        'this query will', 'this will compute', 'this will calculate',
                        'this sql', 'the query', 'explanation:', 'note:', 'result:',
                        'this finds', 'this returns', 'output:', 'the result'
                    ]
                    
                    # Check if after_semicolon starts with explanatory text
                    after_lower = after_semicolon.lower()
                    is_explanatory = any(after_lower.startswith(phrase) for phrase in explanatory_phrases)
                    
                    if is_explanatory:
                        sql_response = sql_part
                    else:
                        # Check if it might be another SQL statement
                        has_sql_keyword = any(keyword in after_semicolon.upper() for keyword in sql_keywords)
                        if not has_sql_keyword:
                            sql_response = sql_part
                        # If it has SQL keywords, keep the full response (multiple statements)
                else:
                    sql_response = sql_part
            
            # Clean up the SQL by removing explanatory lines
            lines = sql_response.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip comment lines
                if line.startswith('--') or line.startswith('#'):
                    continue
                
                # Skip lines that start with explanatory text
                line_lower = line.lower()
                explanatory_starts = [
                    'this query', 'this sql', 'this will', 'explanation:', 'note:',
                    'the result', 'this finds', 'output:', 'result:', 'the query',
                    'this computes', 'this calculates', 'this returns'
                ]
                
                if any(line_lower.startswith(start) for start in explanatory_starts):
                    continue
                
                clean_lines.append(line)
            
            # Join the clean lines with spaces (SQL can be on multiple lines)
            final_sql = ' '.join(clean_lines)
            
            # Final cleanup
            final_sql = final_sql.strip()
            
            # Ensure it ends with semicolon if it doesn't already
            if final_sql and not final_sql.endswith(';'):
                if any(keyword in final_sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                    final_sql += ';'
            
            # Remove any trailing explanatory text that might still be there
            if final_sql.endswith(';'):
                # Check if there's any text after the last semicolon
                last_semicolon = final_sql.rfind(';')
                if last_semicolon < len(final_sql) - 1:
                    after_last_semicolon = final_sql[last_semicolon + 1:].strip()
                    if after_last_semicolon:
                        # If it's explanatory text, remove it
                        after_lower = after_last_semicolon.lower()
                        if any(word in after_lower for word in ['this', 'the', 'query', 'will', 'compute', 'result']):
                            final_sql = final_sql[:last_semicolon + 1]
            
            return final_sql.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning SQL response: {str(e)}")
            # Fallback: try to extract the first line with SQL keywords
            lines = sql_response.replace('\\n', '\n').split('\n')
            for line in lines:
                line = line.strip()
                if line and any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                    # Clean the line and ensure it ends with semicolon
                    if not line.endswith(';'):
                        line += ';'
                    return line
            
            return sql_response.strip()
    
    @staticmethod
    def validate_sql_syntax(sql_query: str) -> dict:
        """Basic SQL syntax validation"""
        try:
            # Basic checks for SQL keywords
            sql_upper = sql_query.upper().strip()
            
            if not any(keyword in sql_upper for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP']):
                return {
                    "valid": False,
                    "error": "No valid SQL keywords found"
                }
            
            # Check for basic SQL structure
            if sql_upper.startswith('SELECT') and 'FROM' not in sql_upper:
                return {
                    "valid": False,
                    "error": "SELECT statement missing FROM clause"
                }
            
            return {
                "valid": True,
                "message": "Basic SQL syntax appears valid"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }

@router.post("/create-session")
async def create_session():
    """Create a new session for the user"""
    session_id = SessionManager.create_new_session()
    return {
        "success": True,
        "session_id": session_id,
        "message": "New session created successfully"
    }

def create_comprehensive_tools(session_id: str) -> List[Tool]:
    """Create comprehensive tools for all available data sources with enhanced features"""
    tools = []
    
    if session_id not in user_sessions:
        return tools
    
    data_sources = user_sessions[session_id]["data_sources"]
    
    # 1. Enhanced Document QA Tool with semantic search and fuzzy matching
    if session_id in document_stores:
        retriever = document_stores[session_id].as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for semantic search
            search_kwargs={
                "k": 5,
                "lambda_mult": 0.5,  # Diversity vs relevance balance
                "score_threshold": 0.3  # Minimum similarity score
            }
        )
        
        doc_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template="""Use the following context to answer the question. 
                    Consider similar concepts and synonyms in your analysis.
                    
                    Context: {context}
                    
                    Question: {question}
                    
                    Answer in a detailed, semantic-aware manner:""",
                    input_variables=["context", "question"]
                )
            }
        )
        
        def document_search(query: str) -> str:
            try:
                # First try exact match
                result = doc_qa({"query": query})
                
                # If low confidence, try fuzzy matching
                if len(result["source_documents"]) == 0:
                    all_texts = [doc.page_content for doc in document_stores[session_id].docstore._dict.values()]
                    fuzzy_matches = FuzzyMatcher.jaccard_match(query, all_texts)
                    if fuzzy_matches:
                        result = doc_qa({"query": fuzzy_matches[0]["match"]})
                
                return result["result"]
            except Exception as e:
                return f"Document search error: {str(e)}"
        
        tools.append(Tool(
            name="Enhanced_Document_Search",
            func=document_search,
            description="Semantic search across documents with fuzzy matching capabilities. Use this for questions about document content, policies, procedures, or any text-based information."
        ))
    
    # 2. Smart SQL Query Generation Tool (unchanged)
    if session_id in dataframe_stores:
        def smart_sql_generator(query: str) -> str:
            try:
                # Extract table name from query if specified
                table_name = None
                available_tables = list(dataframe_stores[session_id].keys())
                
                # Simple table name extraction
                for table in available_tables:
                    if table.lower() in query.lower():
                        table_name = table
                        break
                
                if not table_name and len(available_tables) == 1:
                    table_name = available_tables[0]
                elif not table_name:
                    return f"Multiple tables available: {available_tables}. Please specify which table to use."
                
                df = dataframe_stores[session_id][table_name]
                result = SQLQueryGenerator.generate_sql_query(query, df, table_name)
                
                if result["success"]:
                    return f"Generated SQL Query for {table_name}:\n\n{result['sql_query']}"
                else:
                    return f"SQL generation failed: {result.get('error', 'Unknown error')}"
                    
            except Exception as e:
                return f"SQL generation error: {str(e)}"
        
        tools.append(Tool(
            name="SQL_Generator",
            func=smart_sql_generator,
            description="Generate SQL queries from natural language questions for uploaded datasets. Use this ONLY when user explicitly asks for SQL query generation or SQL syntax."
        ))
    
    # 3. Enhanced Data Analysis Tools for each DataFrame with fuzzy matching
    if session_id in dataframe_stores:
        for df_name, df in dataframe_stores[session_id].items():
            # Create pandas agent for each DataFrame
            try:
                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True
                )
                
                def create_enhanced_df_tool(df_agent, df_name, df_shape, df_ref):
                    def df_query(query: str) -> str:
                        try:
                            # Add context about the data
                            enhanced_query = f"""
                            You are analyzing data from {df_name} with {df_shape[0]} rows and {df_shape[1]} columns.
                            Columns available: {', '.join(df_ref.columns.tolist())}
                            
                            Query: {query}
                            
                            Please provide direct answers with actual data analysis results. 
                            Include specific numbers, calculations, and insights.
                            If showing data, format it clearly.
                            Do not just generate SQL - actually analyze the data and provide results.
                            when they say specificaly usr ask about sql query gearet it
                            If the query contains potential typos or approximate matches,
                            use fuzzy matching techniques to find the closest matches.
                            """
                            
                            # First try exact analysis
                            try:
                                result = df_agent.run(enhanced_query)
                                return f"Analysis from {df_name}:\n{result}"
                            except Exception as e:
                                # If analysis fails, try fuzzy matching on column names
                                logger.warning(f"Initial analysis failed, trying fuzzy matching: {str(e)}")
                                fuzzy_col_matches = FuzzyMatcher.levenshtein_match(
                                    query, 
                                    df_ref.columns.tolist()
                                )
                                if fuzzy_col_matches:
                                    new_query = query
                                    for match in fuzzy_col_matches[:3]:  # Top 3 matches
                                        new_query = new_query.replace(
                                            match['match'].lower(), 
                                            match['match']
                                        )
                                    if new_query != query:
                                        logger.info(f"Retrying with fuzzy matched columns: {new_query}")
                                        result = df_agent.run(enhanced_query.replace(query, new_query))
                                        return f"Analysis (with fuzzy matching) from {df_name}:\n{result}"
                                raise e
                        except Exception as e:
                            return f"Error analyzing {df_name}: {str(e)}"
                    return df_query
                
                tools.append(Tool(
                    name=f"Analyze_{df_name.replace(' ', '_').replace('.', '_')}",
                    func=create_enhanced_df_tool(agent, df_name, df.shape, df),
                    description=f"Analyze and answer questions about data in {df_name} ({df.shape[0]} rows, {df.shape[1]} columns) with fuzzy matching capabilities. Use this for data analysis, calculations, statistics, filtering, and getting actual results from the data."
                ))
                
            except Exception as e:
                logger.warning(f"Could not create agent for {df_name}: {str(e)}")
    
    # 4. Cross-source Analysis Tool with enhanced semantic understanding
    if session_id in dataframe_stores and len(dataframe_stores[session_id]) > 1:
        def cross_source_analysis(query: str) -> str:
            try:
                available_data = dataframe_stores[session_id]
                data_summary = {}
                for name, df in available_data.items():
                    data_summary[name] = {
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "sample": df.head(2).to_dict('records') if len(df) > 0 else [],
                        "tags": getattr(df, 'tags', [])  # Include auto-generated tags
                    }
                
                prompt = f"""
                You have access to multiple datasets: {list(available_data.keys())}
                
                Data summary: {json.dumps(data_summary, indent=2)}
                
                Query: {query}
                
                Provide analysis that considers:
                - Relationships or comparisons across these datasets
                - Semantic similarities between columns in different tables
                - Potential fuzzy matches for column names or values
                - Auto-generated tags for each dataset
                
                Give actual results and insights, not just SQL queries.
                """
                
                response = llm.predict(prompt)
                return f"Cross-source analysis: {response}"
            except Exception as e:
                return f"Cross-source analysis error: {str(e)}"
        
        tools.append(Tool(
            name="Enhanced_Cross_Source_Analysis",
            func=cross_source_analysis,
            description="Perform intelligent analysis across multiple data sources with semantic understanding and fuzzy matching. Compare datasets or provide insights that require information from multiple tables/files."
        ))
    
    return tools
@router.post("/ask")
async def ask_question(
    question: str = Form(...),
    session_id: str = Form(...),
    generate_graph: bool = Form(False)
):
    """Enhanced ask endpoint with intelligent query routing"""
    try:
        logger.info(f"Processing question: {question} for session: {session_id}")
        
        if not SessionManager.validate_session(session_id):
            raise HTTPException(status_code=400, detail="Session not found")
        
        # Update session timestamp
        SessionManager.update_session_timestamp(session_id)
        
        # Get available data sources
        data_sources = user_sessions[session_id]["data_sources"]
        
        # Classify the query intent
        intent_result = QueryIntentClassifier.classify_query_intent(question, data_sources)
        logger.info(f"Query intent classification: {intent_result}")
        
        response_data = {
            "question": question,
            "session_id": session_id,
            "intent_classification": intent_result
        }
        
        # Route based on intent
        if intent_result['intent'] == 'sql_generation' and session_id in dataframe_stores:
            # Handle SQL generation with human-readable response
            available_tables = list(dataframe_stores[session_id].keys())
            
            if len(available_tables) == 1:
                table_name = available_tables[0]
                df = dataframe_stores[session_id][table_name]
                sql_result = SQLQueryGenerator.generate_sql_query(question, df, table_name)
                
                if sql_result["success"]:
                    # Generate human-readable explanation instead of raw SQL
                    human_explanation = SQLQueryGenerator.generate_human_readable_sql_explanation(
                        question, sql_result['sql_query'], table_name, df
                    )
                    
                    response_data.update({
                        "success": True,
                        "response": human_explanation,
                        "sql_query": sql_result['sql_query'],  # Keep SQL available for other endpoints
                        "table_name": table_name,
                        "type": "sql_generation"
                    })
                else:
                    response_data.update({
                        "success": False,
                        "error": sql_result.get('error', 'SQL generation failed'),
                        "type": "sql_generation_error"
                    })
            else:
                response_data.update({
                    "success": False,
                    "message": f"Multiple tables available: {available_tables}. Please specify which table to use for SQL generation.",
                    "available_tables": available_tables,
                    "type": "sql_generation_ambiguous"
                })
        
        elif intent_result['intent'] == 'data_analysis' or intent_result['intent'] == 'general_question':
            # Handle data analysis or general questions using tools
            tools = create_comprehensive_tools(session_id)
            logger.info(f"Created {len(tools)} tools for session {session_id}")
            
            if not tools:
                response_data.update({
                    "success": False,
                    "message": "No data sources available. Please upload files or connect to databases first.",
                    "available_sources": list(data_sources.keys())
                })
            else:
                # Use tools to answer the question
                try:
                    # Create memory for conversation
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                    
                    # Try to get the react prompt from hub, with fallback
                    try:
                        prompt = hub.pull("hwchase17/react")
                        logger.info("Successfully loaded prompt from hub")
                    except Exception as e:
                        logger.warning(f"Failed to load prompt from hub: {e}, using fallback")
                        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
                        
                        prompt = PromptTemplate.from_template(template)
                    
                    # Create and run agent
                    try:
                        agent = create_react_agent(llm, tools, prompt)
                        agent_executor = AgentExecutor(
                            agent=agent,
                            tools=tools,
                            verbose=True,
                            memory=memory,
                            max_iterations=30,
                            handle_parsing_errors=True,
                            return_intermediate_steps=True
                        )
                        
                        # Execute the query
                        logger.info("Executing agent query for data analysis")
                        response_result = agent_executor.invoke({"input": question})
                        logger.info("Agent execution completed successfully")
                        
                        answer = response_result.get("output", "No response generated")
                        
                        response_data.update({
                            "success": True,
                            "response": answer,
                            "type": "data_analysis",
                            "tools_used": [tool.name for tool in tools]
                        })
                        
                    except Exception as agent_error:
                        logger.error(f"Agent execution failed: {str(agent_error)}")
                        # Fallback: try each tool individually
                        answer = await fallback_tool_execution(question, tools, session_id, intent_result)
                        
                        response_data.update({
                            "success": True,
                            "response": answer,
                            "type": "fallback_analysis",
                            "tools_available": [tool.name for tool in tools]
                        })
                        
                except Exception as setup_error:
                    logger.error(f"Tool setup failed: {str(setup_error)}")
                    # Simple fallback without agent
                    answer = await simple_fallback_response(question, session_id)
                    
                    response_data.update({
                        "success": True,
                        "response": answer,
                        "type": "simple_fallback"
                    })
        
        elif intent_result['intent'] == 'document_search':
            # Handle document search specifically
            if session_id in document_stores:
                retriever = document_stores[session_id].as_retriever(search_kwargs={"k": 5})
                doc_qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                
                try:
                    result = doc_qa({"query": question})
                    answer = result["result"]
                    
                    response_data.update({
                        "success": True,
                        "response": answer,
                        "type": "document_search",
                        "sources_found": len(result.get("source_documents", []))
                    })
                except Exception as e:
                    response_data.update({
                        "success": False,
                        "error": f"Document search failed: {str(e)}",
                        "type": "document_search_error"
                    })
            else:
                response_data.update({
                    "success": False,
                    "message": "No documents available for search. Please upload PDF or text files first.",
                    "type": "no_documents"
                })
        
        # Store conversation
        if "conversation_history" not in user_sessions[session_id]:
            user_sessions[session_id]["conversation_history"] = []
            
        user_sessions[session_id]["conversation_history"].append({
            "question": question,
            "response": response_data.get("response", response_data.get("error", "No response")),
            "intent": intent_result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add data source information
        data_sources_info = []
        for name, source in data_sources.items():
            data_sources_info.append({
                "name": name,
                "type": source.source_type,
                "metadata": source.metadata
            })
        
        response_data["data_sources_available"] = data_sources_info
        
        return response_data
        
    except Exception as e:
        logger.error(f"Question processing failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "question": question,
            "session_id": session_id,
            "message": "An error occurred while processing your question. Please try again."
        }
        
        
async def fallback_tool_execution(question: str, tools: List[Tool], session_id: str, intent_result: Dict) -> str:
    """Enhanced fallback method to execute tools based on intent"""
    try:
        logger.info(f"Attempting fallback tool execution with intent: {intent_result['intent']}")
        
        # Prioritize tools based on intent
        if intent_result['intent'] == 'data_analysis':
            # Try data analysis tools first
            for tool in tools:
                if "Analyze" in tool.name or "Cross_Source" in tool.name:
                    try:
                        result = tool.func(question)
                        if result and "error" not in result.lower()[:50]:  # Check first 50 chars for error
                            return f"Data Analysis Result:\n{result}"
                    except Exception as e:
                        logger.warning(f"Analysis tool {tool.name} failed: {e}")
                        continue
        
        elif intent_result['intent'] == 'document_search':
            # Try document search tools first
            for tool in tools:
                if "Document" in tool.name:
                    try:
                        result = tool.func(question)
                        if result and "error" not in result.lower()[:50]:
                            return f"Document Search Result:\n{result}"
                    except Exception as e:
                        logger.warning(f"Document tool {tool.name} failed: {e}")
                        continue
        
        # If specific tools failed, try all tools in order
        for tool in tools:
            try:
                result = tool.func(question)
                if result and "error" not in result.lower()[:50]:
                    return f"Analysis Result ({tool.name}):\n{result}"
            except Exception as e:
                logger.warning(f"Tool {tool.name} failed: {e}")
                continue
        
        # If all tools fail, return summary
        return await simple_fallback_response(question, session_id)
        
    except Exception as e:
        logger.error(f"Fallback tool execution failed: {e}")
        return await simple_fallback_response(question, session_id)

async def simple_fallback_response(question: str, session_id: str) -> str:
    """Enhanced simple fallback response"""
    try:
        # Get data sources summary
        data_sources = user_sessions[session_id]["data_sources"]
        
        summary = f"I have access to the following data sources:\n"
        for name, source in data_sources.items():
            if source.source_type in ['postgresql', 'iceberg', 'file_data', 'csv', 'excel']:
                summary += f"- {name} ({source.source_type}): {source.metadata.get('shape', 'Unknown size')}\n"
            else:
                summary += f"- {name} ({source.source_type})\n"
        
        # Classify the question to provide better guidance
        intent_result = QueryIntentClassifier.classify_query_intent(question, data_sources)
        
        # Use LLM directly for a simple response with guidance
        prompt = f"""
        Based on the following data sources available:
        {summary}
        
        Question: {question}
        Question Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})
        
        The user is asking about their data. Please provide a helpful response that:
        1. Acknowledges what data is available
        2. Explains what kind of analysis could be performed
        3. Suggests how to rephrase the question for better results
        4. If it's a data analysis question, provide some general insights about what could be found
        
        Be helpful and specific about the available data sources.
        """
        
        response = llm.predict(prompt)
        return f"I can help you analyze your data. {response}\n\nNote: I encountered some technical issues with the analysis tools, but I can still assist you. Try rephrasing your question or being more specific about what you'd like to know."
        
    except Exception as e:
        logger.error(f"Simple fallback failed: {e}")
        return f"I have access to your uploaded data but encountered technical difficulties processing your question '{question}'. Please try:\n\n1. Being more specific about what you want to know\n2. Asking about a particular dataset or table\n3. Using simpler language\n\nFor SQL generation, explicitly ask 'generate SQL query for...' or 'create SQL to...'"

# Additional utility endpoints

@router.post("/upload-files")
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None)
):
    """Upload multiple files (PDF, Excel, CSV) and integrate with RAG system"""
    try:
        # Get or create session
        session_id = SessionManager.get_or_create_session(session_id)
        
        results = []
        all_documents = []
        file_dataframes = {}
        
        for file in files:
            file_id = str(uuid.uuid4())
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_path = tmp.name
            
            try:
                # Process file based on type
                if file_ext == '.pdf':
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    split_docs = text_splitter.split_documents(docs)
                    all_documents.extend(split_docs)
                    
                elif file_ext in ['.xlsx', '.xls']:
                    # Load Excel as DataFrame for structured queries
                    try:
                        df = pd.read_excel(temp_path)
                        file_dataframes[f"{file.filename}"] = df
                        logger.info(f"Loaded Excel file {file.filename} with shape {df.shape}")
                        
                        # Convert DataFrame to documents for text search
                        excel_text = f"File: {file.filename}\n\n"
                        excel_text += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
                        excel_text += f"Columns: {', '.join(df.columns)}\n\n"
                        
                        # Add summary statistics
                        excel_text += "Summary:\n"
                        excel_text += df.describe(include='all').to_string() + "\n\n"
                        
                        # Add first few rows as sample
                        excel_text += "Sample Data:\n"
                        excel_text += df.head(10).to_string() + "\n"
                        
                        # Create document from text representation
                        doc = Document(
                            page_content=excel_text,
                            metadata={"source": file.filename, "type": "excel"}
                        )
                        split_docs = text_splitter.split_documents([doc])
                        all_documents.extend(split_docs)
                        
                    except Exception as e:
                        logger.warning(f"Could not load {file.filename} as Excel: {str(e)}")
                
                elif file_ext == '.csv':
                    # Load CSV as DataFrame for structured queries
                    try:
                        df = pd.read_csv(temp_path)
                        file_dataframes[f"{file.filename}"] = df
                        logger.info(f"Loaded CSV file {file.filename} with shape {df.shape}")
                        
                        # Convert DataFrame to documents for text search
                        csv_text = f"File: {file.filename}\n\n"
                        csv_text += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
                        csv_text += f"Columns: {', '.join(df.columns)}\n\n"
                        
                        # Add summary statistics
                        csv_text += "Summary:\n"
                        csv_text += df.describe(include='all').to_string() + "\n\n"
                        
                        # Add first few rows as sample
                        csv_text += "Sample Data:\n"
                        csv_text += df.head(10).to_string() + "\n"
                        
                        # Create document from text representation
                        doc = Document(
                            page_content=csv_text,
                            metadata={"source": file.filename, "type": "csv"}
                        )
                        split_docs = text_splitter.split_documents([doc])
                        all_documents.extend(split_docs)
                        
                    except Exception as e:
                        logger.warning(f"Could not load {file.filename} as CSV: {str(e)}")
                        # Fallback to langchain CSV loader for text extraction
                        try:
                            loader = CSVLoader(temp_path)
                            docs = loader.load()
                            split_docs = text_splitter.split_documents(docs)
                            all_documents.extend(split_docs)
                        except Exception as e2:
                            logger.error(f"Both CSV loading methods failed for {file.filename}: {str(e2)}")
                
                elif file_ext == '.txt':
                    # Handle text files manually
                    try:
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        doc = Document(
                            page_content=content,
                            metadata={"source": file.filename, "type": "text"}
                        )
                        split_docs = text_splitter.split_documents([doc])
                        all_documents.extend(split_docs)
                    except Exception as e:
                        logger.error(f"Failed to load text file {file.filename}: {str(e)}")
                
                else:
                    logger.warning(f"Unsupported file type: {file_ext} for file {file.filename}")
                    continue
                
                # Store file metadata
                file_storage[file_id] = {
                    "original_name": file.filename,
                    "session_id": session_id,
                    "size": len(content),
                    "type": file_ext,
                    "processed_at": datetime.now().isoformat()
                }
                
                results.append({
                    "file_id": file_id,
                    "filename": file.filename,
                    "chunks_created": len(split_docs) if file_ext in ['.pdf', '.txt'] else 0,
                    "dataframe_loaded": file.filename in file_dataframes
                })
                
            finally:
                os.unlink(temp_path)
        
        # Create or update vector store for documents
        if all_documents:
            if session_id in document_stores:
                document_stores[session_id].add_documents(all_documents)
            else:
                document_stores[session_id] = FAISS.from_documents(all_documents, embeddings)
            
            # Store document source
            user_sessions[session_id]["data_sources"]["documents"] = DataSource(
                source_type="document",
                name="uploaded_documents",
                data=document_stores[session_id],
                metadata={"total_chunks": len(all_documents)}
            )
        
        # Store DataFrames
        if file_dataframes:
            if session_id not in dataframe_stores:
                dataframe_stores[session_id] = {}
            dataframe_stores[session_id].update(file_dataframes)
            
            # Store dataframe sources
            for filename, df in file_dataframes.items():
                user_sessions[session_id]["data_sources"][f"file_data_{filename}"] = DataSource(
                    source_type="file_data",
                    name=filename,
                    data=df,
                    metadata={"shape": df.shape, "columns": list(df.columns)}
                )
        
        # Update session timestamp
        SessionManager.update_session_timestamp(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "session_created": session_id not in user_sessions or not user_sessions[session_id].get("data_sources"),
            "files_processed": len(results),
            "total_document_chunks": len(all_documents),
            "dataframes_loaded": len(file_dataframes),
            "results": results,
            "capabilities": {
                "sql_generation": len(file_dataframes) > 0,
                "document_search": len(all_documents) > 0,
                "data_analysis": len(file_dataframes) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@router.post("/generate-sql")
async def generate_sql_query(
    question: str = Form(...),
    session_id: str = Form(...),
    table_name: Optional[str] = Form(None)
):
    """Generate SQL query from natural language question for uploaded datasets"""
    try:
        logger.info(f"Generating SQL for question: {question} in session: {session_id}")
        
        if not SessionManager.validate_session(session_id):
            raise HTTPException(status_code=400, detail="Session not found")
        
        # Check if session has dataframes
        if session_id not in dataframe_stores or not dataframe_stores[session_id]:
            return {
                "success": False,
                "message": "No datasets available for SQL generation. Please upload CSV or Excel files first.",
                "available_sources": list(user_sessions[session_id]["data_sources"].keys())
            }
        
        available_dataframes = dataframe_stores[session_id]
        
        # If table name specified, use that specific dataframe
        if table_name:
            if table_name not in available_dataframes:
                return {
                    "success": False,
                    "error": f"Table '{table_name}' not found",
                    "available_tables": list(available_dataframes.keys())
                }
            
            df = available_dataframes[table_name]
            sql_result = SQLQueryGenerator.generate_sql_query(question, df, table_name)
            
            # Update session timestamp
            SessionManager.update_session_timestamp(session_id)
            
            return sql_result
        
        # If no table specified, try to determine the best table or use all
        if len(available_dataframes) == 1:
            # Only one table, use it
            table_name = list(available_dataframes.keys())[0]
            df = available_dataframes[table_name]
            sql_result = SQLQueryGenerator.generate_sql_query(question, df, table_name)
            
        else:
            # Multiple tables - generate SQL for each or ask user to specify
            results = []
            for table_name, df in available_dataframes.items():
                sql_result = SQLQueryGenerator.generate_sql_query(question, df, table_name)
                results.append(sql_result)
            
            # Update session timestamp
            SessionManager.update_session_timestamp(session_id)
            
            return {
                "success": True,
                "message": f"Generated SQL queries for {len(results)} tables",
                "question": question,
                "multiple_queries": results,
                "note": "Multiple tables found. Consider specifying table_name parameter for specific table."
            }
        
        # Update session timestamp
        SessionManager.update_session_timestamp(session_id)
        
        return sql_result
        
    except Exception as e:
        logger.error(f"SQL generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQL generation error: {str(e)}")

@router.get("/available-tables/{session_id}")
async def get_available_tables(session_id: str):
    """Get list of available tables/datasets for SQL generation"""
    try:
        if not SessionManager.validate_session(session_id):
            raise HTTPException(status_code=400, detail="Session not found")
        
        if session_id not in dataframe_stores:
            return {
                "success": True,
                "tables": [],
                "message": "No datasets available"
            }
        
        tables_info = []
        for table_name, df in dataframe_stores[session_id].items():
            tables_info.append({
                "table_name": table_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "sample_data": df.head(2).to_dict('records') if len(df) > 0 else []
            })
        
        return {
            "success": True,
            "session_id": session_id,
            "total_tables": len(tables_info),
            "tables": tables_info
        }
        
    except Exception as e:
        logger.error(f"Error getting available tables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching tables: {str(e)}")

@router.get("/session-info/{session_id}")
async def get_session_info(session_id: str):
    """Get comprehensive information about a session"""
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = user_sessions[session_id]
    
    # Compile data sources information
    data_sources_summary = {}
    for name, source in session_data["data_sources"].items():
        data_sources_summary[name] = {
            "type": source.source_type,
            "created_at": source.created_at.isoformat(),
            "metadata": source.metadata
        }
    
    # Get available tools count
    tools = create_comprehensive_tools(session_id)
    
    return {
        "session_id": session_id,
        "created_at": session_data["created_at"],
        "data_sources": data_sources_summary,
        "total_data_sources": len(data_sources_summary),
        "tools_available": len(tools),
        "conversation_count": len(session_data.get("conversation_history", [])),
        "has_documents": session_id in document_stores,
        "has_dataframes": session_id in dataframe_stores,
        "dataframe_count": len(dataframe_stores.get(session_id, {})),
        "capabilities": {
            "sql_generation": session_id in dataframe_stores,
            "document_search": session_id in document_stores,
            "data_analysis": session_id in dataframe_stores,
            "cross_source_analysis": session_id in dataframe_stores and len(dataframe_stores.get(session_id, {})) > 1
        }
    }

@router.post("/debug-classify-intent")
async def debug_classify_query_intent(
    question: str = Form(...),
    session_id: str = Form(...)
):
    """Debug endpoint to see detailed intent classification breakdown"""
    try:
        if not SessionManager.validate_session(session_id):
            raise HTTPException(status_code=400, detail="Session not found")
        
        data_sources = user_sessions[session_id]["data_sources"]
        question_lower = question.lower().strip()
        
        # Detailed pattern matching (same as in QueryIntentClassifier)
        sql_generation_patterns = [
            r'generate\s+sql', r'create\s+sql', r'write\s+sql', r'sql\s+query\s+for',
            r'give\s+me\s+sql', r'show\s+me\s+sql', r'sql\s+to\s+', r'convert\s+to\s+sql', r'sql\s+statement'
        ]
        
        data_analysis_patterns = [
            r'what\s+is\s+the\s+(average|mean|sum|total|count|maximum|minimum)',
            r'how\s+many\s+', r'show\s+me\s+(top|bottom|highest|lowest)', r'(analyze|analysis)\s+',
            r'what\s+are\s+the\s+(trends|patterns)', r'compare\s+', r'find\s+(all|records|data)\s+where',
            r'list\s+(all|the)\s+', r'get\s+(me\s+)?(all|the)\s+', r'which\s+.*\s+has\s+the\s+(highest|lowest|most|least)',
            r'summarize\s+', r'group\s+by', r'calculate\s+', r'percentage\s+of', r'distribution\s+of',
            r'list\s+.*\s+(customers|users|records|data|entries)', r'show\s+.*\s+(customers|users|records|data|entries)',
            r'find\s+.*\s+(customers|users|records|data|entries)', r'get\s+.*\s+(customers|users|records|data|entries)',
            r'display\s+.*\s+(customers|users|records|data|entries)', r'retrieve\s+.*\s+(customers|users|records|data|entries)',
            r'(list|show|find|get|display)\s+.*\s+(from|in|where|who|that)', r'.*\s+who\s+(are|have|meet)',
            r'.*\s+that\s+(are|have|meet)', r'.*\s+with\s+', r'.*\s+where\s+'
        ]
        
        # Pattern matching results
        sql_matches = []
        analysis_matches = []
        
        for pattern in sql_generation_patterns:
            if re.search(pattern, question_lower):
                sql_matches.append(pattern)
        
        for pattern in data_analysis_patterns:
            if re.search(pattern, question_lower):
                analysis_matches.append(pattern)
        
        # Get the actual classification
        intent_result = QueryIntentClassifier.classify_query_intent(question, data_sources)
        
        return {
            "success": True,
            "question": question,
            "question_lower": question_lower,
            "session_id": session_id,
            "pattern_analysis": {
                "sql_matches": sql_matches,
                "sql_match_count": len(sql_matches),
                "analysis_matches": analysis_matches,
                "analysis_match_count": len(analysis_matches)
            },
            "intent_classification": intent_result,
            "available_data_sources": list(data_sources.keys()),
            "has_structured_data": any(source.source_type in ['postgresql', 'iceberg', 'file_data', 'csv', 'excel'] 
                                     for source in data_sources.values()),
            "recommendation": f"This query should be classified as '{intent_result['intent']}' and should {'generate SQL' if intent_result['intent'] == 'sql_generation' else 'return actual data results'}"
        }
        
    except Exception as e:
        logger.error(f"Debug intent classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug intent classification error: {str(e)}")

@router.post("/test-data-analysis")
async def test_data_analysis(
    question: str = Form(...),
    session_id: str = Form(...),
    table_name: Optional[str] = Form(None)
):
    """Test endpoint to force data analysis instead of SQL generation"""
    try:
        if not SessionManager.validate_session(session_id):
            raise HTTPException(status_code=400, detail="Session not found")
        
        if session_id not in dataframe_stores or not dataframe_stores[session_id]:
            return {
                "success": False,
                "message": "No datasets available for analysis."
            }
        
        available_dataframes = dataframe_stores[session_id]
        
        # Determine which table to use
        if table_name and table_name in available_dataframes:
            target_table = table_name
        elif len(available_dataframes) == 1:
            target_table = list(available_dataframes.keys())[0]
        else:
            return {
                "success": False,
                "message": f"Multiple tables available: {list(available_dataframes.keys())}. Please specify table_name.",
                "available_tables": list(available_dataframes.keys())
            }
        
        df = available_dataframes[target_table]
        
        # Create a pandas agent specifically for this analysis
        try:
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            
            # Enhanced query for better results
            enhanced_query = f"""
            You are analyzing data from {target_table} with {df.shape[0]} rows and {df.shape[1]} columns.
            Columns available: {', '.join(df.columns.tolist())}
            
            User Query: {question}
            
            Please provide direct answers with actual data analysis results. 
            Show the actual data that matches the criteria.
            Include specific records, counts, and insights.
            Format the results clearly and readably.
            Do not generate SQL - analyze the data directly and return the results.
            
            If the query asks for a list of records, show the actual records with relevant columns.
            If it asks for counts or statistics, provide the actual numbers.
            """
            
            result = agent.run(enhanced_query)
            
            return {
                "success": True,
                "question": question,
                "table_analyzed": target_table,
                "table_shape": df.shape,
                "analysis_result": result,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Data analysis failed: {str(e)}")
            return {
                "success": False,
                "error": f"Data analysis failed: {str(e)}",
                "question": question,
                "session_id": session_id
            }
        
    except Exception as e:
        logger.error(f"Test data analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test data analysis error: {str(e)}")

@router.post("/classify-intent")
async def classify_query_intent_endpoint(
    question: str = Form(...),
    session_id: str = Form(...)
):
    """Endpoint to test query intent classification"""
    try:
        if not SessionManager.validate_session(session_id):
            raise HTTPException(status_code=400, detail="Session not found")
        
        data_sources = user_sessions[session_id]["data_sources"]
        intent_result = QueryIntentClassifier.classify_query_intent(question, data_sources)
        
        return {
            "success": True,
            "question": question,
            "session_id": session_id,
            "intent_classification": intent_result,
            "available_data_sources": list(data_sources.keys()),
            "recommendations": {
                "sql_generation": intent_result['intent'] == 'sql_generation',
                "data_analysis": intent_result['intent'] == 'data_analysis',
                "document_search": intent_result['intent'] == 'document_search'
            }
        }
        
    except Exception as e:
        logger.error(f"Intent classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Intent classification error: {str(e)}")

@router.post("/get-raw-sql")
async def get_raw_sql_query(
    question: str = Form(...),
    session_id: str = Form(...),
    table_name: Optional[str] = Form(None)
):
    """New endpoint to get the raw SQL query without human-readable explanation"""
    try:
        logger.info(f"Getting raw SQL for question: {question} in session: {session_id}")
        
        if not SessionManager.validate_session(session_id):
            raise HTTPException(status_code=400, detail="Session not found")
        
        # Check if session has dataframes
        if session_id not in dataframe_stores or not dataframe_stores[session_id]:
            return {
                "success": False,
                "message": "No datasets available for SQL generation. Please upload CSV or Excel files first.",
                "available_sources": list(user_sessions[session_id]["data_sources"].keys())
            }
        
        available_dataframes = dataframe_stores[session_id]
        
        # If table name specified, use that specific dataframe
        if table_name:
            if table_name not in available_dataframes:
                return {
                    "success": False,
                    "error": f"Table '{table_name}' not found",
                    "available_tables": list(available_dataframes.keys())
                }
            
            df = available_dataframes[table_name]
            sql_result = SQLQueryGenerator.generate_sql_query(question, df, table_name)
            
            if sql_result["success"]:
                # Return ONLY the raw SQL query
                response_data = {
                    "success": True,
                    "sql_query": sql_result['sql_query'],
                    "table_name": table_name,
                    "question": question,
                    "columns_available": sql_result['columns_available']
                }
            else:
                response_data = {
                    "success": False,
                    "error": sql_result.get('error', 'SQL generation failed'),
                    "question": question,
                    "table_name": table_name
                }
            
            # Update session timestamp
            SessionManager.update_session_timestamp(session_id)
            
            return response_data
        
        # If no table specified, try to determine the best table or use all
        if len(available_dataframes) == 1:
            # Only one table, use it
            table_name = list(available_dataframes.keys())[0]
            df = available_dataframes[table_name]
            sql_result = SQLQueryGenerator.generate_sql_query(question, df, table_name)
            
            if sql_result["success"]:
                response_data = {
                    "success": True,
                    "sql_query": sql_result['sql_query'],
                    "table_name": table_name,
                    "question": question,
                    "columns_available": sql_result['columns_available']
                }
            else:
                response_data = {
                    "success": False,
                    "error": sql_result.get('error', 'SQL generation failed'),
                    "question": question,
                    "table_name": table_name
                }
        
        else:
            # Multiple tables - generate SQL for each
            results = []
            for table_name, df in available_dataframes.items():
                sql_result = SQLQueryGenerator.generate_sql_query(question, df, table_name)
                if sql_result["success"]:
                    results.append({
                        "table_name": table_name,
                        "sql_query": sql_result['sql_query'],
                        "columns_available": sql_result['columns_available']
                    })
                else:
                    results.append({
                        "table_name": table_name,
                        "error": sql_result.get('error', 'SQL generation failed')
                    })
            
            response_data = {
                "success": True,
                "message": f"Generated SQL queries for {len(results)} tables",
                "question": question,
                "multiple_queries": results,
                "note": "Multiple tables found. Consider specifying table_name parameter for specific table."
            }
        
        # Update session timestamp
        SessionManager.update_session_timestamp(session_id)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Raw SQL generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Raw SQL generation error: {str(e)}")

logger.info("✅ Enhanced Multi-source RAG chatbot with human-readable SQL responses created successfully")