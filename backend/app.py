import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.warning(
        "python-dotenv not installed. Using system environment variables only."
    )

# Try to import AWS Bedrock
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    HAS_BEDROCK = True
except ImportError:
    HAS_BEDROCK = False
    logging.warning("boto3 not installed. Bedrock functionality will be disabled.")

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

BAKU_TZ = timezone(timedelta(hours=4))

# Configuration from environment variables
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "JGMPKF6VEI")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
CLAUDE_MODEL_ID = os.getenv(
    "CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Application settings
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
SESSION_CLEANUP_HOURS = int(os.getenv("SESSION_CLEANUP_HOURS", "24"))

# Retry settings
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))  # seconds between retries
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

logger.info(f"Starting application with AWS Region: {AWS_REGION}")
logger.info(f"Knowledge Base ID: {KNOWLEDGE_BASE_ID}")
logger.info(
    f"AWS credentials configured: {bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)}"
)

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot API with AWS Bedrock Knowledge Base",
    description="REST API for AI chatbot using AWS Bedrock Knowledge Base and Claude Sonnet",
    version="2.1.0",
    docs_url="/docs",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    session_id: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None


# In-memory session storage
chat_sessions = {}

# Initialize Bedrock client with explicit credentials
bedrock_client = None
if HAS_BEDROCK and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    try:
        bedrock_client = boto3.client(
            "bedrock-agent-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        logger.info(
            "AWS Bedrock client initialized successfully with explicit credentials"
        )
    except (NoCredentialsError, Exception) as e:
        logger.warning(
            f"Failed to initialize Bedrock client with explicit credentials: {e}"
        )
        bedrock_client = None
elif HAS_BEDROCK:
    try:
        bedrock_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
        logger.info(
            "AWS Bedrock client initialized successfully with default credentials"
        )
    except (NoCredentialsError, Exception) as e:
        logger.warning(
            f"Failed to initialize Bedrock client with default credentials: {e}"
        )
        bedrock_client = None
else:
    logger.warning("boto3 not available or AWS credentials not configured")


async def query_knowledge_base_with_retry(
    query: str, session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Query the AWS Bedrock Knowledge Base with retry logic."""
    if not bedrock_client:
        logger.info("Bedrock client not available, using mock response")
        return create_mock_chat_response(query)

    # Validate configuration
    if not KNOWLEDGE_BASE_ID or not AWS_REGION or not CLAUDE_MODEL_ID:
        logger.error(
            "Missing required environment variables: KNOWLEDGE_BASE_ID, AWS_REGION, or CLAUDE_MODEL_ID"
        )
        return {
            "success": False,
            "error": "Missing required configuration parameters",
            "timestamp": datetime.now(BAKU_TZ).isoformat(),
        }

    # Prepare the request
    request_body = {
        "input": {"text": query},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{CLAUDE_MODEL_ID}",
            },
        },
    }

    if session_id and session_id in chat_sessions:
        request_body["sessionId"] = session_id
        logger.info(f"Using existing Bedrock session ID: {session_id}")
    else:
        logger.info("Starting new Bedrock session (no session ID provided or invalid)")

    # Retry logic
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Querying Bedrock (attempt {attempt + 1}/{MAX_RETRIES})")

            response = bedrock_client.retrieve_and_generate(**request_body)
            logger.info("Successfully received response from Bedrock")

            returned_session_id = response.get("sessionId")
            if returned_session_id:
                chat_sessions[returned_session_id] = {
                    "created_at": datetime.now(BAKU_TZ).isoformat(),
                    "last_activity": datetime.now(BAKU_TZ).isoformat(),
                    "message_count": 1,
                }

            return {
                "success": True,
                "answer": response["output"]["text"],
                "session_id": returned_session_id,
                "citations": response.get("citations", []),
                "timestamp": datetime.now(BAKU_TZ).isoformat(),
            }

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(
                f"Bedrock ClientError (attempt {attempt + 1}): {error_code} - {error_message}"
            )

            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                return {
                    "success": False,
                    "error": f"AWS error after {MAX_RETRIES} attempts: {error_message}",
                    "timestamp": datetime.now(BAKU_TZ).isoformat(),
                }

        except Exception as e:
            logger.error(
                f"Unexpected error querying knowledge base (attempt {attempt + 1}): {str(e)}"
            )
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.info(f"Unexpected error, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                return {
                    "success": False,
                    "error": f"Unexpected error after {MAX_RETRIES} attempts: {str(e)}",
                    "timestamp": datetime.now(BAKU_TZ).isoformat(),
                }

    return {
        "success": False,
        "error": "Maximum retry attempts exceeded",
        "timestamp": datetime.now(BAKU_TZ).isoformat(),
    }


def create_mock_chat_response(query: str) -> Dict[str, Any]:
    mock_responses = {
        "hello": "Hello! I'm your AI assistant (mock mode).",
        "help": "I can help you with various questions (mock mode).",
        "status": "I'm running in mock mode.",
        "test": "This is a test response (mock mode).",
        "default": f"I received your message: '{query}'. I'm currently in mock mode.",
    }

    query_lower = query.lower()
    response = mock_responses.get("default")
    for keyword, mock_response in mock_responses.items():
        if keyword != "default" and keyword in query_lower:
            response = mock_response
            break

    return {
        "success": True,
        "answer": response,
        "session_id": str(uuid.uuid4()),
        "citations": [],
        "timestamp": datetime.now(BAKU_TZ).isoformat(),
        "is_mock": True,
    }


def manage_session(session_id: Optional[str]) -> str:
    current_time = datetime.now(BAKU_TZ).isoformat()

    if session_id and session_id in chat_sessions:
        chat_sessions[session_id]["last_activity"] = current_time
        chat_sessions[session_id]["message_count"] += 1
        return session_id
    else:
        new_session_id = str(uuid.uuid4())
        chat_sessions[new_session_id] = {
            "created_at": current_time,
            "last_activity": current_time,
            "message_count": 1,
        }
        return new_session_id


def cleanup_old_sessions():
    cutoff_time = datetime.now(BAKU_TZ) - timedelta(hours=SESSION_CLEANUP_HOURS)
    sessions_to_remove = []

    for session_id, session_data in chat_sessions.items():
        session_time = datetime.fromisoformat(session_data["last_activity"])
        if session_time < cutoff_time:
            sessions_to_remove.append(session_id)

    for session_id in sessions_to_remove:
        del chat_sessions[session_id]

    if sessions_to_remove:
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")


@app.get("/config")
def get_config() -> Dict[str, Any]:
    return {
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
        "aws_region": AWS_REGION,
        "claude_model_id": CLAUDE_MODEL_ID,
        "debug": DEBUG,
        "session_cleanup_hours": SESSION_CLEANUP_HOURS,
        "has_bedrock": HAS_BEDROCK,
        "bedrock_client_available": bedrock_client is not None,
        "allowed_origins": ALLOWED_ORIGINS,
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "AI Chatbot API with AWS Bedrock Knowledge Base",
        "version": "2.1.0",
        "docs": "/docs",
        "features": ["Retry logic", "Session management"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        session_id = manage_session(request.session_id)
        result = await query_knowledge_base_with_retry(request.message, session_id)

        return ChatResponse(
            success=result["success"],
            answer=result.get("answer"),
            session_id=session_id,
            citations=result.get("citations", []),
            error=result.get("error"),
            timestamp=result.get("timestamp"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return ChatResponse(
            success=False,
            error=f"Internal server error: {str(e)}",
            timestamp=datetime.now(BAKU_TZ).isoformat(),
        )


@app.get("/sessions")
def get_sessions() -> Dict[str, Any]:
    cleanup_old_sessions()
    sessions_info = []
    for session_id, session_data in chat_sessions.items():
        sessions_info.append(
            {
                "session_id": session_id,
                "created_at": session_data["created_at"],
                "last_activity": session_data["last_activity"],
                "message_count": session_data["message_count"],
            }
        )

    return {"total_sessions": len(sessions_info), "sessions": sessions_info}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {
            "success": True,
            "message": f"Session {session_id} deleted successfully",
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/sessions")
def clear_all_sessions() -> Dict[str, Any]:
    session_count = len(chat_sessions)
    chat_sessions.clear()
    return {
        "success": True,
        "message": f"All {session_count} sessions cleared successfully",
    }
