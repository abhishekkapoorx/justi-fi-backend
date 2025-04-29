# Required imports
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langgraph.graph import StateGraph, END
import uuid
import datetime
import requests
import json
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

# Environment variables
os.environ["GROQ_API_KEY"] = "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"
NOTION_MCP_SERVER_URL = "https://your-notion-mcp-server-url.com"

# Pydantic models for data structures
class User(BaseModel):
    user_id: str
    name: str
    email: str

class Space(BaseModel):
    space_id: str
    name: str
    user_id: str
    created_at: str
    messages: List[Message] = []

class Thread(BaseModel):
    thread_id: str
    space_id: str
    title: str
    created_at: str

class Message(BaseModel):
    message_id: str
    thread_id: str
    content: str
    role: str  # "user" or "assistant"
    created_at: str

class Document(BaseModel):
    document_id: str
    space_id: str
    name: str
    file_path: str
    uploaded_at: str

class CaseInsight(BaseModel):
    insight_id: str
    space_id: str
    case_summary: str
    opposition_summary: str
    points_in_favor: List[str]
    points_against: List[Dict[str, str]]  # {"point": "...", "suggestion": "..."}
    generated_at: str

# State classes for LangGraph
class AgentState(BaseModel):
    user_id: str
    space_id: Optional[str] = None
    thread_id: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    current_task: Optional[str] = None
    task_result: Optional[Dict[str, Any]] = None
    user_memory: Optional[Dict[str, Any]] = None
    space_memory: Optional[Dict[str, Any]] = None
    thread_memory: Optional[Dict[str, Any]] = None

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Create embeddings and vector store
model_name = "sentence-transformers/all-mpnet-base-v2"  # Using a publicly available model
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Initialize ChromaDB with proper settings
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Create vector store
vector_store = Chroma(
    client=chroma_client,
    collection_name="legal_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Helper functions
def load_document(file_path: str) -> List[Document]:
    """Load document and split into chunks"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(docs)

def get_memory(user_id: str, space_id: Optional[str] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Get memory for user, space, or thread"""
    # In a real implementation, this would fetch from a database
    # Here we're simulating memory retrieval
    memory = {}
    
    if user_id:
        memory["user"] = {"preferences": "prefers detailed explanations", "history": "has 5 active cases"}
    
    if space_id:
        memory["space"] = {"case_type": "intellectual property", "court": "Federal Court"}
    
    if thread_id:
        memory["thread"] = {"last_discussed": "patent infringement arguments"}
        
    return memory

def update_memory(memory_type: str, id: str, data: Dict[str, Any]) -> None:
    """Update memory for user, space, or thread"""
    # In a real implementation, this would update a database
    # Here we're simulating memory updates
    print(f"Updating {memory_type} memory for {id}")
    print(f"New data: {data}")

# LangGraph Nodes
def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant documents based on the current query"""
    if not state.documents:
        # Get documents from vector store based on the last message
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(state.messages[-1]["content"])
        state.documents = [{"id": str(i), "content": doc.page_content} for i, doc in enumerate(docs)]
    
    return state

def answer_question(state: AgentState) -> AgentState:
    """Generate an answer to the user's query using retrieved documents"""
    # Build system prompt with context
    context = "\n\n".join([doc["content"] for doc in state.documents])
    
    # Incorporate memories
    memory_context = ""
    if state.user_memory:
        memory_context += f"User Context: {json.dumps(state.user_memory)}\n\n"
    if state.space_memory:
        memory_context += f"Case Context: {json.dumps(state.space_memory)}\n\n"
    if state.thread_memory:
        memory_context += f"Thread Context: {json.dumps(state.thread_memory)}\n\n"
    
    system_prompt = f"""You are an AI legal assistant helping lawyers manage their cases.
    
    Use the following context to answer the question:
    {context}
    
    {memory_context}
    
    Always be professional, accurate, and cite specific references from the documents when possible.
    If you don't know the answer or can't find it in the context, say so clearly.
    """
    
    # Format conversation for the LLM
    messages = [SystemMessage(content=system_prompt)]
    for msg in state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # Get response
    response = llm.invoke(messages)
    
    # Update state
    state.task_result = {"answer": response.content}
    
    return state

def generate_case_insights(state: AgentState) -> AgentState:
    """Generate insights about the case based on all available documents"""
    # Combine all document content
    all_content = "\n\n".join([doc["content"] for doc in state.documents])
    
    # Create a prompt for case insights
    system_prompt = """You are a legal expert analyzing case documents. 
    Based on the provided documents, generate a comprehensive case analysis with the following structure:
    1. Case Summary - A concise overview of the case
    2. Opposition's Case Summary - Analysis of the opposing party's arguments
    3. Points in Favor - List key points supporting your client's position
    4. Points Not in Favor - Identify potential weaknesses, with suggestions for addressing each
    
    Format your response as a JSON object with these four sections.
    """
    
    # Generate insights
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here are the case documents to analyze:\n\n{all_content}")
    ]
    
    response = llm.invoke(messages)
    
    # Parse JSON response
    try:
        insights = json.loads(response.content)
        state.task_result = {
            "case_summary": insights.get("Case Summary", ""),
            "opposition_summary": insights.get("Opposition's Case Summary", ""),
            "points_in_favor": insights.get("Points in Favor", []),
            "points_against": insights.get("Points Not in Favor", []),
        }
    except json.JSONDecodeError:
        # Fallback if the response isn't valid JSON
        state.task_result = {
            "error": "Failed to generate structured insights",
            "raw_response": response.content
        }
    
    return state

def generate_document(state: AgentState) -> AgentState:
    """Generate a document for Notion based on the case insights"""
    if not state.task_result or "case_summary" not in state.task_result:
        # We need insights first
        state = generate_case_insights(state)
    
    insights = state.task_result
    
    # Format document for Notion
    notion_doc = {
        "title": f"Case Analysis - {datetime.datetime.now().strftime('%Y-%m-%d')}",
        "content": [
            {
                "type": "heading_1",
                "content": "Case Summary"
            },
            {
                "type": "paragraph",
                "content": insights["case_summary"]
            },
            {
                "type": "heading_1",
                "content": "Opposition's Case"
            },
            {
                "type": "paragraph",
                "content": insights["opposition_summary"]
            },
            {
                "type": "heading_1",
                "content": "Points in Favor"
            },
            {
                "type": "bullet_list",
                "items": insights["points_in_favor"]
            },
            {
                "type": "heading_1",
                "content": "Potential Challenges and Solutions"
            }
        ]
    }
    
    # Add challenges and solutions
    for point in insights["points_against"]:
        notion_doc["content"].append({
            "type": "heading_2",
            "content": point.get("point", "Challenge")
        })
        notion_doc["content"].append({
            "type": "paragraph",
            "content": point.get("suggestion", "No suggestion provided")
        })
    
    # In a real implementation, we would send this to the Notion MCP server
    # For this example, we'll just update the state
    state.task_result = {"notion_document": notion_doc}
    
    return state

def schedule_meeting(state: AgentState) -> AgentState:
    """Schedule a meeting in Notion Calendar"""
    # Extract meeting details from the last message
    last_message = state.messages[-1]["content"]
    
    # Use the LLM to extract meeting details
    system_prompt = """Extract meeting details from the text. Return a JSON object with:
    1. title: The meeting title
    2. date: The meeting date in YYYY-MM-DD format
    3. time: The meeting time in HH:MM format
    4. duration: Meeting duration in minutes
    5. attendees: List of attendee names
    6. notes: Any additional notes
    
    If any field is missing, use null or empty string/list as appropriate.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message)
    ]
    
    response = llm.invoke(messages)
    
    # Parse meeting details
    try:
        meeting_details = json.loads(response.content)
        
        # In a real implementation, we would send this to the Notion MCP server
        # For this example, we'll just update the state
        state.task_result = {"meeting": meeting_details}
        
    except json.JSONDecodeError:
        state.task_result = {
            "error": "Failed to extract meeting details",
            "raw_response": response.content
        }
    
    return state

def router(state: AgentState) -> str:
    """Route to the appropriate node based on the task"""
    if state.current_task == "answer_question":
        return "answer_question"
    elif state.current_task == "generate_insights":
        return "generate_insights"
    elif state.current_task == "generate_document":
        return "generate_document"
    elif state.current_task == "schedule_meeting":
        return "schedule_meeting"
    else:
        return "retrieve_context"

# Create the LangGraph
def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("answer_question", answer_question)
    workflow.add_node("generate_insights", generate_case_insights)
    workflow.add_node("generate_document", generate_document)
    workflow.add_node("schedule_meeting", schedule_meeting)
    
    # Add edges
    workflow.add_conditional_edges(
        "retrieve_context",
        router,
        {
            "answer_question": "answer_question",
            "generate_insights": "generate_insights",
            "generate_document": "generate_document",
            "schedule_meeting": "schedule_meeting"
        }
    )
    
    # All nodes lead to END
    workflow.add_edge("answer_question", END)
    workflow.add_edge("generate_insights", END)
    workflow.add_edge("generate_document", END)
    workflow.add_edge("schedule_meeting", END)
    
    # Set the entry point
    workflow.set_entry_point("retrieve_context")
    
    return workflow.compile()

# Create API
app = FastAPI()
agent_graph = create_agent_graph()

# API Routes
@app.post("/api/spaces/")
async def create_space(space: Space):
    # In a real implementation, this would create a space in the database
    return {"status": "success", "space_id": space.space_id}

@app.post("/api/threads/")
async def create_thread(thread: Thread):
    # In a real implementation, this would create a thread in the database
    return {"status": "success", "thread_id": thread.thread_id}

@app.post("/api/documents/upload/")
async def upload_document(document: Document):
    # In a real implementation, this would upload the document and process it
    try:
        docs = load_document(document.file_path)
        vector_store.add_documents(docs)
        return {"status": "success", "document_id": document.document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/api/messages/")
async def create_message(message: Message):
    # Get the current space
    space = spaces.get(message.space_id)
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    
    # Add the message to the space
    space.messages.append(message)
    
    # Execute the graph
    result = graph.invoke({"messages": space.messages})
    
    # Create the response message
    response_message = Message(
        space_id=message.space_id,
        role="assistant",
        content=result["answer"] if "answer" in result else "I'm sorry, I couldn't process your request.",
        timestamp=datetime.now()
    )
    
    # Add the response to the space
    space.messages.append(response_message)
    
    return response_message

@app.get("/api/insights/{space_id}")
async def get_insights(space_id: str):
    # Initialize state for insight generation
    user_id = "user-123"  # In a real app, this would come from the authenticated user
    state = AgentState(
        user_id=user_id,
        space_id=space_id,
        current_task="generate_insights",
        # We'd normally fetch all documents for this space
        documents=[{"id": "1", "content": "Sample document content"}]
    )
    
    # Run agent
    result = agent_graph.invoke(state)
    
    # Create insight
    insight = CaseInsight(
        insight_id=str(uuid.uuid4()),
        space_id=space_id,
        case_summary=result.task_result["case_summary"],
        opposition_summary=result.task_result["opposition_summary"],
        points_in_favor=result.task_result["points_in_favor"],
        points_against=result.task_result["points_against"],
        generated_at=datetime.datetime.now().isoformat()
    )
    
    return {"status": "success", "insight": insight}

@app.post("/api/documents/generate/{space_id}")
async def generate_notion_document(space_id: str):
    # Initialize state for document generation
    user_id = "user-123"  # In a real app, this would come from the authenticated user
    state = AgentState(
        user_id=user_id,
        space_id=space_id,
        current_task="generate_document",
        # We'd normally fetch all documents for this space
        documents=[{"id": "1", "content": "Sample document content"}]
    )
    
    # Run agent
    result = agent_graph.invoke(state)
    
    # In a real implementation, we would send this to the Notion MCP server
    # notion_response = requests.post(
    #     f"{NOTION_MCP_SERVER_URL}/create_page",
    #     json=result.task_result["notion_document"]
    # )
    
    return {"status": "success", "document": result.task_result["notion_document"]}

@app.post("/api/meetings/schedule")
async def schedule_notion_meeting(meeting_request: Dict[str, Any]):
    # Initialize state for meeting scheduling
    user_id = "user-123"  # In a real app, this would come from the authenticated user
    state = AgentState(
        user_id=user_id,
        current_task="schedule_meeting",
        messages=[{"role": "user", "content": meeting_request["description"]}]
    )
    
    # Run agent
    result = agent_graph.invoke(state)
    
    # In a real implementation, we would send this to the Notion MCP server
    # notion_response = requests.post(
    #     f"{NOTION_MCP_SERVER_URL}/create_calendar_event",
    #     json=result.task_result["meeting"]
    # )
    
    return {"status": "success", "meeting": result.task_result["meeting"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)