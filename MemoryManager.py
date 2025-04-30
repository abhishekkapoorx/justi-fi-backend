import operator
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
import dotenv
import uuid
from langgraph.graph.message import add_messages

dotenv.load_dotenv()

# Custom string reducer function (prefer first non-empty value)
def string_reducer(a: str, b: str) -> str:
    """Return the first non-empty string, or empty string if both are empty."""
    if a:
        return a
    return b

# Custom max function with explicit signature
def max_value(a, b):
    """Return the maximum of two values."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)

# -- State Schema Definition with proper annotations --
class MemoryState(TypedDict):
    input: Annotated[str, string_reducer]
    user_id: Annotated[str, string_reducer]
    space_id: Annotated[str, string_reducer]
    thread_id: Annotated[str, string_reducer]
    timestamp: Annotated[int, max_value]
    short_term_history: Annotated[List[str], operator.add]
    long_term_snippets: Annotated[List[str], operator.add]
    merged_context: Annotated[List[str], operator.add]
    new_memory: Annotated[str, string_reducer]

class MemoryInputState(TypedDict):
    input: Annotated[str, string_reducer]
    user_id: Annotated[str, string_reducer]
    space_id: Annotated[str, string_reducer]
    thread_id: Annotated[str, string_reducer] 

class MemoryOutputState(TypedDict):
    merged_context: Annotated[List[str], operator.add]

# -- External Clients Initialization --
# Initialize Llama 3 embeddings via Ollama
embeddings = OllamaEmbeddings(
    model="llama3"
)

# Initialize ChromaDB from LangChain for short-term memory
short_term_db = Chroma(
    collection_name=f"short_term_memory_{uuid.uuid4()}",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Initialize Pinecone from LangChain for long-term memory
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_ENVIRONMENT"] = "us-ease-1"
index_name = "case-index"

try:
    long_term_db = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        text_key="text"
    )
except Exception as e:
    print(f"Warning: Pinecone initialization failed: {e}")
    # Fallback to Chroma for long-term storage too
    long_term_db = Chroma(
        collection_name=f"long_term_memory_{uuid.uuid4()}",
        embedding_function=embeddings,
        persist_directory="./chroma_db_long_term"
    )

# Initialize MemorySaver for state persistence
memory_path = "./memory_data"
os.makedirs(memory_path, exist_ok=True)
memory_saver = MemorySaver(
    # path=memory_path
)

# -- Node Functions --
def context_enricher(state: MemoryState) -> MemoryState:
    """Enrich the state with additional context information."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    # Ensure ID fields have default values if missing
    for key in ("user_id", "space_id", "thread_id"):
        if key not in new_state or not new_state[key]:
            new_state[key] = ""
    
    return new_state

def short_term_retrieval(state: MemoryState) -> MemoryState:
    """Retrieve relevant information from short-term memory."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    tid = new_state["thread_id"]
    
    try:
        # Query ChromaDB for recent memories using LangChain integration
        filter_dict = {"thread_id": tid}
        results = short_term_db.similarity_search_with_score(
            query=new_state["input"],
            k=10,
            filter=filter_dict
        )
        
        # Extract documents and their scores
        new_state["short_term_history"] = [doc.page_content for doc, _ in results]
    except Exception as e:
        print(f"Error retrieving from short-term memory: {e}")
        new_state["short_term_history"] = []
        
    return new_state

def long_term_retrieval(state: MemoryState) -> MemoryState:
    """Retrieve relevant information from long-term memory."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    try:
        # Use Pinecone through LangChain
        filter_dict = {
            "user_id": new_state["user_id"],
            "space_id": new_state["space_id"]
        }
        
        results = long_term_db.similarity_search(
            query=new_state["input"],
            k=5,
            filter=filter_dict
        )
        
        new_state["long_term_snippets"] = [doc.page_content for doc in results]
    except Exception as e:
        print(f"Error retrieving from long-term memory: {e}")
        new_state["long_term_snippets"] = []
        
    return new_state

def context_merger(state: MemoryState) -> MemoryState:
    """Combine information from different memory sources."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    merged = []
    merged.extend(new_state.get("short_term_history", []))
    merged.extend(new_state.get("long_term_snippets", []))
    new_state["merged_context"] = merged
    
    return new_state

def memory_updater(state: MemoryState) -> MemoryState:
    """Update memory with new information."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    # Heuristic: store full input
    new_state["new_memory"] = new_state["input"]
    
    return new_state

def persistence_writer(state: MemoryState) -> MemoryState:
    """Write information to persistent storage."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    try:
        # Create unique ID based on thread and timestamp
        unique_id = f"{new_state['thread_id']}-{new_state.get('timestamp', uuid.uuid4())}"
        
        # Metadata for both storage systems
        metadata = {
            "thread_id": new_state["thread_id"],
            "user_id": new_state["user_id"],
            "space_id": new_state["space_id"],
            "timestamp": str(new_state.get("timestamp", ""))
        }
        
        # Short-term: ChromaDB via LangChain
        short_term_db.add_texts(
            texts=[new_state["input"]],
            metadatas=[metadata],
            ids=[f"st-{unique_id}"]
        )
        
        # Long-term: Pinecone via LangChain
        long_term_db.add_texts(
            texts=[new_state["new_memory"]],
            metadatas=[metadata],
            ids=[f"lt-{unique_id}"]
        )
    except Exception as e:
        print(f"Error writing to memory stores: {e}")
    
    return new_state

# -- Build Memory Manager StateGraph --
def build_memory_manager():
    """Construct and return the memory manager graph."""
    # Define the graph with proper input type
    builder = StateGraph(MemoryState, input=MemoryInputState, output=MemoryOutputState)
    
    # Define Nodes
    builder.add_node("ContextEnricher", context_enricher)
    builder.add_node("ShortTermRetrieval", short_term_retrieval)
    builder.add_node("LongTermRetrieval", long_term_retrieval)
    builder.add_node("ContextMerger", context_merger)
    builder.add_node("MemoryUpdater", memory_updater)
    builder.add_node("PersistenceWriter", persistence_writer)

    # Define Edges - add each edge individually
    builder.add_edge(START, "ContextEnricher")
    builder.add_edge("ContextEnricher", "ShortTermRetrieval")
    builder.add_edge("ContextEnricher", "LongTermRetrieval")
    builder.add_edge("ShortTermRetrieval", "ContextMerger")
    builder.add_edge("LongTermRetrieval", "ContextMerger")
    builder.add_edge("ContextMerger", "MemoryUpdater")
    builder.add_edge("MemoryUpdater", "PersistenceWriter")
    builder.add_edge("PersistenceWriter", END)

    # Compile the graph with proper configuration
    return builder

# -- Example Usage --
if __name__ == "__main__":
    # Build the memory graph
    mem_graph = build_memory_manager().compile(checkpointer=memory_saver)
    
    # Create a sample input state
    state = {
        'input': "List upcoming case deadlines.",
        'user_id': "user123",
        'space_id': "space456",
        'thread_id': "thread789",
        'timestamp': 1711795200,
        'short_term_history': [],
        'long_term_snippets': [],
        'merged_context': []
    }
    
    # Process the input and get the result
    try:
        result = mem_graph.invoke(state)
        print("Merged Context:", result.get('merged_context'))
    except Exception as e:
        print(f"Error during graph execution: {e}")
        import traceback
        traceback.print_exc()
