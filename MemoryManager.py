import operator
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
import os
import dotenv
import uuid
from langgraph.graph.message import add_messages
from lib.VectorStore import VectorStore
from lib.embeddings import embeddings
from lib.llm import llm
from langchain_core.documents import Document
from lib.api_keys import pinecone_api_key

dotenv.load_dotenv()

long_term_db = VectorStore("long-term-db", embeddings, pinecone_api_key)
long_term_db = long_term_db.getVectorStore()

short_term_db = VectorStore("short-term-db", embeddings, pinecone_api_key)
short_term_db = short_term_db.getVectorStore()


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
    clarified_query: Annotated[str, string_reducer]
    memory_summary: Annotated[str, string_reducer]


class MemoryInputState(TypedDict):
    input: Annotated[str, string_reducer]
    user_id: Annotated[str, string_reducer]
    space_id: Annotated[str, string_reducer]
    thread_id: Annotated[str, string_reducer]


class MemoryOutputState(TypedDict):
    memory_summary: Annotated[str, string_reducer]


# Initialize MemorySaver for state persistence
memory_path = "./memory_data"
os.makedirs(memory_path, exist_ok=True)
memory_saver = MemorySaver()


# -- Node Functions --
def context_enricher(state: MemoryInputState) -> MemoryState:
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
        # Define system message and user message for LLM
        system_message = """
        You are a query clarification assistant helping to improve search results.
        Your task is to analyze user queries and rewrite them to enhance semantic search relevance.
        Consider entities mentioned, timeframes, and specific information needs.
        Rewrite the query to be more specific and detailed for better search results.
        Return ONLY the rewritten query without explanations or additional text.
        """
        
        user_message = f"""
        Original query: {new_state["input"]}
        
        Please rewrite this query to:
        1. Identify specific entities (people, cases, documents)
        2. Clarify relevant timeframes
        3. Specify the exact information needed
        
        Make it optimized for semantic search against recent conversation history.
        """
        
        # Get clarified query from LLM with separated messages
        llm_response = llm.invoke(system=system_message, user=user_message)
        clarified_query = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        print(f"Original query: {new_state['input']}")
        print(f"Clarified query: {clarified_query}")
        
        # Query ChromaDB for recent memories using LangChain integration with clarified query
        filter_dict = {"thread_id": tid}
        results = short_term_db.similarity_search_with_score(
            query=clarified_query, k=10, filter=filter_dict
        )
        
        # Extract documents and their scores
        new_state["short_term_history"] = [doc.page_content for doc, _ in results]
        
        # Store the clarified query for potential use in long-term retrieval
        new_state["clarified_query"] = clarified_query
        
    except Exception as e:
        print(f"Error retrieving from short-term memory: {e}")
        new_state["short_term_history"] = []
        
    return new_state


def long_term_retrieval(state: MemoryState) -> MemoryState:
    """Retrieve relevant information from long-term memory."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    try:
        # Use the clarified query if available, otherwise create one
        if "clarified_query" in new_state and new_state["clarified_query"]:
            query_text = new_state["clarified_query"]
        else:
            # Define system message and user message for LLM
            system_message = """
            You are a historical context assistant specializing in legal case information retrieval.
            Your task is to analyze user queries and optimize them for retrieving relevant historical data.
            Consider case history, precedents, and long-term context that might be relevant.
            Return ONLY the rewritten query without explanations or additional text.
            """
            
            user_message = f"""
            Original query: {new_state["input"]}
            
            Please rewrite this query to:
            1. Identify historical case information that might be relevant
            2. Include possible precedents or similar past situations
            3. Expand abbreviations and legal terminology
            4. Consider longer-term context and relationships
            
            Make it optimized for semantic search against historical case records.
            """
            
            # Get clarified query from LLM with separated messages
            llm_response = llm.invoke(system=system_message, user=user_message)
            query_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            print(f"Original query: {new_state['input']}")
            print(f"Long-term clarified query: {query_text}")
        
        # Use Pinecone through LangChain
        filter_dict = {
            "user_id": new_state["user_id"],
            "space_id": new_state["space_id"],
        }
        
        results = long_term_db.similarity_search(
            query=query_text, k=5, filter=filter_dict
        )
        
        new_state["long_term_snippets"] = [doc.page_content for doc in results]
    except Exception as e:
        print(f"Error retrieving from long-term memory: {e}")
        new_state["long_term_snippets"] = []
        
    return new_state


def context_merger(state: MemoryState) -> MemoryState:
    """Combine information from different memory sources with formatted XML-style tags."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    merged = []
    
    # Add short-term memories with XML-style tags
    for memory in new_state.get("short_term_history", []):
        formatted_memory = f'<Memory type="short_term" content="{memory}" />'
        merged.append(formatted_memory)
    
    # Add long-term memories with XML-style tags
    for memory in new_state.get("long_term_snippets", []):
        formatted_memory = f'<Memory type="long_term" content="{memory}" />'
        merged.append(formatted_memory)
    
    new_state["merged_context"] = merged
    
    return new_state


def memory_updater(state: MemoryState) -> MemoryState:
    """Update memory with new information in properly formatted XML structure."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    # Extract original input for storage
    user_input = new_state["input"]
    
    # Format as new memory with XML tag
    new_state["new_memory"] = user_input
    
    # Create a summary of retrieved context combined with input for future reference
    context_summary = f"""
    <MemoryUpdate timestamp="{new_state.get('timestamp', '')}">
        <UserQuery>{user_input}</UserQuery>
        <RetrievedContext>
            {''.join(new_state.get('merged_context', []))}
        </RetrievedContext>
    </MemoryUpdate>
    """
    
    # Store this formatted summary for persistence
    new_state["memory_summary"] = context_summary
    
    return new_state


def persistence_writer(state: MemoryState) -> MemoryOutputState:
    """Write information to persistent storage with XML-formatted memory structure."""
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    try:
        # Create unique ID based on thread and timestamp
        unique_id = (
            f"{new_state['thread_id']}-{new_state.get('timestamp', uuid.uuid4())}"
        )
        
        # Metadata for both storage systems
        metadata = {
            "thread_id": new_state["thread_id"],
            "user_id": new_state["user_id"],
            "space_id": new_state["space_id"],
            "timestamp": str(new_state.get("timestamp", "")),
        }
        
        # Get the content to store - either use the memory_summary if available or just the input
        memory_content = new_state.get("input", f'<Memory type="input" content="{new_state["input"]}" />')
        
        documents = [
            Document(
                page_content=memory_content,
                metadata=metadata,
            )
        ]
        uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Short-term: ChromaDB via LangChain - store the formatted memory
        short_term_db.add_documents(documents=documents, ids = uuids)
        
        # Extract essential information for long-term storage using LLM
        system_message = """
        You are a legal memory extraction specialist. Your task is to identify and extract ONLY the most critical information 
        from user inputs that should be preserved in long-term memory across a legal workspace.
        
        Focus on extracting:
        1. Key facts and dates related to cases
        2. Important legal precedents or citations mentioned
        3. Client information and critical requirements
        4. Deadlines, hearing dates, and other time-sensitive information
        5. Specific legal strategies discussed
        6. Essential case outcomes or decisions
        
        Format the output as an XML structure with appropriate tags to categorize the information.
        DO NOT include general chit-chat, pleasantries, or non-essential information.
        Be concise but comprehensive in capturing only what would be valuable for long-term reference.
        """
        
        user_message = f"""
        Please extract only the essential legal information that should be preserved in long-term memory from this input:
        
        USER INPUT: {new_state["input"]}
        
        CONTEXT: {new_state.get('clarified_query', '')}
        
        If there is no essential legal information worth preserving long-term, respond with: <NoEssentialInformation />
        
        Otherwise, format your response as:
            <Memory category="[category]">[extracted information]</Memory>
            <!-- Include multiple Item tags as needed -->
        """
        class Response(TypedDict):
            """Joke to tell user."""
            memories: Annotated[List[str], ..., 'List of different memories formated like <Memory category="[category]">[extracted information]</Memory>']



        # Get extraction from LLM
        try:
            llm_response = llm.with_structured_output(Response).invoke(system=system_message, user=user_message)
            essential_content = llm_response
            memories = essential_content.memories
            
            # Add extraction metadata
            metadata["content_type"] = "essential_extraction"
            
            documents = [
                Document(
                    page_content=memories[i],
                    metadata=metadata,
                ) for i in memories
            ]
            uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
            # Short-term: ChromaDB via LangChain - store the formatted memory
            long_term_db.add_documents(documents=documents, ids = uuids)


        except Exception as e:
            print(f"Error extracting essential information: {e}, falling back to storing full memory")
            
        
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
        "input": "List upcoming case deadlines.",
        "user_id": "user123",
        "space_id": "space456",
        "thread_id": "thread789",
        "timestamp": 1711795200,
        "short_term_history": [],
        "long_term_snippets": [],
        "merged_context": [],
    }

    # Process the input and get the result
    try:
        result = mem_graph.invoke(state)
        print("Merged Context:", result.get("merged_context"))
    except Exception as e:
        print(f"Error during graph execution: {e}")
        import traceback

        traceback.print_exc()
