from token import STAR
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Dict, Any, List, TypedDict

from MemoryManager import string_reducer
from lib.VectorStore import VectorStore
from lib.embeddings import embeddings_e5_large, embeddings_mpnet_base
from lib.api_keys import pinecone_api_key_anshul, pinecone_api_key_madhav

law_store = VectorStore("law", embeddings_e5_large, pinecone_api_key_anshul).getVectorStore()
preceeding_store = VectorStore("cases", embeddings_mpnet_base, pinecone_api_key_madhav).getVectorStore()

class ChatSubgraphState(TypedDict):
    """State for the chat subgraph."""
    user_id: str
    thread_id: str
    space_id: str
    input: str
    intents: List[str]
    chat_results: Annotated[str, string_reducer]
    memory_summary: Annotated[str, string_reducer]
    rag_law_context: Annotated[str, string_reducer]
    rag_preceedings: Annotated[str, string_reducer]
    doc_context: Annotated[str, string_reducer]

class ChatSubgraphInput(TypedDict):
    """Input for the chat subgraph."""
    input: str
    user_id: str
    space_id: str
    thread_id: str
    memory_summary: Annotated[str, string_reducer]

class ChatSubgraphOutput(TypedDict):
    """Output for the chat subgraph."""
    chat_results: Annotated[str, string_reducer]



def rag_law(state: ChatSubgraphInput) -> ChatSubgraphState:
    # 1. Query vector DB with combined user input and memory summary
    # 2. Inject retrieved text into state['rag_context']
    user_query = state["input"]
    memory_context = state.get("memory_summary", "")
    
    # Merge user input with memory summary for richer context
    if memory_context:
        from lib.llm import llm
        from langchain_core.messages import SystemMessage, HumanMessage
        
        try:
            # Use LLM to create an optimized query that combines user input with memory context
            messages = [
                SystemMessage(content="""
                You are a legal query optimizer. Your task is to combine a user's question with relevant context 
                from their conversation history to create a more effective search query. Focus on:
                1. Legal terminology from both inputs
                2. Case references and citations
                3. Specific legal questions or requirements
                4. Relevant dates, names, and entities
                
                Return ONLY the optimized search query without explanations.
                """),
                HumanMessage(content=f"""
                USER QUESTION: {user_query}
                
                CONVERSATION CONTEXT: {memory_context}
                
                Create an optimized search query that combines the user's question with relevant context.
                """)
            ]
            
            enhanced_query = llm.invoke(messages).content
            print(f"Enhanced query: {enhanced_query}")
            query = enhanced_query
        except Exception as e:
            print(f"Error creating enhanced query: {e}. Using original query.")
            # Fallback to simple concatenation if LLM enhancement fails
            query = f"{user_query} {memory_context}"
    else:
        query = user_query
    
    # Retrieve relevant documents along with similarity scores
    results = law_store.similarity_search_with_score(query, k=5)
    
    # Format and store the retrieved context using XML style tags
    law_context = []
    
    for i, (doc, score) in enumerate(results):
        law_context.append(f"""<Document rank="{i+1}" relevance="{score:.4f}">
      <Source>{doc.metadata.get('source_file', 'Legal Document')}</Source>
      <Url>{doc.metadata.get('document_url', 'Legal Document')}</Url>
      <Content>{doc.page_content}</Content>
    </Document>""")
    
    # Add retrieved context to state
    state['rag_law_context'] = "\n".join(law_context)
    
    return state

def rag_previous_cases(state: ChatSubgraphInput) -> ChatSubgraphState:
    # 1. Query vector DB with combined user input and memory summary
    # 2. Inject retrieved text into state['rag_context']
    user_query = state["input"]
    memory_context = state.get("memory_summary", "")
    
    # Merge user input with memory summary for richer context
    if memory_context:
        from lib.llm import llm
        from langchain_core.messages import SystemMessage, HumanMessage
        
        try:
            # Use LLM to create an optimized query that combines user input with memory context
            messages = [
                SystemMessage(content="""
                You are a legal query optimizer. Your task is to combine a user's question with relevant context 
                from their conversation history to create a more effective search query. Focus on:
                1. Legal terminology from both inputs
                2. Case references and citations
                3. Specific legal questions or requirements
                4. Relevant dates, names, and entities
                
                Return ONLY the optimized search query without explanations.
                """),
                HumanMessage(content=f"""
                USER QUESTION: {user_query}
                
                CONVERSATION CONTEXT: {memory_context}
                
                Create an optimized search query that combines the user's question with relevant context.
                """)
            ]
            
            enhanced_query = llm.invoke(messages).content
            print(f"Enhanced query: {enhanced_query}")
            query = enhanced_query
        except Exception as e:
            print(f"Error creating enhanced query: {e}. Using original query.")
            # Fallback to simple concatenation if LLM enhancement fails
            query = f"{user_query} {memory_context}"
    else:
        query = user_query
    
    # Retrieve relevant documents along with similarity scores
    results = preceeding_store.similarity_search_with_score(query, k=5)
    
    # Format and store the retrieved context using XML style tags
    law_context = []
    
    for i, (doc, score) in enumerate(results):
        law_context.append(f"""<Document rank="{i+1}" relevance="{score:.4f}">
      <Source>{doc.metadata.get('source_file', 'Legal Document')}</Source>
      <Url>{doc.metadata.get('document_url', 'Legal Document')}</Url>
      <Content>{doc.page_content}</Content>
    </Document>""")
    
    # Add retrieved context to state
    state["rag_preceedings"] = "\n".join(law_context)
    
    return state



def doc_ret(state: ChatSubgraphInput) -> ChatSubgraphState:
    # 1. Semantic search over uploaded case docs
    # 2. Inject retrieved text into state['doc_context']
    # state['doc_context'] = "…relevant document excerpts…"
    return state

def context_merger(state: ChatSubgraphState) -> ChatSubgraphState:
    
    return state

def llm_chat_response(state: ChatSubgraphState) -> ChatSubgraphOutput:
    # 1. Build prompt using state['input'], state['rag_context'], state['doc_context']
    # 2. Call LLM, store result in state['chat_reply']
    state['chat_reply'] = "llm response based on input and context"
    return state

def build_chat_subgraph():
    g = StateGraph(ChatSubgraphState, input=ChatSubgraphInput, output=ChatSubgraphOutput)

    # Define nodes
    g.add_node("RAGLaw", rag_law)
    g.add_node("RAGPreviousCases", rag_previous_cases)
    g.add_node("DocRetrieval", doc_ret)

    g.add_node("LLMChat", llm_chat_response)

    # Wiring: START → RAG → Doc → LLM → END
    g.add_edge(START, "RAGLaw")
    g.add_edge(START, "DocRetrieval")
    g.add_edge(START, "RAGPreviousCases")

    g.add_edge("RAGLaw", "LLMChat")
    g.add_edge("RAGPreviousCases", "LLMChat")
    g.add_edge("DocRetrieval", "LLMChat")
    g.add_edge("LLMChat", END)

    return g