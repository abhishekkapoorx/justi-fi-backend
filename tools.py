from lib.llm import llm
from lib.VectorStore import VectorStore
from lib.embeddings import embeddings_e5_large, embeddings_mpnet_base
from lib.api_keys import pinecone_api_key_anshul, pinecone_api_key_madhav
from typing import List

law_store = VectorStore("law", embeddings_e5_large, pinecone_api_key_anshul).getVectorStore()
preceeding_store = VectorStore("cases", embeddings_mpnet_base, pinecone_api_key_madhav).getVectorStore()


def rag_law(query: str) -> List[str]:
    """Retrieve legal statutes, regulations, and legislative documents related to a query.
    
    This tool searches a vector database of legal statutes and regulations to find
    documents relevant to the user's query. It returns structured XML results with
    source information, relevance scores, and content excerpts from legal documents.
    
    Args:
        query: A string containing the legal question or topic to search for
        
    Returns:
        A list of XML-formatted document entries with source, URL, and content
    """
    # 1. Query vector DB with combined user input

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
    return law_context


def rag_previous_cases(query: str) -> List[str]:
    """Retrieve previous legal cases and precedents related to a query.
    
    This tool searches a vector database of legal case precedents to find
    similar cases relevant to the user's query. It returns structured XML results
    with case names, relevance scores, and key excerpts from previous legal cases.
    
    Args:
        query: A string containing the legal question or situation to find relevant cases for
        
    Returns:
        A list of XML-formatted document entries with source, URL, and content
    """
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
    
    return law_context


tools = [rag_law, rag_previous_cases]
llm_with_tools = llm.bind_tools(tools)