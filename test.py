
from langchain.schema import Document


def retrieve_schema_context(user_query: str, top_k: int = 10) -> str:
    """Query vector store for relevant schema docs and join them into single string context."""
    global VECTOR_STORE
    if VECTOR_STORE is None or VECTOR_STORE.vectorstore is None:
        return ""
    
    # Use LangChain's similarity_search which returns LangChain Document objects
    results = VECTOR_STORE.vectorstore.similarity_search(user_query, k=top_k)
    
    if not results:
        return ""
    
    # Results are LangChain Document objects; extract page_content (the schema string)
    joined = "\n\n".join([r.page_content for r in results if isinstance(r, Document)])
    return joined

schema_context = retrieve_schema_context(query, top_k=10)