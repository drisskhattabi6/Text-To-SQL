# ai_agent.py
"""
AI Agent module (LangGraph nodes) + vector-store ingestion utilities.

FIX: This version explicitly converts SchemaDocument into LangChain Document 
objects within ingest_schema_from_db, ensuring page_content is a string 
and metadata is simple (just the table name) before passing to the manager.
The VectorStoreManager is simplified to expect LangChain Documents or strings.
"""

import re
# import typing as t
from dataclasses import dataclass, field
from typing import Optional, List, Dict, TypedDict

import pandas as pd
import pymysql
import psycopg2

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
import pymysql.cursors # Added for execute_sql fix

from langgraph.graph import StateGraph

# --- helpers you already have; keep these pointing to your actual helper implementations
from helpers import (
    call_openai,
    call_gemini,
    call_ollama,
    get_embeddings,
)

import chromadb

# LangChain imports (used for Document and Chroma wrapper)
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Try community util to filter complex metadata; fallback to simple sanitizer if missing
try:
    from langchain_community.vectorstores.utils import filter_complex_metadata
except Exception:
    def filter_complex_metadata(md: dict) -> dict:
        """Remove non-primitive metadata values (keep str/int/float/bool/None)."""
        safe = {}
        for k, v in md.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                safe[k] = v
        return safe


# ------------------------
# Config & small helpers
# ------------------------
CHROMA_DEFAULT_DIR = "./chroma_store"

def sanitize_sql(sql: str) -> str:
    """Remove markdown fences and multiple statements; return single SELECT-like SQL."""
    if sql is None:
        return ""
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql, flags=re.IGNORECASE)
    sql = sql.strip()
    if sql.count(";") > 1:
        sql = sql.split(";")[0]
    return sql.rstrip(";")

# ------------------------
# SchemaDocument dataclass (Kept for extract_schema output structure)
# ------------------------
@dataclass
class SchemaDocument:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)

# ------------------------
# VectorStoreManager (Simplified to expect LangChain Documents)
# ------------------------
class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_store", collection_name="db_schema"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.vectorstore = None
        self._embedding_model = None
        
    # --- NEW METHOD 1: Check if collection exists and has data ---
    def collection_is_ingested(self) -> bool:
        """Checks if the Chroma collection exists and contains any documents."""
        if chromadb is None:
            return False
        try:
            # Get the collection reference
            collection = self.client.get_collection(self.collection_name)
            # Check if the collection has documents
            if collection.count() > 0:
                return True
            return False
        except Exception:
            # Collection does not exist or access failed
            return False

    # --- NEW METHOD 2: Load existing collection ---
    def load_existing_vectorstore(self, embedding_model):
        """Loads the existing Chroma vector store into the manager's state."""
        self._embedding_model = embedding_model
        from langchain.vectorstores import Chroma
        # Re-initializes self.vectorstore by connecting to the existing collection
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_model
        )
        print(f"VectorStoreManager loaded existing collection '{self.collection_name}'.")
        return self.vectorstore

    def add_documents(self, docs: List[Document], embedding_model, upsert=False):
        """
        Add a list of LangChain Document objects into Chroma safely.
        """
        self._embedding_model = embedding_model

        # Delete existing collection if upsert=False
        if not upsert:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass

        # Initialize Chroma vectorstore
        from langchain.vectorstores import Chroma
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_model
        )

        # Add documents safely (assuming docs is now List[Document])
        if docs:
            self.vectorstore.add_documents(docs)

# ------------------------
# Module-level singletons
# ------------------------
VECTOR_STORE: Optional[VectorStoreManager] = None
EMBEDDING_MODEL = None

# ------------------------
# SQLAlchemy URL builder (No change)
# ------------------------
def build_sqlalchemy_url(db_type: str, user: str, password: str, host: str, port: str, dbname: str) -> str:
    if db_type.lower() == "mysql":
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
    elif db_type.lower() in ("postgres", "postgresql"):
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    else:
        raise ValueError("Unsupported db_type. Use 'mysql' or 'postgres'.")

# ------------------------
# Schema extraction (No change)
# ------------------------
# (extract_schema function remains unchanged)
def extract_schema(engine: Engine) -> List[SchemaDocument]:
    """
    Use SQLAlchemy inspector to get tables, columns, types, PKs and FKs.
    Returns list of SchemaDocument (one per table).
    """
    insp = inspect(engine)
    docs: List[SchemaDocument] = []
    try:
        table_names = insp.get_table_names()
    except Exception:
        table_names = []

    for table in table_names:
        try:
            cols = insp.get_columns(table)
        except Exception:
            cols = []
        try:
            pk = insp.get_pk_constraint(table) or {}
        except Exception:
            pk = {}
        try:
            fks = insp.get_foreign_keys(table) or []
        except Exception:
            fks = []

        content_lines = [f"Table: {table}", "Columns:"]
        for c in cols:
            ctype = str(c.get("type"))
            nullable = c.get("nullable", True)
            content_lines.append(f"  - {c.get('name')}: {ctype}, nullable={nullable}")
        if pk and pk.get("constrained_columns"):
            content_lines.append(f"Primary Key: {pk.get('constrained_columns')}")
        if fks:
            for fk in fks:
                referred_table = fk.get("referred_table")
                referred_cols = fk.get("referred_columns")
                constrained = fk.get("constrained_columns")
                content_lines.append(f"Foreign Key: {constrained} -> {referred_table}.{referred_cols}")

        doc_text = "\n".join(content_lines)
        # simple metadata only
        metadata = {"table": table, "num_columns": len(cols)}
        docs.append(SchemaDocument(id=f"{table}", content=doc_text, metadata=metadata))

    return docs


# ------------------------
# Ingest schema into Chroma
# ------------------------
def ingest_schema_from_db(
    user: str,
    password: str,
    host: str,
    port: str,
    dbname: str,
    db_type: str,
    persist_directory: str = CHROMA_DEFAULT_DIR
):
    """
    Connect to DB, extract schema, and ingest into Chroma using the chosen embedding model.
    """
    global VECTOR_STORE, EMBEDDING_MODEL
    if chromadb is None:
        raise RuntimeError("chromadb not installed. pip install chromadb")

    # Build engine
    db_uri = build_sqlalchemy_url(db_type, user, password, host, port, dbname)
    engine = create_engine(db_uri)
    print(f"Connected to {db_type} database at {host}:{port}/{dbname}")

    # Extract schema. docs is a list of SchemaDocument objects
    schema_docs = extract_schema(engine)
    print(f"Extracted {len(schema_docs)} schema documents.")
    if not schema_docs:
        return {"ingested_tables": [], "count": 0}

    # Initialize embedding model
    # Note: If called from load_or_create, EMBEDDING_MODEL might already be set.
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = get_embeddings()
        if EMBEDDING_MODEL is None:
            raise RuntimeError("Failed to instantiate embedding model")

    # convert SchemaDocument to LangChain Document
    lc_docs = []
    for d in schema_docs:
        # Convert content to string and metadata to only include 'table'
        lc_docs.append(Document(
            page_content=d.content,
            metadata={"table": d.metadata.get("table", d.id)} # Use d.id as fallback for table name
        ))
    
    # Initialize manager and add documents.
    manager = VectorStoreManager(persist_directory=persist_directory, collection_name="db_schema")
    # Pass the list of LangChain Document objects directly.
    manager.add_documents(lc_docs, EMBEDDING_MODEL, upsert=False)

    print(f"Ingested {len(lc_docs)} schema documents into Chroma at {persist_directory}")

    # expose module-level manager
    VECTOR_STORE = manager

    return {"ingested_tables": [d.id for d in schema_docs], "count": len(schema_docs)}

# ------------------------
# Load or Create Vector DB
# ------------------------
def load_or_create_vector_store(
    user: str,
    password: str,
    host: str,
    port: str,
    dbname: str,
    db_type: str,
    persist_directory: str = CHROMA_DEFAULT_DIR
):
    """
    Checks if the vector store exists and is ingested. If yes, loads it. 
    If no, extracts the schema and ingests it.
    """
    global VECTOR_STORE, EMBEDDING_MODEL
    
    if chromadb is None:
        raise RuntimeError("chromadb not installed. pip install chromadb")

    # 1. Initialize Embedding Model (Needed for both loading and ingesting)
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = get_embeddings()
        if EMBEDDING_MODEL is None:
            raise RuntimeError("Failed to instantiate embedding model")
            
    # 2. Initialize Manager and Check Persistence
    manager = VectorStoreManager(persist_directory=persist_directory, collection_name="db_schema")
    
    if manager.collection_is_ingested():
        # 3. Load Existing
        manager.load_existing_vectorstore(EMBEDDING_MODEL)
        print("Schema loaded from existing Chroma store.")
        
    else:
        # 4. Ingest Schema (The slow part, only runs if needed)
        print("Chroma store not found or empty. Ingesting schema...")
        ingest_schema_from_db(user, password, host, port, dbname, db_type, persist_directory)

    # Expose module-level manager regardless of whether it was loaded or created
    VECTOR_STORE = manager
    return VECTOR_STORE

# ------------------------
# Retrieval helper
# ------------------------
# (retrieve_schema_context function remains unchanged)
def retrieve_schema_context(user_query: str, top_k: int = 10) -> str:
    """Query vector store for relevant schema docs and join them into single string context."""
    global VECTOR_STORE
    
    if VECTOR_STORE is None or VECTOR_STORE.vectorstore is None:
        print("[ai_agent] DEBUG: VECTOR_STORE or vectorstore is None.")
        return ""
    
    try:
        # Use LangChain's similarity_search which returns LangChain Document objects
        results = VECTOR_STORE.vectorstore.similarity_search(user_query, k=top_k)
    except Exception as e:
        print(f"[ai_agent] ERROR during similarity_search: {e}")
        return ""
    
    if not results:
        print(f"[ai_agent] DEBUG: No documents found for query: {user_query}")
        return ""
    
    # Results are LangChain Document objects; extract page_content (the schema string)
    joined = "\n\n".join([r.page_content for r in results if isinstance(r, Document)])
    
    # Add a debug print to see what's retrieved
    print(f"[ai_agent] DEBUG: Retrieved schema context length: {len(joined)}")
    
    return joined

# ------------------------
# LangGraph node types and logic 
# ------------------------
# (AgentState, SQL_PROMPT_TEMPLATE, generate_sql, execute_sql, build_graph, and app remain unchanged)

class AgentState(TypedDict):
    query: str
    sql: Optional[str]
    result: Optional[str]
    db_uri: str
    llm_choice: str
    user: str
    password: str
    host: str
    port: str
    dbname: str
    db_type: str

SQL_PROMPT_TEMPLATE = """
You are an expert SQL developer. Use only the provided database schema information to generate a single, valid SQL SELECT query (no explanations).
Schema context:
{schema_context}

User request:
{user_query}

Constraints:
- Use only tables/columns present in schema_context.
- Output a single SELECT statement (no multiple statements).
- If the user requests counts/aggregations, use appropriate SQL aggregations.
- If you are unsure about types, avoid unsafe casts.
"""

def generate_sql(state: AgentState) -> AgentState:
    """LangGraph node: build RAG prompt using vector DB retrieved schema and call LLM chosen by user."""
    query = state["query"]
    llm_choice = state.get("llm_choice", "gemini").lower()
    schema_context = retrieve_schema_context(query, top_k=10)

    print("[ai_agent] Retrieved schema context length:", len(schema_context))
    print(schema_context)
    
    if not schema_context:
        print("[ai_agent] WARNING: No schema context retrieved. Check ingestion.")
        schema_context = "No relevant schema found. The database connection might be wrong or ingestion failed."
    
    prompt = SQL_PROMPT_TEMPLATE.format(schema_context=schema_context, user_query=query)

    print("[ai_agent] Generated prompt:", prompt)

    if llm_choice == "openai":
        sql_out = call_openai(prompt)
    elif llm_choice == "gemini":
        sql_out = call_gemini(prompt)
    elif llm_choice == "ollama":
        sql_out = call_ollama(prompt)
    else:
        sql_out = "-- ERROR: Unknown LLM choice"

    sql_out = sanitize_sql(sql_out)
    print("[ai_agent] Generated SQL:", sql_out)
    state["sql"] = sql_out
    return state

def execute_sql(state: AgentState) -> AgentState:
    """LangGraph node: execute SQL via pymysql or psycopg2 using credentials passed in state."""
    sql = state.get("sql")
    conn = None  # Initialize conn to None for safe access in the finally block

    # 1. Check for failed or missing SQL (e.g., SELECT NULL)
    if not sql or sql.strip().upper() in ("", "SELECT NULL", "-- ERROR"):
        state["result"] = "SQL generation failed or resulted in a non-executable query."
        print(f"[ai_agent] Skipping SQL execution. SQL: {sql}")
        return state

    db_type = state.get("db_type", "mysql").lower()
    user = state.get("user")
    password = state.get("password")
    host = state.get("host")
    port = state.get("port")
    dbname = state.get("dbname")

    # Ensure port is handled safely, assuming standard defaults if missing/None
    try:
        safe_port = int(port) if port is not None else (3306 if db_type == "mysql" else 5432)
    except ValueError:
        state["result"] = f"Invalid port value: {port}"
        return state

    try:
        # 2. Connection and execution logic
        if db_type == "mysql":
            # FIX: Added 'charset' and 'cursorclass' for robust connection with PyMySQL
            conn = pymysql.connect(
                host=host,
                user=user,
                password=password,
                db=dbname,
                port=safe_port,
                # Use a widely compatible character set
                charset='utf8mb4', 
                # This ensures consistent cursor behavior, often required for robust connections
                # NOTE: The cursorclass must be imported at the top of the file: `import pymysql.cursors`
                cursorclass=pymysql.cursors.Cursor
            )
            df = pd.read_sql(sql, conn)
        elif db_type in ("postgres", "postgresql"):
            # psycopg2.connect accepts port as string or integer
            conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=safe_port)
            df = pd.read_sql(sql, conn)
        else:
            state["result"] = f"Unsupported DB type: {db_type}"
            return state

        # Convert result DataFrame to a list of dicts
        state["result"] = df.to_dict(orient="records")
        print(f"[ai_agent] SQL executed successfully. {len(df)} rows returned.")

    except Exception as e:
        # This catches the OperationalError (1045) and other execution errors
        state["result"] = f"Error executing SQL: {type(e).__name__}: {e}"
        print(f"[ai_agent] SQL execution failed: {e}")

    finally:
        # 3. Guaranteed Connection Closure
        if conn is not None:
            try:
                conn.close()
            except Exception as close_e:
                # Log a warning if closing fails, but don't stop the agent
                print(f"[ai_agent] Warning: Failed to close connection gracefully: {close_e}")

    return state

# ------------------------
# Build LangGraph flow 
# ------------------------
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.set_entry_point("generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.set_finish_point("execute_sql")
    return graph.compile()

# compiled app
app = build_graph()
