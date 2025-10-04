# streamlit_app.py
import streamlit as st
import pandas as pd
from ai_agent import app, load_or_create_vector_store, ingest_schema_from_db, CHROMA_DEFAULT_DIR, VectorStoreManager
from helpers import check_mysql_connection, check_postgres_connection
import ai_agent # Kept for explicit calls

st.set_page_config(page_title="Chat with DB", layout="wide")
st.sidebar.title("Chat with DB")

# --- Streamlit Session State Initialization ---
# Initialize flags and credentials needed across reruns
if 'connection_successful' not in st.session_state:
    st.session_state.connection_successful = False
if 'schema_ingested' not in st.session_state:
    st.session_state.schema_ingested = False
if 'db_details' not in st.session_state:
    st.session_state.db_details = {}


# --- Streamlit Caching for Vector Store ---
# This function will run ONCE per Streamlit session (or until credentials/params change)
@st.cache_resource(show_spinner=False)
def setup_vector_store_cached(db_details: dict, embedding_choice: str, force_reingest: bool = False):
    """Caches the heavy operation of loading or creating the vector store."""
    
    # If forced re-ingestion, invalidate the cache explicitly before running.
    # NOTE: Streamlit's cache management handles the actual cache key.
    if force_reingest:
        st.warning("Forcing schema re-ingestion...")
        # To truly force a complete re-run and cache clear, we must call the
        # primary ingestion function directly, which handles the collection deletion.
        return ingest_schema_from_db(
            user=db_details.get('user'),
            password=db_details.get('password'),
            host=db_details.get('host'),
            port=db_details.get('port'),
            dbname=db_details.get('dbname'),
            db_type=db_details.get('db_type'),
            embedding_choice=embedding_choice,
            persist_directory=CHROMA_DEFAULT_DIR
        )

    st.info("Attempting to load existing schema or ingest new schema...")
    # Calls your function, which handles load (quick) or ingest (slow)
    vector_store_manager = load_or_create_vector_store(
        user=db_details.get('user'),
        password=db_details.get('password'),
        host=db_details.get('host'),
        port=db_details.get('port'),
        dbname=db_details.get('dbname'),
        db_type=db_details.get('db_type'),
        embedding_choice=embedding_choice,
        persist_directory=CHROMA_DEFAULT_DIR
    )
    return vector_store_manager


# -------------------------
# Sidebar: Database config
# -------------------------
st.sidebar.header("Database Connection")

db_type = st.sidebar.selectbox("DB Type", ["Postgres", "MySQL"])
user = st.sidebar.text_input("Username", value=st.session_state.db_details.get('user', ''))
password = st.sidebar.text_input("Password", type="password", value=st.session_state.db_details.get('password', ''))
host = st.sidebar.text_input("Host", value=st.session_state.db_details.get('host', 'localhost'))
default_port = "5432" if db_type == "Postgres" else "3306"
port = st.sidebar.text_input("Port", value=st.session_state.db_details.get('port', default_port))
dbname = st.sidebar.text_input("Database Name", value=st.session_state.db_details.get('dbname', ''))

# Determine DB URI once based on current inputs
if db_type == "Postgres":
    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
else:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"

# -------------------------
# DB Connection Check & Credential Saving
# -------------------------
if st.sidebar.button("Check Connection & Save Credentials"):
    check_func = check_postgres_connection if db_type == "Postgres" else check_mysql_connection
    
    with st.sidebar:
        with st.spinner("Checking connection..."):
            success = check_func(user, password, host, port, dbname)

    if success:
        st.session_state.connection_successful = True
        st.session_state.db_details = {
            "db_type": db_type,
            "user": user,
            "password": password,
            "host": host,
            "port": port,
            "dbname": dbname,
            "db_uri": db_uri
        }
        st.sidebar.success("✅ Connection successful! Credentials saved.")
        # Reset schema flag to trigger load/ingest attempt
        st.session_state.schema_ingested = False 
    else:
        st.session_state.connection_successful = False
        st.sidebar.error("❌ Connection failed. Check your credentials.")


st.sidebar.header("AI Configuration")
llm_choice = st.sidebar.selectbox("Choose LLM", ["gemini", "openai", "ollama"])
embedding_choice = st.sidebar.selectbox("Embedding Model", ["ollama", "hf"])

# -------------------------
# Schema Ingestion/Loading/Re-Ingestion Area (The new button is here)
# -------------------------
if st.session_state.connection_successful:
    
    # 1. ADD THE RE-INGEST BUTTON
    col1, col2 = st.sidebar.columns([1, 2])
    force_ingest_button = col1.button("Re-Ingest Schema", help="Force deletion and re-creation of the vector database (slow operation).")
    
    # Check if we should attempt to load/ingest
    if force_ingest_button or not st.session_state.schema_ingested:
        
        # Clear the cache for the specific setup function if forced
        if force_ingest_button:
            # We must clear the cache to ensure the function runs again
            setup_vector_store_cached.clear()
        
        try:
            with st.sidebar:
                # Pass the force flag to the cached function.
                # If force_ingest_button is True, the function bypasses the load_or_create logic
                # and calls ingest_schema_from_db directly.
                result = setup_vector_store_cached(
                    st.session_state.db_details,
                    embedding_choice,
                    force_reingest=force_ingest_button
                )
            
            # Check the result type to see if it was a load or an ingest
            if isinstance(result, VectorStoreManager):
                # Auto-load or initial ingest path
                if result.collection_is_ingested():
                    st.session_state.schema_ingested = True
                    st.sidebar.success("✅ Schema Context is READY (Loaded from disk).")
                else:
                    st.sidebar.error("❌ Schema ingestion failed or no tables found.")
            
            elif isinstance(result, dict) and 'ingested_tables' in result:
                # Force re-ingest path (returns the result dictionary from ingest_schema_from_db)
                st.session_state.schema_ingested = True
                st.sidebar.success(f"✅ Schema Re-Ingested! {result['count']} tables updated.")
                col2.write(f"Tables: {result['ingested_tables'][:5]}...")
            
        except Exception as e:
            st.session_state.schema_ingested = False
            st.sidebar.error(f"Failed to load/ingest schema: {e}")
            st.exception(e) # Show full traceback for debugging

else:
    st.sidebar.warning("Connect to the DB first to load the schema context.")


# -------------------------
# Main: User Query
# -------------------------
# st.subheader("💬 Ask a Question in Natural Language")
query = st.text_area("Enter your request:", height=68)

if st.button("Run Query"):
    if query.strip() == "":
        st.warning("Please enter a query first.")
    elif not st.session_state.connection_successful:
        st.error("Please establish a successful database connection first.")
    elif not st.session_state.schema_ingested:
        st.error("Please ensure the schema context is loaded/ingested successfully.")
    else:
        # 1. Create a dictionary with ONLY the required arguments for check_connection
        db_creds = {
            "user": st.session_state.db_details['user'],
            "password": st.session_state.db_details['password'],
            "host": st.session_state.db_details['host'],
            "port": st.session_state.db_details['port'],
            "dbname": st.session_state.db_details['dbname']
        }
        
        # 2. Get the correct checking function
        check_func = check_postgres_connection if st.session_state.db_details['db_type'] == "Postgres" else check_mysql_connection
        
        # 3. Perform the connection check using only the cleaned dictionary
        if not check_func(**db_creds):
            st.error("Database connection failed during query execution. Please recheck credentials.")
            st.stop()
            
        # Prepare inputs for AI agent (This part is fine as the agent expects all details)
        inputs = {
            "query": query,
            "llm_choice": llm_choice,
            "embedding_choice": embedding_choice,
            # Pass all DB details for the execute_sql node to use
            **st.session_state.db_details
        }

        with st.spinner("Running AI Agent (Generating SQL and Executing)..."):
            # Invoke AI Agent
            final_state = app.invoke(inputs)

        # Display results...
        st.markdown("##### Generated SQL")
        st.code(final_state.get("sql", "No SQL generated."), language="sql")

        st.markdown("##### Query Result")
        result = final_state.get("result")
        if isinstance(result, list) and result:
            st.dataframe(pd.DataFrame(result))
        else:
            st.write(result)