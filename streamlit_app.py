# streamlit_app.py
import streamlit as st
from ai_agent import app
from helpers import check_mysql_connection, check_postgres_connection
import ai_agent

st.set_page_config(page_title="Chat with DB", layout="wide")
st.sidebar.title("Chat with DB")

# -------------------------
# Sidebar: Database config
# -------------------------
st.sidebar.header("Database Connection")

db_type = st.sidebar.selectbox("DB Type", ["Postgres", "MySQL"])
user = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
host = st.sidebar.text_input("Host", "localhost")
port = st.sidebar.text_input("Port", "5432" if db_type == "Postgres" else "3306")
dbname = st.sidebar.text_input("Database Name")

if db_type == "Postgres":
    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
else:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"

# -------------------------
# DB Connection Check
# -------------------------
if st.sidebar.button("Check Connection"):
    if db_type == "MySQL":
        success = check_mysql_connection(user, password, host, port, dbname)
    else:
        success = check_postgres_connection(user, password, host, port, dbname)

    if success:
        st.sidebar.success("✅ Connection successful!")
    else:
        st.sidebar.error("❌ Connection failed. Check your credentials.")

st.sidebar.header("AI Configuration")
llm_choice = st.sidebar.selectbox("Choose LLM", ["gemini", "openai", "ollama"])
embedding_choice = st.sidebar.selectbox("Embedding Model", ["ollama", "hf"])

if st.sidebar.button("Ingest Schema to Vector DB"):
    # Ensure connection check succeeded first
    if db_type == "MySQL":
        ok = check_mysql_connection(user, password, host, port, dbname)
    else:
        ok = check_postgres_connection(user, password, host, port, dbname)

    if not ok:
        st.sidebar.error("DB connection failed — cannot ingest schema. Please check credentials.")
    else:
        with st.sidebar:
            with st.spinner("Extracting schema and ingesting into vector DB..."):
                try:
                    print('Extracting schema and ingesting into vector DB...')
                    res = ai_agent.ingest_schema_from_db(
                        user=user,
                        password=password,
                        host=host,
                        port=port,
                        dbname=dbname,
                        db_type=db_type,
                        embedding_choice=embedding_choice,
                        persist_directory="./chroma_store"
                    )
                    st.success(f"Ingested {res['count']} tables.")
                    st.write(res["ingested_tables"])
                except Exception as e:
                    st.error(f"Failed to ingest schema: {e}")


# -------------------------
# Main: User Query
# -------------------------
# st.subheader("💬 Ask a Question in Natural Language")
query = st.text_area("Enter your request:", height=68)

if st.button("Run Query"):
    if query.strip() == "":
        st.warning("Please enter a query first.")
    else:
        # Check DB connection first
        if db_type == "MySQL":
            if not check_mysql_connection(user, password, host, port, dbname):
                st.error("MySQL connection failed. Please check credentials.")
                st.stop()
        elif db_type == "Postgres":
            if not check_postgres_connection(user, password, host, port, dbname):
                st.error("Postgres connection failed. Please check credentials.")
                st.stop()

        # Prepare inputs for AI agent
        inputs = {
            "query": query,
            "db_uri": db_uri,
            "llm_choice": llm_choice,
            "embedding_choice": embedding_choice
        }

        # Invoke AI Agent
        final_state = app.invoke(inputs)

        # Display generated SQL
        st.subheader("Generated SQL")
        st.code(final_state.get("sql", "No SQL generated."), language="sql")

        # Display query results
        st.subheader("Query Result")
        result = final_state.get("result")
        if isinstance(result, list) and result:
            st.dataframe(result)
        else:
            st.write(result)
            