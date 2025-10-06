# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from ai_agent import app, load_or_create_vector_store, ingest_schema_from_db, CHROMA_DEFAULT_DIR, VectorStoreManager
from helpers import check_mysql_connection, check_postgres_connection
# import ai_agent

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
if 'df_query_result' not in st.session_state:
    st.session_state.df_query_result = pd.DataFrame()
if 'generated_sql' not in st.session_state: # 🆕 Store SQL persistently too!
    st.session_state.generated_sql = ""


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

# -----------------------------------------------------------------
# 💬 AI Agent Query Section
# -----------------------------------------------------------------
query = st.text_area("Enter your request:", height=68)

if st.button("Run Query"):
    if query.strip() == "":
        st.warning("Please enter a query first.")
    elif not st.session_state.connection_successful:
        st.error("Please establish a successful database connection first.")
    elif not st.session_state.schema_ingested:
        st.error("Please ensure the schema context is loaded/ingested successfully.")
    else:
        # 1. Create a dictionary with ONLY the required arguments for connection check
        db_creds = {
            "user": st.session_state.db_details['user'],
            "password": st.session_state.db_details['password'],
            "host": st.session_state.db_details['host'],
            "port": st.session_state.db_details['port'],
            "dbname": st.session_state.db_details['dbname']
        }
        
        # 2. Get the correct checking function
        check_func = check_postgres_connection if st.session_state.db_details['db_type'] == "Postgres" else check_mysql_connection
        
        # 3. Perform the connection check
        if not check_func(**db_creds):
            st.error("Database connection failed during query execution. Please recheck credentials.")
            st.stop()
            
        # Prepare inputs for AI agent (Assuming 'app' is your application)
        inputs = {
            "query": query,
            "llm_choice": llm_choice,
            "embedding_choice": embedding_choice,
            **st.session_state.db_details
        }

        with st.spinner("Running AI Agent (Generating SQL and Executing)..."):
            final_state = app.invoke(inputs)

        # Get results
        sql = final_state.get("sql", "No SQL generated.")
        result = final_state.get("result")

        # --- FIX: Store both SQL and DataFrame in session state ---
        st.session_state.generated_sql = sql
        
        if isinstance(result, list) and result:
            df = pd.DataFrame(result)
            st.session_state.df_query_result = df
        else:
            st.session_state.df_query_result = pd.DataFrame()
            if result:
                st.warning(f"Query returned a non-tabular result: {result}")
            else:
                st.info("Query executed successfully but returned no rows.")


# -----------------------------------------------------------------
# 💾 Persistent Query Output (NEW LOCATION FOR TABLE & SQL)
# -----------------------------------------------------------------

if st.session_state.generated_sql:
    # Display SQL persistently
    with st.expander("Generated SQL") :
        st.code(st.session_state.generated_sql, language="sql")

if not st.session_state.df_query_result.empty:
    # Display DataFrame persistently
    st.markdown("##### Query Result")
    st.dataframe(st.session_state.df_query_result)
    
# -----------------------------------------------------------------
# 📈 Data Visualization Section - Functions (UNCHANGED)
# -----------------------------------------------------------------

# Helper function to get columns by data type
def get_column_options(df, dtype_filter='number'):
    """Filters dataframe columns based on a list of pandas dtypes."""
    if dtype_filter == 'number':
        return df.select_dtypes(include=np.number).columns.tolist()
    elif dtype_filter == 'categorical':
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    elif dtype_filter == 'temporal':
        return df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()
    return df.columns.tolist()

# Plotting function using Altair for flexibility and type-checking
def create_altair_plot(df, x_col, y_col, chart_type, color_col=None):
    """Creates an Altair chart based on user selection and chart type."""
    
    requires_y = chart_type in ["Scatter", "Line", "Bar", "Area", "Box Plot"]
    if requires_y and (not x_col or not y_col):
        st.info("Please select both X-axis and Y-axis columns.")
        return
    elif chart_type == "Histogram" and not x_col:
        st.info("Please select the X-axis column.")
        return

    x_dtype = 'Q' if x_col and df[x_col].dtype.kind in 'fi' else 'N'
    y_dtype = 'Q' if y_col and df[y_col].dtype.kind in 'fi' else 'N'
    
    base = alt.Chart(df).interactive()
    chart = None

    if chart_type == "Scatter":
        if not (x_dtype == 'Q' and y_dtype == 'Q'):
            st.warning("Scatter Plot requires two **numeric** (Quantitative) columns.")
            return
        chart = base.mark_circle().encode(
            x=alt.X(f'{x_col}:Q', title=x_col),
            y=alt.Y(f'{y_col}:Q', title=y_col),
            color=color_col if color_col else alt.value("steel blue"),
            tooltip=[x_col, y_col]
        ).properties(title='Scatter Plot')
        
    elif chart_type == "Bar":
        if not (x_dtype == 'N' and y_dtype == 'Q'):
            st.warning("Bar Chart requires a **Categorical** X-axis and a **Numeric** Y-axis.")
            return
        chart = base.mark_bar().encode(
            x=alt.X(f'{x_col}:N', title=x_col),
            y=alt.Y(f'{y_col}:Q', title=y_col),
            color=x_col if color_col else alt.value("teal"),
            tooltip=[x_col, y_col]
        ).properties(title='Bar Chart')
        
    elif chart_type == "Line":
        if not (y_dtype == 'Q'):
            st.warning("Line Chart requires a **Numeric** Y-axis.")
            return
        chart = base.mark_line().encode(
            x=alt.X(f'{x_col}', title=x_col), 
            y=alt.Y(f'{y_col}:Q', title=y_col),
            color=color_col if color_col else alt.value("blue"),
            tooltip=[x_col, y_col]
        ).properties(title='Line Chart')

    elif chart_type == "Area":
        if not (y_dtype == 'Q'):
            st.warning("Area Chart requires a **Numeric** Y-axis.")
            return
        chart = base.mark_area().encode(
            x=alt.X(f'{x_col}', title=x_col), 
            y=alt.Y(f'{y_col}:Q', title=y_col),
            color=color_col if color_col else alt.value("lightblue"),
            tooltip=[x_col, y_col]
        ).properties(title='Area Chart')
        
    elif chart_type == "Histogram":
        if not (x_dtype == 'Q'):
            st.warning("Histogram requires the X-axis to be a **single numeric** (Quantitative) column.")
            return
        chart = base.mark_bar().encode(
            x=alt.X(f'{x_col}:Q', bin=True, title=x_col), 
            y=alt.Y('count()', title='Frequency'), 
            color=alt.value("darkgreen"),
            tooltip=[x_col, 'count()']
        ).properties(title=f'Histogram of {x_col}')

    elif chart_type == "Box Plot":
        if not (x_dtype == 'N' and y_dtype == 'Q'):
            st.warning("Box Plot requires a **Categorical** X-axis and a **Numeric** Y-axis.")
            return
        chart = base.mark_boxplot(extent="min-max").encode(
            x=alt.X(f'{x_col}:N', title=x_col),
            y=alt.Y(f'{y_col}:Q', title=y_col),
            color=alt.value("purple"),
            tooltip=[x_col, y_col]
        ).properties(title='Box Plot')
        
    else:
        st.error("Invalid chart type selected.")
        return

    if chart:
        st.altair_chart(chart, use_container_width=True)

# -----------------------------------------------------------------
# 📈 Data Visualization Section - UI
# -----------------------------------------------------------------

df = st.session_state.df_query_result 

if not df.empty:
    st.markdown("---")
    st.markdown("### 📈 Data Visualization")
    
    # Create tabs for different plot types
    tab_scatter, tab_bar, tab_line, tab_area, tab_hist, tab_box = st.tabs([
        "Scatter Plot", "Bar Chart", "Line Chart", "Area Chart", "Histogram", "Box Plot"
    ])

    # Define column sets based on type for better UX
    numeric_cols = get_column_options(df, 'number')
    categorical_cols = get_column_options(df, 'categorical')
    all_cols = df.columns.tolist()

    # --- Scatter Plot Tab (Requires 2 Numeric) ---
    with tab_scatter:
        st.markdown("A **Scatter Plot** shows the relationship between two **numeric** variables.")
        if len(numeric_cols) < 2:
            st.warning("Scatter plots require at least two numeric columns.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                scatter_x = st.selectbox("X-axis (Numeric)", numeric_cols, key="scatter_x", index=0)
            with col2:
                default_y_index = (numeric_cols.index(scatter_x) + 1) % len(numeric_cols) if scatter_x in numeric_cols and len(numeric_cols) > 1 else 0
                scatter_y = st.selectbox("Y-axis (Numeric)", numeric_cols, index=default_y_index, key="scatter_y")
            
            create_altair_plot(df, scatter_x, scatter_y, "Scatter")

    # --- Bar Chart Tab (Requires 1 Categorical X, 1 Numeric Y) ---
    with tab_bar:
        st.markdown("A **Bar Chart** compares categories (X-axis) against a measure (Y-axis).")
        if not categorical_cols or not numeric_cols:
            st.warning("Bar charts require at least one categorical and one numeric column.")
        else:
            col3, col4 = st.columns(2)
            with col3:
                bar_x = st.selectbox("X-axis (Categorical)", categorical_cols, key="bar_x")
            with col4:
                bar_y = st.selectbox("Y-axis (Numeric)", numeric_cols, key="bar_y")
            
            create_altair_plot(df, bar_x, bar_y, "Bar")


    # --- Line Chart Tab (Requires 1 Ordered X, 1 Numeric Y) ---
    with tab_line:
        st.markdown("A **Line Chart** is best for showing trends over an ordered variable (like time or sequence).")
        if not numeric_cols:
             st.warning("Line charts require at least one numeric column for the Y-axis.")
        else:
            col5, col6 = st.columns(2)
            with col5:
                line_x = st.selectbox("X-axis (Ordered/Time)", all_cols, key="line_x")
            with col6:
                default_y_index = all_cols.index(numeric_cols[0]) if numeric_cols else 0
                line_y = st.selectbox("Y-axis (Numeric)", all_cols, index=default_y_index, key="line_y")

            create_altair_plot(df, line_x, line_y, "Line")
            
    # --- Area Chart Tab (Requires 1 Ordered X, 1 Numeric Y) ---
    with tab_area:
        st.markdown("An **Area Chart** is similar to a Line Chart, emphasizing the total magnitude.")
        if not numeric_cols:
             st.warning("Area charts require at least one numeric column for the Y-axis.")
        else:
            col7, col8 = st.columns(2)
            with col7:
                area_x = st.selectbox("X-axis (Ordered/Time)", all_cols, key="area_x")
            with col8:
                default_y_index = all_cols.index(numeric_cols[0]) if numeric_cols else 0
                area_y = st.selectbox("Y-axis (Numeric)", all_cols, index=default_y_index, key="area_y")

            create_altair_plot(df, area_x, area_y, "Area")

    # --- Histogram Tab (Requires 1 Numeric X) ---
    with tab_hist:
        st.markdown("A **Histogram** shows the frequency distribution of a **single numeric** column.")
        if not numeric_cols:
            st.warning("Histograms require at least one numeric column.")
        else:
            hist_x = st.selectbox("Column to Plot (Numeric)", numeric_cols, key="hist_x")
            create_altair_plot(df, hist_x, None, "Histogram")

    # --- Box Plot Tab (Requires 1 Categorical X, 1 Numeric Y) ---
    with tab_box:
        st.markdown("A **Box Plot** compares the distribution and outliers of a numeric variable across categories.")
        if not categorical_cols or not numeric_cols:
            st.warning("Box plots require at least one categorical and one numeric column.")
        else:
            col9, col10 = st.columns(2)
            with col9:
                box_x = st.selectbox("X-axis (Category)", categorical_cols, key="box_x")
            with col10:
                box_y = st.selectbox("Y-axis (Numeric)", numeric_cols, key="box_y")
            
            create_altair_plot(df, box_x, box_y, "Box Plot")
