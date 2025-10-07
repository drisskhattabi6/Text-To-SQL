import os
import ollama
import google.generativeai as genai
from openai import OpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
import pymysql
import psycopg2

# Load keys from .env
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def check_mysql_connection(user, password, host, port, dbname) -> bool:
    """Check MySQL DB connection using pymysql"""
    try:
        print(f"Trying MySQL connection: {user}@{host}:{port}/{dbname}")
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=dbname,
            port=int(port)
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchall()
        conn.close()
        return True
    except Exception as e:
        print("MySQL connection error:", e)
        return False


def check_postgres_connection(user, password, host, port, dbname) -> bool:
    """Check Postgres DB connection using psycopg2"""
    try:
        print(f"Trying Postgres connection: {user}@{host}:{port}/{dbname}")
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchall()
        conn.close()
        return True
    except Exception as e:
        print("Postgres connection error:", e)
        return False


def call_openai(prompt, model="gpt-4o-mini"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an SQL expert."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def call_gemini(prompt, model="gemini-2.5-flash"):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    return response.text.strip()

def call_ollama(prompt, model="llama3"):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def get_embeddings():
    model_name="mxbai-embed-large:latest"
    return OllamaEmbeddings(model=model_name)

    
# def get_embeddings(model_choice="hf"):
#     if model_choice == "hf":
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#         return HuggingFaceEmbeddings(model_name=model_name)
#     elif model_choice == "ollama":
#         model_name="mxbai-embed-large:latest"
#         return OllamaEmbeddings(model=model_name)
#     else:
#         raise ValueError("Unknown embedding model choice")

