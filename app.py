import streamlit as st
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import textwrap

# --- LOAD CONFIG ---
load_dotenv()
TENANT_ID = "40c1b80f-7071-4cf6-8a06-cda221ff3f4d"
TENANT_SCHEMA = f"tenant_{TENANT_ID}"
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", 5432),
}
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# --- RAG CONFIG ---
model = SentenceTransformer("all-MiniLM-L6-v2")
SCHEMA_INDEX_FILE = "schema_index.faiss"
SCHEMA_CHUNKS_FILE = "schema_chunks.json"
RULES_INDEX_FILE = "rules_index.faiss"
RULES_CHUNKS_FILE = "rules_chunks.json"

# --- Build or load schema index ---
def build_schema_index():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s
            """, (TENANT_SCHEMA,))
            rows = cur.fetchall()

            cur.execute("""
                SELECT
                    tc.table_name AS source_table,
                    kcu.column_name AS source_column,
                    ccu.table_name AS target_table,
                    ccu.column_name AS target_column
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                WHERE constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s;
            """, (TENANT_SCHEMA,))
            fk_rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No tables found in schema '{TENANT_SCHEMA}'.")

    table_docs = {}
    for table, col, dtype in rows:
        table_docs.setdefault(table, []).append(f"{col} ({dtype})")

    relationships = []
    for src_table, src_col, tgt_table, tgt_col in fk_rows:
        relationships.append(f"{src_table}.{src_col} → {tgt_table}.{tgt_col}")

    chunks = []
    for table, cols in table_docs.items():
        rels = [r for r in relationships if r.startswith(table)]
        chunk = f"{table}: {', '.join(cols)}"
        if rels:
            chunk += f"\nRelationships: {', '.join(rels)}"
        chunks.append(chunk)

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, SCHEMA_INDEX_FILE)
    with open(SCHEMA_CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)
    return index, chunks

def load_schema_index():
    if os.path.exists(SCHEMA_INDEX_FILE) and os.path.exists(SCHEMA_CHUNKS_FILE):
        index = faiss.read_index(SCHEMA_INDEX_FILE)
        with open(SCHEMA_CHUNKS_FILE) as f:
            chunks = json.load(f)
        return index, chunks
    return build_schema_index()

# --- Build or load rules index ---
def build_rules_index():
    # Load your strict rules from file
    with open("strict_rules.txt", "r", encoding="utf-8") as f:
        rules_text = f.read()

    # Split into smaller chunks (e.g., ~500 characters each)
    chunks = textwrap.wrap(rules_text, width=500, break_long_words=False)

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, RULES_INDEX_FILE)
    with open(RULES_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    return index, chunks

def load_rules_index():
    if os.path.exists(RULES_INDEX_FILE) and os.path.exists(RULES_CHUNKS_FILE):
        index = faiss.read_index(RULES_INDEX_FILE)
        with open(RULES_CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks
    return build_rules_index()

# --- Load indexes ---
schema_index, schema_chunks = load_schema_index()
rules_index, rules_chunks = load_rules_index()

# --- Retrieval helpers ---
def get_relevant_chunks(question, index, chunks, top_k=5):
    q_emb = model.encode([question])
    D, I = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in I[0]]

# --- DB execution ---
def run_sql(query):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            return cur.fetchall()

# --- Generate SQL ---
def generate_sql(user_question, conversation_history):
    relevant_schema = "\n".join(get_relevant_chunks(user_question, schema_index, schema_chunks))
    relevant_rules = "\n".join(get_relevant_chunks(user_question, rules_index, rules_chunks))

    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for i, (q, a) in enumerate(conversation_history[-3:]):
            context += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"

    prompt = f"""
You are an expert PostgreSQL assistant for a multi-tenant Workdesk database.
All tables are inside the schema "{TENANT_SCHEMA}".
Use schema-qualified table names: "{TENANT_SCHEMA}".table_name
Do NOT use tenant_id filters.

STRICT RULES (retrieved for this query):
{relevant_rules}

{context}

Relevant schema:
{relevant_schema}

When the question is unclear, ask clarifying questions.
If generating SQL, return ONLY the SQL query with no explanations.
Current Question: {user_question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# --- Streamlit UI ---
st.title("Workdesk AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Workdesk data assistant. How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Workdesk data"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            conversation_history = []
            for i in range(0, len(st.session_state.messages)-1, 2):
                if i+1 < len(st.session_state.messages):
                    conversation_history.append(
                        (st.session_state.messages[i]["content"],
                         st.session_state.messages[i+1]["content"])
                    )

            response = generate_sql(prompt, conversation_history)

            if response.lower().startswith(("select", "with", "insert", "update", "delete")):
                message_placeholder.markdown("Running query...")
                results = run_sql(response)
                if results:
                    summary_prompt = f"Question: {prompt}\nSQL Result: {results}\nProvide a concise natural language answer."
                    summary_response = client.chat.completions.create(
                        model="gpt-4.1-nano",
                        messages=[{"role": "system", "content": summary_prompt}],
                        temperature=0.1
                    )
                    answer = summary_response.choices[0].message.content
                    full_response = f"**Query Executed:**\n```sql\n{response}\n```\n\n**Answer:**\n{answer}"
                else:
                    full_response = f"Query executed but returned no results:\n```sql\n{response}\n```"
            else:
                full_response = response
        except Exception as e:
            full_response = f"Error: {str(e)}"

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
