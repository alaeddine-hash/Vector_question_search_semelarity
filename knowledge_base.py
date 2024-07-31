import mysql.connector
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="193.203.15.45",
        user="alaeddine",
        password="apaia",
        database="cache_db"
    )

# Fetch questions from the database
def fetch_questions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, language, query FROM question")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# Generate embeddings for questions
def generate_embeddings(questions):
    model = SentenceTransformer('intfloat/e5-large-v2')
    embeddings = model.encode(questions)
    return embeddings

# Store embeddings in Faiss index
def store_embeddings(embeddings, question_ids, languages):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, "vector_store.index")
    np.save("question_ids.npy", np.array(question_ids))
    np.save("languages.npy", np.array(languages))

def knowledge_base():
    rows = fetch_questions()
    question_ids = [row[0] for row in rows]
    languages = [row[1] for row in rows]
    questions = [f"{row[1]}: {row[2]}" for row in rows]  # Concatenate language and question
    embeddings = generate_embeddings(questions)
    store_embeddings(embeddings, question_ids, languages)
    print("Embeddings generated and stored successfully.")

if __name__ == "__main__":
    knowledge_base()
