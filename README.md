# Vector_question_search_semelarity
# Knowledge Base and Similarity Search

This project involves creating a knowledge base from questions stored in a MySQL database, generating embeddings using the SentenceTransformer model, and storing these embeddings in a Faiss index for efficient similarity searches. Additionally, it includes a Flask web service to perform similarity searches on the stored embeddings.

## Requirements

- Python 3.11+
- MySQL Server
- Faiss
- SentenceTransformers
- Flask
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/knowledge-base-search.git
    cd knowledge-base-search
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install mysql-connector-python numpy faiss-cpu sentence-transformers flask
    ```

4. Set up your MySQL database and update the connection details in the `get_db_connection` function.

## Usage

### Step 1: Generate Embeddings and Store in Faiss Index

Run the following script to fetch questions from the database, generate embeddings, and store them in a Faiss index:

