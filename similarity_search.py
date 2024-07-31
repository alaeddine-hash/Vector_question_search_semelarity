from flask import Flask, request, jsonify
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load your vector store and other necessary files
index = faiss.read_index("vector_store.index")
question_ids = np.load("question_ids.npy")
languages = np.load("languages.npy")

# Embedding model
model = SentenceTransformer('intfloat/e5-large-v2')

@app.route('/similarity_search', methods=['POST'])
def similarity_search():
    data = request.get_json()
    query = data.get('query')
    language = data.get('language')

    if not query or not language:
        return jsonify({'error': 'Query and language must be provided'}), 400

    # Create the query embedding including the language
    query_vector = model.encode([f"{language}: {query}"]).astype('float32')

    # Perform the search
    D, I = index.search(query_vector, 1)  # Find the most similar vector
    similarity_score = 1 - D[0][0]  # Convert distance to similarity score
    question_id = int(question_ids[I[0][0]])
    result_language = languages[I[0][0]]

    return jsonify({"question_id": question_id, "similarity_score": similarity_score, "language": result_language})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
