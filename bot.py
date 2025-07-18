from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json
import torch

app = Flask(__name__)

# Load your data
with open("App_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract file names
file_names = [item["file_name"] for item in data]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(file_names, convert_to_tensor=True)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing ?q= parameter"}), 400

    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    top_k = 5
    top_indices = torch.topk(similarities, top_k).indices

    results = [data[i] for i in top_indices]

    return jsonify(results)

# For local testing
if __name__ == "__main__":
    app.run(debug=True, port=5000)
