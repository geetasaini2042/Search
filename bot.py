from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load App_data.json
with open("App_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

file_names = [item["file_name"] for item in data]

# Build TF-IDF index
vectorizer = TfidfVectorizer().fit(file_names)
vectors = vectorizer.transform(file_names)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing query ?q="}), 400

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, vectors).flatten()
    top_k = 5
    top_indices = scores.argsort()[::-1][:top_k]

    results = [data[i] for i in top_indices if scores[i] > 0.1]
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
