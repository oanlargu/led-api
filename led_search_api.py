from flask import Flask, request, jsonify
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load your embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load embedded devices
with open("led_device_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

vectors = np.array([item["embedding"] for item in data]).astype("float32")
metadata = [
    {"brand": item["brand"], "model": item["model"], "device_type": item["device_type"]}
    for item in data
]

# Build similarity index
nn = NearestNeighbors(n_neighbors=5, metric="cosine")
nn.fit(vectors)

@app.route("/search", methods=["POST"])
def search():
    payload = request.get_json()
    query = payload.get("query", "")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    q_vec = model.encode([query])
    distances, indices = nn.kneighbors(q_vec)

    results = [metadata[i] for i in indices[0]]
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
