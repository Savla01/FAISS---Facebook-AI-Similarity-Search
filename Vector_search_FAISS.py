#the vector is being found but how kaise kya need to be understod

import numpy as np
import faiss
import time

# -----------------------
# Configuration
# -----------------------
NUM_IMAGES = 10_000
DIM = 512
COSINE_THRESHOLD = 0.80

# print full vector? (large output)

PRINT_FULL_VECTOR = False

# -----------------------
# Utility
# -----------------------

def normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)
    
# -----------------------
# Generate embeddings
# -----------------------
print("Generating embeddings...")

embeddings = np.random.rand(NUM_IMAGES, DIM).astype("float32")
embeddings = normalize(embeddings)

names = [f"person_{i}.jpg" for i in range(NUM_IMAGES)]

# -----------------------
# Build FAISS index
# -----------------------
print("Building FAISS index...")

index = faiss.IndexFlatIP(DIM)
index.add(embeddings)

print("Vectors indexed:", index.ntotal)

# -----------------------
# Query simulation
# -----------------------
query_id = np.random.randint(NUM_IMAGES)
query_vec = embeddings[query_id].reshape(1, -1)
print("\nQuery vector generated (in DB)")
print("Vector shape:", query_vec.shape)

print("\nQuery vector chosen:")
print("Index:", query_id)
print("Image:", names[query_id])
print("Vector shape:", query_vec.shape)

print("\nFirst 10 embedding values:")
print(query_vec[0][:10])

if PRINT_FULL_VECTOR:
    print("\nFull embedding:")
    print(query_vec[0])

# -----------------------
# Search
# -----------------------
print("\nSearching...")

start = time.perf_counter()

k = 1
scores, indices = index.search(query_vec, k)

end = time.perf_counter()
elapsed_ms = (end - start) * 1000

best_score = scores[0][0]
best_idx = indices[0][0]

print("\nSearch result:")

if best_score >= COSINE_THRESHOLD:
    print("Match found:", names[best_idx])
else:
    print("No match found")

print("Similarity:", best_score)
print(f"Search time: {elapsed_ms:.3f} ms")

print("\nVerification:")
print("Query name :", names[query_id])
print("Result name:", names[best_idx])

