import numpy as np
import faiss
import time
import os

NUM_IMAGES = 10_000
DIM = 512
COSINE_THRESHOLD = 0.80         #If metric set to L2 Euclidean Distance, Cosine threshold should be 0.4

PRINT_FULL_VECTOR = False

print(os.cpu_count())           #printing number of CPU cores

def normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)
    
faiss.omp_set_num_threads(6)    #fixing number of threads to 6

print("Generating embeddings...")

embeddings = np.random.rand(NUM_IMAGES, DIM).astype("float32")
embeddings = normalize(embeddings)

names = [f"person_{i}.jpg" for i in range(NUM_IMAGES)]

print("Building FAISS index...")

# index = faiss.IndexFlatL2(DIM)  #Sets the metric to L2 euclidean distance
index = faiss.IndexFlatIP(DIM)    #Sets the metric to Inner Product Similarity
index.add(embeddings)             #Vector file getting stored on RAM

print("Vectors indexed:", index.ntotal)

query_id = np.random.randint(NUM_IMAGES)
query_vec = embeddings[query_id].reshape(1, -1)
print("\nQuery vector generated (in DB)")
print("Vector shape:", query_vec.shape)

print("\nQuery vector chosen:")
print("Index:", query_id)
print("Image:", names[query_id])

#warmup search to load data to CPU & stabilise the search time
k = 1
for _ in range(50):
    index.search(query_vec, k)

print("\nSearching...")

start = time.perf_counter()

k = 1
scores, indices = index.search(query_vec, k)    #search for the query vector

end = time.perf_counter()
elapsed_ms = (end - start) * 1000

best_score = scores[0][0]
best_idx = indices[0][0]

print("\nSearch result:")

if best_score >= COSINE_THRESHOLD:            #If metric set to L2 Euclidean Distance, best_score < Cosine_threshold
    print("Match found:", names[best_idx])
else:
    print("No match found")

print("Similarity:", best_score)
print(f"Search time: {elapsed_ms:.3f} ms")

print("\nVerification: ")
print("Query name :", names[query_id])
print("Result name:", names[best_idx])

