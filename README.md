# FAISS

[FAISS DOCUMENTATION](https://ai.meta.com/tools/faiss/#:~:text=FAISS%20(Facebook%20AI%20Similarity%20Search,more%20scalable%20similarity%20search%20functions)

<b><u>Choose FAISS index type based on size of vector datasets:</u></b>

Use <b>Flat Index</b>, When dataset < 100K vectors <br>
Use <b>HNSW Index</b>, When dataset > 500k - 1M vectors <br>
Use <b>IVF Index</b>, When dataset > 1M vectors <br>

<b><u>Metric Configuration: </u></b><br>
Flat Index:

Similarity metric is configured when creating an index. <br>
For L2 Euclidean distance, use faiss.IndexFlatL2(DIM) <br>
For Inner Product Similarity, use faiss.IndexFlatIP(DIM)<br>

HNSW Index:

HNSW uses L2 Euclidean distance by default.<br>
To switch to Inner Product Similarity, use index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT) 
Where DIM = Vector Dimensions,
      M = No of neighbours per graph
<br>

<b><u>Similarity score Calculations:</u></b>

For L2 Euclidean distance, if the similarity score is 0, its an exact match in the database. Smaller score indicates closer match in the database.

For Inner Product Similarity, if the similarity score is 1, its an exact match in the database. Higher score indicates closer match in the database.



