# print(qdrant_client.__version__)
# print(inspect.signature(QdrantClient.query_points))


import inspect
from qdrant_client import QdrantClient

print(inspect.signature(QdrantClient.query_points))