from sentence_transformers import SentenceTransformer

try:
    # This will attempt to load the model
    model = SentenceTransformer("BAAI/bge-m3")
    print("Success: BAAI/bge-m3 is inside your environment!")
except Exception as e:
    print(f"Failed to load the model. Error: {e}")