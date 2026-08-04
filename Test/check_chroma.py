import chromadb

client = chromadb.PersistentClient(path="saved-embeddings")

collections = client.list_collections()

print("Collections:", collections)

for collection in collections:
    print(f"Collection: {collection.name}")
    c = client.get_collection(collection.name)
    print("Count:", c.count())