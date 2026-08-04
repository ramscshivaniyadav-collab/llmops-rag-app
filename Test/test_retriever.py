from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024
)
vs = Chroma(
    collection_name="rag_demo",
    embedding_function=embedder,
    persist_directory="saved-embeddings",
)

print("Chunks:", vs._collection.count())

retriever = vs.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke("What is this project about?")

print("Retrieved:", len(docs))

for d in docs:
    print("=" * 50)
    print(d.page_content[:300])