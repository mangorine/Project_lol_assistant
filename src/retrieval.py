import os
import chromadb
from google import genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from vector_db import GoogleGenAIEmbeddingFunction

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIRECTORY = "data/chroma_db"

def search_knowledge_base(query_text, n_results=5):
    
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    google_ef = GoogleGenAIEmbeddingFunction(
        api_key=GOOGLE_API_KEY,
        model_name="text-embedding-004"
    )
    
    try:
        collection = client.get_collection(
            name="lol_knowledge",
            embedding_function=google_ef
        )
    except Exception as e:
        return [], f"Erreur: Impossible de lire la base de données. Lance vector_db.py ? ({e})"

    print(f"Question : '{query_text}'")

    # LA RECHERCHE (QUERY)
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    # Mise en forme des résultats
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    formatted_results = []
    
    for i in range(len(documents)):
        item = {
            "content": documents[i],
            "source": metadatas[i].get("source", "Inconnu"),
            "title": metadatas[i].get("title", "Sans titre"),
            "category": metadatas[i].get("category", "General"),
            "relevance_score": distances[i]
        }
        formatted_results.append(item)

    return formatted_results
