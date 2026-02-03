import os
from pyexpat import model
import chromadb
import json
from google import genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = input("Entrez votre clé API Google Generative AI: ").strip()

INPUT_FILE = "data/processed_knowledge.json"
PERSIST_DIRECTORY = "data/chroma_db"

class GoogleGenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input):
        # Chroma nous envoie une liste de textes (input)
        # On demande à Google de les transformer en vecteurs
        
        # Le nouveau SDK permet d'envoyer une liste directement
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=input,
            config={'output_dimensionality': 768}
        )
        
        if response.embeddings is None:
            return []
        res = [e.values for e in response.embeddings]
        return res

# --- 3. FONCTION PRINCIPALE ---

def create_vector_database():
    print("Initialisation")
    
    google_ef = GoogleGenAIEmbeddingFunction(
        api_key=GOOGLE_API_KEY,
        model_name="text-embedding-004"
    )
    
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Création de la collection
    collection = client.get_or_create_collection(
        name="lol_knowledge",
        embedding_function=google_ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Chargement JSON
    if not os.path.exists(INPUT_FILE):
        print(f"Erreur : {INPUT_FILE} introuvable.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"{len(data)} documents prêts. Vectorisation en cours...")

    # Batch Processing
    BATCH_SIZE = 50
    
    ids_batch = []
    documents_batch = []
    metadatas_batch = []
    count = 0

    for item in data:
        # Nettoyage et sécu
        if not item.get("content") or len(item["content"]) < 5:
            continue

        ids_batch.append(item["id"])
        documents_batch.append(item["content"])
        
        meta = {
            "source": item.get("source", "Unknown"),
            "category": item.get("category", "General"),
            "title": item.get("title") or item.get("name", "Unknown")
        }
        metadatas_batch.append(meta)
        
        # Envoi par paquets
        if len(ids_batch) >= BATCH_SIZE:
            try:
                collection.upsert(
                    ids=ids_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch
                )
                count += len(ids_batch)
                print(f"Embeddings générés... {count}/{len(data)}", end="\r")
            except Exception as e:
                print(f"\nErreur sur un batch : {e}")
            
            ids_batch = []
            documents_batch = []
            metadatas_batch = []

    # Dernier paquet
    if ids_batch:
        try:
            collection.upsert(ids=ids_batch, documents=documents_batch, metadatas=metadatas_batch)
        except Exception as e:
            print(f"\nErreur finale : {e}")

    print(f"Stocké dans : {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    create_vector_database()
    print("Base de données vectorielle créée avec succès.")
