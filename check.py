import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

print(f"🔍 Clé utilisée : {api_key[:5]}...")
print("\n📋 LISTE DES MODÈLES (Brute) :")
print("---------------------------------")

try:
    # On itère simplement sur la liste sans demander de détails complexes
    pager = client.models.list()
    
    for model in pager:
        # On affiche juste le nom, c'est le seul truc dont on a besoin
        print(f"🔹 {model.name}")
        
except Exception as e:
    print(f"❌ Erreur critique : {e}")