import os
from google import genai
from dotenv import load_dotenv
from retrieval import search_knowledge_base
from vector_db import GoogleGenAIEmbeddingFunction

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GENERATION_MODEL = "gemini-2.5-flash"

class LoLCoachBot:
    def __init__(self, rank="Silver"):
        self.rank = rank
        self.genai_client = genai.Client(api_key=GOOGLE_API_KEY)

    def ask(self, question):
        
        try :
            results = search_knowledge_base(question, n_results=5)
        except Exception as e:
            return f"Error DB not found : {e}"
        
        if not results:
            return "Sorry, I couldn't find any relevant information in my knowledge base."
        
        context_str = "" 
        sources_used = []

        for i, res in enumerate(results):
            src_info = f"{res['source']} - {res['title']}"
            context_str += f"Source {i + 1} ({src_info}): {res['content']}\n\n"
            sources_used.append(src_info)
        
        prompt = f"""You are a challenger coach for League of Legends. Following these rules :
        - Always answer in English, even if the question is in another language.
        - If possible Use only the provided sources to answer the question. Do not use any other information.
        - Answer in a concise and clear manner, suitable for a {self.rank} player.

        Your mission : Use the following extracted information from various sources to answer the question as best as possible:
        {context_str}
        Question: {question}
        """

        response = self.genai_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt)
        
        return response.text, sources_used


if __name__ == "__main__":
    bot = LoLCoachBot()
    
    print("\n LoL Coach RAG (Tape 'q' pour quitter)")
    while True:
        user_input = input("\nToi : ").strip()
        if user_input.lower() in ['q', 'quit']:
            break
        if not user_input:
            continue
            
        answer, sources = bot.ask(user_input)
        
        print(f"\nResponse :\n{answer}")
        
        if sources:
            print("\nSources :")
            for src in sources:
                print(f"- {src}")
