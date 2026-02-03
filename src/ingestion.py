import os
import json
from bs4 import BeautifulSoup
import youtube_transcript_api
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

BASE_DATA_PATH = "data/dragontail-16.2.1"
VERSION_LOL = "16.2.1"
LANG = "en_US"

# Construction des chemins
BASE_JSON_PATH = os.path.join(BASE_DATA_PATH, VERSION_LOL, "data", LANG)

def load_json(filename, subdir=None):
    """Charge un fichier JSON (gère les sous-dossiers comme 'champion/')"""
    if subdir:
        path = os.path.join(BASE_JSON_PATH, subdir, filename)
    else:
        path = os.path.join(BASE_JSON_PATH, filename)
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Fichier manquant : {path}")
        return None

def clean_html(html_text):
    """Nettoie les balises HTML (<br>, <font>, <active>...) des descriptions Riot."""
    if not html_text:
        return "Aucune description."
    text = html_text.replace("<br>", "\n").replace("<br />", "\n")
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text().strip()

def process_runes():
    """Extrait toutes les runes (Runes Reforged)."""
    print("🔮 Traitement des Runes...")
    data = load_json("runesReforged.json")
    if not data:
        return []

    docs = []
    for tree in data:  # Domination, Precision, etc.
        tree_name = tree['name']
        
        for slot in tree['slots']:
            for rune in slot['runes']:
                clean_desc = clean_html(rune['longDesc'])
                
                content = (
                    f"RUNE NAME: {rune['name']}\n"
                    f"TREE: {tree_name}\n"
                    f"TYPE: {'Keystone' if slot == tree['slots'][0] else 'Minor Rune'}\n"
                    f"EFFECT: {clean_desc}"
                )
                
                docs.append({
                    "id": f"rune_{rune['id']}",
                    "name": rune['name'],
                    "type": "rune_info",
                    "content": content
                })
    
    print(f"{len(docs)} runes extraites.")
    return docs

def process_items():
    """Extrait les items avec stats et prix."""
    print("Traitement des Items...")
    data = load_json("item.json")
    if not data:
        return []

    docs = []
    for item_id, info in data['data'].items():
        if not info.get('name'):
            continue

        desc = clean_html(info.get('description', ''))
        
        # (ex: {'FlatPhysicalDamageMod': 40})
        stats_text = ", ".join([f"{k}: {v}" for k, v in info.get('stats', {}).items()])
        
        content = (
            f"ITEM NAME: {info['name']}\n"
            f"GOLD COST: Total {info['gold']['total']} (Sell {info['gold']['sell']})\n"
            f"STATS: {stats_text}\n"
            f"DETAILS/PASSIVE: {desc}"
        )
        
        docs.append({
            "id": f"item_{item_id}",
            "name": info['name'],
            "type": "item_info",
            "content": content
        })
        
    print(f"{len(docs)} items extraits.")
    return docs

def process_champions_detailed():
    """
    Extrait les détails COMPLETS des champions.
    Ouvre chaque fichier individuel (ex: champion/Aatrox.json).
    """
    print("Traitement détaillé des Champions (cela peut prendre quelques secondes)...")
    
    # 1. On charge la liste sommaire pour avoir les noms/IDs
    summary_data = load_json("champion.json")
    if not summary_data:
        return []

    docs = []
    
    for champ_id in summary_data['data'].keys():
        detail_data = load_json(f"{champ_id}.json", subdir="champion")

        if not detail_data:
            continue

        champ = detail_data['data'][champ_id]
        
        base_stats = (
            f"HP: {champ['stats']['hp']} (+{champ['stats']['hpperlevel']}/lvl)\n"
            f"Mana: {champ['stats']['mp']} (+{champ['stats']['mpperlevel']}/lvl)\n"
            f"Move Speed: {champ['stats']['movespeed']}\n"
            f"Armor: {champ['stats']['armor']} (+{champ['stats']['armorperlevel']}/lvl)\n"
            f"MR: {champ['stats']['spellblock']} (+{champ['stats']['spellblockperlevel']}/lvl)\n"
            f"Attack Range: {champ['stats']['attackrange']}"
        )
        
        passive_desc = clean_html(champ['passive']['description'])
        passive_text = f"PASSIVE - {champ['passive']['name']}: {passive_desc}"
        
        # Sorts (Q, W, E, R)
        spells_text = ""
        keys = ['Q', 'W', 'E', 'R']
        
        for idx, spell in enumerate(champ['spells']):
            key = keys[idx]
            spell_desc = clean_html(spell['description'])
            
            cooldown = spell.get('cooldownBurn', 'N/A')
            cost = spell.get('costBurn', '0')
            range_val = spell.get('rangeBurn', 'N/A')
            
            spells_text += (
                f"\n--- SPELL {key}: {spell['name']} ---\n"
                f"Cooldown: {cooldown}s\n"
                f"Cost: {cost} Resource\n"
                f"Range: {range_val}\n"
                f"Description: {spell_desc}\n"
            )

        full_content = (
            f"CHAMPION: {champ['name']} ({champ['title']})\n"
            f"LORE: {champ['lore']}\n"
            f"TIPS (Ally): {', '.join(champ.get('allytips', []))}\n"
            f"TIPS (Enemy/Counter): {', '.join(champ.get('enemytips', []))}\n\n"
            f"--- BASE STATS ---\n{base_stats}\n\n"
            f"--- ABILITIES ---\n"
            f"{passive_text}\n"
            f"{spells_text}"
        )
        
        docs.append({
            "id": f"champ_detailed_{champ['key']}",
            "name": champ['name'],
            "type": "champion_detailed",
            "content": full_content
        })

    print(f" {len(docs)} champions traités avec détails sorts/stats.")
    return docs


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

VIDEO_DATABASE = [
    {"id": "Jh-eYTB1Ij0", "title": "Split Pushing Guide", "category": "Macro"},
    {"id": "GmGT1aQj1ek", "title": "The Ultimate Wave Management Guide", "category": "Laning"},
    {"id": "L7qPSGmS9ik", "title": "How to Trade Like a Challenger", "category": "Laning"},
    {"id": "cL6cWGQtocw", "title": "Minion Wave Control for Every Lane", "category": "Laning"},
    {"id": "ZzKpPJ_BNHM", "title": "How to Cs Like a Challenger", "category": "Laning"},
    {"id": "HN2qjqeGAFM", "title": "How to End Games (Macro)", "category": "Macro"},
    {"id": "34kHa32NIKQ", "title": "Mid Game Macro Guide", "category": "Macro"},
    {"id": "YwibbNDg7kM", "title": "Side Lane Macro", "category": "Macro"},
    {"id": "MjXxRn5sQiQ", "title": "Jungle Tracking Guide", "category": "Jungle"},
    {"id": "bYU3Y4G4uKc", "title": "How to Teamfight", "category": "Teamfight"},
    {"id": "1b2uLRx8miE", "title": "ADC Positioning Guide", "category": "Teamfight"},
    {"id": "3qAr9J2lBhw", "title": "Dodge Skillshots", "category": "Micro"},
    {"id": "ApJLn9Iaq9Q", "title": "Low Elo Macro", "category": "Psychology"},
    {"id": "l8vZCDhT9JE", "title": "TopLane Matchups", "category": "Laning"},
    {"id": "gPJ0zUI2yyU", "title": "TopLane Fundamentals", "category": "Laning"}
]

def fetch_clean_transcript(video_id):
    """Récupère et nettoie la transcription d'une vidéo."""
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        
        full_text = " ".join([t.text for t in transcript])
        
        # Nettoyage basique
        full_text = full_text.replace("\n", " ")
        full_text = full_text.replace("  ", " ")
        return full_text
        
    except TranscriptsDisabled:
        print(f"Sous-titres désactivés pour la vidéo {video_id}")
        return None
    except Exception as e:
        print(f"Erreur {video_id}: {str(e)}")
        return None

def process_youtube_videos():
    print(f"Démarrage de l'extraction de {len(VIDEO_DATABASE)} vidéos...")
    knowledge_base = []
    
    for video in VIDEO_DATABASE:
        text = fetch_clean_transcript(video['id'])
        
        if text:
            chunk_size = 1000
            overlap = 100
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                
                if len(chunk) < 100 :
                    continue
                
                doc = {
                    "id": f"yt_{video['id']}_{i}",
                    "source": "SkillCapped_YouTube",
                    "type": "coach_guide",
                    "category": video['category'],  # Important pour filtrer plus tard !
                    "title": video['title'],
                    "content": f"COACH ADVICE ({video['category']} - {video['title']}): {chunk}"
                }
                knowledge_base.append(doc)
    return knowledge_base


if __name__ == "__main__":
    all_local_data = []
    runes_docs = process_runes()
    items_docs = process_items()
    champs_docs = process_champions_detailed()
    videos_docs = process_youtube_videos()
    all_local_data.extend(runes_docs)
    all_local_data.extend(items_docs)
    all_local_data.extend(champs_docs)
    all_local_data.extend(videos_docs)

    output_file = "data/processed_knowledge.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(all_local_data, f, ensure_ascii=False, indent=4)
    print(f"Total documents extraits: {len(runes_docs) + len(items_docs) + len(champs_docs) + len(videos_docs)}")
