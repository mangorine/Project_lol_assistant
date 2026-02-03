import os
import json
import requests
import time
from bs4 import BeautifulSoup
from riotwatcher import LolWatcher, ApiError

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
    # On remplace les <br> par des sauts de ligne pour la lisibilité
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

        # Nettoyage description
        desc = clean_html(info.get('description', ''))
        
        # Stats (ex: {'FlatPhysicalDamageMod': 40})
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
    
    # Pour chaque champion dans la liste
    for champ_id in summary_data['data'].keys():
        # 2. On charge le fichier DÉTAILLÉ du champion
        # Fichier situé dans data/en_US/champion/{Nom}.json
        detail_data = load_json(f"{champ_id}.json", subdir="champion")

        if not detail_data:
            continue

        # Riot encapsule les données dans data[champ_id]
        champ = detail_data['data'][champ_id]
        
        # --- A. Stats de base ---
        base_stats = (
            f"HP: {champ['stats']['hp']} (+{champ['stats']['hpperlevel']}/lvl)\n"
            f"Mana: {champ['stats']['mp']} (+{champ['stats']['mpperlevel']}/lvl)\n"
            f"Move Speed: {champ['stats']['movespeed']}\n"
            f"Armor: {champ['stats']['armor']} (+{champ['stats']['armorperlevel']}/lvl)\n"
            f"MR: {champ['stats']['spellblock']} (+{champ['stats']['spellblockperlevel']}/lvl)\n"
            f"Attack Range: {champ['stats']['attackrange']}"
        )
        
        # --- B. Passif ---
        passive_desc = clean_html(champ['passive']['description'])
        passive_text = f"PASSIVE - {champ['passive']['name']}: {passive_desc}"
        
        # --- C. Sorts (Q, W, E, R) ---
        spells_text = ""
        keys = ['Q', 'W', 'E', 'R']
        
        for idx, spell in enumerate(champ['spells']):
            key = keys[idx]
            spell_desc = clean_html(spell['description'])
            
            # Récupération des données variables ("burn" = valeurs pré-calculées par niveau comme 60/70/80)
            cooldown = spell.get('cooldownBurn', 'N/A')
            cost = spell.get('costBurn', '0')
            range_val = spell.get('rangeBurn', 'N/A')

            # Info-bulle détaillée (Tooltip)
            # Riot utilise des variables {{ e1 }} dans les tooltips.
            # Pour un RAG simple, on garde le texte descriptif général qui est souvent plus clair sémantiquement.
            
            spells_text += (
                f"\n--- SPELL {key}: {spell['name']} ---\n"
                f"Cooldown: {cooldown}s\n"
                f"Cost: {cost} Resource\n"
                f"Range: {range_val}\n"
                f"Description: {spell_desc}\n"
            )

        # Assemblage final du document Champion
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


def scrape_mobafire_guide(champion_name):
    """Scraping léger pour la stratégie (optionnel)."""
    url = f"https://www.mobafire.com/league-of-legends/champion/{champion_name}"
    try:
        time.sleep(2)
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        return None
    except Exception as e:
        print(f"Erreur scraping Mobafire pour {champion_name}: {e}")
        return None


if __name__ == "__main__":
    all_local_data = []
    runes_docs = process_runes()
    items_docs = process_items()
    champs_docs = process_champions_detailed()
    all_local_data.extend(runes_docs)
    all_local_data.extend(items_docs)
    all_local_data.extend(champs_docs)

    output_file = "data/processed_knowledge.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(all_local_data, f, ensure_ascii=False, indent=4)
    print(f"Total documents extraits: {len(runes_docs) + len(items_docs) + len(champs_docs)}")
