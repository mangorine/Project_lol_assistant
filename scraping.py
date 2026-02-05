import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US",  # On force l'anglais pour matcher "loses more against"
}


def fetch_raw_matchups(CHAMP: str, RANK: str) -> None:
    URL = f"https://www.leagueofgraphs.com/champions/counters/{CHAMP}/{RANK}"

    response = requests.get(URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    extracted_data = []

    section_titles = soup.find_all("h3", class_="box-title")

    for title in section_titles:
        title_text = title.get_text().strip().lower()

        matchup_type = "Neutre"
        if "loses more against" in title_text:
            matchup_type = "Game"
        elif "wins more against" in title_text:
            matchup_type = "Game"
        elif "loses lane against" in title_text:
            matchup_type = "Lane"
        elif "wins lane against" in title_text:
            matchup_type = "Lane"
        else:
            continue

        target_table = title.find_next("table")

        if target_table:
            rows = target_table.find_all("tr")

            for row in rows:
                name_span = row.find("span", class_="name")
                progressbar = row.find("progressbar")

                if name_span and progressbar:
                    try:
                        enemy_name = name_span.get_text().strip()
                        raw_value = float(progressbar.get("data-value"))
                        if matchup_type == "Lane":
                            value = raw_value / 100
                        else:
                            value = raw_value * 100
                        extracted_data.append(
                            {
                                "Champion": CHAMP,
                                "Enemy": enemy_name,
                                "Matchup Type": matchup_type,
                                "Value": value,
                            }
                        )
                    except Exception:
                        continue
    if extracted_data:
        df = pd.DataFrame(extracted_data)
        df = df.sort_values(by=["Matchup Type", "Value"], ascending=[True, False])
        df = df.drop_duplicates()
        output_filename = f"data/raw_matchups/{CHAMP}_{RANK}.csv"
        df.to_csv(output_filename, index=False, sep=";")
        # print(f"Data saved to {output_filename}")
    else:
        print("No data extracted.")


def get_champion_slugs():
    try:
        # 1. Récupérer le numéro du dernier patch officiel (ex: "14.23.1")
        version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
        versions = requests.get(version_url).json()
        latest_version = versions[0]
        print(f"🔹 Patch détecté : {latest_version}")

        champs_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion.json"
        data = requests.get(champs_url).json()

        champion_list = []

        for champ_id, champ_info in data["data"].items():

            slug = champ_id.lower()

            # exceptions manuelles
            if slug == "monkeyking":
                slug = "wukong"
            if slug == "renata":
                slug = "renata"

            champion_list.append(slug)

        champion_list.sort()
        return champion_list

    except Exception as e:
        print(f"Erreur lors de la récupération : {e}")
        return []


def fetch_combos(CHAMP: str, RANK: str):
    URL = f"https://www.leagueofgraphs.com/champions/counters/{CHAMP}/{RANK}"

    try:
        response = requests.get(URL, headers=HEADERS)
        # On utilise lxml si possible (plus rapide), sinon html.parser
        try:
            soup = BeautifulSoup(response.text, "lxml")
        except:
            soup = BeautifulSoup(response.text, "html.parser")

        extracted_data = []

        # Recherche de la section
        titles = soup.find_all("h3", class_="box-title")
        target_table = None

        for title in titles:
            if "best with" in title.get_text().strip().lower():
                target_table = title.find_next("table")
                break

        if target_table:
            # On prend toutes les lignes, y compris celles dans tbody
            rows = target_table.find_all("tr")

            for i, row in enumerate(rows):
                # On saute les entêtes (souvent pas de td, ou th)
                cells = row.find_all("td")
                if not cells:
                    continue

                # --- NOM DU CHAMPION ---
                ally_name = "Inconnu"
                # On regarde d'abord l'image (Naafiri)
                img = row.find("img")
                if img and img.get("alt"):
                    ally_name = img.get("alt").strip()
                # Sinon le texte
                elif row.find("span", class_="name"):
                    ally_name = row.find("span", class_="name").get_text().strip()

                # --- SCORE (La méthode Texte) ---
                # On cherche directement le texte "+5.1%" dans la classe progressBarTxt
                # C'est visible dans votre code HTML : <div class="progressBarTxt">+5.1%</div>
                score_div = row.find("div", class_="progressBarTxt")
                synergy_score = 0.0

                if score_div:
                    text_score = score_div.get_text().strip()  # ex: "+5.1%"
                    # On nettoie le texte pour garder juste le chiffre (5.1)
                    clean_score = re.sub(r"[^\d.-]", "", text_score)
                    try:
                        synergy_score = float(clean_score)
                    except:
                        synergy_score = 0.0
                else:
                    # Plan B : Si pas de texte, on tente l'attribut data-value
                    # Utile si le texte est caché
                    pbar = row.find("progressbar")
                    if pbar and pbar.get("data-value"):
                        try:
                            raw = float(pbar.get("data-value"))
                            synergy_score = raw * 100 if abs(raw) < 1 else raw
                        except:
                            pass

                # --- VALIDATION ---
                # On ne garde que si on a un nom valide
                if ally_name != "Inconnu":
                    extracted_data.append(
                        {
                            "Champion": CHAMP,
                            "Ally": ally_name,
                            "Score": round(synergy_score, 2),
                        }
                    )

            # --- SAUVEGARDE ---
            if extracted_data:
                df = pd.DataFrame(extracted_data)
                df = df.sort_values(by="Score", ascending=False)
                filename = f"data/combos/{CHAMP}_{RANK}.csv"
                df.to_csv(filename, index=False, sep=";")
            else:
                print("no data extracted.")

        else:
            print("Section 'Best With' introuvable (Vérifiez l'anglais/headers).")

    except Exception as e:
        print(f"Erreur critique : {e}")


ranks = ["iron", "bronze", "silver", "gold", "platinum", "diamond", "master"]
all_champs = get_champion_slugs()

if "__main__" == __name__:
    fetch_raw_matchups("kayle", "diamond")
    fetch_combos("irelia", "diamond")
