import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US",  # On force l'anglais pour matcher "loses more against"
}


def fetch_raw_matchups(CHAMP: str, RANK: str) -> None:
    URL = f"https://www.leagueofgraphs.com/champions/counters/{CHAMP}/{RANK}"
    print(f"Connexion à {URL} ...")

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
            df = df.sort_values(by=["Value", "Matchup Type"], ascending=[True, True])
            df = df.drop_duplicates()
            output_filename = f"data/raw_matchups_{CHAMP}_{RANK}.csv"
            df.to_csv(output_filename, index=False, sep=";")
        else:
            print("No data extracted.")


if __name__ == "__main__":
    fetch_raw_matchups("kayle", "bronze")
