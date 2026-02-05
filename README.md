# LOL RAG assistant

premiere version "manuelle", je projette d'apprendre et d'utiliser langchain.

## Ingestion

On process des donnees DW en locale de datadragon ( data officielles de riot games sur les donnees du jeu ). [Latest](https://ddragon.leagueoflegends.com/cdn/dragontail-16.2.1.tgz)

On ajoute a cela du texte de subtitles de videos youtube. Des difficultes peuvent etre rencontres due au limitations de youtube qui peut ban votre IP si vous faites trop de requetes. Comme solution temporaire, j'utilise un VPN pour changer d'IPs mais youtube ban plus rapidement les IPs clouds.

## Vectorisation ( chunking et embedding )

J'utilise un modele d'encodage de google : text-embedding-004. Le code lit le fichier data/processed_knowledge.json creer par ingestion.py, et stocke les vecteurs dans data/chroma_db et utilise la methode upsert qui permet d'update.

Lancer vector_db uniquement pour initialiser ou update les data. 

Le code n'est pas optimise pour s'update souvent en effet a chaque fois qu'on essaie d'ajouter des donnees on va relire tout le JSON pour encoder, c'est une amelioration a faire.




