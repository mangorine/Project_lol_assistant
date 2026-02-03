# LOL RAG assistant

## Ingestion

On process des donnees DW en locals par datadragon ( data officielles de riot games sur les donnees du jeu ). [Latest](https://ddragon.leagueoflegends.com/cdn/dragontail-16.2.1.tgz)

On ajoute a cela du texte de videos youtube. Des difficultes peuvent etre rencontres due au limitations de youtube qui peut ban votre IP si vous faites trop de requetes. Comme solution temporaires, j'utilise juste un VPN pour changer d'IPs mais youtube ban plus rapidement les IPs clouds.

## Vectorisation ( chunking et embedding )

J'utilise un modele d'encodage de google : GoogleGenerativeAiEmbeddingFunction.

Lancer vector_db uniquement si on update les datas. 

Le code n'est pas optimise pour s'update souvent en effet a chaque fois qu'on essaie d'ajouter des donnees on va relire tout le JSON pour encoder, c'est une amelioration a faire.




