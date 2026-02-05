# LOL RAG assistant

Project LoL Assistant is a personal project exploring Retrieval-Augmented Generation (RAG) applied to a concrete and non-trivial use case: assisting players in the game League of Legends through contextual, knowledge-grounded responses.

## Ingestion

I use official recent data of the game from RIOT GAMES. [Latest](https://ddragon.leagueoflegends.com/cdn/dragontail-16.2.1.tgz)

To this I add texts from subtitles of a custom selection of "guide" videos from youtube. 

## Vectorisation

Uses google text-embedding-004 model. 

The code is poorly optimized, everytime an update is made in the database's json it reads the whole thing and encodes everything one more time.

## Chatbot

Gemini 2-5 flash.

Is really good to answer simple questions but struggles to give the correct answer when asked about complex concept and strategies of the game such as macro timings and mid game decision making even when fetching semanticly coherent documents from database. 

## Future improvements

- Using langchain for more complex architectures 



