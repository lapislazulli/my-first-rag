# Mon Premier RAG

RAG minimal : ChromaDB + sentence-transformers + Groq + agent modérateur.

## Structure

```
config.py              → constantes (modèles, chemins)
vector_db.py           → base vectorielle persistante (ChromaDB)
moderator.py           → agent modérateur anti-injection
rag.py                 → orchestration du pipeline
index_corpus.py        → indexation du corpus (à lancer 1 fois)
main.py                → interface interactive
prompts/               → prompts système (fichiers texte)
05_corpus_rag.csv      → corpus de test
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Créer un `.env` :
```
GROQ_API_KEY=votre_clé
```

## Lancer

```bash
# indexer le corpus (1 seule fois)
python index_corpus.py

# lancer le RAG
python main.py
```
