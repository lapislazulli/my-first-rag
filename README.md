# Mon Premier RAG

mini RAG : ChromaDB + sentence-transformers + Groq + agent modérateur.

## Architecture

```
├── config.py                  # Constantes centralisées (modèles, chemins)
├── vector_db.py               # Brique 1 — Base vectorielle persistante
├── moderator.py               # Brique 2 — Agent modérateur anti-injection
├── rag.py                     # Brique 3 — Orchestrateur du pipeline RAG
├── index_corpus.py            # Script d'indexation (à lancer une seule fois)
├── main.py                    # Boucle interactive
├── prompts/
│   ├── rag_system_prompt.txt  # Prompt système du RAG (avec marqueur {{CHUNKS}})
│   └── moderator_system_prompt.txt  # Prompt du modérateur
└── 05_corpus_rag.csv          # Corpus de faits absurdes (test idéal)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Créer un fichier `.env` à la racine :
```
GROQ_API_KEY=votre_clé_ici
```

## Utilisation

```bash
# Étape 1 — Indexer le corpus (une seule fois)
python index_corpus.py

# Étape 2 — Lancer le RAG interactif
python main.py
```

## Pipeline

1. **Modération** — L'agent modérateur analyse la question (prompt injection ?)
2. **Retrieval** — Les 3 chunks les plus proches sont récupérés via ChromaDB
3. **Prompt** — Le prompt système est construit en injectant les chunks
4. **Génération** — Le LLM Groq répond en se basant uniquement sur le contexte
