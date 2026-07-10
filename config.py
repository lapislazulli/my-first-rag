"""
Module de configuration — constantes centralisées.
Tous les noms de modèles et chemins sont définis ici
pour être modifiés à un seul endroit.
"""

# Modèle d'embedding (multilingue, léger)
EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"

# Modèle de génération (LLM via Groq)
LLM_MODEL = "llama-3.3-70b-versatile"

# Modèle de modération (famille safeguard de Groq)
MODERATION_MODEL = "llama-guard-3-8b"

# Chemin de la base vectorielle persistante
CHROMA_PATH = "./chroma_db"

# Nom de la collection ChromaDB
COLLECTION_NAME = "mon_premier_rag"

# Nombre de chunks à récupérer par requête
TOP_K = 3

# Fichier corpus
CORPUS_PATH = "./05_corpus_rag.csv"
