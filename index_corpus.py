"""
Script d'indexation du corpus CSV dans la base vectorielle.

À exécuter une seule fois pour créer la base ChromaDB persistante.
Les lancements suivants du RAG rechargeront la base sans réindexer.
"""

import pandas as pd
from config import CORPUS_PATH
from vector_db import VectorDB


def main():
    # Charger le corpus CSV
    df = pd.read_csv(CORPUS_PATH)

    print(f"[Indexation] {len(df)} chunks chargés depuis '{CORPUS_PATH}'")

    chunks = df["text"].tolist()
    metadatas = [
        {"source": row["source"], "categorie": row["categorie"], "chunk_id": row["id"]}
        for _, row in df.iterrows()
    ]

    # Créer la base (si elle existe déjà, VectorDB la rechargera)
    db = VectorDB(chunks=chunks, metadatas=metadatas)

    # Test rapide de retrieval
    print("\n--- Test de retrieval ---")
    test_questions = [
        "Quelle est la couleur du chat de Bob ?",
        "Comment s'appelle le chien d'Alice ?",
        "Qui est le maire de Villebrume ?",
    ]

    for q in test_questions:
        results = db.retrieve(q, n=2)
        print(f"\nQ: {q}")
        for r in results:
            print(f"  → {r['text']} (distance: {r['distance']:.4f})")


if __name__ == "__main__":
    main()
