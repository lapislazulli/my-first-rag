import pandas as pd
from config import CORPUS_PATH
from vector_db import VectorDB


def main():
    df = pd.read_csv(CORPUS_PATH)
    print(f"{len(df)} chunks chargés depuis '{CORPUS_PATH}'")

    chunks = df["text"].tolist()
    metadatas = [
        {"source": row["source"], "categorie": row["categorie"], "chunk_id": row["id"]}
        for _, row in df.iterrows()
    ]

    db = VectorDB(chunks=chunks, metadatas=metadatas)

    # petit test rapide
    print("\n--- Test retrieval ---")
    questions = [
        "Quelle est la couleur du chat de Bob ?",
        "Comment s'appelle le chien d'Alice ?",
        "Qui est le maire de Villebrume ?",
    ]

    for q in questions:
        results = db.retrieve(q, n=2)
        print(f"\nQ: {q}")
        for r in results:
            print(f"  -> {r['text']} (dist: {r['distance']:.4f})")


if __name__ == "__main__":
    main()
