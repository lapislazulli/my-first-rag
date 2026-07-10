"""
Brique 1 — Base vectorielle persistante avec ChromaDB.

Cette classe gère la création, le rechargement et la recherche
dans une base vectorielle encodée avec sentence-transformers.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME


class VectorDB:
    """
    Base vectorielle persistante.

    Comportement à l'instanciation :
    - Si une base existe sur disque → la recharge (pas de réindexation).
    - Sinon, si des chunks sont fournis → crée la base et indexe.
    - Sinon → lève une erreur explicite.
    """

    def __init__(self, chunks=None, metadatas=None, path=CHROMA_PATH):
        self.path = path
        base_exists = os.path.isdir(path) and os.listdir(path)

        if base_exists:
            self._load_existing()
        elif chunks is not None:
            self._create_new(chunks, metadatas)
        else:
            raise ValueError(
                f"Aucune base trouvée à '{path}' et aucun chunk fourni. "
                "Impossible de démarrer."
            )

    def _create_new(self, chunks, metadatas):
        """Crée une nouvelle base : encode les chunks et les persiste."""
        print(f"[VectorDB] Création d'une nouvelle base à '{self.path}'...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Client persistant — les données survivent à l'arrêt du programme
        client = chromadb.PersistentClient(path=self.path)

        # On stocke le nom du modèle dans les métadonnées de la collection
        # → au rechargement, on sait quel modèle utiliser
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": EMBEDDING_MODEL}
        )

        # Encodage avec normalisation (similarité cosinus)
        embeddings = self.model.encode(
            chunks,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        # Insertion dans la collection
        ids = [f"id_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas if metadatas else [{"source": "unknown"}] * len(chunks)
        )
        print(f"[VectorDB] {len(chunks)} chunks indexés avec succès.")

    def _load_existing(self):
        """Recharge une base existante sans réindexer."""
        print(f"[VectorDB] Rechargement de la base depuis '{self.path}'...")
        client = chromadb.PersistentClient(path=self.path)
        self.collection = client.get_collection(name=COLLECTION_NAME)

        # Lire le modèle depuis les métadonnées de la collection
        # → évite le bug silencieux d'un mismatch de modèle
        model_name = self.collection.metadata.get("embedding_model", EMBEDDING_MODEL)
        self.model = SentenceTransformer(model_name)
        print(f"[VectorDB] Base rechargée. Modèle : {model_name}")

    def _encode(self, text):
        """Encode un texte avec le modèle chargé (normalisation incluse)."""
        return self.model.encode(
            text,
            normalize_embeddings=True
        ).tolist()

    def retrieve(self, question, n=3):
        """
        Recherche les n chunks les plus proches de la question.

        Retourne une liste de dicts : {"text": ..., "metadata": ..., "distance": ...}
        """
        query_embedding = self._encode(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )

        # Structurer les résultats
        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return retrieved
