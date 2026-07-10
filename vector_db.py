import os
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME


class VectorDB:
    def __init__(self, chunks=None, metadatas=None, path=CHROMA_PATH):
        self.path = path
        base_exists = os.path.isdir(path) and os.listdir(path)

        if base_exists:
            self._load_existing()
        elif chunks is not None:
            self._create_new(chunks, metadatas)
        else:
            raise ValueError(f"Pas de base trouvée à '{path}' et pas de chunks fournis.")

    def _create_new(self, chunks, metadatas):
        print(f"Création de la base à '{self.path}'...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        client = chromadb.PersistentClient(path=self.path)

        # on garde le nom du modèle dans les metadata de la collection
        # comme ça au rechargement on sait quel modèle utiliser
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": EMBEDDING_MODEL}
        )

        embeddings = self.model.encode(
            chunks,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        ids = [f"id_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas if metadatas else [{"source": "unknown"}] * len(chunks)
        )
        print(f"{len(chunks)} chunks indexés.")

    def _load_existing(self):
        print(f"Rechargement de la base depuis '{self.path}'...")
        client = chromadb.PersistentClient(path=self.path)
        self.collection = client.get_collection(name=COLLECTION_NAME)

        # on relit le modèle depuis les metadata de la collection
        model_name = self.collection.metadata.get("embedding_model", EMBEDDING_MODEL)
        self.model = SentenceTransformer(model_name)
        print(f"Base chargée, modèle: {model_name}")

    def _encode(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def retrieve(self, question, n=3):
        query_embedding = self._encode(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return retrieved
