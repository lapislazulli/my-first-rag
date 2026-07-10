import os
from groq import Groq
from dotenv import load_dotenv
from config import LLM_MODEL, TOP_K
from vector_db import VectorDB
from moderator import Moderator

load_dotenv()

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "rag_system_prompt.txt")


class RAG:
    def __init__(self):
        self.client = Groq()
        self.moderator = Moderator()
        self.db = VectorDB()

        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _build_system_prompt(self, chunks):
        chunks_text = "\n\n".join(
            f"[source: {c['metadata'].get('source', '?')}] {c['text']}"
            for c in chunks
        )
        return self.prompt_template.replace("{{CHUNKS}}", chunks_text)

    def answer_question(self, question):
        # 1. modération — si injection détectée, on arrête tout
        moderation_result = self.moderator.moderate(question)
        if moderation_result.get("is_prompt_injection", False):
            return "Votre question a été bloquée (tentative de prompt injection détectée)."

        # 2. retrieval
        chunks = self.db.retrieve(question, n=TOP_K)

        # 3. construction du prompt avec les chunks
        system_prompt = self._build_system_prompt(chunks)

        # 4. appel au LLM
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content
