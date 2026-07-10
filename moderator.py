"""
Brique 2 — Agent modérateur.

Détecte les tentatives de prompt injection avant
que la question n'atteigne le RAG principal.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv

from config import MODERATION_MODEL

load_dotenv()

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "moderator_system_prompt.txt")


class Moderator:
    """
    Agent de modération qui détecte les prompt injections.

    Utilise un modèle dédié (famille safeguard) via Groq
    et retourne un dict {"is_prompt_injection": bool}.
    """

    def __init__(self):
        self.client = Groq()
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def moderate(self, question):
        """
        Analyse une question et détermine si c'est une injection.

        Returns:
            dict: {"is_prompt_injection": True/False}
        """
        response = self.client.chat.completions.create(
            model=MODERATION_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        raw = response.choices[0].message.content

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # En cas de réponse mal formée, on considère la question suspecte
            result = {"is_prompt_injection": True}

        return result
