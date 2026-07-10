import json
import os
from groq import Groq
from dotenv import load_dotenv
from config import MODERATION_MODEL

load_dotenv()

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "moderator_system_prompt.txt")


class Moderator:
    def __init__(self):
        self.client = Groq()
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def moderate(self, question):
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
            result = {"is_prompt_injection": True}

        return result
