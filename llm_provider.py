import os
from typing import Optional

from config import SETTINGS


class OpenAICompatibleProvider:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.api_key = api_key if api_key is not None else SETTINGS.openai_api_key
        self.model = model or SETTINGS.openai_model
        self.base_url = base_url or SETTINGS.openai_base_url or None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2,
             max_tokens: int = 2000) -> str:
        if not self.is_configured:
            raise RuntimeError("LLM API key is not configured.")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {type(exc).__name__}") from exc
