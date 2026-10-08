from __future__ import annotations

import os
from typing import Optional

PARAPHRASE_MODEL = "gpt-4o-mini"
EXPLAINER_MODEL = "gpt-4o-mini"

_client = None


def get_openai_client():
    global _client
    if _client is None:
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Pass --openai-api-key or export "
                "OPENAI_API_KEY before running components that call OpenAI."
            )
        _client = OpenAI(api_key=key)
    return _client


def generate_paraphrased_question(
    question: str,
    model: str = PARAPHRASE_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 128,
    strict: bool = False,
) -> str:
    try:
        response = get_openai_client().chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that rephrases questions into natural and human-like language. Please be creative in paraphrasing and do not use the same sentences before."
                    ),
                },
                {"role": "user", "content": f"Paraphrase this question: {question}"},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001
        if strict:
            raise
        print(f"Error generating paraphrased question: {e}")
        return question


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = EXPLAINER_MODEL,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = get_openai_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content.strip()
