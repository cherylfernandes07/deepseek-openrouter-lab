import os
import time
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
}

def query_model(
    model: str,
    prompt: str,
    system: str = None,
    max_tokens: int = 1000,
    include_reasoning: bool = False,
    retries: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Returns:
        {
            "content":   str | None,   # the answer
            "reasoning": str | None,   # R1 chain-of-thought
            "usage":     dict,         # token counts + cost
            "model":     str,          # model actually used
            "provider":  str,          # backend provider
        }
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "include_reasoning": include_reasoning,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)

            if resp.status_code == 429:
                wait = 2 ** attempt          # 1s, 2s, 4s
                if verbose:
                    print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            msg     = data["choices"][0]["message"]
            usage   = data.get("usage", {})

            result = {
                "content":   msg.get("content"),
                "reasoning": msg.get("reasoning"),
                "usage":     usage,
                "model":     data.get("model"),
                "provider":  data.get("provider"),
            }

            if verbose:
                r_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                print(f"[{result['provider']} · {result['model']}]")
                print(f"  tokens → prompt: {usage.get('prompt_tokens')} | "
                      f"completion: {usage.get('completion_tokens')} | "
                      f"reasoning: {r_tokens} | "
                      f"cost: ${usage.get('cost', 0):.8f}")

            return result

        except requests.exceptions.Timeout:
            if verbose:
                print(f"Timeout on attempt {attempt + 1}")
            if attempt == retries - 1:
                raise

    raise RuntimeError(f"Failed after {retries} attempts")


def extract_answer(result: dict) -> str:
    """
    Returns the best available text from a result dict.
    Prefers content; falls back to reasoning if content is None.
    """
    return result["content"] or result["reasoning"] or ""