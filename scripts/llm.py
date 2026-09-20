"""Provider layer: OpenAI (default, key from .env) with Anthropic fallback.

Model via TABLEREAD_MODEL env var; defaults to gpt-4o (verified working with
the preservation account key).
"""
import copy
import json
import os
import random
import threading
import time

MODEL = os.environ.get("TABLEREAD_MODEL", "gpt-4o")
# $/1M tokens: (input, cached input, output)
PRICES = {"gpt-4o": (2.50, 1.25, 10.00), "gpt-4o-mini": (0.15, 0.075, 0.60),
          "claude-sonnet-5": (2.00, 0.20, 10.00)}


def load_env():
    if os.path.exists(".env"):
        for line in open(".env"):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def _strip_unsupported(schema):
    """OpenAI strict structured outputs rejects numeric/array bound keywords."""
    s = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            for k in ("minimum", "maximum", "minItems", "maxItems"):
                node.pop(k, None)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(s)
    return s


class LLM:
    def __init__(self):
        load_env()
        self.usage = {"in": 0, "cached": 0, "out": 0}
        self._lock = threading.Lock()
        if os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            self.provider = "openai"
            self.client = OpenAI()
        else:
            import anthropic
            self.provider = "anthropic"
            self.client = anthropic.Anthropic()

    def _track(self, tin, cached, tout):
        with self._lock:
            self.usage["in"] += tin
            self.usage["cached"] += cached
            self.usage["out"] += tout

    def cost(self):
        p = PRICES.get(MODEL, (2.50, 1.25, 10.00))
        u = self.usage
        return (u["in"] - u["cached"]) / 1e6 * p[0] + u["cached"] / 1e6 * p[1] + u["out"] / 1e6 * p[2]

    def _openai_call(self, **kwargs):
        """Retry hard on 429s: the org TPM limit is low, so waiting works."""
        import openai
        for attempt in range(10):
            try:
                return self.client.chat.completions.create(**kwargs)
            except openai.RateLimitError:
                if attempt == 9:
                    raise
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 2))

    def complete_json(self, system, user, schema, max_tokens=3000):
        if self.provider == "openai":
            resp = self._openai_call(
                model=MODEL,
                max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "result", "strict": True, "schema": _strip_unsupported(schema)}},
            )
            u = resp.usage
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            self._track(u.prompt_tokens, cached, u.completion_tokens)
            return json.loads(resp.choices[0].message.content)
        resp = self.client.messages.create(
            model=MODEL, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        u = resp.usage
        self._track(u.input_tokens, u.cache_read_input_tokens or 0, u.output_tokens)
        return json.loads(next(b.text for b in resp.content if b.type == "text"))

    def complete_text(self, system, user, max_tokens=500):
        if self.provider == "openai":
            resp = self._openai_call(
                model=MODEL, max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            u = resp.usage
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            self._track(u.prompt_tokens, cached, u.completion_tokens)
            return resp.choices[0].message.content.strip()
        resp = self.client.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        u = resp.usage
        self._track(u.input_tokens, u.cache_read_input_tokens or 0, u.output_tokens)
        return next(b.text for b in resp.content if b.type == "text").strip()
