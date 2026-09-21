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

def load_env():
    if os.path.exists(".env"):
        for line in open(".env"):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


load_env()  # must run before MODEL is resolved
MODEL = os.environ.get("TABLEREAD_MODEL", "gpt-4o")
# $/1M tokens: (uncached input, cache read, cache write, output)
PRICES = {"gpt-4o": (2.50, 1.25, 2.50, 10.00), "gpt-4o-mini": (0.15, 0.075, 0.15, 0.60),
          "claude-sonnet-5": (2.00, 0.20, 2.50, 10.00)}


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
        self.usage = {"in": 0, "cached": 0, "cache_write": 0, "out": 0}
        self._lock = threading.Lock()
        forced = os.environ.get("TABLEREAD_PROVIDER")  # "openai" | "anthropic" | unset
        use_openai = (forced == "openai" if forced
                      else bool(os.environ.get("OPENAI_API_KEY")))
        if use_openai:
            from openai import OpenAI
            self.provider = "openai"
            # Hard 60s request timeout: a hung connection must die, not stall a run.
            self.client = OpenAI(timeout=60.0, max_retries=2)
        else:
            import anthropic
            self.provider = "anthropic"
            # Same discipline as OpenAI: stuck requests must die fast.
            self.client = anthropic.Anthropic(timeout=90.0, max_retries=4)

    def _track(self, tin, cached, tout, cache_write=0):
        with self._lock:
            self.usage["in"] += tin          # uncached input only
            self.usage["cached"] += cached   # cache reads
            self.usage["cache_write"] += cache_write
            self.usage["out"] += tout

    def cost(self):
        p = PRICES.get(MODEL, (2.50, 1.25, 2.50, 10.00))
        u = self.usage
        return (u["in"] / 1e6 * p[0] + u["cached"] / 1e6 * p[1]
                + u["cache_write"] / 1e6 * p[2] + u["out"] / 1e6 * p[3])

    def _openai_call(self, **kwargs):
        """Retry hard on rate-limit 429s (the org TPM limit is low, so waiting
        works) — but fail immediately on exhausted credits, which no retry fixes."""
        import openai
        for attempt in range(10):
            try:
                return self.client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                    raise RuntimeError("OpenAI account is out of credits — add credits or "
                                       "switch provider (TABLEREAD_PROVIDER=anthropic)") from e
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
            self._track(u.prompt_tokens - cached, cached, u.completion_tokens)
            return json.loads(resp.choices[0].message.content)
        resp = self.client.messages.create(
            model=MODEL, max_tokens=max_tokens,
            thinking={"type": "disabled"},  # persona surveys don't need reasoning tokens
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": _strip_unsupported(schema)}},
        )
        u = resp.usage
        self._track(u.input_tokens, u.cache_read_input_tokens or 0, u.output_tokens,
                    u.cache_creation_input_tokens or 0)
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
            self._track(u.prompt_tokens - cached, cached, u.completion_tokens)
            return resp.choices[0].message.content.strip()
        resp = self.client.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": user}])
        u = resp.usage
        self._track(u.input_tokens, u.cache_read_input_tokens or 0, u.output_tokens,
                    u.cache_creation_input_tokens or 0)
        return next(b.text for b in resp.content if b.type == "text").strip()
