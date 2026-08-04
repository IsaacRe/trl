"""
Probe whether GLM 5.1 emits an empty <think></think> block followed by reasoning
*outside* the think block on a later agent turn.

Reproduces the message history of sample id 8efb11fe-4610-49cb-94ce-96cb6b8c1abd
(lambda/hermes-agent-reasoning-traces, glm-5.1 subset) up to — but not including —
its 2nd assistant turn, which in the dataset begins with:

    <think>\n</think>\nNow let me create the comprehensive test file:

then asks z-ai/glm-5.1 (via OpenRouter) to generate the next turn, capturing the
reasoning channel, the visible content, and any structured tool call.

Setup notes (after an initial run degenerated):
  * Tools are sent as a structured `tools` param (parsed from the sample's `tools`
    field). The sample's Hermes system prompt is dropped: it inlines the tool defs
    and mandates a JSON <tool_call> format that conflicts with GLM 5.1's native
    <tool_call>name<arg_key>… format, which the model injects itself when given
    structured tools. Keeping both produced two conflicting <|system|> blocks and a
    degenerate generation.
  * The assistant tool call is sent as a structured `tool_calls` entry and the tool
    result as a proper `tool` message (rendered by GLM as <|observation|>).
  * Provider is pinned to first-party "Z.AI" (no fallbacks) — the first run landed
    on a weak third-party endpoint ("Inceptron") that emitted gibberish.

Before calling, the script renders the exact messages+tools through GLM 5.1's own
chat template (apply_chat_template) and prints a summary so the prompt can be
eyeballed; the full render is saved to /tmp.

Usage:
    OPENROUTER_API_KEY=... python tests/glm51_empty_think_probe.py
"""

import ast
import copy
import json
import os
import re
import urllib.request

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "lambda/hermes-agent-reasoning-traces"
CONFIG = "glm-5.1"
SAMPLE_ID = "8efb11fe-4610-49cb-94ce-96cb6b8c1abd"
TARGET_PREFIX = "<think>\n</think>\nNow let me create the comprehensive test file:"
MODEL = "z-ai/glm-5.1"
PROVIDER = "Z.AI"
TEMPERATURE = 0.6
TOKENIZER = "zai-org/GLM-5.1"
SAMPLE_CACHE = "/tmp/glm51_sample.json"
RENDER_PATH = "/tmp/glm51_local_render.txt"
RESPONSE_PATH = "/tmp/glm51_empty_think_probe_response.json"


def _load_sample() -> dict:
    if os.path.exists(SAMPLE_CACHE):
        return json.load(open(SAMPLE_CACHE))
    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    for row in ds:
        if row["id"] == SAMPLE_ID:
            json.dump(row, open(SAMPLE_CACHE, "w"))
            return row
    raise SystemExit(f"sample {SAMPLE_ID} not found")


def _parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


def _build(row: dict) -> tuple[list[dict], list[dict]]:
    """Return (messages, tools). Tool-call `arguments` are kept as dicts (GLM's
    template iterates them); convert to JSON strings only for the API payload."""
    conv = row["conversations"]
    target_idx = next(
        i for i, m in enumerate(conv)
        if m["from"] == "gpt" and m["value"].startswith(TARGET_PREFIX)
    )
    tools = [
        {"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
        for t in _parse(row["tools"])
    ]

    messages: list[dict] = []
    for m in conv[:target_idx]:
        frm, val = m["from"], m["value"]
        if frm == "system":
            continue  # GLM injects its own tool system block from the structured tools
        if frm == "human":
            messages.append({"role": "user", "content": val})
        elif frm == "gpt":
            call = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", val, re.DOTALL)
            msg = {"role": "assistant", "content": val[: call.start()].rstrip() if call else val}
            if call:
                fn = _parse(call.group(1))
                msg["tool_calls"] = [{"id": None, "type": "function",
                                      "function": {"name": fn["name"], "arguments": fn["arguments"]}}]
            messages.append(msg)
        elif frm == "tool":
            tr = re.search(r"<tool_response>\s*(\{.*\})\s*</tool_response>", val, re.DOTALL)
            payload = _parse(tr.group(1)) if tr else {"content": val}
            tcid = payload.get("tool_call_id")
            for prev in reversed(messages):
                if prev["role"] == "assistant" and prev.get("tool_calls"):
                    prev["tool_calls"][0]["id"] = tcid
                    break
            messages.append({"role": "tool", "tool_call_id": tcid,
                             "content": json.dumps(payload.get("content", payload))})
    return messages, tools


def _render_check(messages: list[dict], tools: list[dict]) -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    rendered = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    open(RENDER_PATH, "w").write(rendered)
    n_tokens = len(tok.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=True)["input_ids"])
    print("=== LOCAL apply_chat_template CHECK (GLM 5.1 template) ===")
    print(f"  tokens={n_tokens} chars={len(rendered)} | "
          f"<|system|>={rendered.count('<|system|>')} <|user|>={rendered.count('<|user|>')} "
          f"<|assistant|>={rendered.count('<|assistant|>')} <|observation|>={rendered.count('<|observation|>')} "
          f"tools={len(tools)}")
    print(f"  ends with: {rendered[-60:]!r}")
    print(f"  full render saved to {RENDER_PATH}\n")


def _call(messages: list[dict], tools: list[dict], api_key: str) -> dict:
    api_messages = copy.deepcopy(messages)
    for m in api_messages:  # OpenAI wire format wants arguments as a JSON string
        for tc in m.get("tool_calls", []):
            tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
    payload = {
        "model": MODEL,
        "messages": api_messages,
        "tools": tools,
        "temperature": TEMPERATURE,
        "reasoning": {"enabled": True},
        "provider": {"only": [PROVIDER], "allow_fallbacks": False},
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY")

    row = _load_sample()
    messages, tools = _build(row)

    print("=== HISTORY SENT ===")
    for m in messages:
        tc = f" +tool_call({m['tool_calls'][0]['function']['name']})" if m.get("tool_calls") else ""
        extra = f" tool_call_id={m.get('tool_call_id')}" if m["role"] == "tool" else ""
        print(f"  {m['role']:9} {len((m['content'] or '')):6} chars{tc}{extra} | {(m['content'] or '')[:64]!r}")
    print(f"  (+ {len(tools)} structured tools; sample's Hermes system prompt dropped)\n")

    _render_check(messages, tools)

    print(f"calling {MODEL} via provider={PROVIDER} (temperature={TEMPERATURE})…\n")
    resp = _call(messages, tools, api_key)
    json.dump(resp, open(RESPONSE_PATH, "w"), indent=2)

    if "choices" not in resp:
        print("=== UNEXPECTED RESPONSE ===")
        print(json.dumps(resp, indent=2)[:2000])
        return

    choice = resp["choices"][0]
    msg = choice["message"]
    print("=== REASONING CHANNEL ===")
    print(msg.get("reasoning") or "(empty)")
    content = msg.get("content") or ""
    print(f"\n=== CONTENT ({len(content)} chars) ===")
    print(content if content else "(empty)")
    if msg.get("tool_calls"):
        print("\n=== STRUCTURED TOOL_CALLS ===")
        for tc in msg["tool_calls"]:
            args = tc["function"]["arguments"]
            print(f"  {tc['function']['name']}({args[:200]}{'…' if len(args) > 200 else ''})")
    print(f"\n=== finish_reason: {choice.get('finish_reason')} | provider: {resp.get('provider')} "
          f"| usage: {resp.get('usage')} ===")
    print(f"full response saved to {RESPONSE_PATH}")


if __name__ == "__main__":
    main()
