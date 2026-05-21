"""
Convert armand0e/kimi-k2.6-claude-code-traces to conversations parquet.

Each JSONL file is one session. We follow the parentUuid chain to reconstruct
the linear conversation, merge consecutive assistant message chunks into single
turns, and format thinking blocks as <think>...</think>.

Usage:
    .venv/bin/python scripts/convert_claude_code_traces.py \
        --output data/kimi-k2.6-claude-code-traces.parquet
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


REPO = "armand0e/kimi-k2.6-claude-code-traces"


def _follow_chain(by_uuid: dict, children: dict) -> list[dict]:
    """
    Walk the parentUuid tree along the main path.

    Some sessions have hidden parent messages (system prompts, etc.) that create
    gaps in the chain.  We handle gaps by finding the longest run through
    messages ordered by timestamp, always preferring user messages over
    redacted_thinking dead-ends when there are multiple children.
    """
    # Sort all messages by timestamp as a fallback ordering
    ts_sorted = sorted(
        by_uuid.values(),
        key=lambda m: m.get("timestamp", ""),
    )

    # Build a set of uuids that are "reachable" dead-ends (redacted_thinking)
    def is_dead_end(uid: str) -> bool:
        m = by_uuid.get(uid)
        if not m:
            return True
        content = m.get("message", {}).get("content", [])
        return isinstance(content, list) and bool(content) and content[0].get("type") == "redacted_thinking"

    # Walk the timestamp-ordered list, but skip messages that are dead-end
    # branches (redacted_thinking messages whose sibling is a user message).
    chain: list[dict] = []
    skip: set[str] = set()

    # Pre-mark dead ends: if a parent has both a user-child and a
    # redacted_thinking-child, mark the latter for skipping.
    for parent_uid, kids in children.items():
        has_user_child = any(by_uuid[k]["type"] == "user" for k in kids if k in by_uuid)
        if has_user_child:
            for k in kids:
                if is_dead_end(k):
                    skip.add(k)

    for msg in ts_sorted:
        if msg["uuid"] not in skip:
            chain.append(msg)

    return chain


def _is_redacted(msg: dict) -> bool:
    content = msg.get("message", {}).get("content", [])
    return (isinstance(content, list) and bool(content)
            and content[0].get("type") == "redacted_thinking")


def _merge_assistant_group(msgs: list[dict]) -> str:
    """Merge consecutive assistant chunks into one <think>…</think>\n\ntext string."""
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    tool_calls: list[str] = []

    for msg in msgs:
        for item in msg.get("message", {}).get("content", []):
            t = item.get("type", "")
            if t == "thinking":
                thinking_parts.append(item.get("thinking", ""))
            elif t == "text":
                v = item.get("text", "")
                if v.strip():
                    text_parts.append(v)
            elif t == "tool_use":
                tool_calls.append(
                    f'<tool_call>\n{{"name": "{item["name"]}", '
                    f'"arguments": {json.dumps(item.get("input", {}))}}}\n</tool_call>'
                )

    thinking = "\n".join(thinking_parts)
    body = "\n".join(text_parts)
    if tool_calls:
        suffix = "\n\n".join(tool_calls)
        body = (body + "\n\n" + suffix).strip() if body else suffix

    return f"<think>\n{thinking}\n</think>\n\n{body}"


def _format_user(msg: dict) -> str | None:
    content = msg.get("message", {}).get("content", "")
    if isinstance(content, str):
        if content.strip().startswith("<task-notification>"):
            return None
        return content
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type", "")
        if t == "tool_result":
            c = item.get("content", "")
            if isinstance(c, list):
                c = " ".join(
                    x.get("text") or x.get("content") or ""
                    for x in c if isinstance(x, dict)
                )
            elif not isinstance(c, str):
                c = str(c)
            tid = item.get("tool_use_id", "")
            parts.append(f"[tool_result id={tid}]\n{c}")
        elif t == "text":
            v = item.get("text", "")
            if v.strip():
                parts.append(v)
    text = "\n".join(parts)
    return text if text.strip() else None


def build_conversation(lines: list[dict]) -> list[dict]:
    relevant = [l for l in lines if l.get("type") in ("user", "assistant") and l.get("uuid")]
    by_uuid = {l["uuid"]: l for l in relevant}
    children: dict[str, list] = defaultdict(list)
    for l in relevant:
        if l.get("parentUuid"):
            children[l["parentUuid"]].append(l["uuid"])

    chain = _follow_chain(by_uuid, children)
    conversations: list[dict] = []
    i = 0

    while i < len(chain):
        msg = chain[i]
        if msg["type"] == "user":
            text = _format_user(msg)
            if text:
                conversations.append({"from": "human", "value": text})
            i += 1
        else:
            group: list[dict] = []
            while i < len(chain) and chain[i]["type"] == "assistant":
                if not _is_redacted(chain[i]):
                    group.append(chain[i])
                i += 1
            if group:
                value = _merge_assistant_group(group)
                if value.strip():
                    conversations.append({"from": "gpt", "value": value})

    return conversations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/kimi-k2.6-claude-code-traces.parquet")
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    jsonl_files = [
        f for f in list_repo_files(REPO, repo_type="dataset")
        if f.endswith(".jsonl") and not f.startswith(".cache")
    ]
    if args.max_sessions:
        jsonl_files = jsonl_files[: args.max_sessions]

    print(f"Processing {len(jsonl_files)} sessions…")
    records = []
    for fname in jsonl_files:
        path = hf_hub_download(REPO, fname, repo_type="dataset")
        lines = [json.loads(l) for l in Path(path).read_text().strip().splitlines()]
        convs = build_conversation(lines)
        # need at least one human + one assistant turn
        if sum(1 for c in convs if c["from"] == "gpt") >= 1:
            records.append({"conversations": convs})

    print(f"Converted {len(records)} conversations")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out, index=False)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
