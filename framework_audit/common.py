"""Shared pieces for the mask-generality audit: a 2-tool-call trajectory, tokenizer loading, stub packages for
importing single framework modules without their heavy package __init__, and the decision-point auditor.

Run every script with `python -I` from this directory's parent; downloaded code lives in ../dl_src (untrusted),
tokenizers in ../dl_gemma4 and ../dl_qwen3 (data only, no remote code)."""
import copy
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dl_src"
GEMMA_DIR = ROOT / "dl_gemma4"
QWEN_DIR = ROOT / "dl_qwen3"

TOOLS = [
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command.",
                                      "parameters": {"type": "object", "properties": {
                                          "command": {"type": "string", "description": "command"}},
                                          "required": ["command"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace text in a file.",
                                      "parameters": {"type": "object", "properties": {
                                          "path": {"type": "string", "description": "file"},
                                          "old": {"type": "string", "description": "old text"},
                                          "new": {"type": "string", "description": "new text"}},
                                          "required": ["path", "old", "new"]}}},
]


def trajectory(final_thought: bool = False) -> list[dict]:
    """system, user, [thought + call] -> tool, [thought + call] -> tool, final answer.
    final_thought: whether the final answer also carries a thought (in many agent datasets it does not)."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test in calc.py."},
        {"role": "assistant", "content": "", "reasoning_content": "I should run the tests first.",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "run_command", "arguments": {"command": "pytest -q"}}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "run_command", "content": "FAILED test_add: expected 3, got 4"},
        {"role": "assistant", "content": "", "reasoning_content": "add() uses a+b+1; remove the +1.",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "edit_file",
                                      "arguments": {"path": "calc.py", "old": "a+b+1", "new": "a+b"}}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "edit_file", "content": "ok"},
        {"role": "assistant", "content": "Fixed the issue: add() now returns a+b."},
    ]
    if final_thought:
        msgs[-1]["reasoning_content"] = "The edit is done; report."
    return msgs


def load_tok(path, template: str | None = None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
    if template is not None:
        tok.chat_template = template
    return tok


def stub(name: str, **attrs):
    """Insert a stub module (or package, if it has __path__) into sys.modules."""
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def namespace_pkg(name: str, path: Path):
    """Register a package without executing its __init__.py, so submodules can be imported one by one."""
    return stub(name, __path__=[str(path)])


# ----------------------------------------------------------------------------------------------- the auditor

def decision_points(tok, ids: list[int], marker_seqs: list[list[int]], after_tool_only=None) -> list[int]:
    """Indices k such that ids[k] is the first token the model generates at a turn start: the token right after
    any occurrence of one of the marker token sequences (e.g. '<tool_response|>' for Gemma 4, the
    '<|im_start|>assistant\\n' that follows a tool turn for Qwen3)."""
    out = []
    for seq in marker_seqs:
        n = len(seq)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == seq and (after_tool_only is None or after_tool_only(ids, i)):
                if i + n < len(ids):
                    out.append(i + n)
    return sorted(set(out))


def audit(tok, ids: list[int], labels: list[int], points: list[int], width: int = 3) -> list[dict]:
    """For each decision point: the first `width` tokens and whether each is a label, and the offset to the first
    supervised token at or after the point."""
    rows = []
    for k in points:
        toks = tok.convert_ids_to_tokens(ids[k:k + width])
        sup = [labels[j] != -100 for j in range(k, min(k + width, len(ids)))]
        first = next((j - k for j in range(k, len(ids)) if labels[j] != -100), None)
        rows.append({"pos": k, "next_tokens": toks, "supervised": sup, "first_supervised_offset": first})
    return rows


def prefix_check(tok, train_ids: list[int], msgs: list[dict], render_prompt) -> list[dict]:
    """Train/inference consistency: for every assistant message that follows a tool (or user) message, render the
    inference prompt for the history before it and check that it is a token prefix of the training sequence.
    Returns, per point, prefix ok / first divergence (decoded), and where the prompt ends."""
    out = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or msgs[i - 1]["role"] not in ("tool", "user"):
            continue
        p = render_prompt(msgs[:i])
        n = 0
        while n < len(p) and n < len(train_ids) and p[n] == train_ids[n]:
            n += 1
        ok = n == len(p)
        out.append({"msg": i, "prompt_len": len(p), "prefix_ok": ok,
                    "divergence": None if ok else {"at": n, "prompt_has": tok.decode(p[n:n + 12]),
                                                   "train_has": tok.decode(train_ids[n:n + 12])},
                    "prompt_tail": tok.decode(p[-6:])})
    return out


def show(title: str, tok, ids, labels, points, extra=None):
    print(f"\n=== {title}")
    for r in audit(tok, ids, labels, points):
        print(f"  pos {r['pos']:4d} next={r['next_tokens']} supervised={r['supervised']} "
              f"first_supervised_offset={r['first_supervised_offset']}")
    if extra:
        for e in extra:
            print("  prefix:", json.dumps(e, ensure_ascii=False))


def labelled_text(tok, ids, labels) -> str:
    """Rendered sequence with supervised spans in [[...]] for eyeballing."""
    out, cur = [], None
    for t, l in zip(ids, labels):
        s = l != -100
        if s != cur:
            out.append("[[" if s else ("]]" if cur else ""))
            cur = s
        out.append(tok.decode([t]))
    if cur:
        out.append("]]")
    return "".join(out)
