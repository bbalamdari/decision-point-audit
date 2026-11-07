"""Decision-point audit for agent SFT data: is the next token trained wherever an inference prompt can end?

Rule (from our R9/R11 failure): at every position where an inference-time request can end -- right after a tool
response, or after the generation prompt -- the model's next token decides "keep working" vs "stop". If the label
mask leaves that token untrained for one kind of decision (e.g. every mid-trajectory "continue") while the other kind
is trained (the final "stop" answer), the model is supervised toward one side only. Validation loss is computed only
on labelled tokens, so it cannot see this. This tool checks the encoded data directly, in seconds, on CPU.

Two ways to find the decision points (both can be used together):
  * prompt mode (needs a tokenizer with a chat template and the example's `messages`): render the inference prompt
    for every assistant turn with `add_generation_prompt=True`; the decision point is the first token after that
    prompt, if the prompt is a token prefix of the training sequence (a mismatch is itself reported). The kind is
    taken from the teacher's turn: "continue" if it makes a tool call, else "stop".
  * marker mode: the position right after each occurrence of a marker token sequence (e.g. `<tool_response|>`),
    i.e. where *your* server ends a request. The kind is "continue" if a tool-call marker follows before the next
    decision point (with --call-marker), otherwise the last point of an example is "stop" and the rest "continue".

Labels follow the Hugging Face convention: `labels[k]` is the target for `input_ids[k]` (the model shifts), and
-100 means "not trained". A 0/1 mask field (`assistant_masks`, `completion_mask`, `loss_mask`) is accepted instead
of labels. Use --shifted if your labels are already shifted (labels[k] is the target for input_ids[k + 1]).

Only the standard library is needed for encoded input; `transformers` is imported only when --tokenizer is given.
Exit status: 0 = pass, 1 = audit failed (see --fail-on), 2 = usage or input error.
"""
from __future__ import annotations

import argparse
import collections
import importlib
import importlib.util
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

IGNORE = -100
CONTINUE, STOP = "continue", "stop"
MASK_FIELDS = ("assistant_masks", "completion_mask", "loss_mask")


@dataclass
class Point:
    """One decision point: the token at `pos` is the first one the model generates after a prompt that ends at pos."""
    example: int
    pos: int
    kind: str                         # "continue" | "stop"
    source: str                       # "prompt" | "marker:<name>"
    trained: bool
    next_id: int
    next_token: str = ""
    first_label_offset: int | None = None   # tokens from pos to the first trained token (None = none after pos)


@dataclass
class Report:
    points: list[Point] = field(default_factory=list)
    prefix_mismatches: list[dict] = field(default_factory=list)   # prompt mode: prompt is not a prefix of training
    marker_warnings: list[str] = field(default_factory=list)
    n_examples: int = 0


# ------------------------------------------------------------------------------------------------ core functions

def trained_flags(row: dict, shifted: bool = False) -> list[bool]:
    """Per input position k: is input_ids[k] a training target? From `labels` (-100 = no) or a 0/1 mask field."""
    ids = row["input_ids"]
    if "labels" in row and row["labels"] is not None:
        lab = list(row["labels"])
        flags = [l != IGNORE for l in lab]
    else:
        name = next((f for f in MASK_FIELDS if row.get(f) is not None), None)
        if name is None:
            raise ValueError("row has neither 'labels' nor a mask field " + "/".join(MASK_FIELDS))
        flags = [bool(x) for x in row[name]]
    if shifted:   # flags[k] says whether input_ids[k + 1] is a target
        flags = [False] + flags[:-1]
    if len(flags) != len(ids):
        raise ValueError(f"labels/mask length {len(flags)} != input_ids length {len(ids)}")
    return flags


def find_marker_points(ids: Sequence[int], marker: Sequence[int]) -> list[int]:
    """Positions right after every occurrence of the token sequence `marker` (if a token follows)."""
    n, out = len(marker), []
    if n == 0:
        return out
    first = marker[0]
    for i in range(len(ids) - n + 1):
        if ids[i] == first and list(ids[i:i + n]) == list(marker) and i + n < len(ids):
            out.append(i + n)
    return out


def contains_seq(ids: Sequence[int], seq: Sequence[int], start: int, end: int) -> bool:
    n = len(seq)
    return any(list(ids[i:i + n]) == list(seq) for i in range(start, max(start, end - n + 1)))


def classify(ids: Sequence[int], positions: list[int], call_markers: list[Sequence[int]] | None) -> list[str]:
    """continue/stop per sorted position. With call markers: continue iff a call marker starts before the next
    decision point (or the end). Without: the last point is the stop, all earlier ones continue."""
    kinds = []
    for j, p in enumerate(positions):
        nxt = positions[j + 1] if j + 1 < len(positions) else len(ids)
        if call_markers:
            kinds.append(CONTINUE if any(contains_seq(ids, c, p, nxt) for c in call_markers) else STOP)
        else:
            kinds.append(STOP if j == len(positions) - 1 else CONTINUE)
    return kinds


def first_label_offset(flags: Sequence[bool], pos: int) -> int | None:
    return next((j - pos for j in range(pos, len(flags)) if flags[j]), None)


def _as_ids(x: Any) -> list[int]:
    """Normalise apply_chat_template output (list, nested list, BatchEncoding/dict) to a flat list of ints."""
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    x = list(x)
    if x and isinstance(x[0], (list, tuple)):
        x = list(x[0])
    return [int(t) for t in x]


def prompt_points(tok, messages: list[dict], ids: Sequence[int], tools=None,
                  template_kwargs: dict | None = None) -> tuple[list[tuple[int, str, int]], list[dict]]:
    """For every assistant message that follows a user or tool message, render the inference prompt with
    add_generation_prompt=True. Returns ([(pos, kind, msg_index)], mismatches). kind = continue if the teacher's turn
    has tool calls, else stop. Assistant turns that directly follow another assistant turn are not request ends."""
    kw = dict(template_kwargs or {})
    pts, bad = [], []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or i == 0 or messages[i - 1].get("role") not in ("user", "tool"):
            continue
        prompt = _as_ids(tok.apply_chat_template(messages[:i], tools=tools, add_generation_prompt=True,
                                                 tokenize=True, **kw))
        n = 0
        while n < len(prompt) and n < len(ids) and prompt[n] == ids[n]:
            n += 1
        kind = CONTINUE if m.get("tool_calls") else STOP
        if n == len(prompt) and n < len(ids):
            pts.append((n, kind, i))
        else:
            d = {"message": i, "kind": kind, "prompt_len": len(prompt), "diverges_at": n}
            if hasattr(tok, "decode"):
                d["prompt_has"] = tok.decode(prompt[n:n + 12])
                d["training_has"] = tok.decode(list(ids[n:n + 12]))
            bad.append(d)
    return pts, bad


def audit_row(row: dict, example: int = 0, *, tok=None, markers: dict[str, Sequence[int]] | None = None,
              call_markers: list[Sequence[int]] | None = None, use_prompts: bool = True,
              template_kwargs: dict | None = None, shifted: bool = False, report: Report | None = None) -> Report:
    """Audit one encoded example. `row` needs input_ids and labels (or a mask field); `messages` (+ `tools`) and a
    tokenizer enable prompt mode; `markers` maps a name to a token sequence for marker mode."""
    rep = report or Report()
    rep.n_examples += 1
    ids = [int(t) for t in row["input_ids"]]
    flags = trained_flags(row, shifted)
    name_of = (lambda i: tok.convert_ids_to_tokens([i])[0]) if tok is not None else (lambda i: str(i))
    seen: dict[int, Point] = {}

    def add(pos, kind, source):
        if pos in seen:   # the same position found twice (prompt and marker): keep one, note both sources
            if source not in seen[pos].source.split("+"):
                seen[pos].source += "+" + source
            return
        p = Point(example, pos, kind, source, flags[pos], ids[pos], name_of(ids[pos]), first_label_offset(flags, pos))
        seen[pos] = p
        rep.points.append(p)

    if use_prompts and tok is not None and row.get("messages"):
        pts, bad = prompt_points(tok, row["messages"], ids, row.get("tools"), template_kwargs)
        for d in bad:
            d["example"] = example
        rep.prefix_mismatches.extend(bad)
        for pos, kind, _ in pts:
            add(pos, kind, "prompt")
    if markers:
        allpos = sorted({p for seq in markers.values() for p in find_marker_points(ids, seq)})
        kind_at = dict(zip(allpos, classify(ids, allpos, call_markers)))
        for mname, seq in markers.items():
            for pos in find_marker_points(ids, seq):
                add(pos, kind_at[pos], "marker:" + mname)
    return rep


def summarize(rep: Report) -> dict:
    """Counts per kind, the next tokens of untrained points, and the verdict flags."""
    by_kind: dict[str, dict] = {}
    for kind in (CONTINUE, STOP):
        ps = [p for p in rep.points if p.kind == kind]
        tr = [p for p in ps if p.trained]
        offs = [p.first_label_offset for p in ps if not p.trained and p.first_label_offset is not None]
        by_kind[kind] = {
            "points": len(ps), "trained": len(tr), "untrained": len(ps) - len(tr),
            "untrained_next_tokens": dict(collections.Counter(p.next_token for p in ps if not p.trained).most_common(10)),
            "trained_next_tokens": dict(collections.Counter(p.next_token for p in tr).most_common(10)),
            "median_offset_to_first_label_when_untrained": statistics.median(offs) if offs else None,
            "by_source": dict(collections.Counter(p.source for p in ps)),
        }
    c, s = by_kind[CONTINUE], by_kind[STOP]
    flags = []
    for kind, other in ((CONTINUE, STOP), (STOP, CONTINUE)):
        k, o = by_kind[kind], by_kind[other]
        if k["points"] and k["trained"] == 0 and o["trained"] > 0:
            flags.append(f"ONE_SIDED: no '{kind}' decision point is trained, but {o['trained']} '{other}' points are "
                         f"-- the model is supervised only toward '{other}'")
        elif k["points"] and k["trained"] == 0:
            flags.append(f"UNTRAINED_KIND: no '{kind}' decision point is trained")
    if c["untrained"] + s["untrained"] and not any(f.startswith(("ONE_SIDED", "UNTRAINED_KIND")) for f in flags):
        flags.append(f"PARTIAL: {c['untrained']} continue and {s['untrained']} stop decision points are untrained")
    if rep.prefix_mismatches:
        flags.append(f"PREFIX_MISMATCH: {len(rep.prefix_mismatches)} inference prompts are not a token prefix of the "
                     f"training sequence (train/inference format differs; those points were not audited by prompt)")
    if not rep.points:
        flags.append("NO_POINTS: no decision point found (check markers, messages and the tokenizer)")
    ratio = (c["trained"] / s["trained"]) if s["trained"] else None
    return {"examples": rep.n_examples, "continue": c, "stop": s,
            "trained_continue_to_stop_ratio": ratio, "flags": flags,
            "marker_warnings": rep.marker_warnings, "prefix_mismatch_examples": rep.prefix_mismatches[:5]}


def failed(summary: dict, fail_on: str) -> bool:
    fl = summary["flags"]
    if fail_on == "never":
        return False
    if any(f.startswith("NO_POINTS") for f in fl):
        return True
    if fail_on == "one-sided":
        return any(f.startswith("ONE_SIDED") for f in fl)
    return summary["continue"]["untrained"] + summary["stop"]["untrained"] > 0   # "any"


def format_summary(summary: dict) -> str:
    lines = [f"decision-point audit: {summary['examples']} examples"]
    for kind in (CONTINUE, STOP):
        k = summary[kind]
        lines.append(f"  {kind:8s} points {k['points']:7d}  trained {k['trained']:7d}  untrained {k['untrained']:7d}"
                     f"  sources {k['by_source']}")
        if k["untrained"]:
            lines.append(f"           untrained next tokens: {k['untrained_next_tokens']}"
                         f"  (median offset to first label: {k['median_offset_to_first_label_when_untrained']})")
        if k["trained"]:
            lines.append(f"           trained next tokens:   {k['trained_next_tokens']}")
    r = summary["trained_continue_to_stop_ratio"]
    lines.append(f"  trained continue : stop = {summary['continue']['trained']} : {summary['stop']['trained']}"
                 + (f" (~{r:.1f}:1)" if r is not None else ""))
    for w in summary["marker_warnings"]:
        lines.append("  WARNING " + w)
    for d in summary["prefix_mismatch_examples"]:
        lines.append("  prefix mismatch: " + json.dumps(d, ensure_ascii=False))
    if summary["flags"]:
        lines.extend("  FLAG " + f for f in summary["flags"])
    else:
        lines.append("  OK: every audited decision point is trained")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ input helpers

def load_labeler(spec: str) -> Callable[[Any, dict], dict]:
    """'hf-assistant-mask' or 'path/to/file.py:function' or 'package.module:function'. The function is called as
    fn(tokenizer, example) and returns a dict with input_ids and labels (or a mask field)."""
    if spec == "hf-assistant-mask":
        return hf_assistant_mask
    mod, _, fn = spec.rpartition(":")
    if not mod or not fn:
        raise ValueError("--labeler must be 'hf-assistant-mask' or 'module_or_file.py:function'")
    if mod.endswith(".py"):
        path = Path(mod).resolve()
        spec_ = importlib.util.spec_from_file_location(path.stem, path)
        m = importlib.util.module_from_spec(spec_)
        sys.path.insert(0, str(path.parent))
        spec_.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, fn)


def hf_assistant_mask(tok, ex: dict, template_kwargs: dict | None = None) -> dict:
    """Labels from the chat template's {% generation %} blocks (what TRL's assistant_only_loss uses)."""
    out = tok.apply_chat_template(ex["messages"], tools=ex.get("tools"), tokenize=True, return_dict=True,
                                  return_assistant_tokens_mask=True, **(template_kwargs or {}))
    ids = _as_ids(out)
    mask = out["assistant_masks"]
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    if mask and isinstance(mask[0], (list, tuple)):
        mask = mask[0]
    return {"input_ids": ids, "labels": [t if m else IGNORE for t, m in zip(ids, mask)]}


def encode_markers(tok, names: Iterable[str]) -> dict[str, list[int]]:
    out = {}
    for s in names:
        if all(part.strip().lstrip("-").isdigit() for part in s.split(",")):   # raw token ids "51" or "2,3,4"
            out[s] = [int(x) for x in s.split(",")]
        else:
            if tok is None:
                raise ValueError(f"marker {s!r} is text; pass --tokenizer or give token ids like '51' or '2,3'")
            out[s] = list(tok.encode(s, add_special_tokens=False))
    return out


def check_marker_tokenization(tok, row: dict, markers: dict[str, list[int]], text_markers: list[str]) -> list[str]:
    """If a text marker occurs more often in the decoded text than its token sequence does in input_ids, the marker
    is tokenized differently in context and marker mode would miss points."""
    if tok is None or not hasattr(tok, "decode"):
        return []
    text = tok.decode(row["input_ids"])
    warn = []
    for s in text_markers:
        n_text = text.count(s)
        n_tok = len(find_marker_points(row["input_ids"], markers[s])) + (1 if text.endswith(s) else 0)
        if n_text != n_tok:
            warn.append(f"marker {s!r}: {n_text} occurrences in text but {n_tok} as tokens {markers[s]} "
                        f"(tokenized differently in context?)")
    return warn


def unescape(s: str) -> str:
    return s.encode("utf-8").decode("unicode_escape") if "\\" in s else s


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("data", help="JSONL: rows with input_ids + labels (or a mask field), optionally messages/tools; "
                                 "or rows with messages (+ tools) when --labeler is given")
    ap.add_argument("--tokenizer", help="HF tokenizer path or hub id (enables prompt mode and text markers)")
    ap.add_argument("--chat-template", help="file with a Jinja chat template to use instead of the tokenizer's")
    ap.add_argument("--labeler", help="'hf-assistant-mask' or 'file.py:function' -> fn(tokenizer, example) "
                                      "returning input_ids + labels; use this to audit your own encoder")
    ap.add_argument("--marker", action="append", default=[],
                    help="where a request ends, as text (needs --tokenizer) or token ids '51' / '2,3'. Repeatable. "
                         "Backslash escapes are honoured ('<|turn>model\\n')")
    ap.add_argument("--call-marker", action="append", default=[],
                    help="start of a tool call (text or ids); a point is 'continue' iff one follows before the next "
                         "point. Without it, the last point of each example is 'stop'")
    ap.add_argument("--no-prompts", action="store_true", help="marker mode only, even if messages are present")
    ap.add_argument("--template-kwargs", default="{}", help='JSON passed to apply_chat_template, '
                                                            'e.g. \'{"enable_thinking": true}\'')
    ap.add_argument("--shifted", action="store_true", help="labels are pre-shifted (labels[k] targets ids[k+1])")
    ap.add_argument("--limit", type=int, default=0, help="audit at most N examples (0 = all)")
    ap.add_argument("--fail-on", choices=("any", "one-sided", "never"), default="any",
                    help="exit 1 if any decision point is untrained (default), only if one kind is never trained, "
                         "or never")
    ap.add_argument("--json", help="write the summary (and every untrained point) to this file")
    a = ap.parse_args(argv)

    try:
        tok = None
        if a.tokenizer:
            from transformers import AutoTokenizer   # only needed in this mode
            tok = AutoTokenizer.from_pretrained(a.tokenizer)
            if a.chat_template:
                tok.chat_template = Path(a.chat_template).read_text(encoding="utf-8")
        tkw = json.loads(a.template_kwargs)
        marker_names = [unescape(m) for m in a.marker]
        markers = encode_markers(tok, marker_names)
        calls = list(encode_markers(tok, [unescape(m) for m in a.call_marker]).values())
        text_markers = [m for m in marker_names if not all(p.strip().lstrip("-").isdigit() for p in m.split(","))]
        labeler = load_labeler(a.labeler) if a.labeler else None
        if labeler is not None and tok is None:
            raise ValueError("--labeler needs --tokenizer")
    except (ValueError, OSError, ImportError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    rep = Report()
    warned = set()
    for i, row in enumerate(iter_jsonl(a.data)):
        if a.limit and i >= a.limit:
            break
        if labeler is not None:
            enc = labeler(tok, row, tkw) if labeler is hf_assistant_mask else labeler(tok, row)
            if enc is None:
                continue
            row = {**row, **enc}
        for w in check_marker_tokenization(tok, row, markers, text_markers) if i < 20 else []:
            if w not in warned:
                warned.add(w)
                rep.marker_warnings.append(w)
        audit_row(row, i, tok=tok, markers=markers, call_markers=calls, use_prompts=not a.no_prompts,
                  template_kwargs=tkw, shifted=a.shifted, report=rep)
    s = summarize(rep)
    print(format_summary(s))
    if a.json:
        out = dict(s, untrained_points=[asdict(p) for p in rep.points if not p.trained])
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if failed(s, a.fail_on) else 0


if __name__ == "__main__":
    sys.exit(main())
