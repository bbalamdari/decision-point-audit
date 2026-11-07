"""The label mask from the paper, in isolation: the bug (R9, "mask") and the fix (R13, "mask_open").

The training text is the Gemma 4 chat template's rendering of an agent trajectory. Loss is computed only on the
character spans this function returns (mapped to tokens with the tokenizer's offsets). Everything else is context.

- "mask" (R9, the bug): whole thought blocks '<|channel>thought ... <channel|>' are removed from the labels,
  delimiters included. The first token after every tool response is '<|channel>' (the agent keeps working), so that
  decision is never a label; the final answer ("Fixed the issue ...") after the last tool response is. The model is
  supervised only toward stopping.
- "mask_open" (R13, the fix): the opener '<|channel>thought\\n' is trained; the thought text and its closer stay
  context only.
- "train": thoughts are trained too.

Run `python mask_example/mask.py` to see the labelled spans of a two-step toy trajectory under each mode.
"""
import re


def trained_char_spans(text: str, mode: str = "mask_open") -> list[tuple[int, int]]:
    """Model-generated regions: from each '<|turn>model\\n' to its '<turn|>', minus inline tool responses and,
    in "mask" mode, minus thought blocks, and in "mask_open" mode, minus the thought text and its closer."""
    excl = r"<\|tool_response>.*?<tool_response\|>" + {
        "mask": r"|<\|channel>thought\n.*?<channel\|>",
        "mask_open": r"|(?<=<\|channel>thought\n).*?<channel\|>"}.get(mode, "")
    spans = []
    for m in re.finditer(r"<\|turn>model\n(.*?)(<turn\|>|$)", text, re.S):
        start, end = m.start(1), m.end(2)
        cur = start
        for r in re.finditer(excl, text[start:end], re.S):
            if r.start() + start > cur:
                spans.append((cur, r.start() + start))
            cur = r.end() + start
        if cur < end:
            spans.append((cur, end))
    return spans


# A two-step toy trajectory in the Gemma 4 format: the agent searches, then (after the tool response) answers.
TOY = ("<|turn>user\nFix the bug.<turn|>\n<|turn>model\n<|channel>thought\nSearch first.<channel|>"
       "<|tool_call>call:run_command{command:grep -n bug}<tool_call|>"
       "<|tool_response>response:run_command{output:a.py:3}<tool_response|>"
       "<|channel>thought\nEdit a.py.<channel|><|tool_call>call:edit_file{filepath:a.py}<tool_call|>"
       "<|tool_response>response:edit_file{status:ok}<tool_response|>"
       "Fixed the issue in a.py and submitted the patch.<turn|>")


def next_after_tool_response(text: str, spans: list[tuple[int, int]]) -> list[tuple[str, bool]]:
    """For every decision point (right after '<tool_response|>'), the next 12 characters and whether they are a label."""
    out = []
    for m in re.finditer(r"<tool_response\|>", text):
        i = m.end()
        out.append((text[i:i + 12], any(a <= i < b for a, b in spans)))
    return out


if __name__ == "__main__":
    for mode in ("mask", "mask_open"):
        print(f"{mode}:")
        for nxt, trained in next_after_tool_response(TOY, trained_char_spans(TOY, mode)):
            print(f"  after <tool_response|> -> {nxt!r:16} {'TRAINED' if trained else 'not a label'}")
