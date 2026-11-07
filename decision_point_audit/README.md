# Decision-point audit

A pre-training check for agent SFT data. It answers one question: **at every position where an inference request
can end, is the model's next token a training label?**

## Why

In a multi-turn tool-calling trajectory, every time a tool result comes back the model makes a decision with its
first generated token: keep working (open a thought, call a tool) or stop (write the final answer). If the label
mask leaves that token untrained at the *continue* points but trains it at the final *stop* point, the model is
supervised toward stopping only. Validation loss is computed only on labelled tokens, so it cannot see this.

We hit exactly this: a "don't train reasoning" mask removed the whole thought block of Gemma 4, opener `<|channel>`
included. In our 3,000 training trajectories every mid-trajectory continue token (41,923) was masked and every final
"Fixed …" answer (3,000) was trained. Validation loss fell from 0.409 to 0.145, and the fine-tuned agent stopped
working and claimed success right after tool results. The same one-sided pattern comes out of several common
"don't train thinking" or "last response only" options in Axolotl, TRL, LLaMA-Factory and Unsloth
(see `framework_audit/`).

## Install

Nothing to install for encoded data (standard library only, Python 3.9+). Prompt mode and text markers need
`transformers` (and its `tokenizers`), which any HF training setup already has.

## Quick start

**1. You have encoded data** (JSONL with `input_ids` and `labels`, -100 = not trained, or a 0/1 field
`assistant_masks` / `completion_mask` / `loss_mask`):

```bash
python decision_point_audit/audit.py encoded.jsonl \
    --tokenizer google/gemma-4-E2B-it \
    --marker '<tool_response|>' --marker '<|turn>model\n' \
    --call-marker '<|tool_call>'
```

`--marker` is where *your server* ends a request (Qwen3: `'<|im_start|>assistant\n'`). Without `--tokenizer`, give
markers as token ids (`--marker 51` or `--marker 2,3,4`).

**2. You have messages and want to test your encoder** (or a framework's): give a labeler,
`file.py:function` called as `fn(tokenizer, example) -> {"input_ids": [...], "labels": [...]}`:

```python
# my_labeler.py
from my_training_code import encode
def labeler(tok, ex):
    return encode(tok, ex)          # whatever your trainer does to produce input_ids + labels
```

```bash
python decision_point_audit/audit.py messages.jsonl --tokenizer path/to/tokenizer \
    --labeler my_labeler.py:labeler --template-kwargs '{"enable_thinking": true}'
```

With `--labeler hf-assistant-mask`, labels come from the chat template's `{% generation %}` blocks (what TRL's
`assistant_only_loss` uses).

When the rows carry `messages` (and optional `tools`) and a tokenizer is given, **prompt mode** runs as well: for every
assistant turn that follows a user or tool message, the inference prompt is rendered with
`add_generation_prompt=True`, and the decision point is the first token after it. The point's kind comes from the
teacher: *continue* if that turn makes a tool call, *stop* otherwise. If a prompt is not a token prefix of the
training sequence (history thoughts dropped, a different tool format, a generation prompt that opens the thought),
the audit reports a `PREFIX_MISMATCH`: training and inference disagree on format there.

## Output

```
decision-point audit: 30 examples
  continue points     438  trained       0  untrained     438  sources {'marker:<tool_response|>': 408, ...}
           untrained next tokens: {'<|channel>': 438}  (median offset to first label: 25.0)
  stop     points      30  trained      30  untrained       0  sources {'marker:<tool_response|>': 30}
           trained next tokens:   {'Fixed': 30}
  trained continue : stop = 0 : 30 (~0.0:1)
  FLAG ONE_SIDED: no 'continue' decision point is trained, but 30 'stop' points are -- the model is supervised only toward 'stop'
```

(That is our R9 encoder on the first 30 rows of our validation file. With the fix, `mask_open`, which trains the
thought opener, the same command reports 438 : 30 trained and `OK`.)

Flags:

| Flag | Meaning |
|---|---|
| `ONE_SIDED` | One kind (continue or stop) is never trained while the other is. The R9 failure. |
| `UNTRAINED_KIND` | One kind is never trained, and neither is the other (symmetric; less dangerous, still a gap). |
| `PARTIAL` | Some decision points are untrained. |
| `PREFIX_MISMATCH` | An inference prompt is not a prefix of the training sequence (prompt mode). |
| `NO_POINTS` | Nothing found: check markers, messages, tokenizer. |

Exit status: 0 pass, 1 fail, 2 usage error. `--fail-on any` (default) fails on any untrained point,
`--fail-on one-sided` only on `ONE_SIDED` (and `NO_POINTS`), `--fail-on never` only reports. `--json out.json` writes
the summary plus every untrained point (example, position, next token, offset to the first label). Put it in CI or
in front of the training launch.

## Options worth knowing

- `--shifted`: your labels are already shifted (`labels[k]` is the target for `input_ids[k+1]`).
- `--no-prompts`: marker mode only.
- `--chat-template file.jinja`: audit against a different template than the tokenizer's (e.g. the one your server
  uses).
- `--limit N`: first N examples only.

## Where the decision point is depends on serving

The check is only as good as the definition of "where a request ends". Our harness sent requests that end right
after `<tool_response|>`, so the opener `<|channel>` was the decision token. With the Hugging Face Gemma 4 template and
`enable_thinking=True`, the generation prompt can already end in `<|channel>thought\n`; the decision then moves to the
first thought token. Use the markers or chat template that match your server, and run both modes if unsure.

## Limits

- It checks supervision, not behaviour. A passing audit does not mean the model will act well; a
  failing one means one decision gets no gradient. Pair it with a cheap behaviour probe at the first checkpoint
  (sample the next action at a few decision points and count false "done" claims).
- Marker matching is on token sequences. If a text marker tokenizes differently in context, the tool prints a
  `WARNING` for the first 20 examples; prefer special tokens or prompt mode.
- Kinds in marker mode are inferred (a tool-call marker before the next point = continue). Prompt mode uses the
  teacher's messages and is exact.

## Tests

```bash
python -m unittest tests.test_decision_point_audit -v
```

CPU only, no downloads: a toy tokenizer and a tiny in-memory Hugging Face tokenizer with a Gemma-4-like template, which
reproduces the R9 mask (thought block outside `{% generation %}`) and the fix.

## Provenance

Generalises the checks we ran during the investigation (paper, Section 4) into one standalone tool.
