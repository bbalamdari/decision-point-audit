"""Tests for tools/decision_point_audit/audit.py. CPU only, no downloads: a toy duck-typed tokenizer and, if
`transformers`/`tokenizers` are installed, a tiny in-memory Hugging Face tokenizer with a Gemma-4-like template.

Run:  .venv/Scripts/python -m unittest tests.test_decision_point_audit -v
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "decision_point_audit"))
import audit as A  # noqa: E402

# ------------------------------------------------------------------------------------------- toy trajectory


def trajectory(final_thought=False):
    """system, user, [thought + call] -> tool, [thought + call] -> tool, final answer (no thought: R9 data shape)."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": "", "reasoning_content": "run the tests first",
         "tool_calls": [{"type": "function", "function": {"name": "run_command", "arguments": {"command": "pytest"}}}]},
        {"role": "tool", "content": "FAILED test_add"},
        {"role": "assistant", "content": "", "reasoning_content": "remove the plus one",
         "tool_calls": [{"type": "function", "function": {"name": "edit_file", "arguments": {"path": "calc.py"}}}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "Fixed the issue."},
    ]
    if final_thought:
        msgs[-1]["reasoning_content"] = "done report"
    return msgs


class ToyTokenizer:
    """Word-level tokenizer with a Gemma-4-like chat format. After a tool response the generation prompt adds nothing
    (the request ends at '<tool_response|>', as in our harness); after a user turn it opens '<|turn> model'.
    gen_thought=True mimics HF Gemma 4 with enable_thinking: the generation prompt also opens '<|channel>'.
    drop_history_thoughts=True leaves past thoughts out of inference prompts (a train/inference mismatch)."""

    def __init__(self, gen_thought=False, drop_history_thoughts=False):
        self.vocab, self.inv = {}, {}
        self.gen_thought, self.drop = gen_thought, drop_history_thoughts

    def _id(self, t):
        if t not in self.vocab:
            self.vocab[t] = len(self.vocab)
            self.inv[self.vocab[t]] = t
        return self.vocab[t]

    def render_tokens(self, messages, add_generation_prompt=False, for_prompt=False):
        out = ["<bos>"]
        for i, m in enumerate(messages):
            r = m["role"]
            if r in ("system", "user"):
                out += ["<|turn>", r, *m["content"].split(), "<turn|>"]
            elif r == "assistant":
                if i == 0 or messages[i - 1]["role"] not in ("assistant", "tool"):
                    out += ["<|turn>", "model"]
                if m.get("reasoning_content") and not (for_prompt and self.drop):
                    out += ["<|channel>", *m["reasoning_content"].split(), "<channel|>"]
                for c in m.get("tool_calls") or []:
                    out += ["<|tool_call>", c["function"]["name"], "<tool_call|>"]
                if m.get("content"):
                    out += [*m["content"].split(), "<turn|>"]
            elif r == "tool":
                out += ["<|tool_response>", *m["content"].split(), "<tool_response|>"]
        if add_generation_prompt:
            if messages[-1]["role"] == "user":
                out += ["<|turn>", "model"]
            if self.gen_thought:
                out += ["<|channel>"]
        return out

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=True, **kw):
        toks = self.render_tokens(messages, add_generation_prompt, for_prompt=True)
        return [self._id(t) for t in toks] if tokenize else " ".join(toks)

    def encode(self, text, add_special_tokens=False):
        return [self._id(t) for t in text.split()]

    def convert_ids_to_tokens(self, ids):
        return [self.inv[i] for i in ids]

    def decode(self, ids):
        return " ".join(self.inv[i] for i in ids)

    def encode_example(self, messages, mode):
        """Our training script's masking on the toy format: model-turn tokens are trained except tool responses; 'mask'
        also drops whole thought blocks incl. delimiters (R9), 'mask_open' trains only the opener (R13)."""
        toks = self.render_tokens(messages)
        ids = [self._id(t) for t in toks]
        labels, in_model, in_resp, in_thought = [], False, False, False
        for t, i in zip(toks, ids):
            trained = False
            if t == "<|turn>":
                in_model = False
            if t == "<|tool_response>":
                in_resp = True
            if t == "<|channel>":
                in_thought = True
                trained = in_model and mode in ("mask_open", "train")
            elif in_thought:
                trained = in_model and mode == "train"
            elif in_model and not in_resp:
                trained = True
            if t == "<channel|>":
                in_thought = False
            if t == "<tool_response|>":
                in_resp = False
            if t == "model":
                in_model = True
            labels.append(i if trained else A.IGNORE)
        return {"input_ids": ids, "labels": labels}


def toy_row(mode, final_thought=False, tok=None):
    tok = tok or ToyTokenizer()
    msgs = trajectory(final_thought)
    return tok, dict(tok.encode_example(msgs, mode), messages=msgs)


# ------------------------------------------------------------------------------------------- core logic


class CoreTests(unittest.TestCase):
    def test_marker_points_and_classify(self):
        ids = [5, 1, 9, 2, 1, 9, 3, 1, 9, 4]
        self.assertEqual(A.find_marker_points(ids, [1, 9]), [3, 6, 9])
        self.assertEqual(A.find_marker_points([1, 9], [1, 9]), [])           # marker at the end: no next token
        self.assertEqual(A.classify(ids, [3, 6, 9], None), ["continue", "continue", "stop"])
        self.assertEqual(A.classify(ids, [3, 6, 9], [[2], [3]]), ["continue", "continue", "stop"])
        self.assertEqual(A.classify(ids, [3, 6, 9], [[3]]), ["stop", "continue", "stop"])

    def test_flags_from_labels_mask_and_shift(self):
        row = {"input_ids": [1, 2, 3], "labels": [-100, 2, -100]}
        self.assertEqual(A.trained_flags(row), [False, True, False])
        self.assertEqual(A.trained_flags({"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]}), [False, True, True])
        self.assertEqual(A.trained_flags({"input_ids": [1, 2, 3], "labels": [2, -100, -100]}, shifted=True),
                         [False, True, False])
        with self.assertRaises(ValueError):
            A.trained_flags({"input_ids": [1, 2], "labels": [1]})
        with self.assertRaises(ValueError):
            A.trained_flags({"input_ids": [1, 2]})

    def test_r9_mask_is_one_sided_in_marker_mode(self):
        tok, row = toy_row("mask")
        markers = {"<tool_response|>": tok.encode("<tool_response|>")}
        rep = A.audit_row(row, tok=tok, markers=markers, use_prompts=False)
        s = A.summarize(rep)
        self.assertEqual((s["continue"]["points"], s["continue"]["trained"]), (1, 0))
        self.assertEqual((s["stop"]["points"], s["stop"]["trained"]), (1, 1))
        self.assertEqual(s["continue"]["untrained_next_tokens"], {"<|channel>": 1})
        self.assertEqual(s["stop"]["trained_next_tokens"], {"Fixed": 1})
        self.assertTrue(any(f.startswith("ONE_SIDED") for f in s["flags"]), s["flags"])
        self.assertTrue(A.failed(s, "one-sided") and A.failed(s, "any") and not A.failed(s, "never"))
        # offset to the first label: '<|channel> remove the plus one <channel|> <|tool_call>' -> 6
        self.assertEqual(s["continue"]["median_offset_to_first_label_when_untrained"], 6)

    def test_mask_open_and_train_pass(self):
        for mode in ("mask_open", "train"):
            tok, row = toy_row(mode)
            markers = {"tr": tok.encode("<tool_response|>"), "mt": tok.encode("<|turn> model")}
            s = A.summarize(A.audit_row(row, tok=tok, markers=markers, use_prompts=False))
            self.assertEqual(s["flags"], [], mode)
            self.assertEqual(s["continue"]["points"], 2, mode)   # first turn + one mid point
            self.assertFalse(A.failed(s, "any"))

    def test_final_thought_masked_is_symmetric_not_one_sided(self):
        tok, row = toy_row("mask", final_thought=True)
        s = A.summarize(A.audit_row(row, tok=tok, markers={"tr": tok.encode("<tool_response|>")}, use_prompts=False))
        self.assertTrue(any(f.startswith("UNTRAINED_KIND") for f in s["flags"]), s["flags"])
        self.assertFalse(any(f.startswith("ONE_SIDED") for f in s["flags"]))
        self.assertTrue(A.failed(s, "any"))
        self.assertFalse(A.failed(s, "one-sided"))

    def test_partial(self):
        rep = A.Report(points=[A.Point(0, 1, "continue", "m", True, 1), A.Point(0, 2, "continue", "m", False, 1),
                               A.Point(0, 3, "stop", "m", True, 1)], n_examples=1)
        s = A.summarize(rep)
        self.assertEqual(len(s["flags"]), 1)
        self.assertTrue(s["flags"][0].startswith("PARTIAL"))

    def test_no_points_fails(self):
        s = A.summarize(A.Report(n_examples=1))
        self.assertTrue(A.failed(s, "one-sided"))


class PromptModeTests(unittest.TestCase):
    def test_prompt_mode_finds_request_ends_and_kinds(self):
        tok, row = toy_row("mask")
        rep = A.audit_row(row, tok=tok)
        kinds = [(p.kind, p.next_token, p.trained) for p in rep.points]
        self.assertEqual(kinds, [("continue", "<|channel>", False), ("continue", "<|channel>", False),
                                 ("stop", "Fixed", True)])
        self.assertEqual(rep.prefix_mismatches, [])
        s = A.summarize(rep)
        self.assertTrue(s["flags"][0].startswith("ONE_SIDED"))

    def test_prompt_and_marker_points_merge(self):
        tok, row = toy_row("mask_open")
        rep = A.audit_row(row, tok=tok, markers={"tr": tok.encode("<tool_response|>")})
        self.assertEqual(len(rep.points), 3)
        self.assertEqual(sorted(p.source for p in rep.points), ["prompt", "prompt+marker:tr", "prompt+marker:tr"])
        self.assertEqual(A.summarize(rep)["flags"], [])

    def test_generation_prompt_that_opens_the_thought_moves_the_decision(self):
        """HF Gemma 4 with enable_thinking: the prompt already ends in '<|channel>', so the decision is the first
        thought token, which 'mask' and 'mask_open' both leave untrained, while the stop answer is not even reachable
        (the training sequence has no thought there): reported as a prefix mismatch."""
        tok = ToyTokenizer(gen_thought=True)
        tok, row = toy_row("mask_open", tok=tok)
        rep = A.audit_row(row, tok=tok)
        s = A.summarize(rep)
        self.assertEqual(s["continue"]["points"], 2)
        self.assertEqual(s["continue"]["trained"], 0)
        self.assertEqual(len(rep.prefix_mismatches), 1)
        self.assertEqual(rep.prefix_mismatches[0]["kind"], "stop")
        self.assertTrue(any(f.startswith("PREFIX_MISMATCH") for f in s["flags"]))

    def test_prompt_that_drops_history_thoughts_is_a_mismatch(self):
        tok = ToyTokenizer(drop_history_thoughts=True)
        tok, row = toy_row("train", tok=tok)
        rep = A.audit_row(row, tok=tok)
        # the first prompt (after the user turn) has no history thought, so it still matches; later ones do not
        self.assertEqual(len(rep.points), 1)
        self.assertEqual(len(rep.prefix_mismatches), 2)
        self.assertIn("prompt_has", rep.prefix_mismatches[0])


class CliTests(unittest.TestCase):
    def run_cli(self, rows, *args):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            js = Path(d) / "out.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = A.main([str(p), *args, "--json", str(js)])
            return code, buf.getvalue(), json.loads(js.read_text(encoding="utf-8"))

    def test_cli_encoded_jsonl_with_token_id_markers(self):
        tok = ToyTokenizer()
        rows = []
        for mode in ("mask", "mask"):
            _, r = toy_row(mode, tok=tok)
            rows.append({"input_ids": r["input_ids"], "labels": r["labels"]})
        marker = str(tok.vocab["<tool_response|>"])
        call = str(tok.vocab["<|tool_call>"])
        code, out, js = self.run_cli(rows, "--marker", marker, "--call-marker", call)
        self.assertEqual(code, 1, out)
        self.assertEqual(js["continue"]["points"], 2)
        self.assertEqual(js["stop"]["trained"], 2)
        self.assertIn("ONE_SIDED", out)
        self.assertEqual(len(js["untrained_points"]), 2)
        code, out, _ = self.run_cli(rows, "--marker", marker, "--fail-on", "never")
        self.assertEqual(code, 0)

    def test_cli_text_marker_without_tokenizer_is_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, _, _ = self.run_cli([{"input_ids": [1], "labels": [1]}], "--marker", "<tool_response|>") \
                if False else (A.main(["nonexistent.jsonl", "--marker", "<tool_response|>"]), None, None)
        self.assertEqual(code, 2)


# ------------------------------------------------------------------------------------------- real HF tokenizer

try:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    HAVE_HF = True
except Exception:  # pragma: no cover
    HAVE_HF = False

SPECIAL = ["<pad>", "<bos>", "<eos>", "[UNK]", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|tool_call>",
           "<tool_call|>", "<|tool_response>", "<tool_response|>"]
THOUGHT = "{% if m.reasoning_content %}<|channel>{{ m.reasoning_content }}<channel|>{% endif %}"
TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m.role in ['system', 'user'] %}<|turn>{{ m.role }} {{ m.content }}<turn|>"
    "{% elif m.role == 'assistant' %}"
    "{% if loop.index0 == 0 or messages[loop.index0 - 1].role not in ['assistant', 'tool'] %}<|turn>model {% endif %}"
    "THOUGHT_OUT{% generation %}THOUGHT_IN"
    "{% for c in (m.tool_calls or []) %}<|tool_call>{{ c.function.name }}<tool_call|>{% endfor %}"
    "{% if m.content %}{{ m.content }}<turn|>{% endif %}{% endgeneration %}"
    "{% elif m.role == 'tool' %}<|tool_response>{{ m.content }}<tool_response|>{% endif %}"
    "{% endfor %}{% if add_generation_prompt and messages[-1].role == 'user' %}<|turn>model {% endif %}")


def hf_tokenizer(thought_trained: bool):
    words = sorted({w.strip(".") for m in trajectory() for w in (m.get("content") or "").split()} |
                   {w for m in trajectory() for w in (m.get("reasoning_content") or "").split()} |
                   {"system", "user", "model", "run_command", "edit_file", "Fixed", "the", "issue.", "test."})
    vocab = {t: i for i, t in enumerate(SPECIAL + [w for w in words if w not in SPECIAL])}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
                                  unk_token="[UNK]", additional_special_tokens=SPECIAL[4:])
    tok.chat_template = TEMPLATE.replace("THOUGHT_OUT", "" if thought_trained else THOUGHT) \
                                .replace("THOUGHT_IN", THOUGHT if thought_trained else "")
    return tok


@unittest.skipUnless(HAVE_HF, "transformers/tokenizers not installed")
class HuggingFaceTests(unittest.TestCase):
    def audit(self, thought_trained):
        tok = hf_tokenizer(thought_trained)
        ex = {"messages": trajectory()}
        row = dict(ex, **A.hf_assistant_mask(tok, ex))
        return A.summarize(A.audit_row(row, tok=tok, markers={"tr": tok.encode("<tool_response|>",
                                                                               add_special_tokens=False)}))

    def test_thought_outside_generation_reproduces_r9(self):
        """The 'natural' TRL patch for Gemma 4 (thought block outside {% generation %}) = the R9 mask."""
        s = self.audit(thought_trained=False)
        self.assertEqual((s["continue"]["points"], s["continue"]["trained"]), (2, 0))
        self.assertEqual((s["stop"]["points"], s["stop"]["trained"]), (1, 1))
        self.assertTrue(s["flags"][0].startswith("ONE_SIDED"), s["flags"])
        self.assertEqual(s["continue"]["by_source"], {"prompt": 1, "prompt+marker:tr": 1})

    def test_thought_inside_generation_passes(self):
        s = self.audit(thought_trained=True)
        self.assertEqual(s["flags"], [])
        self.assertEqual(s["continue"]["trained"], 2)

    def test_cli_with_saved_tokenizer_and_labeler(self):
        for trained, want in ((False, 1), (True, 0)):
            with tempfile.TemporaryDirectory() as d:
                hf_tokenizer(trained).save_pretrained(d)
                data = Path(d) / "msgs.jsonl"
                data.write_text(json.dumps({"messages": trajectory()}) + "\n", encoding="utf-8")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    code = A.main([str(data), "--tokenizer", d, "--labeler", "hf-assistant-mask",
                                   "--marker", "<tool_response|>", "--call-marker", "<|tool_call>"])
                self.assertEqual(code, want, buf.getvalue())
                self.assertNotIn("WARNING", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
