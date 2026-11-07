"""LLaMA-Factory 0.9.5 (PyPI wheel): real template.py / formatter.py / tool_utils.py / processor/supervised.py,
imported one module at a time with stubs for mm_plugin, logging, constants, datasets, fsspec."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

LF = SRC / "llamafactory-0.9.5-py3-none-any" / "llamafactory"
namespace_pkg("llamafactory", LF)
namespace_pkg("llamafactory.data", LF / "data")
namespace_pkg("llamafactory.data.processor", LF / "data" / "processor")
namespace_pkg("llamafactory.extras", LF / "extras")
class _Log:
    def __getattr__(self, k): return lambda *a, **kw: None
stub("llamafactory.extras.logging", get_logger=lambda *a: _Log())
stub("llamafactory.extras.constants", IGNORE_INDEX=-100)
stub("fsspec")
stub("datasets", DatasetDict=object, concatenate_datasets=None, interleave_datasets=None)
class _MM:
    def process_messages(self, messages, *a): return messages
    def process_token_ids(self, input_ids, labels, *a): return input_ids, labels
stub("llamafactory.data.mm_plugin", get_mm_plugin=lambda *a, **k: _MM())
import importlib
TPL = importlib.import_module("llamafactory.data.template")
SUP = importlib.import_module("llamafactory.data.processor.supervised")


def lf_messages(thought_words, final_thought=False):
    o, c = thought_words
    def fn(t, name, args): return {"role": "function", "content": f"{o}{t}{c}" + json.dumps({"name": name, "arguments": args})}
    msgs = [{"role": "user", "content": "Fix the failing test in calc.py."},
            fn("I should run the tests first.", "run_command", {"command": "pytest -q"}),
            {"role": "observation", "content": "FAILED test_add: expected 3, got 4"},
            fn("add() uses a+b+1; remove the +1.", "edit_file", {"path": "calc.py", "old": "a+b+1", "new": "a+b"}),
            {"role": "observation", "content": "ok"},
            {"role": "assistant", "content": (f"{o}The edit is done; report.{c}" if final_thought else "") +
             "Fixed the issue: add() now returns a+b."}]
    return msgs


def run(name, tokdir, markers, hf_kwargs):
    for final_thought in (False, True):
        for cfg in [dict(enable_thinking=True, preserve_thinking=False, mask_history=False),
                    dict(enable_thinking=False, preserve_thinking=False, mask_history=False),
                    dict(enable_thinking=True, preserve_thinking=False, mask_history=True),
                    dict(enable_thinking=True, preserve_thinking=True, mask_history=False)]:
            tok = load_tok(tokdir)
            hf_tpl = tok.chat_template
            da = SimpleNamespace(template=name, train_on_prompt=False, tool_format=None, default_system=None,
                                 cutoff_len=100000, **cfg)
            template = TPL.get_template_and_fix_tokenizer(tok, da)
            proc = SUP.SupervisedDatasetProcessor(template=template, tokenizer=tok, processor=None, data_args=da)
            msgs = lf_messages(template.thought_words, final_thought)
            ids, labels = proc._encode_data_example(prompt=msgs[:-1], response=msgs[-1:],
                                                     system="You are a coding agent.",
                                                     tools=json.dumps(TOOLS), images=[], videos=[], audios=[])
            mk = [tok.encode(m, add_special_tokens=False) for m in markers]
            pts = decision_points(tok, ids, mk)
            # inference consistency vs the model's own HF chat template (what vLLM / transformers serve)
            def render_prompt(h):
                return tok.apply_chat_template(h, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                               chat_template=hf_tpl, **hf_kwargs(cfg))
            msgs_oai = trajectory(final_thought)
            pc = []
            for i, m in enumerate(msgs_oai):
                if m["role"] == "assistant" and msgs_oai[i - 1]["role"] == "tool":
                    p = tok.encode(render_prompt(msgs_oai[:i]), add_special_tokens=False)
                    n = 0
                    while n < min(len(p), len(ids)) and p[n] == ids[n]: n += 1
                    pc.append({"msg": i, "prefix_ok": n == len(p), "diverge_at": n,
                               "hf_prompt_has": tok.decode(p[n:n + 8]), "lf_train_has": tok.decode(ids[n:n + 8])})
            show(f"LLaMA-Factory template={name} {cfg} final_thought={final_thought}", tok, ids, labels, pts, pc[:1])
            if cfg["enable_thinking"] and not cfg["mask_history"] and not cfg["preserve_thinking"] and not final_thought:
                print("  " + labelled_text(tok, ids, labels)[-700:].replace("\n", "\n"))
            if cfg["mask_history"] and not final_thought:
                print("  " + labelled_text(tok, ids, labels)[-700:].replace("\n", "\n"))


run("gemma4", GEMMA_DIR, ["<|turn>model\n"], lambda cfg: {"enable_thinking": cfg["enable_thinking"]})
run("qwen3", QWEN_DIR, ["<|im_start|>assistant\n"], lambda cfg: {"enable_thinking": cfg["enable_thinking"]})
