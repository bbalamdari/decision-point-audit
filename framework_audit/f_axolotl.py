"""Axolotl 0.21.1: real prompt_strategies/chat_template.py (ChatTemplatePrompter + ChatTemplateStrategy), imported with
stubs for pydantic / datasets / addict / axolotl.utils.chat_templates / axolotl.utils.schemas. Defaults follow
StrategyLoader._get_strategy_params: roles_to_train=['assistant'], train_on_eos='turn'.
Variants: default; reasoning parts marked train:false (the documented per-part way to not train on reasoning);
reasoning train:false + content train:true."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

AX = SRC / "axolotl-0.21.1-py3-none-any" / "axolotl"
namespace_pkg("axolotl", AX)
namespace_pkg("axolotl.utils", AX / "utils")
namespace_pkg("axolotl.utils.schemas", AX / "utils" / "schemas")
namespace_pkg("axolotl.prompt_strategies", AX / "prompt_strategies")
stub("pydantic", BaseModel=type("BaseModel", (), {}))
stub("datasets", Dataset=type("Dataset", (), {}))
stub("addict", Dict=dict)
stub("axolotl.utils.distributed", is_main_process=lambda *a, **k: True)
stub("axolotl.utils.chat_templates", get_chat_template_from_config=None)
stub("axolotl.utils.schemas.datasets", DatasetConfig=type("DatasetConfig", (), {}))
import importlib
CT = importlib.import_module("axolotl.prompt_strategies.chat_template")


def variant(msgs, kind):
    msgs = copy.deepcopy(msgs)
    for m in msgs:
        if m["role"] != "assistant":
            continue
        if kind in ("reasoning_false", "reasoning_false_content_true") and m.get("reasoning_content"):
            m["reasoning_content"] = [{"type": "text", "text": m["reasoning_content"], "train": False}]
        if kind == "reasoning_false_content_true":
            m["content"] = [{"type": "text", "text": m["content"], "train": True}]
    return msgs


def run(label, tokdir, markers, kwargs, eot, template=None, mappings=None):
    for final_thought in (False, True):
        for kind in ("default", "reasoning_false", "reasoning_false_content_true"):
            tok = load_tok(tokdir)
            prompter = CT.ChatTemplatePrompter(tok, chat_template=template or tok.chat_template, max_length=100000,
                                               chat_template_kwargs=kwargs, message_property_mappings=mappings)
            strat = CT.ChatTemplateStrategy(prompter, tok, train_on_inputs=False, sequence_len=100000,
                                            roles_to_train=["assistant"], train_on_eos="turn", eot_tokens=eot)
            try:
                out = strat.tokenize_prompt({"messages": variant(trajectory(final_thought), kind), "tools": TOOLS})
            except Exception as e:
                print(f"\n=== Axolotl {label} {kind} final_thought={final_thought}: RAISES {type(e).__name__}: {e}")
                continue
            ids, labels = out["input_ids"], out["labels"]
            pts = decision_points(tok, ids, [tok.encode(x, add_special_tokens=False) for x in markers])
            show(f"Axolotl {label} {kind} final_thought={final_thought}", tok, ids, labels, pts)
            if not final_thought:
                print("  " + labelled_text(tok, ids, labels)[-560:].replace("\n", "\\n"))


run("gemma4 hub", GEMMA_DIR, ["<tool_response|>", "<|turn>model\n"], {"enable_thinking": True}, ["<turn|>"])
run("qwen3 hub", QWEN_DIR, ["<|im_start|>assistant\n"], {"enable_thinking": True}, ["<|im_end|>"])
TPL = AX / "utils" / "chat_templates" / "templates"
run("gemma4 axolotl-bundled gemma4.jinja", GEMMA_DIR, ["<tool_response|>", "<|turn>model\n"], {"enable_thinking": True},
    ["<turn|>"], (TPL / "gemma4.jinja").read_text(encoding="utf-8"))
run("qwen3 axolotl-bundled qwen3.jinja", QWEN_DIR, ["<|im_start|>assistant\n"], {"enable_thinking": True},
    ["<|im_end|>"], (TPL / "qwen3.jinja").read_text(encoding="utf-8"))
MAP = {k: k for k in ("role", "content", "reasoning_content", "tool_calls", "name", "tool_call_id")}
GM = ["<tool_response|>", "<|turn>model\n"]
run("gemma4 hub + explicit message_property_mappings", GEMMA_DIR, GM, {"enable_thinking": True}, ["<turn|>"], None, MAP)
run("gemma4 axolotl-bundled gemma4.jinja + explicit message_property_mappings", GEMMA_DIR, GM, {"enable_thinking": True},
    ["<turn|>"], (TPL / "gemma4.jinja").read_text(encoding="utf-8"), MAP)
