"""Unsloth: unsloth_zoo 2026.10.11 dataset_utils.train_on_responses_only (real code, loaded by file path with a
`datasets` stub) + chat templates: the model's own HF template and Unsloth's 'gemma-4' / 'gemma-4-thinking' templates
(string literals extracted from unsloth/chat_templates.py with ast, not executed)."""
import ast, sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

ZOO = SRC / "unsloth_zoo-2026.10.11-py3-none-any" / "unsloth_zoo"
namespace_pkg("unsloth_zoo", ZOO)
stub("datasets", Dataset=type("Dataset", (), {}), IterableDataset=type("IterableDataset", (), {}))
spec = importlib.util.spec_from_file_location("unsloth_zoo.dataset_utils", ZOO / "dataset_utils.py")
DU = importlib.util.module_from_spec(spec); sys.modules[spec.name] = DU; spec.loader.exec_module(DU)

CT = SRC / "unsloth-2026.10.11-py3-none-any" / "unsloth" / "chat_templates.py"
lit = {}
for node in ast.parse(CT.read_text(encoding="utf-8")).body:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        for t in node.targets:
            if isinstance(t, ast.Name): lit[t.id] = node.value.value

def run(label, tokdir, template, parts, markers, kw, msgs_fn=trajectory):
    for final_thought in (False, True):
        for opts in (dict(), dict(last_response_only=True)):
            tok = load_tok(tokdir, template)
            msgs = msgs_fn(final_thought)
            text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, **kw)
            ids = tok(text, add_special_tokens=False).input_ids
            if parts == "auto":
                fn = DU.train_on_responses_only(None, tokenizer=tok, return_function=True, **opts)
            else:
                fn = DU.train_on_responses_only(None, instruction_part=parts[0], response_part=parts[1],
                                                tokenizer=tok, return_function=True, **opts)
            labels = fn({"input_ids": [ids]})["labels"][0]
            pts = decision_points(tok, ids, [tok.encode(m, add_special_tokens=False) for m in markers])
            def rp(h): return tok.encode(tok.apply_chat_template(h, tools=TOOLS, add_generation_prompt=True,
                                                                  tokenize=False, **kw), add_special_tokens=False)
            pc = prefix_check(tok, ids, msgs, rp)
            show(f"Unsloth {label} parts={parts} {opts} final_thought={final_thought}", tok, ids, labels, pts,
                 [p for p in pc if p["msg"] > 2][:1])
            if not opts and not final_thought:
                print("  " + labelled_text(tok, ids, labels)[-650:].replace("\n", "\n"))

G = ["<tool_response|>", "<|turn>model\n"]
hub_g = load_tok(GEMMA_DIR).chat_template
run("gemma4 HF-hub template", GEMMA_DIR, hub_g, ("<|turn>user\n", "<|turn>model\n"), G, {"enable_thinking": True})
run("gemma4 HF-hub template", GEMMA_DIR, hub_g, "auto", G, {"enable_thinking": True})
run("gemma4 unsloth 'gemma-4' template", GEMMA_DIR, lit["gemma4_template"], ("<|turn>user\n", "<|turn>model\n"), G, {"enable_thinking": True})
run("gemma4 unsloth 'gemma-4-thinking' template", GEMMA_DIR, lit["gemma4_thinking_template"], ("<|turn>user\n", "<|turn>model\n"), G, {"enable_thinking": True})
run("qwen3 HF-hub template", QWEN_DIR, None, ("<|im_start|>user\n", "<|im_start|>assistant\n"), ["<|im_start|>assistant\n"], {"enable_thinking": True})
