"""TRL 1.17.1: real chat_template_utils.get_training_chat_template (loaded by file path) + the same
apply_chat_template(return_assistant_tokens_mask=True) call SFTTrainer's tokenize_fn makes (sft_trainer.py ~1420),
and the prompt-completion path (completion_only_loss) for one example per step."""
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

TRL = SRC / "trl-1.17.1-py3-none-any" / "trl"
namespace_pkg("trl", TRL)
stub("trl.data_utils", prepare_multimodal_messages=None)
spec = importlib.util.spec_from_file_location("trl.chat_template_utils", TRL / "chat_template_utils.py")
CTU = importlib.util.module_from_spec(spec); sys.modules[spec.name] = CTU; spec.loader.exec_module(CTU)

def assistant_only(label, tokdir, markers, kw):
    tok = load_tok(tokdir)
    print(f"\n### {label}: has_generation_markers(hub template) = {CTU.has_generation_markers(tok.chat_template)}")
    try:
        tpl = CTU.get_training_chat_template(tok)
        print(f"    get_training_chat_template -> {'None (no patch needed)' if tpl is None else 'patched template, len %d' % len(tpl)}")
    except Exception as e:
        print(f"    get_training_chat_template RAISES: {type(e).__name__}: {e}")
        return
    for final_thought in (False, True):
        msgs = trajectory(final_thought)
        out = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=True, return_dict=True,
                                      return_assistant_tokens_mask=True, chat_template=tpl, **kw)
        ids, am = out["input_ids"], out["assistant_masks"]
        labels = [t if m else -100 for t, m in zip(ids, am)]
        pts = decision_points(tok, ids, [tok.encode(m, add_special_tokens=False) for m in markers])
        def rp(h): return tok.encode(tok.apply_chat_template(h, tools=TOOLS, add_generation_prompt=True, tokenize=False, **kw), add_special_tokens=False)
        show(f"TRL assistant_only_loss {label} final_thought={final_thought}", tok, ids, labels, pts,
             [p for p in prefix_check(tok, ids, msgs, rp) if p["msg"] > 2][:1])
        if not final_thought:
            print("  " + labelled_text(tok, ids, labels)[-600:].replace("\n", "\n"))

def prompt_completion(label, tokdir, kw):
    """completion_only_loss with one example per decision point: prompt = history, completion = next assistant msg."""
    tok = load_tok(tokdir)
    msgs = trajectory(False)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or msgs[i - 1]["role"] != "tool":
            continue
        p = tok.apply_chat_template(msgs[:i], tools=TOOLS, add_generation_prompt=True, tokenize=False, **kw)
        pc = tok.apply_chat_template(msgs[:i + 1], tools=TOOLS, tokenize=False, **kw)
        p_ids = tok(p, add_special_tokens=False).input_ids
        pc_ids = tok(pc, add_special_tokens=False).input_ids
        n = CTU.common_prefix_length(p_ids, pc_ids) if hasattr(CTU, "common_prefix_length") else None
        if n is None:
            n = 0
            while n < min(len(p_ids), len(pc_ids)) and p_ids[n] == pc_ids[n]: n += 1
        print(f"  [{label} prompt-completion, msg {i}] prompt is prefix: {n == len(p_ids)}; prompt tail "
              f"{tok.decode(p_ids[-4:])!r}; first completion tokens {tok.convert_ids_to_tokens(pc_ids[n:n+3])} (trained);"
              f" prompt-only tail dropped: {tok.decode(p_ids[n:])!r}")

assistant_only("gemma4 hub", GEMMA_DIR, ["<tool_response|>", "<|turn>model\n"], {"enable_thinking": True})
assistant_only("qwen3 hub", QWEN_DIR, ["<|im_start|>assistant\n"], {"enable_thinking": True})
for et in (True, False):
    prompt_completion(f"gemma4 enable_thinking={et}", GEMMA_DIR, {"enable_thinking": et})
    prompt_completion(f"qwen3 enable_thinking={et}", QWEN_DIR, {"enable_thinking": et})
