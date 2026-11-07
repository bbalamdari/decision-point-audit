import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
exec(open(Path(__file__).resolve().parent / "f_axolotl.py", encoding="utf-8").read().split("def variant")[0])
TPL = AX / "utils" / "chat_templates" / "templates"
tok = load_tok(GEMMA_DIR)
t = (TPL / "gemma4.jinja").read_text(encoding="utf-8")
try:
    print(tok.apply_chat_template(trajectory(False), tools=TOOLS, tokenize=False, chat_template=t, enable_thinking=True)[-400:])
except Exception as e:
    print("plain HF render with bundled template:", type(e).__name__, e)
p = CT.ChatTemplatePrompter(tok, chat_template=t, max_length=100000, chat_template_kwargs={"enable_thinking": True})
s = CT.ChatTemplateStrategy(p, tok, train_on_inputs=False, sequence_len=100000, roles_to_train=["assistant"], train_on_eos="turn", eot_tokens=["<turn|>"])
turns = s.get_conversation_thread({"messages": trajectory(False), "tools": TOOLS})
print([sorted(x) for x in turns])
try:
    s.tokenize_prompt({"messages": trajectory(False), "tools": TOOLS})
except Exception:
    tb = traceback.format_exc().splitlines()
    print("\n".join(tb[-14:]))
