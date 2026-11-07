"""Which message keys does Axolotl's JinjaTemplateAnalyzer think each template reads? transform_message keeps only those."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
exec(open(Path(__file__).resolve().parent / "f_axolotl.py", encoding="utf-8").read().split("def variant")[0])
for name, d in (("gemma4", GEMMA_DIR), ("qwen3", QWEN_DIR)):
    tok = load_tok(d)
    p = CT.ChatTemplatePrompter(tok, chat_template=tok.chat_template)
    print(name, sorted(p.chat_template_msg_variables))
    strat = CT.ChatTemplateStrategy(p, tok, train_on_inputs=False, sequence_len=100000, roles_to_train=["assistant"], train_on_eos="turn")
    turns = strat.get_conversation_thread({"messages": trajectory(False), "tools": TOOLS})
    print("  kept keys per turn:", [sorted(t) for t in turns])
