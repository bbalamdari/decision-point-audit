"""TRL on Gemma 4 needs a hand-patched template (get_training_chat_template raises). Two plausible user patches of the
hub template: A = {% generation %} around thought + tool calls + content + <turn|>; B = the same but WITHOUT the thought
block (the natural way to 'not train on reasoning'). Masks come from apply_chat_template(return_assistant_tokens_mask),
which is what SFTTrainer(assistant_only_loss=True) uses."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

tok = load_tok(GEMMA_DIR)
hub = tok.chat_template


def sub(t, old, new):
    assert t.count(old) == 1, (t.count(old), old[:60])
    return t.replace(old, new)


TH = "{{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}"
FOR = "{%- for tool_call in message.get('tool_calls') -%}"
END = "{%- set ns.prev_message_type = 'tool_call' -%}"
CC = "{{- captured_content -}}"
base = sub(hub, FOR, "{% generation %}" + FOR)
base = sub(base, END, "{% endgeneration %}" + END)
base = sub(base, CC, "{%- if role == 'model' -%}{% generation %}" + CC + "{% endgeneration %}{%- else -%}" + CC + "{%- endif -%}")
m = list(re.finditer(r"\{%- elif not \(ns_tr_out\.flag and not has_content and not next_nt\.found\) -%\}\s*(\{\{- '<turn\|>\\n' -\}\})", base))
assert len(m) == 1, len(m)
a, b = m[0].span(1)
base = base[:a] + ("{%- if role == 'model' -%}{% generation %}{{- '<turn|>\\n' -}}{% endgeneration %}"
                   "{%- else -%}{{- '<turn|>\\n' -}}{%- endif -%}") + base[b:]
A = sub(base, TH, "{% generation %}" + TH + "{% endgeneration %}")
B = base
for name, tpl in (("A: thought inside {% generation %}", A), ("B: thought outside {% generation %}", B)):
    for final_thought in (False, True):
        msgs = trajectory(final_thought)
        out = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=True, return_dict=True,
                                      return_assistant_tokens_mask=True, chat_template=tpl, enable_thinking=True)
        ids, am = out["input_ids"], out["assistant_masks"]
        hub_ids = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=True, return_dict=True, enable_thinking=True)["input_ids"]
        assert ids == hub_ids, "patched template must render the same tokens as the hub template"
        labels = [t if k else -100 for t, k in zip(ids, am)]
        pts = decision_points(tok, ids, [tok.encode(x, add_special_tokens=False) for x in ("<tool_response|>", "<|turn>model\n")])
        show(f"TRL + user-patched Gemma 4 template {name} final_thought={final_thought}", tok, ids, labels, pts)
        if not final_thought:
            print("  " + labelled_text(tok, ids, labels)[-520:].replace("\n", "\\n"))
