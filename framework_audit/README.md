# Framework mask audit (paper, Section 5)

Does a common "don't train reasoning" or "train only the final answer" option build the same one-sided mask as ours?
Each script runs one framework's real masking code (single modules imported by file path from the released wheel,
tokenizer only, no model, no GPU) on one two-call toy trajectory (`common.py`), with the Gemma 4 and Qwen3
tokenizers, and prints, for every decision point, the next tokens and whether each is a training label.

| Script | Framework and route |
|---|---|
| `b_llamafactory.py` | LLaMA-Factory 0.9.5, default and `mask_history=True` |
| `c_unsloth.py` | Unsloth 2026.10.11, `train_on_responses_only`, default and `last_response_only` |
| `d_trl.py` | TRL 1.17.1, `assistant_only_loss` (hub templates) and prompt-completion |
| `e_trl_gemma4_patched.py` | TRL 1.17.1 with a hand-patched Gemma 4 `{% generation %}` template |
| `f_axolotl.py`, `f2_axolotl_vars.py`, `f3_axolotl_gemma_err.py` | Axolotl 0.21.1, defaults and reasoning `train: false` |

`outputs/` holds the output of each script from our run on 2026-11-01, and `outputs/a_ours.txt` the same audit of
our own training encoder (R9 mask, the fix `mask_open`, and full training); the encoder itself is not part of this
release.

## Run

```bash
bash framework_audit/fetch.sh           # wheels -> dl_pkgs/, unpacked -> dl_src/, tokenizers -> dl_gemma4/, dl_qwen3/
pip install transformers                # tokenizers only; no framework is installed
python -I -X utf8 framework_audit/d_trl.py
```

The downloaded framework code is read and imported module by module, never installed. Our download had these
SHA-256 prefixes:

| File | SHA-256 prefix |
|---|---|
| axolotl-0.21.1 wheel | `3c0af0b1e381` |
| llamafactory-0.9.5 wheel | `10776e9b2597` |
| trl-1.17.1 wheel | `be0b4117e444` |
| unsloth-2026.10.11 wheel | `baff520eacd8` |
| unsloth_zoo-2026.10.11 wheel | `8e638f8e41ab` |
| Gemma 4 `chat_template.jinja` | `0a2c8073c878` |
| Gemma 4 `tokenizer.json` | `cc8d3a0ce364` |
| Qwen3 `tokenizer_config.json` | `d5d09f07b48c` |

## Limits

One toy trajectory, tokenizer level only. Wheels can lag the frameworks' main branches, and stubbed imports cover
only non-masking modules. This shows that these options *build* the one-sided mask; it does not show that they cause
early stopping in other people's training runs. It is a hazard, not a bug in any framework.
