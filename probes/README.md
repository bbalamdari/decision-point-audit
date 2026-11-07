# Probes (paper, Sections 4 and 6)

Both probes score a base model and LoRA checkpoints on one vLLM server (adapters loaded with `--enable-lora`), at
decision points of held-out validation trajectories. They are the scripts we ran; paths follow our setup (a Kaggle
4×L4 notebook and a single-A100 pod) and need adapting to yours.

**`behaviour_probe_notebook.py`: behaviour probe.** Generates a Kaggle notebook. At three decision-point types
(early, before the first edit, after an edit) in 30 validation examples, it samples the next action twice
(temperature 0.2, top_p 0.95) and classifies it: edit, command, read, submit, `claim_fixed` (text claiming the fix
with no tool call), other text, truncated. The same run reports validation loss by token category (closing message,
edits, commands, reads, install, submit) from prompt log-probs. Results: `results/behaviour_probe_*.json`.

**`decision_probe_logprobs.py`: teacher-forced decision probe.** The probability of the thought opener `<|channel>`
at the 391 mid-trajectory decision points of 30 validation examples (where the teacher continued) and at the 30
final points (where it stopped), from vLLM prompt log-probs. The pre-registered stop rule fires when the mean mid-point
probability falls below half the base value. Results: `results/decision_probe_logprobs.json`. In the paper this
probe ran inside the training loop for R13 and was scored afterwards on R9's saved checkpoints with this script.
Served W4A16 log-probs differ from the training-time bf16 forward pass; the served base stands in for step 0.

## Not included

The inputs: the validation trajectories (derived from third-party SWE-agent datasets under their own licences), the
token ids built from them, and the adapters. The result files hold counts and summary statistics only.
