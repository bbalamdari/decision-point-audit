"""Track 2, test 3: a Kaggle 4xL4 notebook that scores the lora_v3 checkpoints on the served W4A16 model with vLLM.

Inputs (one private dataset): probe_inputs.json from make_probe_inputs.py and adapters/<name>/ (adapter_config.json +
adapter_model.safetensors) for each checkpoint. One vLLM server serves the base model and every adapter at once.
  loss:      per model, the mean loss of each trained-token category (closing message, edits, commands, reads,
             install, submit) from prompt log-probs of the exact training token ids;
  behaviour: per model and decision point (early / pre_edit / post_edit), the class of the generated next action
             (2 samples, the agents' sampling: temperature 0.2, top_p 0.95): edit, command, read, submit,
             claim_fixed (text claiming the fix with no tool call), other_text, truncated.
--max-num-batched-tokens is small because prompt log-probs need full-vocabulary logits for every scheduled token.
The JSON (/kaggle/working/loss_probe.json) is rewritten after every model and phase; a time box stops early.

Usage: python make_loss_probe.py --canonical <getting-started notebook> --canonical-metadata <its kernel-metadata.json>
           --dataset <owner/slug> --out <new folder>
"""
import argparse
import json
from pathlib import Path

PROBE = r'''
import json, re, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T0 = time.time()
MAX_MIN = 105
MODEL_PATH = str(next(c.parent for c in sorted(Path('/kaggle/input').rglob('config.json')) if 'gemma-4-31b-it-qat-w4a16-ct' in str(c)))
print('model', MODEL_PATH)
found = sorted(Path('/kaggle/input').rglob('probe_inputs.json'))
assert len(found) == 1, found
ROOT = found[0].parent
INP = json.loads(found[0].read_text())
ADAPTERS = sorted((p.parent.name, str(p.parent)) for p in (ROOT / 'adapters').glob('*/adapter_model.safetensors'))
print('adapters', ADAPTERS, 'examples', len(INP['loss']), 'probes', len(INP['probes']))
OUT = Path('/kaggle/working/loss_probe.json')
rep = {'adapters': [a for a, _ in ADAPTERS], 'loss': {}, 'behaviour': {}, 'notes': []}
def save(): OUT.write_text(json.dumps(rep, indent=1))

cmd = ['/usr/bin/python3', '-m', 'vllm.entrypoints.openai.api_server', '--model', MODEL_PATH, '--host', '127.0.0.1',
       '--port', '8000', '--max-model-len', '32768', '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.85',
       '--tensor-parallel-size', '4', '--no-scheduler-reserve-full-isl', '--max-num-batched-tokens', '1024',
       '--max-num-seqs', '16', '--enable-lora', '--max-loras', str(len(ADAPTERS)), '--max-lora-rank', '16',
       '--lora-modules'] + [f'{n}={p}' for n, p in ADAPTERS]
rep['cmd'] = cmd
log = open('/kaggle/working/vllm.log', 'w')
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
URL = 'http://127.0.0.1:8000/v1'
for _ in range(240):
    try:
        with urllib.request.urlopen(URL + '/models', timeout=10) as r:
            served = [m['id'] for m in json.loads(r.read())['data']]
        break
    except Exception:
        assert srv.poll() is None, Path('/kaggle/working/vllm.log').read_text()[-3000:]
        time.sleep(10)
else:
    raise SystemExit('vLLM did not start')
BASE = next(m for m in served if m not in rep['adapters'])
MODELS = [BASE] + rep['adapters']
NAMES = {BASE: 'base'}
rep['startup_s'] = round(time.time() - T0)
print('served', served, 'startup', rep['startup_s'], flush=True)

def post(body, timeout=1800):
    req = urllib.request.Request(URL + '/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def left_min(): return MAX_MIN - (time.time() - T0) / 60

# phase 1: loss by category
def loss_one(model, ex):
    out = post({'model': model, 'prompt': ex['ids'], 'max_tokens': 1, 'temperature': 0, 'prompt_logprobs': 0})
    pl = out['choices'][0]['prompt_logprobs']
    by = {}
    for i, c in ex['trained']:
        d = pl[i]
        lp = d[str(ex['ids'][i])]['logprob']
        s, n = by.get(c, (0.0, 0))
        by[c] = (s - lp, n + 1)
    return by

for model in MODELS:
    if left_min() < 60:
        rep['notes'].append(f'loss skipped for {model}: time'); break
    t = time.time(); tot = {}
    try:
        with ThreadPoolExecutor(4) as pool:
            for by in pool.map(lambda ex: loss_one(model, ex), INP['loss']):
                for c, (s, n) in by.items():
                    S, N = tot.get(c, (0.0, 0)); tot[c] = (S + s, N + n)
    except Exception as e:
        rep['notes'].append(f'loss error {model}: {str(e)[:300]}'); save(); continue
    allS = sum(s for s, _ in tot.values()); allN = sum(n for _, n in tot.values())
    rep['loss'][NAMES.get(model, model)] = {'all': round(allS / allN, 5), 'tokens': allN,
        'by_category': {c: {'loss': round(s / n, 5), 'tokens': n, 'loss_sum': round(s, 2)} for c, (s, n) in tot.items()},
        'seconds': round(time.time() - t)}
    save(); print('loss', model, rep['loss'][NAMES.get(model, model)]['all'], flush=True)

# phase 2: behaviour at decision points
CALL = re.compile(r'<\|tool_call>call:(\w+)\{(.*)', re.S)
def classify(text, finish):
    m = CALL.search(text)
    if m:
        name, body = m.group(1), m.group(2)
        if name == 'run_command':
            return 'edit' if 'safe_edit/apply.py' in body else 'command'
        return {'read_file': 'read', 'submit_patch': 'submit', 'run_skill_script': 'install',
                'edit_file': 'edit', 'write_file': 'edit'}.get(name, 'other_call')
    if finish == 'length':
        return 'truncated'
    after = text.split('<channel|>')[-1]
    return 'claim_fixed' if re.search(r'Fixed the issue|submitted the patch', after) else 'other_text'

def gen_one(model, p):
    out = post({'model': model, 'prompt': p['prompt_ids'], 'max_tokens': 3000, 'temperature': 0.2, 'top_p': 0.95,
                'n': 2, 'stop': ['<tool_call|>', '<turn|>'], 'include_stop_str_in_output': True,
                'skip_special_tokens': False, 'seed': 7})
    return [(classify(c['text'], c.get('finish_reason')), c['text'][-400:]) for c in out['choices']]

points = ['early', 'pre_edit', 'post_edit']
for point in points:
    ps = [p for p in INP['probes'] if p['point'] == point]
    for model in MODELS:
        if left_min() < 8:
            rep['notes'].append(f'behaviour stopped at {point}/{model}: time'); break
        t = time.time()
        try:
            with ThreadPoolExecutor(6) as pool:
                res = list(pool.map(lambda p: gen_one(model, p), ps))
        except Exception as e:
            rep['notes'].append(f'gen error {point}/{model}: {str(e)[:300]}'); save(); continue
        counts = {}
        for r in res:
            for c, _ in r:
                counts[c] = counts.get(c, 0) + 1
        rep['behaviour'].setdefault(point, {})[NAMES.get(model, model)] = {
            'counts': counts, 'n': sum(counts.values()), 'seconds': round(time.time() - t),
            'teacher_next': {k: sum(1 for p in ps if p['teacher_next'] == k) for k in {p['teacher_next'] for p in ps}},
            'examples': [{'id': p['id'], 'teacher': p['teacher_next'], 'got': [c for c, _ in r], 'tail': r[0][1]}
                         for p, r in zip(ps, res)][:30]}
        save(); print(point, model, counts, flush=True)
srv.terminate()
rep['total_min'] = round((time.time() - T0) / 60, 1); save()
print(json.dumps({'loss': {k: v['all'] for k, v in rep['loss'].items()},
                  'behaviour': {pt: {m: v['counts'] for m, v in d.items()} for pt, d in rep['behaviour'].items()},
                  'notes': rep['notes']}, indent=1))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--canonical-metadata", dest="canonical_metadata", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; choose a new folder")
    canon = json.loads(a.canonical.read_text(encoding="utf-8"))
    install = canon["cells"][3]
    assert "WHEELHOUSE_DIR" in "".join(install["source"]), "cell 3 is not the install cell"
    code = lambda src: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
    # the mount layout of /kaggle/input can differ between kernels: point the install cell at wherever the wheels are
    src = "".join(install["source"])
    old = "WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')"
    assert old in src
    install = dict(install, outputs=[], source=src.replace(old, "WHEELHOUSE_DIR = next(p.parent for p in sorted(Path('/kaggle/input').rglob('adk_submission-*.whl')))\nprint('wheelhouse', WHEELHOUSE_DIR)"))
    listing = "!find /kaggle/input -maxdepth 3 -not -path '*/snapshots/*' -not -path '*/sample_submission/*' | head -60"
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": "# Loss split and behaviour probe of LoRA checkpoints (Track 2, test 3)"},
                    code("!nvidia-smi"), code(listing), install, code(PROBE)],
          "metadata": canon["metadata"], "nbformat": canon["nbformat"], "nbformat_minor": canon["nbformat_minor"]}
    meta = json.loads(a.canonical_metadata.read_text(encoding="utf-8"))
    slug = "loss-probe-lora-checkpoints-gemma-4"
    meta.update(id=f"{meta['id'].split('/')[0]}/{slug}", title="Loss probe LoRA checkpoints Gemma 4", code_file=f"{slug}.ipynb",
                dataset_sources=[s for s in meta["dataset_sources"] if "wheelhouse" in s] + [a.dataset])
    meta.pop("id_no", None)
    a.out.mkdir(parents=True)
    (a.out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8", newline="\n")
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
