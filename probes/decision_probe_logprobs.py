"""G3 (Track 2 gap): R13's in-training decision probe (train_lora.decision_probe) scored with vLLM prompt logprobs on the
served base and on saved R9 / R13 LoRA checkpoints, on the pod (one A100). Would the in-loop stop rule
(p_continue_mid < 0.5 x step 0) have fired on R9 at step 48?

Inputs: g3_inputs.json (scripts/lora/make_g3_inputs.py; token ids taken from the R13 pretokenized cache, so no
tokenisation here) and adapters/<name>/ in the same dataset folder, found under /kaggle/input (symlinks followed).
For each model, per validation example: one request with prompt_logprobs=1 over the ids up to the last mid point
(teacher continued: the actual next token is '<|channel>', so its logprob and rank are returned), and one request per
final point (teacher stopped) with the prefix + '<|channel>' appended (its logprob is the last prompt logprob).
Caveat: served W4A16 logits differ from the training-time bf16 forward; base vs adapters on this server is the comparison
(the base stands in for step 0).
Usage: /opt/kaggle-eval-venv/bin/python g3_probe_pod.py OUTDIR
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_MIN = 80          # hard limit for the whole job (watchdog kills the server and exits at MAX_MIN + 3)
ORDER = ["r9_s48", "r13_s48", "r9_s96", "r13_s96", "r9_s192", "r13_s192", "r9_s375", "r13_s375"]
URL = "http://127.0.0.1:8000/v1"
NOTE = ("served W4A16 (vLLM) logprobs, not the training-time bf16 forward; the served base stands in for step 0; "
        "compare base vs adapters on this server")


def _rglob(name):
    return [Path(d) / name for d, _, fs in os.walk("/kaggle/input", followlinks=True) if name in fs]


def http_post(body, timeout=900):
    req = urllib.request.Request(URL + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _entry(pl, i, tid):
    """(probability, rank) of token tid at prompt position i; rank 1 = argmax."""
    d = pl[i][str(tid)]
    lp = d["logprob"]
    rank = d.get("rank")
    if rank is None:  # older format: argmax iff no other listed token is more likely
        rank = 1 + sum(1 for k, v in pl[i].items() if k != str(tid) and v["logprob"] > lp)
    return math.exp(lp), rank


def score_example(post, model, ex, cont):
    """Mid values [(p, argmax_is_stop)] and final values [p] for one example, as decision_probe defines them."""
    mids, fins = [], []
    if any(m for _, m in ex["points"]):
        pl = post({"model": model, "prompt": ex["ids"], "max_tokens": 1, "temperature": 0,
                   "prompt_logprobs": 1})["choices"][0]["prompt_logprobs"]
        assert len(pl) == len(ex["ids"]), (len(pl), len(ex["ids"]))
        for k, m in ex["points"]:
            if m:
                assert ex["ids"][k + 1] == cont
                p, rank = _entry(pl, k + 1, cont)
                mids.append((p, rank != 1))
    for pre in ex["final_prefixes"]:
        assert pre[-1] == cont
        pl = post({"model": model, "prompt": pre, "max_tokens": 1, "temperature": 0,
                   "prompt_logprobs": 0})["choices"][0]["prompt_logprobs"]
        fins.append(_entry(pl, len(pre) - 1, cont)[0])
    return mids, fins


def score_model(post, model, inp, workers=4):
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(lambda ex: score_example(post, model, ex, inp["cont_id"]), inp["examples"]))
    mid = [p for ms, _ in res for p, _ in ms]
    stop = sum(s for ms, _ in res for _, s in ms)
    fin = [p for _, fs in res for p in fs]
    mean = lambda xs: round(sum(xs) / len(xs), 5) if xs else None
    return {"p_continue_mid": mean(mid), "p_continue_final": mean(fin), "mid_points": len(mid),
            "final_points": len(fin), "mid_argmax_stop_share": round(stop / len(mid), 4) if mid else None,
            "p_continue_mid_min": round(min(mid), 6) if mid else None,
            "per_example_mid_mean": [mean([p for p, _ in ms]) for ms, _ in res],
            "mid_values": [round(p, 6) for p in mid], "final_values": [round(p, 6) for p in fin]}


def apply_rule(rep):
    """The in-loop stop rule: fire when p_continue_mid < 0.5 x base (step-0 stand-in)."""
    b = rep["probe"].get("base", {}).get("p_continue_mid")
    if b is None:
        return {}
    return {m: {"ratio_to_base": round(v["p_continue_mid"] / b, 5), "rule_fires": v["p_continue_mid"] < 0.5 * b}
            for m, v in rep["probe"].items() if m != "base" and v.get("p_continue_mid") is not None}


def main(outdir):
    T0 = time.time()
    out = Path(outdir)
    rep = {"note": NOTE, "probe": {}, "notes": [], "max_min": MAX_MIN}
    OUT = out / "g3_decision_probe.json"
    save = lambda: OUT.write_text(json.dumps(rep, indent=1))
    srv = None

    def watchdog():
        rep["notes"].append("watchdog: hard time limit reached")
        try:
            save()
        finally:
            if srv is not None:
                srv.kill()
            os._exit(9)
    wd = threading.Timer((MAX_MIN + 3) * 60, watchdog)
    wd.daemon = True  # never keeps a failed job alive
    wd.start()
    try:
        model_path = str(next(c.parent for c in sorted(_rglob("config.json")) if "gemma-4-31b-it-qat-w4a16-ct" in str(c)))
        found = sorted(_rglob("g3_inputs.json"))
        assert len(found) == 1, found
        root = found[0].parent
        inp = json.loads(found[0].read_text())
        ads = {p.parent.name: str(p.parent) for p in (root / "adapters").glob("*/adapter_model.safetensors")}
        assert sorted(ads) == sorted(ORDER), sorted(ads)
        rep.update(model=model_path, inputs=str(found[0]), mid_points_expected=inp["mid_points"],
                   final_points_expected=inp["final_points"], sources=inp["sources"])
        cmd = ["/opt/kaggle-eval-venv/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", model_path,
               "--host", "127.0.0.1", "--port", "8000", "--max-model-len", "32768", "--dtype", "bfloat16",
               "--gpu-memory-utilization", "0.85", "--tensor-parallel-size", "1", "--no-scheduler-reserve-full-isl",
               "--max-num-batched-tokens", "1024", "--max-num-seqs", "16", "--enable-lora", "--max-loras",
               str(len(ads)), "--max-lora-rank", "16", "--lora-modules"] + [f"{n}={ads[n]}" for n in ORDER]
        rep["cmd"] = cmd
        save()
        log = open(out / "vllm.log", "w")
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(240):
            try:
                with urllib.request.urlopen(URL + "/models", timeout=10) as r:
                    served = [m["id"] for m in json.loads(r.read())["data"]]
                break
            except Exception:
                assert srv.poll() is None, (out / "vllm.log").read_text()[-3000:]
                time.sleep(10)
        else:
            raise SystemExit("vLLM did not start in 40 min")
        base = next(m for m in served if m not in ads)
        rep["startup_s"] = round(time.time() - T0)
        print("served", served, "startup", rep["startup_s"], flush=True)
        for name, model in [("base", base)] + [(n, n) for n in ORDER]:
            if MAX_MIN - (time.time() - T0) / 60 < 4:
                rep["notes"].append(f"stopped before {name}: time"); break
            t = time.time()
            try:
                r = score_model(http_post, model, inp)
            except Exception as e:
                rep["notes"].append(f"error {name}: {str(e)[:300]}"); save(); continue
            r["seconds"] = round(time.time() - t)
            if r["mid_points"] != inp["mid_points"] or r["final_points"] != inp["final_points"]:
                rep["notes"].append(f"{name}: point count mismatch")
            rep["probe"][name] = r
            rep["rule"] = apply_rule(rep)
            save()
            print(name, {k: r[k] for k in ("p_continue_mid", "p_continue_final", "mid_argmax_stop_share", "seconds")},
                  flush=True)
    except BaseException as e:
        rep["notes"].append(f"fatal: {type(e).__name__}: {str(e)[:500]}")
        raise
    finally:
        rep["total_min"] = round((time.time() - T0) / 60, 1)
        save()
        if srv is not None:
            srv.terminate()
            try:
                srv.wait(60)
            except Exception:
                srv.kill()
        print(json.dumps({"probe": {m: {k: v[k] for k in ("p_continue_mid", "p_continue_final", "mid_argmax_stop_share")}
                                    for m, v in rep["probe"].items()}, "rule": rep.get("rule"), "notes": rep["notes"]},
                         indent=1), flush=True)
    os._exit(0)  # the watchdog timer thread must not keep the job alive


if __name__ == "__main__":
    main(sys.argv[1])
