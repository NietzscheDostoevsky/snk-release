#!/usr/bin/env python3
"""Runs an LLM judge over a blind item file (item_id, word, output). Resumable.

The census in results/census_verdicts.csv was produced with this runner, through Ollama,
at temperature 0, several items per request, with code/judge_rubric.md as the system prompt:

  python code/run_judge.py --provider ollama --model gemma4:31b-cloud --items census_blind.csv --out verdicts_gemma4.csv
  python code/run_judge.py --provider ollama --model gpt-oss:20b      --items census_blind.csv --out verdicts_gptoss20b.csv

census_blind.csv is the item_id, word and output columns of census_verdicts.csv in
shuffled order, so that a batch carries no signal about artifact or level (see README).

Every verdict is appended to the output CSV as it arrives, and a re-run skips any
item_id already present, so an interrupted run continues where it stopped. Each row
records the model string and a timestamp.

  --limit N      stop after N new items
  --batch N      items per request, default 10
  --dry-run      print the first request and exit

For a cloud-hosted model (e.g. gemma4:31b-cloud) set OLLAMA_HOST=https://ollama.com
and OLLAMA_API_KEY.
"""
import argparse, csv, json, os, re, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def load_prompt():
    md = open(os.path.join(HERE, "judge_rubric.md"), encoding="utf-8").read()
    parts = md.split("\n---\n")
    if len(parts) < 2:
        sys.exit("judge_rubric.md: expected a '---' ruler before the rubric")
    return parts[-1].strip()


def post(url, payload, headers, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call_ollama(model, system, user, repeat_penalty=None):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    headers = {}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"
    options = {"temperature": 0}
    if repeat_penalty:
        options["repeat_penalty"] = repeat_penalty
    d = post(f"{host}/api/chat",
             {"model": model, "stream": False, "options": options,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]}, headers)
    return d["message"]["content"]


CALLERS = {"ollama": call_ollama}

# Cloud endpoints drop calls sporadically (read timeouts, 502s). Retrying the
# batch in place is far cheaper than exiting and re-reading the verdict file.
RETRY_WAITS = [5, 15, 45, 90, 180]


def parse(text, expected):
    """Pull {'item_id':..., 'verdict':...} objects out of the reply. Tolerant of
    fences and stray prose; silently drops anything not asked for, so unanswered
    items are simply retried on the next run."""
    got = {}
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        iid, v = str(o.get("item_id", "")), o.get("verdict")
        if iid in expected and v in (0, 1, "0", "1"):
            got[iid] = int(v)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=sorted(CALLERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between requests")
    ap.add_argument("--repeat-penalty", type=float, default=None,
                    help="Raise above the 1.1 default (e.g. 1.3) to break "
                         "deterministic repeat-token loops that abort with HTTP 500 "
                         "'prediction aborted, token repeat limit reached'.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    system = load_prompt()
    items = list(csv.DictReader(open(a.items, newline="", encoding="utf-8")))
    out_path = a.out

    done = set()
    if os.path.exists(out_path):
        done = {r["item_id"] for r in csv.DictReader(open(out_path, newline="", encoding="utf-8"))}
    todo = [i for i in items if i["item_id"] not in done]
    print(f"{len(items)} items, {len(done)} already judged, {len(todo)} to go")
    if not todo:
        return

    if not os.path.exists(out_path):
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["item_id", "verdict", "model", "provider", "judged_at"])

    n_new, batches = 0, [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    consecutive_failures = 0
    CIRCUIT_BREAKER = 5  # this many irrecoverable batches IN A ROW means the service
    # itself is down, not that we hit a few bad items -- stop instead of silently
    # skipping every remaining batch and finishing having scored almost nothing.
    for bi, batch in enumerate(batches, 1):
        if a.limit and n_new >= a.limit:
            print(f"--limit {a.limit} reached"); break
        user = "\n\n".join(
            f'item_id: {r["item_id"]}\nword: {r["word"]}\ndefinition: {r["output"]}' for r in batch)
        if a.dry_run:
            print("--- SYSTEM ---\n" + system + "\n\n--- USER (batch 1) ---\n" + user)
            return
        reply = None
        quota_exhausted = False
        for attempt, wait in enumerate(RETRY_WAITS + [None], 1):
            try:
                reply = call_ollama(a.model, system, user, repeat_penalty=a.repeat_penalty)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                body = ""
                if isinstance(e, urllib.error.HTTPError):
                    try: body = e.read().decode()[:300]
                    except Exception: pass
                print(f"batch {bi}/{len(batches)}: attempt {attempt}: "
                      f"{type(e).__name__}: {e} {body}", flush=True)
                # A spent quota is not transient: no amount of backoff fixes it,
                # and retrying only delays a clean stop. Bail out at once.
                if "usage limit" in body or "quota" in body.lower():
                    print("QUOTA EXHAUSTED - stopping. Re-run after it resets.", flush=True)
                    wait = None
                    quota_exhausted = True
                if wait is None:
                    break
                time.sleep(wait)
        if reply is None:
            print(f"batch {bi}/{len(batches)}: giving up after {len(RETRY_WAITS) + 1} attempts.")
            if quota_exhausted:
                print("nothing lost. re-run the same command to continue.")
                break
            print("not a quota issue (likely an irrecoverable model error on this batch's "
                  "content, e.g. a repeat-loop abort) -- skipping this batch, continuing with "
                  "the rest. Its items stay unscored and will be retried on the next run.")
            consecutive_failures += 1
            if consecutive_failures >= CIRCUIT_BREAKER:
                print(f"{consecutive_failures} irrecoverable batches in a row -- this looks "
                      f"like the service itself is down, not isolated bad content. Stopping "
                      f"rather than skipping every remaining batch. Re-run once the service "
                      f"is confirmed healthy; already-scored items are untouched.", flush=True)
                break
            continue
        consecutive_failures = 0
        got = parse(reply, {r["item_id"] for r in batch})
        if not got and len(batch) == 1:
            # Singleton batch: no ambiguity about which item this is. A model can
            # echo the item_id back wrong (observed: repeat_penalty suppressing a
            # legitimately repeated character inside the id itself, e.g. "bbbb" ->
            # "bbb") while still getting the verdict right. Salvage that verdict
            # instead of discarding it under a strict id match that only matters
            # when a batch has more than one item to disambiguate.
            for m in re.finditer(r"\{[^{}]*\}", reply):
                try:
                    o = json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
                v = o.get("verdict")
                if v in (0, 1, "0", "1"):
                    got = {batch[0]["item_id"]: int(v)}
                    break
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for iid, v in got.items():
                w.writerow([iid, v, a.model, a.provider, stamp])
        n_new += len(got)
        miss = len(batch) - len(got)
        print(f"batch {bi}/{len(batches)}  +{len(got)}"
              f"{f'  ({miss} unparsed, will retry)' if miss else ''}  total {len(done)+n_new}")
        if a.sleep:
            time.sleep(a.sleep)
    print(f"\n{n_new} new verdicts written to {os.path.relpath(out_path, HERE)}")


if __name__ == "__main__":
    main()
