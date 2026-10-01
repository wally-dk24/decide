#!/usr/bin/env python3
"""decide — a personal decision engine CLI.

Typed noul/choice/score questions over text, answered by a decision model
(Liquid d1, zero output tokens) with an LLM fallback, confidence-threshold
escalation (exactly once), and a signed JSONL decision ledger.

DECIDE_LIQUID_API_KEY (primary) and/or DECIDE_LLM_URL/KEY/MODEL
(OpenAI-compatible fallback + escalation). Keys live only in env vars.
"""
import argparse, hashlib, html, json, os, re, sys
from datetime import datetime, timedelta

try:
    import requests
except ImportError:  # --dry-run and DECIDE_STUB work without it
    requests = None

VERSION = "0.1.0"
DECIDE_DIR = os.path.expanduser("~/.decide")
LEDGER_PATH = os.path.join(DECIDE_DIR, "ledger.jsonl")
LABELS_PATH = os.path.join(DECIDE_DIR, "labels.jsonl")
LIQUID_URL = "https://api.liquid.ai/decisions/v1/systemone"
LIQUID_MODEL = "d1:free"
TIMEOUT = 60


class DecideError(Exception):
    pass


class BackendError(DecideError):
    pass


# --- minimal YAML subset (stdlib only): nested maps, lists, scalars ---
def _strip_comment(line):
    out, quote, i = [], None, 0
    while i < len(line):
        c = line[i]
        if quote:
            out.append(c)
            if c == quote:
                quote = None
        elif c in "\"'":
            quote, out = c, out + [c]
        elif c == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        else:
            out.append(c)
        i += 1
    return "".join(out).rstrip()


def _scalar(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_simple_yaml(text):
    items = []
    for n, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw:
            raise ValueError(f"line {n}: tabs not allowed, use spaces")
        line = _strip_comment(raw)
        if line.strip():
            items.append((len(line) - len(line.lstrip(" ")), line.strip(), n))

    def block(i, indent):
        if i >= len(items) or items[i][0] != indent:
            return {}, i
        if items[i][1].startswith("-"):
            lst = []
            while i < len(items) and items[i][0] == indent and items[i][1].startswith("-"):
                payload, lno = items[i][1][1:].strip(), items[i][2]
                deeper = i + 1 < len(items) and items[i + 1][0] > indent
                if payload == "":
                    child, i = (block(i + 1, items[i + 1][0]) if deeper else ({}, i + 1))
                    lst.append(child)
                elif re.match(r"^[A-Za-z0-9_][A-Za-z0-9_\-]*\s*:", payload):
                    raise ValueError(f"line {lno}: inline maps in lists unsupported; "
                                     "use indented lines instead")
                else:
                    lst.append(_scalar(payload))
                    i += 1
            return lst, i
        d = {}
        while i < len(items) and items[i][0] == indent and not items[i][1].startswith("-"):
            content, lno = items[i][1], items[i][2]
            if ":" not in content:
                raise ValueError(f"line {lno}: expected 'key: value'")
            key, _, val = content.partition(":")
            key, val = _scalar(key.strip()), val.strip()
            deeper = i + 1 < len(items) and items[i + 1][0] > indent
            if val == "":
                child, i = (block(i + 1, items[i + 1][0]) if deeper else ({}, i + 1))
                d[key] = child
            else:
                d[key], i = _scalar(val), i + 1
        return d, i

    return block(0, items[0][0])[0] if items else {}


def load_sets(path):
    try:
        with open(path) as f:
            data = parse_simple_yaml(f.read())
    except FileNotFoundError:
        raise DecideError(f"outcome sets file not found: {path}")
    except ValueError as e:
        raise DecideError(f"could not parse {path}: {e}")
    sets = data.get("sets")
    if not isinstance(sets, dict) or not sets:
        raise DecideError(f"{path}: expected a top-level 'sets:' mapping")
    for name, s in sets.items():
        if not isinstance(s, dict):
            raise DecideError(f"{path}: set '{name}' must be a mapping")
        if s.get("type") not in ("noul", "choice", "score"):
            raise DecideError(f"{path}: set '{name}' needs type: noul|choice|score")
        if not s.get("question"):
            raise DecideError(f"{path}: set '{name}' needs a question:")
        if s["type"] == "choice" and not isinstance(s.get("options"), dict):
            raise DecideError(f"{path}: choice set '{name}' needs an 'options:' mapping")
        if s["type"] == "score" and not isinstance(s.get("levels"), list):
            raise DecideError(f"{path}: score set '{name}' needs a 'levels:' list")
        s.setdefault("threshold", 0.8)
    return sets


def find_sets_file(explicit):
    if explicit:
        return explicit
    for cand in ("./sets.yml", os.path.join(DECIDE_DIR, "sets.yml")):
        if os.path.exists(cand):
            return cand
    raise DecideError("no sets file found; pass --outcomes sets.yml")


# --- backends ---
def liquid_request(state, qtype, question, criteria):
    """Exact d1 request shape per https://docs.liquid.ai/lfm/models/decision-models."""
    q = {"type": qtype, "instructions": question}
    if criteria is not None:
        q["criteria"] = criteria
    return {"method": "POST",
            "url": os.environ.get("DECIDE_LIQUID_URL", LIQUID_URL),
            "headers": {"Authorization": "Bearer <redacted>",
                        "Content-Type": "application/json"},
            "body": {"model": os.environ.get("DECIDE_LIQUID_MODEL", LIQUID_MODEL),
                     "state": state, "questions": {"q": q}}}


def parse_liquid_answer(ans, qtype):
    """answers.q -> {winner, distribution, confidence, [score]}."""
    try:
        if qtype == "noul":
            p = float(ans["noul"])
            dist = {"yes": p, "no": 1.0 - p}
            return {"winner": "yes" if p >= 0.5 else "no", "distribution": dist,
                    "confidence": max(dist.values())}
        dist = {str(k): float(v) for k, v in ans["probabilities"].items()}
        conf = float(ans.get("confidence", max(dist.values())))
        if qtype == "choice":
            return {"winner": str(ans["choice"]), "distribution": dist, "confidence": conf}
        score = float(ans["score"])  # qtype == "score"
        return {"winner": str(int(round(score))), "distribution": dist,
                "confidence": conf, "score": score}
    except (KeyError, TypeError, ValueError) as e:
        raise BackendError(f"unexpected d1 answer shape: {e}")
    raise DecideError(f"unknown question type: {qtype}")


def need_requests():
    if requests is None:
        raise BackendError("the 'requests' package is required (pip install requests)")


def liquid_call(state, qtype, question, criteria):
    key = os.environ.get("DECIDE_LIQUID_API_KEY", "")
    if not key:
        raise BackendError("DECIDE_LIQUID_API_KEY is not set")
    need_requests()
    req = liquid_request(state, qtype, question, criteria)
    try:
        r = requests.post(req["url"], headers={"Authorization": "Bearer " + key,
                                               "Content-Type": "application/json"},
                          json=req["body"], timeout=TIMEOUT)
    except Exception as e:
        raise BackendError(f"Liquid API unreachable: {e}")
    if r.status_code != 200:
        raise BackendError(f"Liquid API HTTP {r.status_code}: {r.text[:200]}")
    try:
        data = r.json()
    except Exception as e:
        raise BackendError(f"Liquid API returned non-JSON: {e}")
    ans = (data.get("answers") or {}).get("q")
    if not ans:
        raise BackendError("Liquid API response contained no answers.q")
    return {**parse_liquid_answer(ans, qtype), "backend": "liquid"}


def llm_prompt(state, qtype, question, criteria, reason):
    if qtype == "choice":
        opts = "\n".join(f'- "{k}": {v}' for k, v in criteria.items())
        shape = '{"choice": "<one of the ids>", "probabilities": {"<id>": <0..1>}, "confidence": <0..1>}'
        extra = f"OPTIONS:\n{opts}"
    elif qtype == "score":
        opts = "\n".join(f"- {i}: {lvl}" for i, lvl in enumerate(criteria))
        shape = '{"score": <weighted position, e.g. 2.4>, "probabilities": {"0": <0..1>}, "confidence": <0..1>}'
        extra = f"ORDERED LEVELS:\n{opts}"
    else:
        shape = '{"p_yes": <0..1 probability the answer is yes>, "confidence": <0..1>}'
        extra = ""
    return (f"{reason}\n\nTEXT:\n{state}\n\nQUESTION ({qtype}): {question}\n{extra}\n\n"
            f"Return ONLY a JSON object, no other text:\n{shape}")


def normalize_llm_json(obj, qtype):
    try:
        if qtype == "noul":
            p = float(obj["p_yes"])
            dist = {"yes": p, "no": 1.0 - p}
            return {"winner": "yes" if p >= 0.5 else "no", "distribution": dist,
                    "confidence": float(obj.get("confidence", max(dist.values())))}
        dist = {str(k): float(v) for k, v in obj["probabilities"].items()}
        conf = float(obj.get("confidence", max(dist.values())))
        if qtype == "choice":
            return {"winner": str(obj["choice"]), "distribution": dist, "confidence": conf}
        score = float(obj["score"])
        return {"winner": str(int(round(score))), "distribution": dist,
                "confidence": conf, "score": score}
    except (KeyError, TypeError, ValueError) as e:
        raise BackendError(f"LLM returned malformed decision JSON: {e}")
    raise DecideError(f"unknown question type: {qtype}")


def llm_call(state, qtype, question, criteria, model, purpose):
    """OpenAI-compatible JSON-mode call. purpose: 'fallback' or 'escalation'."""
    url, key = os.environ.get("DECIDE_LLM_URL", ""), os.environ.get("DECIDE_LLM_KEY", "")
    if not url or not key:
        raise BackendError("DECIDE_LLM_URL and DECIDE_LLM_KEY must both be set")
    need_requests()
    reason = ("You are a precise text classifier."
              if purpose == "fallback" else
              "You are a careful senior classifier. A fast decision model was "
              "uncertain, so you make the final call. When genuinely ambiguous, "
              "spread probability across plausible options instead of overstating "
              "confidence.")
    body = {"model": model,
            "messages": [{"role": "user",
                          "content": llm_prompt(state, qtype, question, criteria, reason)}],
            "response_format": {"type": "json_object"}, "temperature": 0}
    try:
        r = requests.post(url.rstrip("/") + "/chat/completions",
                          headers={"Authorization": "Bearer " + key,
                                   "Content-Type": "application/json"},
                          json=body, timeout=TIMEOUT)
    except Exception as e:
        raise BackendError(f"LLM backend unreachable: {e}")
    if r.status_code != 200:
        raise BackendError(f"LLM backend HTTP {r.status_code}: {r.text[:200]}")
    try:
        obj = json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        raise BackendError(f"LLM backend returned unusable JSON: {e}")
    return {**normalize_llm_json(obj, qtype), "backend": "llm-" + purpose}


def stub_call(state, qtype, question, criteria):
    """Deterministic offline stand-in for tests. DECIDE_STUB=1, or =low to force
    low confidence (exercises the escalation path). No network, no keys."""
    low = os.environ.get("DECIDE_STUB") == "low"
    seed = int(hashlib.sha256(f"{qtype}|{question}|{state}".encode()).hexdigest(), 16)
    conf = 0.55 if low else 0.93
    if qtype == "noul":
        p = 0.55 if low else (0.90 if seed % 2 == 0 else 0.10)
        dist = {"yes": p, "no": 1.0 - p}
        return {"winner": "yes" if p >= 0.5 else "no", "distribution": dist,
                "confidence": max(dist.values()), "backend": "stub"}
    if qtype == "choice":
        ids = list(criteria.keys())
        winner = ids[seed % len(ids)]
        rest = (1.0 - conf) / max(len(ids) - 1, 1)
        return {"winner": winner,
                "distribution": {i: (conf if i == winner else rest) for i in ids},
                "confidence": conf, "backend": "stub"}
    n = len(criteria)
    pos = (n - 1) / 2.0 if low else (seed % 1000) / 1000.0 * (n - 1)
    return {"winner": str(int(round(pos))),
            "distribution": {str(i): (conf if i == int(round(pos))
                                      else (1.0 - conf) / (n - 1)) for i in range(n)},
            "confidence": conf, "score": pos, "backend": "stub"}


# --- decision flow ---
def run_decision(state, set_name, qdef, args):
    qtype, question = qdef["type"], args.question or qdef["question"]
    criteria = {"choice": lambda: dict(qdef["options"]),
                "score": lambda: list(qdef["levels"])}.get(qtype, lambda: None)()
    threshold = float(qdef.get("threshold", 0.8))
    stub = os.environ.get("DECIDE_STUB")

    if args.dry_run:
        print("# dry-run: exact request(s) that would be sent (zero network calls)")
        for label, req in [("liquid/primary", liquid_request(state, qtype, question, criteria))]:
            print(f"\n## {label}\n{req['method']} {req['url']}")
            for k, v in req["headers"].items():
                print(f"{k}: {v}")
            print(json.dumps(req["body"], indent=2)[:4000])
        if os.environ.get("DECIDE_LLM_URL"):
            llm_url = os.environ["DECIDE_LLM_URL"].rstrip("/") + "/chat/completions"
            print(f"\n## llm/fallback+escalation\nPOST {llm_url}\n"
                  "Authorization: Bearer <redacted>\nContent-Type: application/json\n"
                  '{"model": "<DECIDE_LLM_MODEL>", "messages": [{"role": "user", '
                  '"content": "<typed JSON-mode prompt>"}], '
                  '"response_format": {"type": "json_object"}, "temperature": 0}')
        print(f"\n# set '{set_name}': escalate exactly once below confidence {threshold}")
        return None

    llm_model = os.environ.get("DECIDE_LLM_MODEL", "gpt-4o-mini")
    if stub:
        result = stub_call(state, qtype, question, criteria)
    elif os.environ.get("DECIDE_LIQUID_API_KEY"):
        try:
            result = liquid_call(state, qtype, question, criteria)
        except BackendError as e:
            print(f"warning: primary backend failed ({e}); trying LLM fallback", file=sys.stderr)
            result = llm_call(state, qtype, question, criteria, llm_model, "fallback")
    elif os.environ.get("DECIDE_LLM_URL"):
        print("warning: no DECIDE_LIQUID_API_KEY; using LLM fallback as primary", file=sys.stderr)
        result = llm_call(state, qtype, question, criteria, llm_model, "fallback")
    else:
        raise DecideError("no backend configured: set DECIDE_LIQUID_API_KEY and/or "
                          "DECIDE_LLM_URL + DECIDE_LLM_KEY (see README)")

    escalated = False
    if result["confidence"] < threshold:
        esc_model = os.environ.get("DECIDE_ESCALATION_MODEL", llm_model)
        if stub:
            result = {**stub_call(state, qtype, question, criteria),
                      "backend": "stub-escalation"}
            escalated = True
        elif os.environ.get("DECIDE_LLM_URL"):
            print(f"warning: confidence {result['confidence']:.2f} < {threshold:.2f}; "
                  f"escalating once to {esc_model}", file=sys.stderr)
            result = llm_call(state, qtype, question, criteria, esc_model, "escalation")
            escalated = True
        else:
            print(f"warning: confidence {result['confidence']:.2f} < {threshold:.2f} "
                  "but no LLM backend configured for escalation", file=sys.stderr)

    record = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
              "input_sha256": hashlib.sha256(state.encode()).hexdigest(),
              "set": set_name, "question": question, "winner": result["winner"],
              "distribution": {k: round(float(v), 6) for k, v in result["distribution"].items()},
              "confidence": round(float(result["confidence"]), 6),
              "backend": result["backend"], "escalated": escalated}
    if "score" in result:
        record["score"] = round(float(result["score"]), 4)
    if args.keep_input:
        record["input"] = state
    os.makedirs(DECIDE_DIR, exist_ok=True)
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def print_result(r):
    print(f"winner:     {r['winner']}")
    if "score" in r:
        print(f"score:      {r['score']}")
    print(f"confidence: {r['confidence']}\nbackend:    {r['backend']}\n"
          f"escalated:  {str(r['escalated']).lower()}")
    print("distribution: " + ", ".join(f"{k}={v:.3f}" for k, v in r["distribution"].items()))


# --- ledger / calib ---
def read_ledger():
    if not os.path.exists(LEDGER_PATH):
        return []
    out = []
    with open(LEDGER_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def parse_since(s):
    m = re.fullmatch(r"(\d+)([smhd])", s or "")
    if not m:
        raise DecideError("--since must look like 30m, 24h, 7d")
    unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[m.group(2)]
    return timedelta(**{unit: int(m.group(1))})


def cmd_ledger(args):
    recs = read_ledger()
    if args.since:
        cutoff = datetime.now().astimezone() - parse_since(args.since)
        recs = [r for r in recs if datetime.fromisoformat(r["ts"]) >= cutoff]
    if not recs:
        print("(ledger is empty)")
        return
    cols = ["time", "set", "winner", "conf", "backend", "esc", "question"]
    rows = [{"time": r["ts"][:16].replace("T", " "), "set": r["set"][:14],
             "winner": str(r["winner"])[:12], "conf": f"{r['confidence']:.2f}",
             "backend": r["backend"][:14], "esc": "yes" if r["escalated"] else "no",
             "question": r["question"][:40]} for r in recs]
    w = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(r[c].ljust(w[c]) for c in cols))
    print(f"\n{len(rows)} decision(s)")


def cmd_calib(_args):
    recs = read_ledger()
    if not recs:
        print("(ledger is empty)")
        return
    labels = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            for line in f:
                if line.strip():
                    try:
                        l = json.loads(line)
                        # optional "set" scopes the label; unscoped labels apply to any set
                        labels[(l.get("set"), l["input_sha256"])] = str(l["winner"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    by_set = {}
    for r in recs:
        by_set.setdefault(r["set"], []).append(r)
    print(f"{'set':<16}{'n':>5}{'escalated':>11}{'esc_rate':>10}{'labeled':>9}{'accuracy':>10}")
    print("-" * 61)
    for name in sorted(by_set):
        rs, n_esc = by_set[name], sum(1 for r in by_set[name] if r["escalated"])
        lab = []
        for r in rs:
            w = labels.get((name, r["input_sha256"]), labels.get((None, r["input_sha256"])))
            if w is not None:
                lab.append((r, w))
        acc = sum(1 for r, w in lab if str(r["winner"]) == w) / len(lab) if lab else None
        print(f"{name:<16}{len(rs):>5}{n_esc:>11}{(n_esc / len(rs)):>9.1%}"
              f"{len(lab):>9}{(f'{acc:.1%}' if acc is not None else 'n/a'):>10}")


# --- main ---
def cmd_decide(args):
    if not args.input_path:
        raise DecideError("pass --input <file> (or use the 'ledger' / 'calib' subcommands)")
    sets = load_sets(find_sets_file(args.outcomes))
    name = args.set_name
    if name is None:
        if len(sets) == 1:
            name = next(iter(sets))
        else:
            raise DecideError(f"--set is required; available: {', '.join(sorted(sets))}")
    if name not in sets:
        raise DecideError(f"unknown set '{name}'; available: {', '.join(sorted(sets))}")
    try:
        with open(args.input_path) as f:
            state = f.read()
    except FileNotFoundError:
        raise DecideError(f"input file not found: {args.input_path}")
    if not state.strip():
        raise DecideError(f"input file is empty: {args.input_path}")
    record = run_decision(state, name, sets[name], args)
    if record:
        print_result(record)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="decide",
        description="Personal decision engine: typed noul/choice/score questions over "
                    "text, with confidence thresholds, one-shot escalation, and a "
                    "signed decision ledger.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    ap.add_argument("--outcomes", default=None,
                    help="path to sets.yml (default: ./sets.yml, then ~/.decide/sets.yml)")
    ap.add_argument("--set", dest="set_name", default=None, help="outcome set to use")
    ap.add_argument("--question", default=None, help="override the set's question text")
    ap.add_argument("--input", dest="input_path", default=None,
                    help="file containing the text to decide about")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact request(s); make zero network calls")
    ap.add_argument("--keep-input", action="store_true",
                    help="store the full input text in the ledger (default: sha256 only)")
    sub = ap.add_subparsers(dest="command")
    lp = sub.add_parser("ledger", help="show recent decisions")
    lp.add_argument("--since", default=None, help="e.g. 30m, 24h, 7d")
    sub.add_parser("calib", help="escalation rate and labeled accuracy per set")
    sv = sub.add_parser("serve", help="run the branded web UI")
    sv.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0)")
    sv.add_argument("--port", type=int, default=8080, help="port (default 8080)")
    args = ap.parse_args(argv)
    try:
        if args.command == "ledger":
            cmd_ledger(args)
        elif args.command == "calib":
            cmd_calib(args)
        elif args.command == "serve":
            cmd_serve(args)
        else:
            cmd_decide(args)
    except DecideError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


# ----------------------------------------------------------------------------
# Serve mode: a small branded web UI over the same decision engine.
#
#   python3 decide.py serve --port 8080
#
#   GET  /         decision form + recent ledger entries
#   POST /decide   run the decision, render winner + distribution bars
#   GET  /healthz  "ok"
#
# Skin comes from the vendored wally-brand copy (brand/page.py).
# ----------------------------------------------------------------------------

SERVE_TAGLINE = "Typed decisions over text"


def _serve_shell(tool_title, content, page_title=None):
    from brand.page import render
    return render("decide", SERVE_TAGLINE, page_title or tool_title, content,
                  footer_extra="decide " + VERSION)


def _serve_form_content(sets, err=None):
    opts = "\n".join(
        '        <option value="%s">%s (%s)</option>'
        % (n, html.escape(n), html.escape(str(sets[n].get("type", "?"))))
        for n in sorted(sets))
    err_html = ""
    if err:
        err_html = ('<div class="wb-card"><span class="wb-badge b-red">error</span>'
                    '<p>%s</p></div>\n') % html.escape(err)
    recent = read_ledger()[-8:][::-1]
    recent_html = ""
    if recent:
        rows = []
        for r in recent:
            rows.append(
                '<div class="wb-dist-row"><div class="wb-dist-label">%s</div>'
                '<div style="font-size:14px"><span class="wb-badge">%s</span> '
                '<strong>%s</strong> <span style="color:var(--wb-muted)">conf %.2f · %s</span>'
                '</div></div>'
                % (html.escape(r["ts"][:16].replace("T", " ")),
                   html.escape(r["set"]), html.escape(str(r["winner"])),
                   r["confidence"], html.escape(r["backend"])))
        recent_html = ('<div class="wb-card"><h2>Recent decisions</h2>\n'
                       '<div class="wb-dist">\n' + "\n".join(rows) +
                       '\n</div></div>\n')
    return err_html + """<div class="wb-card">
  <h2>New decision</h2>
  <p class="wb-sub">Pick an outcome set, paste the text, and decide.</p>
  <form action="/decide" method="post">
    <div class="wb-field">
      <label for="set">Outcome set</label>
      <select class="wb-select" id="set" name="set">
%s
      </select>
    </div>
    <div class="wb-field">
      <label for="text">Text to decide about</label>
      <textarea class="wb-textarea" id="text" name="text" required
        placeholder="Paste the message, email, or note here&hellip;"></textarea>
    </div>
    <div class="wb-field">
      <label for="question">Question override (optional)</label>
      <input class="wb-input" id="question" name="question"
        placeholder="Defaults to the set&rsquo;s question">
    </div>
    <div class="wb-btn-row">
      <button class="wb-btn wb-btn-primary" type="submit">Decide</button>
    </div>
  </form>
</div>
""" % opts + recent_html


def _serve_result_content(record, qdef, state):
    dist = record["distribution"]
    bars = []
    for k, v in sorted(dist.items(), key=lambda kv: kv[1], reverse=True):
        label = str(k)
        if qdef.get("type") == "score":
            try:
                label = qdef["levels"][int(k)]
            except (IndexError, ValueError, TypeError):
                pass
        cls = ""
        if qdef.get("type") == "noul":
            cls = " fill-green" if k == "yes" else " fill-red"
        bars.append(
            '    <div class="wb-dist-row"><div class="wb-dist-label">%s</div>'
            '<div class="wb-dist-track"><div class="wb-dist-fill%s" style="width:%.1f%%">'
            '</div></div><div class="wb-dist-val">%.3f</div></div>'
            % (html.escape(label), cls, 100 * v, v))
    esc = ' <span class="wb-badge b-amber">escalated</span>' if record["escalated"] else ""
    score_row = ("" if "score" not in record
                 else "<dt>score</dt><dd>%.3f</dd>\n      " % record["score"])
    excerpt = state if len(state) <= 280 else state[:280] + "…"
    raw = html.escape(json.dumps(record, indent=2))
    return """<div class="wb-card">
  <h2><span class="wb-badge b-green">%s</span>%s</h2>
  <p class="wb-sub">%s</p>
  <div class="wb-dist">
%s
  </div>
  <dl class="wb-kv">
      <dt>confidence</dt><dd>%.3f</dd>
      <dt>backend</dt><dd>%s</dd>
      %s<dt>input sha</dt><dd>%s</dd>
  </dl>
</div>
<div class="wb-card">
  <h2>Input</h2>
  <div class="wb-term">%s</div>
</div>
<div class="wb-card">
  <h2>Record</h2>
  <p class="wb-sub">Appended to the decision ledger.</p>
  <div class="wb-term">%s</div>
  <div class="wb-btn-row">
    <a class="wb-btn" href="/">Decide another</a>
  </div>
</div>
""" % (html.escape(str(record["winner"])), esc,
       html.escape(record["question"]),
       "\n".join(bars),
       record["confidence"], html.escape(record["backend"]),
       score_row, record["input_sha256"][:16],
       html.escape(excerpt), raw)


def make_serve_handler(sets):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs
    from brand.page import brand_asset

    class DecideHandler(BaseHTTPRequestHandler):
        server_version = "decide/" + VERSION

        def log_message(self, fmt, *args):  # quieter logs
            sys.stderr.write("decide: %s\n" % (fmt % args))

        def _send(self, body, ctype="text/html; charset=utf-8", code=200):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _page(self, content, title, code=200):
            self._send(_serve_shell(title, content, page_title=title), code=code)

        def do_GET(self):
            path = urlparse(self.path).path
            asset = brand_asset(path)
            if asset:
                ctype, data = asset
                return self._send(data, ctype)
            if path == "/healthz":
                return self._send("ok", "text/plain; charset=utf-8")
            if path in ("/", "/index.html"):
                return self._page(_serve_form_content(sets), "Decide")
            self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/decide":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 2 * 1024 * 1024:
                self.send_error(400, "bad form size")
                return
            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            set_name = (form.get("set", [""])[0] or "").strip()
            text = form.get("text", [""])[0] or ""
            question = (form.get("question", [""])[0] or "").strip() or None
            if set_name not in sets:
                return self._page(
                    _serve_form_content(sets, "unknown set '%s'" % set_name),
                    "Decide", code=400)
            if not text.strip():
                return self._page(
                    _serve_form_content(sets, "paste some text to decide about"),
                    "Decide", code=400)
            ns = argparse.Namespace(question=question, dry_run=False, keep_input=False)
            try:
                record = run_decision(text, set_name, sets[set_name], ns)
            except DecideError as e:
                return self._page(_serve_form_content(sets, str(e)),
                                  "Decide", code=502)
            self._page(_serve_result_content(record, sets[set_name], text),
                       "Decision — " + set_name)

    return DecideHandler


def cmd_serve(args):
    from http.server import ThreadingHTTPServer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sets = load_sets(find_sets_file(args.outcomes))
    if not sets:
        raise DecideError("no outcome sets found")
    handler = make_serve_handler(sets)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    httpd.daemon_threads = True
    host = "localhost" if args.host == "0.0.0.0" else args.host
    print("decide: serving the decision UI at http://%s:%d/  (%d set(s))"
          % (host, args.port, len(sets)))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
