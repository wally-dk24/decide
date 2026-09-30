# decide — a personal decision engine

`decide` replaces "classification wearing a chat costume": all those times an
automation asks a full LLM a question whose answer is really one of N known
options. Instead it asks a **decision model** (Liquid AI `d1`, zero output
tokens) a typed question — **noul** (yes/no), **choice** (pick from a set), or
**score** (ordered rubric) — and gets back calibrated probabilities. Below a
confidence threshold it escalates exactly once to a frontier LLM. Every
decision is appended to a signed JSONL ledger.

## Install

Requires Python 3.12. One dependency:

```bash
pip install requests
git clone https://github.com/wally-dk24/decide
cd decide
```

No daemons, no servers. All state lives under `~/.decide/` plus your
`sets.yml`.

## Configure

Outcome sets live in `sets.yml` (this repo ships an example). Each set has a
`type`, a `question`, a `threshold`, and — for choice/score — `options` or
`levels`. Only a small YAML subset is supported (indented maps, `- ` lists,
quoted/plain scalars, no tabs); the parser errors loudly on anything else.

Backends are chosen by environment variables — **keys are never stored in
files or printed** (dry-run shows `Bearer <redacted>`):

| Variable | Purpose |
|---|---|
| `DECIDE_LIQUID_API_KEY` | Liquid API key (`liquid_…`, from console.liquid.ai) — primary backend |
| `DECIDE_LIQUID_URL` | override, default `https://api.liquid.ai/decisions/v1/systemone` |
| `DECIDE_LIQUID_MODEL` | override, default `d1:free` |
| `DECIDE_LLM_URL` | OpenAI-compatible base URL, e.g. `https://api.openai.com/v1` (fallback + escalation) |
| `DECIDE_LLM_KEY` | key for the above |
| `DECIDE_LLM_MODEL` | model for fallback (default `gpt-4o-mini`) |
| `DECIDE_ESCALATION_MODEL` | stronger model for escalation (default: same as fallback) |

Request shape follows the official docs
(`https://docs.liquid.ai/lfm/models/decision-models`):
`POST {url}` with `{"model": "d1:free", "state": "<your text>",
"questions": {"q": {"type": "noul|choice|score", "instructions": "…",
"criteria": {…} or […]}}}`. Choice/Score answers carry a full
`probabilities` distribution plus `confidence`; `usage.output_tokens` is
always 0.

## Worked examples

### 1. Email triage — spam or not (noul)

```bash
# See exactly what would be sent, with zero network calls and no API key:
./decide.py --outcomes sets.yml --set spam_check --input examples/email_spam.txt --dry-run

# Decide for real (needs DECIDE_LIQUID_API_KEY and/or DECIDE_LLM_*):
DECIDE_STUB=1 ./decide.py --outcomes sets.yml --set spam_check --input examples/email_spam.txt
# winner:     yes
# confidence: 0.90
# backend:    stub
# escalated:  false
# distribution: yes=0.900, no=0.100

DECIDE_STUB=1 ./decide.py --outcomes sets.yml --set spam_check --input examples/email_ham.txt
# winner:     no
# ...
```

(`DECIDE_STUB=1` is the offline test hook — deterministic canned backends,
no network. Use `DECIDE_STUB=low` to force low confidence and exercise the
escalation path.)

### 2. Urgency rubric 0–3 (score)

```bash
DECIDE_STUB=1 ./decide.py --outcomes sets.yml --set urgency --input examples/urgency_note.txt
# winner:     3
# score:      2.61
# confidence: 0.93
# backend:    stub
# escalated:  false
# distribution: 0=0.023, 1=0.023, 2=0.023, 3=0.930
```

Override the question on the fly without editing the file:

```bash
./decide.py --outcomes sets.yml --set intent --input examples/email_ham.txt \
  --question "Which team should handle this message?"
```

## The ledger

Every decision appends one JSON line to `~/.decide/ledger.jsonl`:

```json
{"ts": "2026-09-30T09:30:00-04:00", "input_sha256": "…", "set": "spam_check",
 "question": "Is this message spam?", "winner": "yes",
 "distribution": {"yes": 0.9, "no": 0.1}, "confidence": 0.9,
 "backend": "liquid", "escalated": false}
```

Inputs are hashed, not stored — unless you pass `--keep-input`.

```bash
./decide.py ledger --since 7d   # readable table of recent decisions
./decide.py calib               # per-set escalation rate + accuracy
```

`calib` reports accuracy where you supply labels: append
`{"input_sha256": "<hex>", "winner": "<true winner>"}` lines to
`~/.decide/labels.jsonl` (the sha256 is in the ledger, or compute it with
`sha256sum`). Add `"set": "<name>"` to scope a label to one outcome set when
the same input is decided under several sets.

## Picking thresholds

Start at **0.8** for everything. Then adjust for asymmetric costs:

- **Missing a positive is expensive** (spam slipping into your inbox,
  ignoring an outage): *lower* the threshold for acting — you'd rather
  escalate (or act) than silently decide wrong. Try 0.6–0.7 and watch the
  escalation rate in `decide calib`.
- **False positives are expensive** (auto-deleting mail, paging someone):
  *raise* it toward 0.9, so only near-certain calls act alone.
- If `calib` shows an escalation rate above ~30%, the set's question or
  options are probably ambiguous — rewrite them before lowering the
  threshold. Thresholds can't fix a muddy question.

## Exit codes

`0` on a decision (even an escalated one). `1` on errors: bad config,
missing input, no backend configured, or a backend failure with no working
fallback. Warnings go to stderr; the `winner:`/`confidence:`/`backend:` /
`escalated:` lines on stdout are trivially parseable by scripts.

## Docker

```bash
docker pull wallydk24/decide
docker run --rm -e DECIDE_LIQUID_API_KEY=$KEY wallydk24/decide \
  --set spam_check --question "Is this spam?" --input /data/msg.txt
```

Keys are never baked into the image — pass them at runtime. The bundled
`sets.yml` ships as defaults; mount your own with `-v ./my-sets.yml:/app/sets.yml`.
