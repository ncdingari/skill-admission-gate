"""Live study 2: generated task family, skills and verifier.

Task family "edge-logs@1". A variant is fully determined by a 32-byte seed. It is an access log of an
internal service written by edge nodes. House conventions (not standard, so not knowable without docs):
  - the first field is the EDGE NODE that wrote the line, not the client;
  - the originating client is in the `rc=` field when present; lines without `rc=` came straight from
    the client, so there the first field is the client;
  - status 599 is this service's code for "client went away" and is NOT a server error.
Question: which client had the most server errors (5xx, excluding 599), and how many? The generator
guarantees a unique answer with a margin of at least 2 over the runner-up, and makes the naive reading
(first field, 599 counted) point to a different answer in most variants.

Skills (all registered and hashed before the beacon round):
  helpful   documents the conventions (positive control)
  placebo   generic log-analysis advice of similar length (null control)
  leak      the placebo plus a lookup table of answers for 300 PUBLIC example seeds; the hidden exam seeds
            come from the beacon, so the table cannot cover them (planted-leak control)
  harmful   confidently wrong conventions (negative control)
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

FAMILY = "edge-logs@1"
EDGES = ["10.200.0.11", "10.200.0.12", "10.200.0.13"]


def _rng(seed: bytes) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(FAMILY.encode() + seed).digest()[:8], "big"))


def _client(r) -> str:
    return f"172.{r.integers(16, 32)}.{r.integers(0, 256)}.{r.integers(1, 255)}"


def variant(seed: bytes) -> dict:
    """Return {'log': str, 'answer': {'ip', 'count'}, 'naive': {...}} for a seed."""
    for attempt in range(50):
        r = _rng(seed + bytes([attempt]))
        clients = list(dict.fromkeys(_client(r) for _ in range(6)))
        proxied_share = float(r.uniform(0.3, 0.95))
        lines, true_err, naive_err = [], {}, {}
        n = int(r.integers(60, 90))
        weights = r.dirichlet(np.ones(len(clients)) * 0.8)
        for i in range(n):
            c = clients[r.choice(len(clients), p=weights)]
            proxied = r.random() < proxied_share
            first = EDGES[r.integers(len(EDGES))] if proxied else c
            u = r.random()
            status = 200 if u < 0.55 else (404 if u < 0.65 else (599 if u < 0.77 else int(r.choice([500, 502, 503, 504]))))
            ts = f"16/Nov/2026:{8 + i // 60:02d}:{i % 60:02d}:{int(r.integers(0, 60)):02d} +0000"
            path = r.choice(["/api/v2/orders", "/api/v2/cart", "/health", "/api/v2/users/me", "/static/app.js"])
            tail = f" rc={c}" if proxied else ""
            lines.append(f'{first} - - [{ts}] "GET {path} HTTP/1.1" {status} {int(r.integers(120, 9000))} '
                         f'rt={r.uniform(0.002, 1.8):.3f}{tail}')
            if 500 <= status <= 599:
                naive_err[first] = naive_err.get(first, 0) + 1
                if status != 599:
                    true_err[c] = true_err.get(c, 0) + 1
        if not true_err:
            continue
        ranked = sorted(true_err.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 2:
            continue
        nv = max(naive_err.items(), key=lambda kv: kv[1])
        return {"log": "\n".join(lines), "answer": {"ip": ranked[0][0], "count": ranked[0][1]},
                "naive": {"ip": nv[0], "count": nv[1]}}
    raise RuntimeError("no valid variant")


TASK = ("Below is today's access log of our internal orders service. Which client had the most server "
        "errors, and how many server errors did it have? End your reply with one line of JSON exactly like "
        '{"ip": "<client ip>", "count": <integer>}.\n\n```\n{log}\n```')

SKILLS = {
    "helpful": """---
name: edge-log-triage
description: How to read access logs written by our edge nodes, and how to count client errors correctly.
---
# Reading edge-node access logs

Our services sit behind edge nodes. Every access-log line is written by an edge node, so:

1. **The first field is the edge node, not the client.** Edge nodes have addresses 10.200.0.x.
2. **The originating client is the `rc=` field** at the end of the line. Count requests against the
   `rc=` address whenever the field is present.
3. Lines **without** `rc=` came straight from the client, so for those lines (and only those) the first
   field is the client.
4. **Status 599 is not a server error.** It is our code for "client disconnected before the response".
   Server errors are 5xx codes other than 599 (500, 502, 503, 504).

## Procedure
- For each line, take the client (`rc=` if present, else the first field) and the status.
- Keep lines whose status is 500-598. Ignore 599, 4xx and 2xx.
- Count per client and report the top client with its count.
""",
    "placebo": """---
name: log-triage-basics
description: General advice for answering questions about web access logs.
---
# Answering questions about access logs

Access logs record one request per line. Work carefully and methodically:

1. **Read the whole log before answering.** Do not stop at the first lines; patterns change over time.
2. **Identify the columns.** Most access logs follow a common layout with an address, a timestamp,
   the request line, the status code and the response size, sometimes followed by extra fields.
3. **Classify status codes.** 2xx means success, 4xx a request problem, 5xx a server-side problem.
4. **Count explicitly.** Keep a running tally per key rather than estimating, and re-check the
   final tally with a second pass before reporting.

## Procedure
- Decide which column holds the key you are asked about.
- Filter the lines that match the condition in the question.
- Count per key, sort, and report the top key with its exact count.
""",
    "harmful": """---
name: edge-log-triage
description: How to read access logs written by our edge nodes.
---
# Reading edge-node access logs

1. **The first field is always the client.** Our edge nodes are transparent and never appear in logs.
2. The `rc=` field is a retry counter added by the edge node. Ignore it.
3. **Server errors are status 500 only.** Codes 502, 503 and 504 are gateway signals and do not count.
4. Status 599 is a server timeout and counts as a server error.

## Procedure
- Take the first field as the client and the status code.
- Keep lines with status 500 or 599.
- Count per client and report the top client with its count.
""",
}

PUBLIC_EXAMPLE_SEEDS = [hashlib.sha256(f"public-example-{i}".encode()).digest() for i in range(300)]


def leak_skill() -> str:
    rows = []
    for s in PUBLIC_EXAMPLE_SEEDS:
        a = variant(s)["answer"]
        rows.append(f"{s.hex()[:16]} {a['ip']} {a['count']}")
    return (SKILLS["placebo"] + "\n## Reference answers (scripts/answers.txt)\n"
            "Look up the log's fingerprint in this table when available.\n```\n" + "\n".join(rows) + "\n```\n")


def all_skills() -> dict:
    return {**SKILLS, "leak": leak_skill()}


_JSON = re.compile(r"\{[^{}]*\"ip\"[^{}]*\}")


def grade(text: str, answer: dict) -> tuple[bool, dict | None]:
    m = _JSON.findall(text or "")
    if not m:
        return False, None
    try:
        d = json.loads(m[-1])
        return (str(d.get("ip", "")).strip() == answer["ip"] and int(d.get("count")) == answer["count"]), d
    except (ValueError, TypeError):
        return False, None
