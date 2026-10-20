"""Minimal OpenAI Batch API client, following https://developers.openai.com/api/docs/guides/batch.

    submit FILE [FILE ...]  validate, upload (purpose=batch), create a 24h batch, record ids in a ledger
    status [ID ...]         status + request counts (default: every batch in the ledgers next to FILEs)
    fetch  ID OUT           download the output file (and OUT.errors.jsonl) once the batch completes
    cancel ID

Credentials come from OPENAI_API_KEY (or the variable named by --api-key-env); the base URL is
--base-url, else OPENAI_BASE_URL, else https://api.openai.com/v1. The endpoint must expose the Files and
Batches routes: a gateway that blocks /v1/files for its keys (some LiteLLM proxy deployments return 403
"This route is not publicly accessible" there) fails at the upload step.

Each FILE gets a ledger FILE.batch.json holding its input_file_id and batch_id, so a resubmit is
visible instead of silently paying twice.

Pairs with pi_error_step_label.py:
    python pi_error_step_label.py --cell appworld_dev_4bi --model <judge> --batch-write in/appworld_dev_4bi.jsonl
    OPENAI_API_KEY=... python openai_batch.py submit in/appworld_dev_4bi.jsonl
    OPENAI_API_KEY=... python openai_batch.py fetch <batch_id> out/appworld_dev_4bi.jsonl
    python pi_error_step_label.py --cell appworld_dev_4bi --batch-ingest out/appworld_dev_4bi.jsonl
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import time

from openai import OpenAI

MAX_REQUESTS = 50_000  # per-batch limit in the guide
MAX_BYTES = 200 * 1024 * 1024  # input-file limit in the guide


_API = {"key_env": "OPENAI_API_KEY", "base_url": None}


def client():
    key = os.environ.get(_API["key_env"])
    if not key:
        sys.exit(
            f"{_API['key_env']} is not set (the endpoint must accept Files + Batches uploads; see module docstring)"
        )
    return OpenAI(api_key=key, base_url=_API["base_url"] or os.environ.get("OPENAI_BASE_URL") or None)


def validate(path, endpoint):
    """Reject a file the Batch API would reject, before paying for the upload."""
    size = os.path.getsize(path)
    ids, models, n = set(), set(), 0
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            r = json.loads(line)
            if r.get("url") != endpoint or r.get("method") != "POST":
                sys.exit(f"{path}:{i}: url/method must be POST {endpoint}")
            if r["custom_id"] in ids:
                sys.exit(f"{path}:{i}: duplicate custom_id {r['custom_id']}")
            ids.add(r["custom_id"])
            models.add(r["body"]["model"])
            n = i
    if n > MAX_REQUESTS or size > MAX_BYTES:
        sys.exit(f"{path}: {n:,} requests / {size / 1e6:.0f} MB exceeds {MAX_REQUESTS:,} / 200 MB; split it")
    if len(models) != 1:
        sys.exit(f"{path}: batch mixes models {sorted(models)}")
    return n, size, models.pop()


def rewrite_model(path, model):
    """Copy of `path` with every body.model replaced, for a provider whose model id differs."""
    fd, tmp = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as out, open(path) as fh:
        for line in fh:
            r = json.loads(line)
            r["body"]["model"] = model
            out.write(json.dumps(r) + "\n")
    return tmp


def cmd_submit(a):
    c = client()
    for path in a.files:
        ledger = path + ".batch.json"
        if os.path.exists(ledger) and not a.force:
            print(f"{path}: already submitted ({json.load(open(ledger))['batch_id']}); --force to resubmit")
            continue
        src = rewrite_model(path, a.model) if a.model else path
        n, size, model = validate(src, a.endpoint)
        up = c.files.create(file=open(src, "rb"), purpose="batch")
        b = c.batches.create(
            input_file_id=up.id,
            endpoint=a.endpoint,
            completion_window=a.window,
            metadata={"source": os.path.basename(path)},
        )
        json.dump(
            {
                "input": path,
                "input_file_id": up.id,
                "batch_id": b.id,
                "model": model,
                "requests": n,
                "bytes": size,
                "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            open(ledger, "w"),
            indent=1,
        )
        print(f"{path}: {n:,} requests, {size / 1e6:.1f} MB, model={model} -> {b.id} ({b.status})")
        if src != path:
            os.remove(src)


def cmd_status(a):
    c = client()
    ids = a.ids or [json.load(open(p))["batch_id"] for p in sorted(glob.glob(a.ledger_glob))]
    for bid in ids:
        b = c.batches.retrieve(bid)
        rc = b.request_counts
        print(
            f"{bid}  {b.status:12s} completed {rc.completed:>6,} failed {rc.failed:>5,} total {rc.total:>6,}"
            f"  output={b.output_file_id or '-'}"
        )


def cmd_fetch(a):
    c = client()
    b = c.batches.retrieve(a.id)
    if b.status != "completed":
        sys.exit(f"{a.id} is {b.status}, not completed")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if b.output_file_id:
        open(a.out, "w").write(c.files.content(b.output_file_id).text)
        print(f"wrote {a.out}")
    if b.error_file_id:
        open(a.out + ".errors.jsonl", "w").write(c.files.content(b.error_file_id).text)
        print(f"wrote {a.out}.errors.jsonl ({b.request_counts.failed:,} failed requests)")


def cmd_cancel(a):
    print(client().batches.cancel(a.id).status)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None, help="default $OPENAI_BASE_URL, else https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="NAME of the env var holding the API key")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("submit")
    s.add_argument("files", nargs="+")
    s.add_argument("--endpoint", default="/v1/chat/completions")
    s.add_argument("--window", default="24h")
    s.add_argument("--model", default=None, help="rewrite body.model before upload")
    s.add_argument("--force", action="store_true", help="resubmit even if a ledger exists")
    s.set_defaults(fn=cmd_submit)
    s = sp.add_parser("status")
    s.add_argument("ids", nargs="*")
    s.add_argument("--ledger-glob", default="*.jsonl.batch.json")
    s.set_defaults(fn=cmd_status)
    s = sp.add_parser("fetch")
    s.add_argument("id")
    s.add_argument("out")
    s.set_defaults(fn=cmd_fetch)
    s = sp.add_parser("cancel")
    s.add_argument("id")
    s.set_defaults(fn=cmd_cancel)
    a = ap.parse_args()
    _API.update(key_env=a.api_key_env, base_url=a.base_url)
    a.fn(a)


if __name__ == "__main__":
    main()
