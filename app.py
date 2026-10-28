"""
A2A 1.0 (HTTP+JSON) Invoice Action Agent  -  GA5 "decide and act safely"

Routes (base = path of PUBLIC_BASE_URL, e.g. /a2a/):
  GET  /.well-known/agent-card.json      (public)
  POST {base}message:send                (Bearer)
  GET  {base}tasks/{id}                  (Bearer)
  GET  {base}tasks                       (Bearer)
  POST {base}tasks/{id}:cancel           (Bearer)

Layers: protocol (router/handlers)  |  storage (SQLite)  |  AI decisions (LLM, cached by content)
Run with ONE worker (locks are in-process):  uvicorn app:app --workers 1
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("a2a")

# ----------------------------------------------------------------------------- config
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()  # e.g. https://you-invoice.hf.space/a2a/
BASE_PATH = (urlparse(PUBLIC_BASE_URL).path if PUBLIC_BASE_URL else os.getenv("BASE_PATH", "/a2a/")) or "/"
BASE_PREFIX = BASE_PATH.rstrip("/")  # "/a2a"  ("" when served at root)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_CHUNK = max(1, int(os.getenv("LLM_CHUNK_SIZE", "6")))  # packages per call; chunks run in parallel
MOCK_LLM = os.getenv("MOCK_LLM") == "1"
KEEP_BRACKETS = os.getenv("REF_KEEP_BRACKETS") == "1"  # return refs as "[X]" instead of "X"
DEBUG_LOG_PACKAGES = os.getenv("DEBUG_LOG_PACKAGES") == "1"
DB_PATH = os.getenv("DB_PATH", "/tmp/a2a_agent.db")
MAX_BYTES = 512 * 1024

MT_A2A = "application/a2a+json"
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
MT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
MT_RESULTS = "application/vnd.ga5.invoice-action-results+json"

ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
S_INPUT = "TASK_STATE_INPUT_REQUIRED"
S_DONE = "TASK_STATE_COMPLETED"
S_CANCELED = "TASK_STATE_CANCELED"
TERMINAL = {S_DONE, S_CANCELED, "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

# ----------------------------------------------------------------------------- storage
db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
db.execute("PRAGMA journal_mode=WAL")
db.executescript(
    """
CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY, principal TEXT NOT NULL, created REAL NOT NULL, body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS tasks_principal ON tasks(principal, created);
CREATE TABLE IF NOT EXISTS messages(
  principal TEXT NOT NULL, message_id TEXT NOT NULL, hash TEXT NOT NULL, task_id TEXT NOT NULL,
  PRIMARY KEY(principal, message_id));
CREATE TABLE IF NOT EXISTS decisions(key TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS executions(
  principal TEXT NOT NULL, action_id TEXT NOT NULL, task_id TEXT NOT NULL,
  receipt_nonce TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(principal, action_id));
"""
)

_locks: dict = defaultdict(asyncio.Lock)


def lock(key: str) -> asyncio.Lock:
    return _locks[key]


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_task(task_id: str, principal: str):
    row = db.execute("SELECT body FROM tasks WHERE id=? AND principal=?", (task_id, principal)).fetchone()
    return json.loads(row[0]) if row else None


def put_task(task: dict, principal: str):
    db.execute(
        "INSERT INTO tasks(id, principal, created, body) VALUES(?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET body=excluded.body",
        (task["id"], principal, datetime.now(timezone.utc).timestamp(), json.dumps(task, ensure_ascii=False)),
    )


def put_message(principal: str, message_id: str, h: str, task_id: str):
    db.execute(
        "INSERT INTO messages(principal, message_id, hash, task_id) VALUES(?,?,?,?)",
        (principal, message_id, h, task_id),
    )


def get_decision(key: str):
    row = db.execute("SELECT body FROM decisions WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def put_decision(key: str, d: dict):
    db.execute("INSERT OR REPLACE INTO decisions(key, body) VALUES(?,?)", (key, json.dumps(d, ensure_ascii=False)))


# ----------------------------------------------------------------------------- HTTP helpers
STATUS_NAMES = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
                405: "METHOD_NOT_ALLOWED", 409: "ABORTED", 413: "RESOURCE_EXHAUSTED",
                415: "UNSUPPORTED_MEDIA_TYPE", 500: "INTERNAL"}


def jresp(obj, status=200, media_type=MT_A2A, headers=None) -> Response:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return Response(content=body, status_code=status, media_type=media_type, headers=headers)


def err(status: int, reason: str, message: str, headers=None) -> Response:
    """Generic error body: never echoes task IDs or reveals whether another principal's task exists."""
    return jresp(
        {"error": {"code": status, "status": STATUS_NAMES.get(status, "UNKNOWN"), "message": message,
                   "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                "reason": reason, "domain": "a2a-protocol.org"}]}},
        status, "application/json", headers,
    )


NOT_FOUND = lambda: err(404, "TASK_NOT_FOUND", "Task not found.")  # noqa: E731


def view(task: dict, history_length=None) -> dict:
    if history_length and history_length > 0:
        t = dict(task)
        t["history"] = task.get("history", [])[-history_length:]
        return t
    return task


# ----------------------------------------------------------------------------- agent card
def base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost")).split(",")[0].strip()
    return f"{proto}://{host}{BASE_PREFIX}/"


def agent_card(request: Request) -> dict:
    url = base_url(request)
    p = urlparse(url)
    return {
        "name": "Invoice Action Agent",
        "description": "Reads invoice claim packages, proposes exactly one evidence-backed business action per "
                       "package, executes only grader-accepted actions, and stores task state so retries "
                       "never repeat a payment.",
        "version": "1.0.0",
        "provider": {"organization": "GA5 Invoice Agent", "url": f"{p.scheme}://{p.netloc}"},
        "supportedInterfaces": [{"url": url, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {
            "scheme": "Bearer", "description": "Each bearer token is a separate, isolated principal."}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
        "defaultInputModes": [MT_BATCH, MT_RESULTS],
        "defaultOutputModes": [MT_PROPOSALS, MT_RECEIPTS],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action agent",
            "description": "Chooses settle_invoice, request_approval, hold_invoice, reject_duplicate or "
                           "open_exception for each invoice package, citing the decisive evidence references.",
            "tags": ["invoice", "accounts-payable", "approval-limits", "duplicate-payment", "idempotency"],
            "inputModes": [MT_BATCH, MT_RESULTS],
            "outputModes": [MT_PROPOSALS, MT_RECEIPTS],
        }],
    }


# ----------------------------------------------------------------------------- AI decision layer
SYSTEM_PROMPT = """You are an accounts-payable control agent. For EACH invoice package choose exactly ONE action:

- settle_invoice: the invoice is valid, fully reconciled (PO / goods receipt / amount / vendor agree) AND within the
  agent's autonomous payment authority under the CURRENT policy revision. Only settle when nothing requires
  approval, a hold, a duplicate rejection, or an exception. If in doubt, do NOT settle.
- request_approval: commercially valid and reconciled, but OUTSIDE delegated/autonomous authority (e.g. amount above
  the agent's limit for this policy revision, or a category that policy says needs a human approver).
- hold_invoice: payment must pause until a specific, stated verification completes (e.g. bank-detail change
  call-back, tax/KYC check, pending goods receipt or inspection).
- reject_duplicate: the SAME commercial invoice (same vendor + same invoice number, same amount) was ALREADY PAID.
  A payment for a DIFFERENT invoice number is NOT a duplicate. A cancelled/reversed/failed payment is not a payment.
- open_exception: material records conflict (amount, quantity, price, vendor, PO, currency mismatch, etc.) and need
  an exception workflow.

Reading rules:
- Documents deliberately mix useful facts with OLD/ARCHIVED examples, TRAINING material, templates, decoys,
  superseded policy revisions, negations ("this is not a duplicate", "no hold is required") and irrelevant action
  words. Use only current, applicable facts about THIS invoice. Apply the policy revision given for the batch.
- Find the single paragraph that actually determines the action. Return EXACTLY the three bracketed evidence
  references that appear in that paragraph, copied exactly (without the surrounding square brackets).
  Never return the cover-sheet reference, archive-example references, or training/decoy references.

Facts:
- vendorName and invoiceNumber exactly as written on the invoice under review.
- amountMinor = invoice total in minor currency units as an integer (INR 12,345.67 -> 1234567; JPY has 0 decimals).
- currency = ISO-4217 code (e.g. INR).

Respond with JSON only:
{"decisions":[{"ref":"P1","action":"...","vendorName":"...","invoiceNumber":"...","amountMinor":123,
"currency":"INR","evidenceRefs":["...","...","..."],"reason":"one or two sentences, <= 60 words"}]}
Include one decision for every package ref you were given."""

BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,120})\]")


def package_id(pkg: dict) -> str:
    return str(pkg.get("packageId") or pkg.get("id") or "")


def strip_ids(pkg):
    if isinstance(pkg, dict):
        return {k: v for k, v in pkg.items() if k not in ("packageId", "batchId")}
    return pkg


def bracket_refs(obj, out=None) -> list:
    """All [bracketed] tokens found inside string values of the package, in document order."""
    if out is None:
        out = []
    if isinstance(obj, str):
        for m in BRACKET_RE.findall(obj):
            m = m.strip()
            if m and m not in out:
                out.append(m)
    elif isinstance(obj, dict):
        for v in obj.values():
            bracket_refs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            bracket_refs(v, out)
    return out


def find_key(obj, names):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names and isinstance(v, (str, int, float)) and v != "":
                return v
        for v in obj.values():
            r = find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, names)
            if r is not None:
                return r
    return None


def to_minor(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        return int(v)
    return None


def fallback_decision(pkg: dict, reason: str) -> dict:
    refs = bracket_refs(pkg)
    return {
        "action": "open_exception",
        "vendorName": str(find_key(pkg, {"vendorName", "vendor", "supplier"}) or "UNKNOWN VENDOR"),
        "invoiceNumber": str(find_key(pkg, {"invoiceNumber", "invoiceNo", "invoice_number"}) or "UNKNOWN"),
        "amountMinor": to_minor(find_key(pkg, {"amountMinor"})) or 0,
        "currency": str(find_key(pkg, {"currency"}) or "INR").upper(),
        "evidenceRefs": refs[:3],
        "reason": reason,
        "_fallback": True,
    }


def validate_decision(raw: dict, pkg: dict):
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action", "")).strip()
    if action not in ACTIONS:
        return None
    valid = bracket_refs(pkg)
    lower = {v.lower(): v for v in valid}
    refs = []
    for r in raw.get("evidenceRefs") or []:
        r = str(r).strip()
        if r.startswith("[") and r.endswith("]"):
            r = r[1:-1].strip()
        r = r if r in valid else lower.get(r.lower())
        if r and r not in refs:
            refs.append(r)
    if len(refs) < 2:
        return None
    amount = to_minor(raw.get("amountMinor"))
    vendor = str(raw.get("vendorName") or "").strip()
    inv = str(raw.get("invoiceNumber") or "").strip()
    cur = str(raw.get("currency") or "").strip().upper()
    if amount is None or amount < 0 or not vendor or not inv or not re.fullmatch(r"[A-Z]{3}", cur):
        return None
    return {"action": action, "vendorName": vendor, "invoiceNumber": inv, "amountMinor": amount,
            "currency": cur, "evidenceRefs": refs[:3], "reason": str(raw.get("reason") or "").strip()}


def parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise


async def call_llm(policy: str, items: list, timeout: float) -> dict:
    user = {"policyRevision": policy,
            "packages": [{"ref": f"P{i + 1}", "package": pkg} for i, (_, pkg) in enumerate(items)]}
    payload = {
        "model": LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
    }
    import httpx  # imported lazily so mock/local runs need no extra package

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{LLM_BASE_URL}/chat/completions", json=payload,
                              headers={"Authorization": f"Bearer {LLM_API_KEY}"})
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    data = parse_json(content)
    out = {}
    for d in data.get("decisions", []):
        ref = str(d.get("ref", ""))
        if ref.startswith("P") and ref[1:].isdigit():
            idx = int(ref[1:]) - 1
            if 0 <= idx < len(items):
                out[items[idx][0]] = d
    return out


async def decide_chunk(policy: str, items: list) -> dict:
    """items: [(cache_key, package)] -> {cache_key: decision}"""
    results = {}
    pending = list(items)
    if MOCK_LLM:
        return {k: fallback_decision(p, "Mock mode: no model call was made.") for k, p in items}
    for timeout in (28.0, 12.0):
        if not pending:
            break
        try:
            raw = await call_llm(policy, pending, timeout)
        except Exception as e:  # network/parse failure -> retry once, then fall back
            log.warning("LLM call failed (%s): %s", type(e).__name__, str(e)[:200])
            continue
        still = []
        for k, pkg in pending:
            d = validate_decision(raw.get(k), pkg)
            if d:
                results[k] = d
            else:
                still.append((k, pkg))
        pending = still
    for k, pkg in pending:
        results[k] = fallback_decision(pkg, "Automated review could not verify a safe action, so the package "
                                            "is routed to the exception workflow.")
    return results


async def decide_packages(policy: str, packages: list) -> list:
    """Returns one decision per package (same order). Cached by policy revision + canonical package content."""
    keys = [sha(canon({"policyRevision": policy, "package": strip_ids(p)})) for p in packages]
    uniq = sorted(set(keys))
    held = []
    try:
        for k in uniq:  # sorted acquisition order -> no deadlocks between concurrent batches
            await lock("d:" + k).acquire()
            held.append(k)
        decided = {k: get_decision(k) for k in uniq}
        todo, seen = [], set()
        for k, p in zip(keys, packages):
            if decided[k] is None and k not in seen:
                seen.add(k)
                todo.append((k, p))
        if todo:
            log.info("model call for %d/%d packages", len(todo), len(packages))
            chunks = [todo[i:i + LLM_CHUNK] for i in range(0, len(todo), LLM_CHUNK)]
            for part in await asyncio.gather(*(decide_chunk(policy, c) for c in chunks)):
                for k, d in part.items():
                    decided[k] = d
                    if not d.get("_fallback"):
                        put_decision(k, d)
        return [decided[k] for k in keys]
    finally:
        for k in reversed(held):
            lock("d:" + k).release()


def build_rationale(action: str, refs: list, reason: str) -> str:
    cites = ", ".join(f"[{r}]" for r in refs)
    head = f"Proposed action: {action}. "
    tail = f" Decisive evidence: {cites}."
    reason = reason or "The decisive paragraph of the package supports this action under the current policy."
    room = 1500 - len(head) - len(tail)
    text = head + reason[:max(0, room)].strip() + tail
    if len(text) < 60:
        text += " Decision follows the cited paragraph and the current policy revision."
    return text[:1500]


def build_proposal(pkg: dict, d: dict) -> dict:
    refs = d["evidenceRefs"]
    out_refs = [f"[{r}]" for r in refs] if KEEP_BRACKETS else list(refs)
    return {
        "packageId": package_id(pkg),
        "actionId": "act_" + secrets.token_hex(12),  # durable: stored with the task, reused on every replay
        "action": d["action"],
        "facts": {"vendorName": d["vendorName"], "invoiceNumber": d["invoiceNumber"],
                  "amountMinor": d["amountMinor"], "currency": d["currency"]},
        "evidenceRefs": out_refs,
        "rationale": build_rationale(d["action"], refs, d.get("reason", "")),
    }


# ----------------------------------------------------------------------------- protocol handlers
def principal_from(request: Request):
    m = re.fullmatch(r"Bearer ([^\s]+)", request.headers.get("authorization", ""))
    return ("p_" + sha(m.group(1))[:40]) if m else None


def find_part(msg: dict, media_type: str):
    for p in msg.get("parts") or []:
        if isinstance(p, dict) and p.get("mediaType") == media_type and isinstance(p.get("data"), dict):
            return p
    return None


async def handle_send(principal: str, body: dict) -> Response:
    msg = body.get("message")
    if not isinstance(msg, dict) or not isinstance(msg.get("messageId"), str) or not msg["messageId"]:
        return err(400, "INVALID_MESSAGE", "A message with a messageId is required.")
    mid = msg["messageId"]
    h = sha(canon(msg))  # configuration is ignored for idempotency
    async with lock(f"m:{principal}:{mid}"):
        row = db.execute("SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?",
                         (principal, mid)).fetchone()
        if row:
            if row[0] != h:
                return err(409, "IDEMPOTENCY_CONFLICT", "messageId was already used with different content.")
            task = load_task(row[1], principal)
            return jresp({"task": task}) if task else NOT_FOUND()
        if msg.get("role") != "ROLE_USER":
            return err(400, "INVALID_MESSAGE", "role must be ROLE_USER.")
        if msg.get("taskId"):
            return await handle_continuation(principal, msg, h)
        return await handle_initial(principal, msg, h, body.get("configuration") or {})


async def handle_initial(principal: str, msg: dict, h: str, config: dict) -> Response:
    part = find_part(msg, MT_BATCH)
    if not part:
        return err(400, "UNSUPPORTED_CONTENT", f"Expected a {MT_BATCH} data part.")
    aom = config.get("acceptedOutputModes") if isinstance(config, dict) else None
    if isinstance(aom, list) and aom and MT_PROPOSALS not in aom:
        return err(400, "UNSUPPORTED_OUTPUT_MODE", "acceptedOutputModes must include the proposals media type.")
    data = part["data"]
    batch_id, packages = data.get("batchId"), data.get("packages")
    policy = str(data.get("policyRevision", ""))
    if not isinstance(batch_id, str) or not batch_id or not isinstance(packages, list) or not packages:
        return err(400, "INVALID_BATCH", "batchId and a non-empty packages list are required.")
    ids = [package_id(p) if isinstance(p, dict) else "" for p in packages]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        return err(400, "INVALID_BATCH", "Every package needs a unique packageId.")
    if DEBUG_LOG_PACKAGES:
        log.info("batch %s policy %s first package: %s", batch_id, policy, canon(packages[0])[:3000])

    decisions = await decide_packages(policy, packages)
    proposals = [build_proposal(p, d) for p, d in zip(packages, decisions)]

    task_id = "task_" + secrets.token_hex(16)
    context_id = msg.get("contextId") or ("ctx_" + secrets.token_hex(16))
    hist_msg = dict(msg, taskId=task_id, contextId=context_id)
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": S_INPUT,
            "timestamp": now_iso(),
            "message": {"messageId": "msg_" + secrets.token_hex(12), "role": "ROLE_AGENT",
                        "taskId": task_id, "contextId": context_id,
                        "parts": [{"text": "Proposals ready. A proposal is not permission to act: send "
                                           f"{MT_RESULTS} for this task to execute accepted actions."}]},
        },
        "artifacts": [{"artifactId": "art_proposals_" + secrets.token_hex(8), "name": "invoice-action-proposals",
                       "parts": [{"mediaType": MT_PROPOSALS, "data": {"batchId": batch_id, "proposals": proposals}}]}],
        "history": [hist_msg],
    }
    db.execute("BEGIN")
    try:
        put_task(task, principal)
        put_message(principal, msg["messageId"], h, task_id)
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return jresp({"task": task})


async def handle_continuation(principal: str, msg: dict, h: str) -> Response:
    task_id = msg["taskId"]
    async with lock(f"t:{task_id}"):
        task = load_task(task_id, principal)
        if not task:
            return NOT_FOUND()
        if msg.get("contextId") != task["contextId"]:
            return err(400, "CONTEXT_MISMATCH", "contextId does not match the task.")
        if task["status"]["state"] != S_INPUT:
            return err(409, "TASK_NOT_CONTINUABLE", "Task is not awaiting results.")
        part = find_part(msg, MT_RESULTS)
        if not part:
            return err(400, "UNSUPPORTED_CONTENT", f"Expected a {MT_RESULTS} data part.")
        prop_data = task["artifacts"][0]["parts"][0]["data"]
        if part["data"].get("batchId") != prop_data["batchId"]:
            return err(400, "BATCH_MISMATCH", "batchId does not match the task.")
        results = part["data"].get("results")
        if not isinstance(results, list) or not results:
            return err(400, "INVALID_RESULTS", "results must be a non-empty list.")
        props = {p["packageId"]: p for p in prop_data["proposals"]}
        by_pkg = {}
        for r in results:
            if not isinstance(r, dict):
                return err(400, "INVALID_RESULTS", "Malformed result.")
            p = props.get(r.get("packageId"))
            if not p or r["packageId"] in by_pkg:
                return err(400, "PACKAGE_MISMATCH", "Result package does not match a stored proposal.")
            if r.get("actionId") != p["actionId"] or r.get("action") != p["action"]:
                return err(400, "ACTION_MISMATCH", "Result action does not match the stored proposal.")
            if r.get("outcome") not in ("ACCEPTED", "REJECTED"):
                return err(400, "INVALID_OUTCOME", "outcome must be ACCEPTED or REJECTED.")
            if r["outcome"] == "ACCEPTED" and (not isinstance(r.get("receiptNonce"), str) or not r["receiptNonce"]):
                return err(400, "MISSING_RECEIPT", "Accepted results need a receiptNonce.")
            by_pkg[r["packageId"]] = r
        if set(by_pkg) != set(props):
            return err(400, "INCOMPLETE_RESULTS", "Results must cover every proposal exactly once.")

        executions = []
        for p in prop_data["proposals"]:  # proposal order; only ACCEPTED are executed
            r = by_pkg[p["packageId"]]
            if r["outcome"] != "ACCEPTED":
                continue
            executions.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                               "receiptNonce": r["receiptNonce"], "facts": p["facts"],
                               "evidenceRefs": p["evidenceRefs"]})

        task["history"].append(msg)
        task["artifacts"].append({
            "artifactId": "art_receipts_" + secrets.token_hex(8), "name": "invoice-action-receipts",
            "parts": [{"mediaType": MT_RECEIPTS, "data": {"batchId": prop_data["batchId"], "executions": executions}}],
        })
        task["status"] = {"state": S_DONE, "timestamp": now_iso()}

        db.execute("BEGIN")
        try:
            for e in executions:  # the action ledger: one execution per actionId, ever
                db.execute("INSERT OR IGNORE INTO executions(principal, action_id, task_id, receipt_nonce, body) "
                           "VALUES(?,?,?,?,?)", (principal, e["actionId"], task_id, e["receiptNonce"],
                                                 json.dumps(e, ensure_ascii=False)))
            put_task(task, principal)
            put_message(principal, msg["messageId"], h, task_id)
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        return jresp({"task": task})


async def handle_cancel(principal: str, task_id: str) -> Response:
    async with lock(f"t:{task_id}"):
        task = load_task(task_id, principal)
        if not task:
            return NOT_FOUND()
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task is already in a terminal state.")
        task["status"] = {"state": S_CANCELED, "timestamp": now_iso()}
        put_task(task, principal)
        return jresp(task)


def handle_list(principal: str, q) -> Response:
    rows = db.execute("SELECT body FROM tasks WHERE principal=? ORDER BY created DESC", (principal,)).fetchall()
    tasks = [json.loads(r[0]) for r in rows]
    if q.get("contextId"):
        tasks = [t for t in tasks if t.get("contextId") == q["contextId"]]
    if q.get("status"):
        tasks = [t for t in tasks if t["status"]["state"] == q["status"]]
    total = len(tasks)
    try:
        offset = int(q.get("pageToken") or 0)
        size = min(max(int(q.get("pageSize") or 50), 1), 100)
    except ValueError:
        return err(400, "INVALID_ARGUMENT", "Invalid paging parameters.")
    hl = q.get("historyLength")
    hl = int(hl) if hl and hl.isdigit() else None
    page = []
    for t in tasks[offset:offset + size]:
        t = dict(t)
        t["history"] = t.get("history", [])[-hl:] if hl else []
        page.append(t)
    out = {"tasks": page, "nextPageToken": str(offset + size) if offset + size < total else "",
           "pageSize": size, "totalSize": total}
    while len(json.dumps(out).encode()) > MAX_BYTES - 4096 and len(page) > 1:
        page.pop()
        out["nextPageToken"] = str(offset + len(page))
    return jresp(out)


# ----------------------------------------------------------------------------- router
async def router(request: Request):
    path = request.scope.get("path") or "/"  # already percent-decoded (":" may arrive as %3A)
    try:
        if path == "/.well-known/agent-card.json":
            if request.method != "GET":
                return err(405, "METHOD_NOT_ALLOWED", "Method not allowed.")
            return jresp(agent_card(request), media_type="application/json")

        if BASE_PREFIX and not (path == BASE_PREFIX or path.startswith(BASE_PREFIX + "/")):
            return err(404, "NOT_FOUND", "Not found.")
        rest = path[len(BASE_PREFIX):].strip("/")

        principal = principal_from(request)
        if not principal:
            return err(401, "UNAUTHENTICATED", "A Bearer token is required.",
                       headers={"WWW-Authenticate": "Bearer"})
        if request.headers.get("a2a-version", "").strip() != "1.0":
            return err(400, "VERSION_NOT_SUPPORTED", "A2A-Version: 1.0 is required.")

        raw = await request.body()
        body = {}
        if raw.strip():
            ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if ct != MT_A2A:
                return err(415, "UNSUPPORTED_MEDIA_TYPE", f"Content-Type must be {MT_A2A}.")
            try:
                body = json.loads(raw)
            except Exception:
                return err(400, "INVALID_JSON", "Request body is not valid JSON.")
            if not isinstance(body, dict):
                return err(400, "INVALID_JSON", "Request body must be a JSON object.")

        m = request.method
        if rest == "message:send":
            if m != "POST":
                return err(405, "METHOD_NOT_ALLOWED", "Method not allowed.")
            if not raw.strip():
                return err(400, "INVALID_MESSAGE", "Request body is required.")
            return await handle_send(principal, body)
        if rest == "tasks":
            if m != "GET":
                return err(405, "METHOD_NOT_ALLOWED", "Method not allowed.")
            return handle_list(principal, request.query_params)
        if rest.startswith("tasks/"):
            tail = rest[len("tasks/"):]
            if tail.endswith(":cancel") and "/" not in tail:
                if m != "POST":
                    return err(405, "METHOD_NOT_ALLOWED", "Method not allowed.")
                return await handle_cancel(principal, tail[: -len(":cancel")])
            if "/" not in tail and ":" not in tail and tail:
                if m != "GET":
                    return err(405, "METHOD_NOT_ALLOWED", "Method not allowed.")
                task = load_task(tail, principal)
                if not task:
                    return NOT_FOUND()
                hl = request.query_params.get("historyLength")
                return jresp(view(task, int(hl) if hl and hl.isdigit() else None))
        return err(404, "NOT_FOUND", "Not found.")
    except Exception:
        log.exception("unhandled error")
        return err(500, "INTERNAL", "Internal error.")


# One catch-all route: no framework redirects, no automatic trailing-slash handling.
app = Starlette(routes=[Route("/{full_path:path}", router, methods=["GET", "POST", "PUT", "PATCH", "DELETE"])])
log.info("base path=%s public url=%s model=%s mock=%s db=%s",
         BASE_PATH, PUBLIC_BASE_URL or "(derived)", LLM_MODEL, MOCK_LLM, DB_PATH)
