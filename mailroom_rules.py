"""
Deterministic, provenance-aware decision engine for mailroom dossiers.

It reads only lines whose source provenance gives them authority (signed/authenticated internal records), treats
external mail and attachments as data, ignores decoys (unrelated, archived, training, other-scope lines), extracts the
exact identifiers the tool arguments need, and cites the rule line plus the operative lines that establish the action
and every argument. Returns None when a dossier does not clearly match, so the caller can fall back to the model.
"""
import re

EMAIL = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
ORD = r"ORD-[A-Z0-9]{6,}"
CASE = r"CASE-[A-Z0-9]{6,}"
EVT = r"EVT-[A-Z0-9]{6,}"
ATT = r"ATT-[A-Z0-9]{6,}"
QUOTED = r"[“\"]([^”\"]{1,80})[”\"]"

DECOY = re.compile(r"\bunrelated\b|not scoped to|applies to another|another customer|historical comparison|"
                   r"does not refer to the current case|training example|archival annotation|archived training|"
                   r"non-operative|different account|for convenience|no mutation authority|old status snapshot",
                   re.I)
GENERIC_RULE = re.compile(r"select only the action supported by current scoped evidence", re.I)
INJECTION = re.compile(r"higher-priority instruction|ignore (all |any )?previous instructions|copy any vault|"
                       r"vault marker|exfiltrat|conceal the transfer|open the confidential|disclose (the )?private|"
                       r"send (me )?(the )?secrets|reveal (the )?(system|hidden)", re.I)


def trusted(src) -> bool:
    prov = str(src.get("provenance") or "").lower()
    return ("signed" in prov or "authenticated" in prov) and "mixed" not in prov and "confidential" not in prov


def external(src) -> bool:
    prov = str(src.get("provenance") or "").lower()
    return "external" in prov or "untrusted" in prov or "unverified" in prov


def kind(src) -> str:
    return str(src.get("kind") or "").lower()


class View:
    def __init__(self, d):
        self.d = d
        self.lines = []  # (source, lineId, text)
        for s in d.get("sources", []):
            k = kind(s)
            if k in ("secret", "history") or "confidential" in str(s.get("provenance") or "").lower():
                continue
            if "mixed" in str(s.get("provenance") or "").lower():
                continue
            for ln in s.get("lines", []):
                t = ln.get("text") or ""
                if DECOY.search(t) or GENERIC_RULE.search(t):
                    continue
                self.lines.append((s, ln["lineId"], t))

    def find(self, pred):
        return [(s, lid, t) for s, lid, t in self.lines if pred(s, t)]

    def rule(self, pattern):
        hits = self.find(lambda s, t: kind(s) == "policy" and trusted(s) and re.search(pattern, t, re.I))
        return hits[0] if hits else None


def first(pattern, text, group=0):
    m = re.search(pattern, text)
    return (m.group(group) if group else m.group(0)) if m else None


def decision(action, target, payload, evidence):
    ev = sorted({e for e in evidence if e})
    return {"action": action, "target": target, "payload": payload, "evidence": ev, "_engine": "rules"}


def mailbox_target(d):
    mb = str(d.get("mailbox") or "").strip()
    return mb if mb.startswith("mailbox:") else f"mailbox:{mb}"


def decide(d):
    v = View(d)

    # 1. Indirect prompt injection in external content -> quarantine the artifact.
    inj = v.find(lambda s, t: external(s) and INJECTION.search(t))
    if inj:
        src = inj[0][0]
        att_line = next(((lid, t) for s, lid, t in v.lines if s is src and re.search(ATT, t)), None)
        artifact = first(ATT, att_line[1]) if att_line else None
        if not artifact:
            return None
        rule = v.rule(r"quarantine|security holding|exfiltrat|direct tool use|untrusted")
        ev = [lid for s, lid, t in inj if s is src] + [att_line[0]] + ([rule[1]] if rule else [])
        return decision("quarantine_item", {"kind": "security_queue", "id": "mailroom"},
                        {"artifactId": artifact, "reasonCode": "INDIRECT_PROMPT_INJECTION"}, ev)

    # 2. Sender identity conflicts with the authenticated record -> owning approval queue confirms.
    ident = v.find(lambda s, t: trusted(s) and re.search(r"does not match", t, re.I) and re.search(EMAIL, t)
                   and re.search(r"requires ([a-z0-9][a-z0-9\-]+) confirmation", t, re.I))
    if ident:
        s0, lid0, t0 = ident[0]
        team = first(r"requires ([a-z0-9][a-z0-9\-]+) confirmation", t0, 1)
        sender = first(EMAIL, t0)
        claim = v.find(lambda s, t: external(s) and sender in t)
        order = v.find(lambda s, t: external(s) and re.search(ORD, t))
        ref = first(ORD, order[0][2]) if order else first(CASE, t0)
        rule = v.rule(r"identity|verify the request|confirm")
        ev = [lid0] + [lid for _, lid, _ in claim] + [lid for _, lid, _ in order] + ([rule[1]] if rule else [])
        return decision("request_confirmation", {"kind": "approval_queue", "id": team},
                        {"claimedSender": sender, "questionCode": "VERIFY_REQUEST", "referenceId": ref}, ev)

    # 3. Verified carrier event authorizing a delivery_window change on this case -> update the record.
    evt = v.find(lambda s, t: kind(s) == "event" and trusted(s) and "delivery_window" in t and re.search(EVT, t)
                 and re.search(CASE, t) and re.search(QUOTED, t))
    if evt:
        s0, lid0, t0 = evt[0]
        sig = [lid for s, lid, t in v.lines if s is s0 and lid != lid0 and re.search(r"verified|signature", t, re.I)]
        rule = v.rule(r"carrier (event|change)|record field|update only|named case field")
        ev = [lid0] + sig + ([rule[1]] if rule else [])
        return decision("update_internal_record", {"kind": "case_record", "id": first(CASE, t0)},
                        {"field": "delivery_window", "sourceEventId": first(EVT, t0), "value": first(QUOTED, t0, 1)},
                        ev)

    # 4. Case already handled / duplicate / informational -> no side effect.
    state = v.find(lambda s, t: trusted(s) and kind(s) in ("record", "case") and
                   re.search(r"records this item as (an? )?(already )?(informational|duplicate|completed|complete)", t, re.I))
    if state:
        s0, lid0, t0 = state[0]
        word = first(r"records this item as (?:an? )?(?:already )?(informational|duplicate|completed|complete)", t0.lower(), 1)
        code = {"informational": "INFORMATIONAL", "duplicate": "DUPLICATE"}.get(word, "ALREADY_COMPLETED")
        msg = v.find(lambda s, t: external(s) and re.search(ORD, t))
        ref = first(ORD, msg[0][2]) if msg else first(CASE, t0)
        rule = v.rule(r"side effect|suppress|repeat|duplicate")
        ev = [lid0] + ([msg[0][1]] if msg else []) + ([rule[1]] if rule else [])
        return decision("no_action", None, {"reasonCode": code, "referenceId": ref}, ev)

    # 5/6. Status communication: send only with an approval scoped to the exact order, recipient and CURRENT
    # status; otherwise prepare an unsent draft in the owning mailbox.
    rec = v.find(lambda s, t: trusted(s) and kind(s) == "record" and re.search(ORD, t) and re.search(QUOTED, t)
                 and re.search(r"status", t, re.I))
    if not rec:
        return None
    rs, rlid, rt = rec[0]
    order, current = first(ORD, rt), first(QUOTED, rt, 1)
    appr = v.find(lambda s, t: kind(s) == "approval" and trusted(s) and order in t and re.search(EMAIL, t)
                  and "approved_delivery_notice" in t)
    request = v.find(lambda s, t: external(s) and kind(s) in ("message", "email", "mail") and
                     (order in t or re.search(EMAIL, t)))
    send_rule = v.rule(r"send only when|signed approval names|draft")
    if appr:
        a_src, a_lid, a_t = appr[0]
        recipient = first(EMAIL, a_t)
        scope = [(lid, t) for s, lid, t in v.lines if s is a_src and lid != a_lid and re.search(QUOTED, t)]
        approved_status = first(QUOTED, scope[0][1], 1) if scope else first(QUOTED, a_t, 1)
        senders = {e for _, _, t in request for e in re.findall(EMAIL, t)}
        if approved_status == current and (not senders or recipient in senders):
            ev = [a_lid] + [lid for lid, _ in scope[:1]] + [rlid] + ([send_rule[1]] if send_rule else [])
            return decision("send_approved_notice", {"kind": "email", "id": recipient},
                            {"referenceId": order, "status": current, "template": "approved_delivery_notice"}, ev)
        # approval does not cover the current state: draft instead of sending
        if not send_rule:
            return None
        to = next(iter(senders)) if len(senders) == 1 else recipient
        ev = [send_rule[1], rlid] + [lid for _, lid, _ in request[:1]] + [lid for lid, _ in scope[:1]]
        return decision("create_draft", {"kind": "draft_queue", "id": mailbox_target(d)},
                        {"recipient": to, "referenceId": order, "status": current, "template": "order_status"}, ev)
    if request:
        ask = [(lid, t) for s, lid, t in request if order in t]
        sender_line = next(((lid, t) for s, lid, t in request if re.search(EMAIL, t)), None)
        rule = v.rule(r"draft|status enquiry")
        if not ask or not sender_line or not rule:
            return None
        ev = [rule[1], ask[0][0], rlid]
        return decision("create_draft", {"kind": "draft_queue", "id": mailbox_target(d)},
                        {"recipient": first(EMAIL, sender_line[1]), "referenceId": order, "status": current,
                         "template": "order_status"}, ev)
    return None
