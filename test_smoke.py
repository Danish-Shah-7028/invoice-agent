"""Local smoke test (needs: requests) against a running server (MOCK_LLM=1). python test_smoke.py"""
import concurrent.futures as cf
import json
import requests

B = "http://127.0.0.1:7860/a2a/"
H = {"Authorization": "Bearer alice", "A2A-Version": "1.0", "Content-Type": "application/a2a+json"}
H2 = dict(H, Authorization="Bearer mallory")

pkgs = [{"packageId": f"PKG-{i}", "documents": [
    {"kind": "invoice", "text": f"Invoice INV-{i} from Acme Ltd for INR 1,000.00 [COVER-{i}]"},
    {"kind": "note", "text": f"Decisive paragraph [E{i}-1] [E{i}-2] [E{i}-3]"}]} for i in range(3)]
msg = {"message": {"messageId": "m-1", "role": "ROLE_USER", "parts": [{"mediaType":
       "application/vnd.ga5.invoice-claim-batch+json", "data": {"batchId": "B1", "policyRevision": "r7",
       "packages": pkgs}}]}, "configuration": {"returnImmediately": False, "historyLength": 20}}

c = requests.Session()
print("card", c.get("http://127.0.0.1:7860/.well-known/agent-card.json").status_code)
print("noauth", c.post(B + "message:send", json=msg).status_code)
print("badver", c.post(B + "message:send", headers=dict(H, **{"A2A-Version": "0.3"}), data=json.dumps(msg)).status_code)

with cf.ThreadPoolExecutor(4) as ex:
    rs = list(ex.map(lambda _: c.post(B + "message:send", headers=H, data=json.dumps(msg)), range(4)))
ids = {r.json()["task"]["id"] for r in rs}
print("concurrent same msg ->", [r.status_code for r in rs], "unique tasks", len(ids))
task = rs[0].json()["task"]
print("state", task["status"]["state"], "history", len(task["history"]))

reordered = json.loads(json.dumps(msg)); reordered["configuration"]["returnImmediately"] = True
print("replay changed config", c.post(B + "message:send", headers=H, data=json.dumps(reordered)).json()["task"]["id"] == task["id"])
changed = json.loads(json.dumps(msg)); changed["message"]["parts"][0]["data"]["batchId"] = "B2"
r = c.post(B + "message:send", headers=H, data=json.dumps(changed)); print("conflict", r.status_code, r.text[:80])

print("outsider get", c.get(B + f"tasks/{task['id']}", headers=H2).status_code,
      "outsider list", c.get(B + "tasks", headers=H2).json())

props = task["artifacts"][0]["parts"][0]["data"]["proposals"]
res = {"message": {"messageId": "m-2", "taskId": task["id"], "contextId": task["contextId"], "role": "ROLE_USER",
       "parts": [{"mediaType": "application/vnd.ga5.invoice-action-results+json", "data": {"batchId": "B1",
       "results": [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                    "outcome": "ACCEPTED" if i else "REJECTED", "receiptNonce": f"nonce-{i}"} for i, p in enumerate(props)]}}]}}
with cf.ThreadPoolExecutor(2) as ex:
    f1 = ex.submit(lambda: c.post(B + "message:send", headers=H, data=json.dumps(res)))
    f2 = ex.submit(lambda: c.post(B + f"tasks/{task['id']}:cancel", headers=H))
print("race result/cancel", f1.result().status_code, f2.result().status_code)
final = c.get(B + f"tasks/{task['id']}", headers=H).json()
print("final", final["status"]["state"], "artifacts", len(final["artifacts"]), "history", len(final["history"]))
if len(final["artifacts"]) > 1:
    print("executions", len(final["artifacts"][1]["parts"][0]["data"]["executions"]))
print("proposal sample", json.dumps(props[0], indent=1))
