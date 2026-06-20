"""Interactive inference server (Python stdlib only — no extra deps).

    python -m src.serve.app          # -> http://localhost:8078
GET  /          intake form: enter a patient -> full decision chain, provenance, oversight
POST /predict   JSON intake -> JSON decision (incl. decision_trace, data_provenance, oversight)
POST /override  {audit_id, party, action, reason} -> append to the audit log
GET  /audit     last 20 audit-log entries (decisions + overrides)
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .audit import AuditLog
from .infer import EXAMPLES, TriageEngine

ENGINE = TriageEngine()
AUDIT = AuditLog()

FORM = r"""<!doctype html><meta charset=utf-8><title>Triage Copilot — supervisable decision</title>
<style>body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:#0f1419;color:#e6edf3;margin:0}
.wrap{max-width:980px;margin:0 auto;padding:22px}h1{font-size:18px}a{color:#4aa3ff}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
label{font-size:12px;color:#8b98a8;display:block}input,select{width:100%;background:#161c24;border:1px solid #2a3441;color:#e6edf3;border-radius:6px;padding:7px}
button{margin-top:10px;background:#4aa3ff;color:#04121f;border:0;border-radius:8px;padding:9px 16px;font-weight:600;cursor:pointer}
button.sec{background:#26323f;color:#e6edf3}
.card{background:#161c24;border:1px solid #2a3441;border-radius:10px;padding:14px;margin-top:14px}
.pill{display:inline-block;padding:2px 8px;border-radius:5px;font-weight:700;color:#04121f}
.a1{background:#ff4d4f}.a2{background:#ff8c3b}.a3{background:#f5c518}.a4{background:#3ecf8e}.a5{background:#7d8794;color:#fff}
.flag{color:#ff8c3b;font-weight:700}.k{color:#8b98a8}.req{color:#ff8c3b;font-weight:700}.opt{color:#8b98a8}
ol{margin:6px 0;padding-left:20px}ol li{margin:2px 0}h3{font-size:13px;color:#8b98a8;text-transform:uppercase;letter-spacing:.5px;margin:0 0 8px}
.role{border-left:3px solid #2a3441;padding:4px 10px;margin:6px 0}.role.r{border-left-color:#ff8c3b}
code{background:#0f1419;padding:1px 5px;border-radius:4px}</style>
<div class=wrap><h1>🚑 Triage Copilot — a supervisable decision</h1>
<p class=k>Text-blind structured model · isotonic calibration · conformal prediction · safety override.
Every decision shows its <b>chain</b>, its <b>data provenance</b>, and its <b>multi-party oversight</b>; actions are written to an <a href="/audit" target=_blank>audit log</a>.</p>
<div class=grid id=f></div>
<button onclick=go()>Predict</button> <button class=sec onclick=fill()>Load example</button>
<div id=out></div></div>
<script>
const FIELDS=[["chief_complaint_raw","text","acute confusion and fever, sudden"],
["age","number",82],["sex","select","M,F,Other"],["language","select","Finnish,Swedish,English,Russian,Estonian,Arabic,Somali,Other"],
["insurance_type","select","public,private,none,military,unknown"],["arrival_mode","select","walk-in,ambulance,helicopter,police,transfer,brought_by_family"],
["chief_complaint_system","text","neurological"],["news2_score","number",8],["spo2","number",88],
["heart_rate","number",118],["respiratory_rate","number",24],["systolic_bp","number",92],
["gcs_total","number",12],["pain_score","number",-1],["mental_status_triage","select","alert,drowsy,confused,agitated,unresponsive"]];
const EX=%EXAMPLES%;let LAST=null;
function render(){f.innerHTML=FIELDS.map(([k,t,v])=>{
 if(t==="select"){const o=v.split(",").map(x=>`<option>${x}</option>`).join("");return `<div><label>${k}</label><select id="i_${k}">${o}</select></div>`}
 return `<div><label>${k}</label><input id="i_${k}" type="${t}" value="${v}"></div>`}).join("")}
function fill(){const e=EX[Math.floor(Math.random()*EX.length)].intake;FIELDS.forEach(([k])=>{const el=document.getElementById("i_"+k);if(el&&e[k]!==undefined)el.value=e[k]})}
async function go(){const o={};FIELDS.forEach(([k,t])=>{let v=document.getElementById("i_"+k).value;o[k]=(t==="number")?parseFloat(v):v});
 const d=await(await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(o)})).json();LAST=d;
 const cset=d.conformal_set.map(a=>`<span class="pill a${a}">${a}</span>`).join(" ");
 const cap=d.model_acuity!==d.acuity?` <span class=flag>(model said ${d.model_acuity}, safety override → ${d.acuity})</span>`:"";
 const trace=d.decision_trace.map(s=>`<li><b>${s.stage}</b>: ${s.detail}</li>`).join("");
 const pv=d.data_provenance;
 const ov=d.oversight.map(o=>`<div class="role ${o.required?'r':''}"><b>${o.party}</b> <span class="${o.required?'req':'opt'}">${o.required?'REQUIRED':'optional'}</span><br>
   <span class=k>${o.action} — ${o.why}</span></div>`).join("");
 out.innerHTML=`
 <div class=card><h3>Decision</h3><span class="pill a${d.acuity}">ESI ${d.acuity}</span>${cap}
   ${d.defer_to_human?' <span class=flag>DEFER TO HUMAN</span>':''}
   <div class=k style="margin-top:8px">conformal set (${d.conformal_coverage*100}% coverage): ${cset} ·
   P(critical) ${(d.p_critical*100).toFixed(0)}% · admission ${(d.admission_risk*100).toFixed(0)}% · LOS ${d.predicted_los_h}h · bucket ${d.resource_bucket}</div></div>
 <div class=card><h3>Decision chain</h3><ol>${trace}</ol></div>
 <div class=card><h3>Data processing / provenance</h3>
   <div>${pv.features_provided}/${pv.features_required} features provided · ${pv.features_imputed.length} imputed</div>
   <div class=k>${pv.pain_sentinel_handling}</div>
   <div class=k>no-leakage: ${pv.no_leakage_attestation}</div></div>
 <div class=card><h3>Multi-party oversight</h3>${ov}
   <div style="margin-top:10px"><input id=reason placeholder="reason / note" style="width:60%;display:inline-block">
   <button onclick="act('confirm')">Triage nurse: confirm</button>
   <button class=sec onclick="act('override')">Override</button></div>
   <div id=ack class=k></div></div>`;
}
async function act(action){const reason=document.getElementById("reason").value||"(none)";
 const r=await(await fetch("/override",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({audit_id:LAST.audit_id,party:"Triage nurse",action,reason})})).json();
 document.getElementById("ack").innerHTML=`logged audit #${r.id} — ${action} by ${r.party} @ ${r.ts}. <a href="/audit" target=_blank>view log</a>`}
render();
</script>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.end_headers(); self.wfile.write(body.encode() if isinstance(body, str) else body)

    def _json_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or "{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, FORM.replace("%EXAMPLES%", json.dumps(EXAMPLES)), "text/html; charset=utf-8")
        elif self.path.startswith("/audit"):
            self._send(200, json.dumps(AUDIT.tail(20), ensure_ascii=False, indent=1), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            if self.path == "/predict":
                intake = self._json_body()
                r = ENGINE.predict(intake)
                entry = AUDIT.record("decision", {
                    "complaint": intake.get("chief_complaint_raw"), "acuity": r["acuity"],
                    "model_acuity": r["model_acuity"], "defer": r["defer_to_human"],
                    "override": r["safety_override_triggered"],
                    "required_reviewers": [o["party"] for o in r["oversight"] if o["required"]]})
                r["audit_id"] = entry["id"]
                self._send(200, json.dumps(r, ensure_ascii=False), "application/json")
            elif self.path == "/override":
                b = self._json_body()
                entry = AUDIT.record("human_action", {
                    "decision_id": b.get("audit_id"), "party": b.get("party"),
                    "action": b.get("action"), "reason": b.get("reason")})
                self._send(200, json.dumps(entry, ensure_ascii=False), "application/json")
            else:
                self._send(404, "not found", "text/plain")
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(exc)}), "application/json")

    def log_message(self, *a):
        pass


def main(port=8078):
    print("building engine ..."); ENGINE.build()
    print(f"triage inference server on http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
