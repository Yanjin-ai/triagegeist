"""Interactive inference server (Python stdlib only — no extra deps).

    python -m src.serve.app          # -> http://localhost:8078
GET  /          intake form (enter a patient, get a live triage decision)
POST /predict   JSON intake -> JSON decision (acuity, conformal set, bucket, priority, ...)
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .infer import EXAMPLES, TriageEngine

ENGINE = TriageEngine()

FORM = """<!doctype html><meta charset=utf-8><title>Triage Copilot — live inference</title>
<style>body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:#0f1419;color:#e6edf3;margin:0}
.wrap{max-width:880px;margin:0 auto;padding:22px}h1{font-size:18px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
label{font-size:12px;color:#8b98a8;display:block}input,select{width:100%;background:#161c24;border:1px solid #2a3441;color:#e6edf3;border-radius:6px;padding:7px}
button{margin-top:12px;background:#4aa3ff;color:#04121f;border:0;border-radius:8px;padding:10px 18px;font-weight:600;cursor:pointer}
pre{background:#161c24;border:1px solid #2a3441;border-radius:8px;padding:14px;white-space:pre-wrap}
.pill{display:inline-block;padding:2px 8px;border-radius:5px;font-weight:700;color:#04121f}
.a1{background:#ff4d4f}.a2{background:#ff8c3b}.a3{background:#f5c518}.a4{background:#3ecf8e}.a5{background:#7d8794;color:#fff}
.flag{color:#ff8c3b;font-weight:700}.k{color:#8b98a8}</style>
<div class=wrap><h1>🚑 Triage Copilot — live inference</h1>
<p class=k>Enter a patient intake → text-blind structured model + isotonic calibration + conformal prediction set.</p>
<div class=grid id=f></div>
<button onclick=go()>Predict</button> <button onclick=fill()>Load example</button>
<div id=out></div></div>
<script>
const FIELDS=[["chief_complaint_raw","text","central chest pain radiating to arm, sudden"],
["age","number",68],["sex","select","M,F,Other"],["language","select","Finnish,Swedish,English,Russian,Estonian,Arabic,Somali,Other"],
["insurance_type","select","public,private,none,military,unknown"],["arrival_mode","select","walk-in,ambulance,helicopter,police,transfer,brought_by_family"],
["chief_complaint_system","text","cardiovascular"],["news2_score","number",9],["spo2","number",89],
["heart_rate","number",122],["respiratory_rate","number",26],["systolic_bp","number",95],
["gcs_total","number",14],["pain_score","number",9],["mental_status_triage","select","alert,drowsy,confused,agitated,unresponsive"]];
const EX=%EXAMPLES%;
function render(){f.innerHTML=FIELDS.map(([k,t,v])=>{
 if(t==="select"){const o=v.split(",").map(x=>`<option>${x}</option>`).join("");return `<div><label>${k}</label><select id="i_${k}">${o}</select></div>`}
 return `<div><label>${k}</label><input id="i_${k}" type="${t}" value="${v}"></div>`}).join("")}
function fill(){const e=EX[Math.floor(Math.random()*EX.length)].intake;for(const k in e){const el=document.getElementById("i_"+k);if(el)el.value=e[k]}}
async function go(){const o={};FIELDS.forEach(([k,t])=>{let v=document.getElementById("i_"+k).value;o[k]=(t==="number")?parseFloat(v):v});
 const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(o)});
 const d=await r.json();const cset=d.conformal_set.map(a=>`<span class="pill a${a}">${a}</span>`).join(" ");
 const probs=Object.entries(d.acuity_proba).map(([k,v])=>`${k}:${(v*100).toFixed(0)}%`).join("  ");
 out.innerHTML=`<h3>Decision <span class="pill a${d.acuity}">ESI ${d.acuity}</span> ${d.defer_to_human?'<span class=flag>DEFER TO HUMAN</span>':''}</h3>
 <pre><b>conformal set (${d.conformal_coverage*100}% coverage):</b> ${cset}
<span class=k>acuity probs:</span> ${probs}
<span class=k>P(critical):</span> ${(d.p_critical*100).toFixed(0)}%   <span class=k>admission risk:</span> ${(d.admission_risk*100).toFixed(0)}%   <span class=k>pred LOS:</span> ${d.predicted_los_h}h
<span class=k>resource bucket:</span> ${d.resource_bucket}   <span class=k>priority:</span> ${d.priority}   <span class=k>entropy:</span> ${d.uncertainty_entropy}
<span class=k>red flags:</span> ${d.complaint_trace.red_flags.join(", ")||"none"}</pre>`}
render();
</script>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.end_headers(); self.wfile.write(body.encode() if isinstance(body, str) else body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, FORM.replace("%EXAMPLES%", json.dumps(EXAMPLES)), "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        intake = json.loads(self.rfile.read(n) or "{}")
        try:
            self._send(200, json.dumps(ENGINE.predict(intake)), "application/json")
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(exc)}), "application/json")

    def log_message(self, *a):  # quiet
        pass


def main(port=8078):
    print(f"building engine ..."); ENGINE.build()
    print(f"triage inference server on http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
