# Demo video — production plan & shot-by-shot script

**Honest note:** I cannot record/encode an actual video file. This is a complete, ready-to-shoot script:
exact tools, setup, scenes, on-screen actions, and narration. Following it produces a ~3.5-minute demo. A
GIF-only fallback is at the end.

## Tooling (macOS)
- **Screen recording:** QuickTime Player → File → New Screen Recording (free), or **Loom** / **OBS**.
- **Resolution:** record a single browser window at ~1280×900. Hide bookmarks bar.
- **Two terminals running before you hit record:**
  ```
  python -m http.server 8077 --directory app      # dashboard
  python -m src.serve.app                          # live inference (wait for "server on http://localhost:8078")
  ```
- Open two browser tabs: `http://localhost:8077` and `http://localhost:8078`.

## Narration style
Calm, factual, clinical. The whole point is *honesty + trustworthiness*, so avoid hype. ~140 words/min.
(Chinese narration alternative provided per scene in《》.)

---

## Scene 1 — The hook & the honest core (0:00–0:35)
**Screen:** dashboard Queue page (tab 8077). Slowly scroll the ranked queue.
**Narration:**
> "This is Triage Copilot — an emergency-department triage decision-support system. Before anything else, an
> honest finding shaped the whole project: in this dataset, the chief-complaint text almost perfectly
> determines the acuity label — so a text model scores a near-perfect leaderboard, but that's a property of
> the data, not clinical skill. So our decision engine is *text-blind*: it predicts acuity from physiology
> alone, and we wrap it in the things that actually matter for triage."

《这是 Triage Copilot，一个急诊分诊决策支持系统。一个诚实的发现决定了整个设计：本数据里主诉文本几乎完全决定了 acuity，所以文本模型能刷到近满分——但那是数据的性质，不是临床能力。因此我们的决策核心是"文本盲"的，只用生理信息预测，再围绕它做真正重要的部分。》

**Action:** point at the top banner ("Honest core … text-blind") and the policy toggle; click `throughput`
then back to `safety_first` to show the queue reorder.

## Scene 2 — A patient, fully explained (0:35–1:35)
**Screen:** click the top ESI-1 row (a high-acuity case, e.g. *meningococcal purpura*) → Patient page.
**Narration:**
> "Open any patient. On the left, the intake — note we *show* that pain wasn't assessed instead of hiding it.
> In the middle, a **calibrated** acuity distribution, admission risk, and length-of-stay. But the core of the
> product is below: the full **decision chain** — intake, data processing with a no-leakage attestation, the
> model, calibration, a **conformal prediction set** with a coverage guarantee, the safety override, resource
> mapping, priority, and the disposition. Next to it, **data provenance**: how many features were real versus
> imputed. And the **multi-party oversight** — this case requires both the triage nurse and a senior
> physician, each triggered by a concrete signal."

《点开任意病人。左边是录入信息——注意我们把"疼痛未评估"显示出来而不是藏起来。中间是校准后的 acuity 分布、入院风险、住院时长。产品的核心在下方：完整的决策链——录入、带无泄漏声明的数据处理、模型、校准、带覆盖保证的共形集合、安全网、资源映射、优先级、处置。旁边是数据 provenance，以及多方监督：本例需要分诊护士和高级医师共同复核，每一项都有明确触发条件。》

**Action:** slowly scroll so all three lower cards (Decision chain / Data provenance / Multi-party oversight)
are fully visible.

## Scene 3 — Operations & accountability (1:35–2:20)
**Screen:** Ops console tab, then Audit log tab.
**Narration:**
> "A single patient isn't enough — triage is operational. The Ops console turns the whole cohort into expected
> admissions, bed pressure, a resource-bucket mix, and per-shift flow. And because every recommendation is a
> *decision a human acts on*, the Audit log records what the model suggested and what each clinician did with
> it — here a senior physician overriding with a documented reason. That's accountability built in."

《单个病人不够——分诊是运营问题。运营台把整个队列变成预计入院、床位压力、资源分桶和按班次流量。而且因为每条建议都是"人要据此行动的决策"，审计日志记录了模型建议了什么、医生做了什么——这里是一位高级医师带理由的 override。问责是内建的。》

## Scene 4 — Live, interactive inference (2:20–3:20)
**Screen:** switch to tab 8078. Click **Load example** (or type), then **Predict**.
**Narration:**
> "Finally, it's interactive. I'll enter a patient — an 82-year-old with acute confusion and fever, low oxygen.
> Predict. The model suggested ESI 4, but the **safety override fired** on the abnormal physiology and capped
> it at ESI 3; the **conformal set is wide**, so the system honestly **defers to a human**. We see the same
> chain and provenance. As the nurse, I can confirm — or override with a note — and it's written to the audit
> log. The dashboard and this live engine share the exact same model, so they always agree."

《最后是可交互的。我输入一位 82 岁、急性意识模糊伴发热、低血氧的病人。预测——模型本来给 ESI 4，但异常生理触发了安全网压到 ESI 3；共形集合很宽，系统诚实地"转人工"。同样的链和 provenance 都在。作为护士我可以确认，或带理由 override，并写入审计日志。仪表盘和这个实时引擎用的是同一个模型，所以永远一致。》

**Action:** after Predict, click **Override**, type a reason ("clinical concern — escalate"), show the
"logged audit #…" confirmation, then open `/audit` in a new tab briefly.

## Scene 5 — Honest close (3:20–3:40)
**Screen:** GitHub README (architecture + comparison table) or the WRITEUP limitations section.
**Narration:**
> "To be clear about scope: this is a research prototype on synthetic data — not a clinical device. It has no
> regulatory clearance, EHR integration, or external validation. What it demonstrates is the *discipline* a
> trustworthy triage tool needs: calibrated, uncertainty-aware, fair, explainable, and supervised. Thanks for
> watching."

《说清楚边界：这是合成数据上的研究原型，不是临床设备，没有监管认证、EHR 集成或外部验证。它展示的是一个可信分诊工具该有的"纪律"：校准、知道自己的不确定性、公平、可解释、可监督。感谢观看。》

---

## Shot list (quick reference)
| # | Tab | Action | ~sec |
|---|---|---|---|
| 1 | 8077 Queue | scroll, toggle policy | 35 |
| 2 | 8077 Patient | open ESI-1 row, scroll to governance cards | 60 |
| 3 | 8077 Ops → Audit | show KPIs, then audit table | 45 |
| 4 | 8078 Live | Load example → Predict → Override → /audit | 60 |
| 5 | GitHub/WRITEUP | limitations / comparison | 20 |

## GIF-only fallback (if you can't record voice/video)
Capture 5–6 screenshots (the scenes above) and assemble an animated GIF:
```
# put PNGs as 01.png … 06.png in a folder, then:
# (ImageMagick)   magick -delay 250 -loop 0 0*.png demo.gif
# or use ezgif.com / Kap (https://getkap.co) to record a short silent screen GIF
```
Embed `demo.gif` in the notebook and the GitHub README. Even a silent 30-second GIF of Scene 2 + Scene 4
communicates the decision chain and the live override clearly.

## What to actually submit
- The **video** (or GIF) goes in the **Notebook** (embed/link) and the **Writeup**, as the proof-of-concept.
- Keep it ≤ 4 min; lead with the honesty thesis; end with limitations. That framing is the differentiator.
