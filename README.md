<p align="center">
<img align="center" src="./media/agentic_seek_logo.png" width="200" height="200" alt="AgentGuard Logo">
</p>

<h1 align="center">AgentGuard</h1>
<h3 align="center">Runtime Compliance Layer for Agentic AI in Regulated Industries</h3>

<p align="center">
  <a href="https://agentguard.ai"><img src="https://img.shields.io/static/v1?label=Website&message=agentguard.ai&color=blue&style=flat-square" alt="Website"></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="License">
  <img src="https://img.shields.io/badge/latency-%3C50ms-brightgreen" alt="Latency">
  <img src="https://img.shields.io/badge/status-2%20Binding%20LOIs-orange" alt="Status">
</p>

---

> **"Can you today, in one day, show a regulator the complete decision trail of your AI agent for the past year?"**
>
> If the answer is "no" or "not sure" -- you are exposed.

---

## Why AgentGuard Exists

Every bank deploying AI agents in production faces the same regulatory cliff:

- **EU AI Act** enforcement begins 2026 -- high-risk AI systems require real-time auditability
- **DORA** (Digital Operational Resilience Act) demands resilience testing on production traffic
- **BaFin Memo 2025-12-07** introduces vendor concentration risk requirements for AI providers

There is no product on the market that enforces compliance **at runtime**, in real time, with sub-50ms latency. Until now.

---

## Three Questions Every Regulated AI Deployment Must Answer

| # | Question | Without AgentGuard | With AgentGuard |
|---|----------|--------------------|-----------------|
| 1 | **Who explains to the regulator why AI denied a customer's credit application?** | Manual log review, weeks of preparation, risk of inaccuracies | Automatic immutable audit log + compliant soft-rewrite + one-click PDF report |
| 2 | **How do you prove the agent never uses prohibited attributes (age, gender, ethnicity)?** | Periodic testing; incidents surface after the fact | Real-time check on every response: rewrite / block / escalate **before** it reaches the customer |
| 3 | **How do you manage vendor concentration risk (BaFin)?** | Documentation, SLAs, and hope | Independent runtime layer on top of any LLM -- switch providers without losing compliance |

**This is not "yet another tool." This is mandatory runtime insurance for production AI agents in banking.**

---

## What AgentGuard Does

AgentGuard is a thin **inline proxy** that:

- **Intercepts** every LLM request/response in real time
- **Checks** for bias, PII leakage, hallucinations, and internal policy violations
- **Enforces** the chosen strategy: `monitor-only` | `soft-rewrite` | `hard-block` | `escalate`
- **Logs** a complete immutable audit trail and generates regulator-ready reports

### Latency

| Metric | Value |
|--------|-------|
| Average | **42 ms** |
| p95 | **68 ms** |
| SLA (binding LOI) | **< 50 ms** |

*Benchmarked on GPT-4o + RAG + multi-agent debate pipeline.*

> For context: Deutsche Bank rejected Microsoft Copilot integration when latency exceeded 150 ms. We are 3x under that threshold.

### Unique Capabilities

| Capability | What It Means |
|------------|---------------|
| **Active Compliance Remediation** | Soft-rewrite transforms non-compliant outputs into compliant versions -- preserving UX instead of blocking |
| **Auto-Updating Regulation Base** | EU AI Act, DORA, BaFin rules auto-ingested with full versioning -- no manual updates |
| **Predictive Risk Dashboard** | ROI calculator showing potential fine exposure, saved incidents, and cost avoidance |
| **Vendor Concentration Alerts** | BaFin-specific monitoring of LLM provider drift and dependency thresholds |
| **One-Click Regulator Reports** | PDF generation formatted for BaFin / EU AI Act audits |

---

## Competitive Landscape

| Bank's Decision Criteria | AgentGuard | Lakera + Credo AI | NeMo Guardrails | Why This Matters to the Bank |
|--------------------------|------------|-------------------|-----------------|------------------------------|
| **Prevents fines in real time** | Block + **auto-rewrite** biased outputs | Block only (breaks UX, customer complaints) | Manual rules, no banking focus | One incident = EUR 15-35M + reputational damage. Rewrite preserves UX and compliance. |
| **BaFin vendor risk compliance** | **Predictive vendor concentration alerts** | No | No | Banks must demonstrate reduced dependency on a single AI vendor. |
| **DORA resilience: production latency** | **< 50 ms SLA (42 ms avg, p95 < 70)** | Typically > 120 ms | Unpredictable under load | Above 100 ms, banks cut the agent. Only layer with a binding latency SLA. |
| **Regulatory audit time** | **One-click PDF for BaFin / EU AI Act** | Manual report assembly | No specialized templates | Audit costs EUR 250k-500k/year. Reducing weeks to hours = measurable ROI. |
| **Zero-code integration** | **API proxy, deploy in 1 day** | Requires DevOps and code changes | Requires integration into NVIDIA stack | Banks cannot wait 6+ months for integration. Speed = competitive advantage. |

---

## Architecture (No Magic)

```
User Prompt --> Inline Proxy --> Multi-Agent Core --> Safe Compliant Response + Immutable Log
```

- **Multi-LLM Debate** for hallucination reduction
- **Advanced RAG** over regulations + internal bank policies
- **Policy Engine** -- YAML/JSON configurable rules (tunable enforcement modes)
- **Regulation Updater** -- RSS + scraping of new regulatory publications

**Stack:** Low-Latency Architecture (Python 3.11+, FastAPI, LangGraph Agents) built for SLA guarantees.

### Key Modules (For Tech Due Diligence)

| Module | Function |
|--------|----------|
| `proxy.py` | Inline runtime proxy |
| `policy_engine.py` | Configurable bias, PII, toxicity policies |
| `rag_engine.py` | RAG over regulations and internal policies |
| `predictive_risk.py` | ROI dashboard and fine exposure calculator |
| `updater.py` | Auto-update of regulation base |

*Full project structure and technical documentation available in Data Room.*

---

## Pilot Format for Banks

**4 weeks. 1 high-risk use case** (credit explanations / collections / customer support).

| Week | Activity |
|------|----------|
| 1 | `monitor-only` -- collect baseline incident data |
| 2-4 | `soft-rewrite` on partial traffic -- measure incident reduction |

**Binding SLA:** >= 95% reduction in potential compliance incidents by agreed metrics.

**Success = automatic conversion to paid contract ($150k+ ARR).**

---

## Current Status

| Milestone | Status |
|-----------|--------|
| MVP with live demo (42 ms latency proof) | **Done** |
| 2 binding LOIs from Tier-2 EU banks | **Signed** (names under NDA) |
| First paid contract ($150k ARR) | **Signing January 2026** |
| Latency benchmarks + live demo | **Available on request** |
| Advisory Board: Ex-BaFin AI Officer | **Confirmed** |

---

## For Investors and Partners

AgentGuard closes the critical gap between **"AI agent in production"** and **"AI agent that survives an EU AI Act audit."**

### Investment Thesis: Last Entry Window (Deadline Q1 2026)

| Element | Detail |
|---------|--------|
| **Market Timing** | Regulatory pressure has reached the point of no return. Banks are *obligated* to buy. The window to enter RegTech-AI as the first dedicated runtime player is closing. |
| **Monetization** | SaaS subscription ($150k-500k ARR per bank) -- **pricing based on saved exposure** |
| **Raise** | $4M Seed at $20M pre-money (24-month runway to profitability 2027) |
| **Use of Funds** | Immediate hire of 4-5 core engineers (ML / Low-Latency / RegTech) to build the moat |
| **Exit Path** | Natural acquisition target for ServiceNow (post-Moveworks $2.85B acquisition -- they need a compliance layer for banking) |

### Execution Proof

> We don't just have an idea. We have proven the ability to **execute**.

- **Working MVP** with sub-50ms latency on production-grade pipelines
- **Binding LOIs** with contractual SLA commitments
- **Advisory Board** reinforced with **Ex-BaFin AI Officer** (confirms regulatory defensibility and traction)
- **First revenue date** committed: January 2026

### Why Invest Now

> Not investing = losing share in a market that **is legally required to exist.**

The EU AI Act does not create optional demand. It creates **mandatory procurement budgets** at every financial institution running AI in production. AgentGuard is the only runtime enforcement layer purpose-built for this moment.

---

## Call to Action: 15-Minute Risk Check

We don't discuss hypotheses. We discuss **contractual obligations** (Binding LOI).

| Action | Link |
|--------|------|
| **Request 90-day zero-cost pilot** | [Contact us](mailto:igor@agentguard.ai) |
| **Get 3-slide teaser deck** | [Request access](mailto:igor@agentguard.ai) |
| **Schedule a call** | [Book 15 min](mailto:igor@agentguard.ai) |

> I'll show you the **Live Demo (42ms Latency Proof)** and provide access to the **Data Room** (Latency Benchmarks, LOI, Regulatory Analysis).

---

**Igor Smelyanski**
Founder & CEO | *Visionary & Deep Problem Expert*

[Email](mailto:igor@agentguard.ai) | [LinkedIn](#) | [Phone](#) | [agentguard.ai](https://agentguard.ai)

**Post-Moveworks era starts now. Be ready for Q1 2026.**
