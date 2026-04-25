# Finance Bot — News-Validation Agent Design (v1)

This document specifies the v1 contract for the news-validation agent. It is the
authoritative reference for subsequent implementation commits. No code is
modified by the commit that introduces this document.

---

## 1. Purpose & Scope

The agent evaluates each BUY/SELL ENTRY signal produced by the model against
recent news, producing one of three qualitative decisions per signal —
CONFIRM, VETO, or ABSTAIN. v1 is **information-only**: the agent's output is
written to `data/agent_runs/{run_date}.json` for downstream analysis and does
**not** gate any trading action taken by `live_trader.py`. HOLD signals are
**not evaluated in v1** — only rows where `row_type=ENTRY` and
`model_signal IN (BUY, SELL)` are passed to the agent.

---

## 2. Execution Context

- **Trigger**: GitHub Actions, daily at **20:30 UTC**.
- **Runtime**: GitHub Actions `ubuntu-latest` runner.
- **Persistence**: `agent/runner.py` writes the JSON file to the local
  workspace, then commits it back to the repo using `GITHUB_TOKEN`. The
  workflow must declare `permissions: contents: write` for the commit to
  succeed.
- **Inputs**: `signal_log.csv`, filtered to today's rows where
  `row_type=ENTRY AND model_signal IN (BUY, SELL)`.
- **Outputs**: `data/agent_runs/{run_date}.json`, committed back to the repo
  in the same workflow run.

---

## 3. JSON Schema

The output file is a single JSON object. Comments below are illustrative only
and are **not** part of the file written to disk — the on-disk artifact is
strict JSON.

```jsonc
{
  // ISO date string from datetime.now(timezone.utc).date().isoformat()
  "run_date": "2026-04-25",

  // git short SHA: `git rev-parse --short HEAD`
  "agent_version": "a1b2c3d",

  // Identifier of the LLM serving the decisions (e.g. Groq model id)
  "llm_model": "llama-3.3-70b-versatile",

  // ISO 8601 UTC timestamp when the run began
  "run_started_at": "2026-04-25T20:30:00Z",

  // Wall-clock duration of the run, in seconds (float)
  "run_duration_seconds": 42.7,

  // One entry per ticker the agent intended to process. See per-decision
  // schema below.
  "decisions": [
    {
      // Equity symbol, e.g. "AAPL"
      "ticker": "AAPL",

      // Literal "BUY" or "SELL" copied from signal_log.csv
      "model_signal": "BUY",

      // Model probability for the chosen class (float, 0–1)
      "confidence": 0.71,

      // Ordered list of feature-name strings from signal_log.csv columns
      // shap_driver_1, shap_driver_2, shap_driver_3 (top 3, in order).
      "shap_values": ["RSI_14", "MACD", "sent_rolling_7d"],

      // The query string sent to the news search provider
      "news_query": "AAPL earnings guidance April 2026",

      // News items the agent considered. Each item has url, title, snippet.
      "sources_checked": [
        {
          "url": "https://example.com/article",
          "title": "Apple beats Q2 estimates...",
          "snippet": "Apple Inc. reported quarterly revenue..."
        }
      ],

      // Literal "CONFIRM" | "VETO" | "ABSTAIN", or null on per-decision failure
      "agent_decision": "CONFIRM",

      // Free-text reasoning, MUST follow the template in section 6
      "agent_reasoning": "Evidence for: strong earnings beat and raised guidance reported across two independent outlets. Evidence against: none found. Verdict: CONFIRM.",

      // null on success; populated string when this decision failed
      "error": null
    }
  ],

  // null on success; populated string when the run failed at the top level
  // (in which case `decisions` is an empty array)
  "run_error": null
}
```

Field-by-field rules:

- `run_date` — `datetime.now(timezone.utc).date().isoformat()`.
- `agent_version` — git short SHA, from `git rev-parse --short HEAD`.
- `agent_decision` — literal strings `"CONFIRM"`, `"VETO"`, `"ABSTAIN"`,
  or `null` on per-decision failure.
- `shap_values` — ordered list of feature-name strings (top 3), sourced
  from `signal_log.csv` columns `shap_driver_1`, `shap_driver_2`,
  `shap_driver_3`.
- `sources_checked[]` — each item has keys `url`, `title`, `snippet`.
- `error` — `null` on success; populated string on per-decision failure.
- `run_error` — `null` on success; populated string on top-level failure.

---

## 4. Join Key

Downstream analysis joins the agent JSON to `signal_log.csv` on
`(ticker, run_date)`, filtered to `row_type=ENTRY AND signal IN (BUY, SELL)`.

This composite key is unique because the execution loop in `live_trader.py`
processes each ticker once per pass, and only the model's BUY/SELL ENTRY rows
are agent-relevant. There is therefore at most one ENTRY BUY/SELL row per
`(ticker, run_date)` pair.

The 7-day forward outcome (`result`) is **not** stored in the agent JSON.
It is joined in at analysis time from `signal_log.csv`, since the outcome is
not knowable at the time the agent runs.

---

## 5. Agent Specification

- **Job**: Independent qualitative assessment of recent news against the
  direction of the model's signal. The agent is not retraining the model and
  is not predicting price — it is asking whether the *news context* supports
  or contradicts the model's directional call.
- **News window**: 3 days, anchored to *now* (when the agent runs). This is
  framing **A** — current sentiment vs. the signal direction.
- **Decision rules** (convergent evidence required):
  - **CONFIRM** — News consistently supports the model's signal direction
    across multiple independent items, with no load-bearing contradictions.
  - **VETO** — News consistently contradicts the model's signal direction.
  - **ABSTAIN** — No relevant news found, OR the news is genuinely mixed,
    OR the news is only tangentially relevant and does not speak to the
    trade thesis.
- **Symmetry**: BUY and SELL signals are evaluated identically — the agent
  simply flips the polarity of "supports" vs. "contradicts" depending on the
  signal direction.
- **Default**: ABSTAIN whenever evidence is weak or absent. CONFIRM and VETO
  both require positive evidence; the agent must not guess.

---

## 6. Reasoning Template

`agent_reasoning` MUST follow this exact structural template:

```
Evidence for: <X>. Evidence against: <Y, or 'none found'>. Verdict: <Z>.
```

Where `<X>` and `<Y>` are short natural-language summaries citing the news
items considered, and `<Z>` matches `agent_decision`. This structure is
enforced in two ways:

1. The system prompt explicitly specifies the template.
2. Each of the four few-shot examples (see section 7) demonstrates a
   reasoning string that matches the template.

---

## 7. Prompt Structure

A hybrid system/user split is used:

- **System message** contains:
  - Role definition (independent qualitative news evaluator).
  - Decision rules from section 5.
  - Output format spec (the JSON object the agent must produce, plus the
    reasoning template from section 6).
  - **Four few-shot examples**, covering:
    1. CONFIRM
    2. VETO
    3. ABSTAIN from genuinely mixed evidence
    4. ABSTAIN from peripheral / tangentially-relevant news
- **User message** contains the per-signal instance — `ticker`,
  `model_signal`, `confidence`, `shap_values` — plus minimal task framing
  ("Decide CONFIRM / VETO / ABSTAIN for this signal").

Splitting role + rules + examples into the system message lets the user
message stay short and uniform per ticker, which is friendlier to provider
caching when iterating across many tickers in a single run.

---

## 8. Output Format Enforcement

Strategy: **prompt-based + parse-and-retry**.

- The system prompt instructs the model to emit a JSON object matching the
  per-decision schema.
- The runner parses the response. On parse failure (invalid JSON, missing
  required fields, or `agent_decision` not in the allowed enum), the runner
  retries.
- **Retry budget**: 1 retry, **2 attempts max** per signal.
- **Retry message**: append the parse error to the message history and
  re-request a valid JSON object that matches the schema.
- **Final failure** (both attempts unparseable): write a decision row with
  `agent_decision=null`, populate `error` with the parse failure detail, and
  continue to the next ticker.

---

## 9. Failure Handling

Three layers, from innermost to outermost:

- **API-level retries (Tavily, Groq)** — implemented with `tenacity`,
  **3 attempts max**, exponential backoff. Applies to network failures,
  5xx errors, and 429 rate-limit responses.
- **Per-decision failures** — when API retries are exhausted, OR when the
  JSON-parse retry budget (section 8) is exhausted, the runner writes a
  decision row for that ticker with `agent_decision=null` and a populated
  `error` string, then continues to the next ticker.
- **Top-level failures** — the agent cannot start at all. Examples: missing
  API keys, `signal_log.csv` unreadable, the very first Groq or Tavily call
  fails authentication. In this case the runner writes the JSON file with
  `decisions: []` and `run_error="..."`, sends a Discord notification, and
  exits nonzero.

**Partial-run handling**: every ticker the agent *intended to process* must
appear in `decisions[]`. Successful tickers carry a populated decision;
failed tickers carry `agent_decision=null` and a populated `error`. To make
this guarantee, `runner.py` must enumerate the full ticker work-list
**before** starting the per-ticker iteration loop.

---

## 10. Idempotency

On rerun for the same `run_date`:

- **Existing JSON file is parseable**: rename it to
  `{run_date}.{rename_timestamp_utc}.bak.json`, then proceed with a fresh
  run. Send a minimal Discord notification of the form:

  ```
  ⚠️ Existing {run_date}.json moved to {backup_filename} before rerun
  ```

- **Existing JSON file is unparseable**: **HALT**. Do not rename, do not
  proceed. Send a Discord notification including the parse error. Exit
  nonzero. (Manual investigation required — silently overwriting a corrupt
  artifact would destroy evidence of whatever produced it.)

The backup filename uses the full UTC timestamp at rename time, e.g.

```
2026-04-25.20260425T211500Z.bak.json
```

---

## 11. Secrets & Permissions

Required GitHub Actions secrets:

- `GROQ_API_KEY`
- `TAVILY_API_KEY`
- `DISCORD_WEBHOOK_URL`

Required workflow YAML:

```yaml
permissions:
  contents: write
```

`GITHUB_TOKEN` is provided to the workflow automatically by GitHub Actions;
no separate Personal Access Token is required to commit the JSON artifact
back to the repo.

---

## 12. Out of Scope for v1

Documented explicitly to prevent scope creep:

- HOLD signal evaluation (deferred to v2).
- Multi-query news search per ticker (v1 uses a single query per ticker).
- Feedback loop into trading decisions (v1 is information-only; the agent
  output does not gate `live_trader.py`).
- CrewAI multi-agent orchestration (Phase 2).
- Sentiment veto re-enablement (separate workstream).
