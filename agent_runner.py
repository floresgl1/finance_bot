import json
import os
import string
import subprocess
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from groq import Groq

from agent_prompts import (
    ALLOWED_VERDICTS,
    RETRY_MESSAGE_TEMPLATE,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)
from agent_tools import SearchNewsError, search_news


MAX_AGENT_STEPS = 5
MAX_REASONING_CAPTURE_CHARS = 500
PARSE_RETRY_BUDGET = 1  # one retry, two attempts max

ERROR_CATEGORIES = frozenset({
    "TOOL_FAILURE",
    "PARSE_EXHAUSTED",
    "STEP_CAP",
    "API_FAILURE",
    "UNEXPECTED",
})

LOSS_LIMIT_HALT_CODES = frozenset({
    "HALT_FLAG_PRESENT",
    "PORTFOLIO_HALT_SINGLE_DAY",
    "PORTFOLIO_HALT_ROLLING",
    "REBALANCER_SKIPPED_HALT",
})
PIPELINE_HALT_CODES = frozenset({
    "STALE_MARKET_DATA",
})

GROQ_MODEL = "llama-3.3-70b-versatile"


_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def _git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _truncate_reasoning(text: str) -> str:
    if len(text) > MAX_REASONING_CAPTURE_CHARS:
        return text[:MAX_REASONING_CAPTURE_CHARS] + "..."
    return text


def _parse_action(content: str) -> dict[str, Any]:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}")

    if not isinstance(obj, dict):
        raise ValueError("response must be a JSON object")

    keys = set(obj.keys())
    if keys != {"action", "action_input"}:
        raise ValueError(
            f"response must have exactly keys {{'action', 'action_input'}}; got {sorted(keys)}"
        )

    action = obj["action"]
    action_input = obj["action_input"]

    if not isinstance(action, str) or action not in {"search_news", "decide"}:
        raise ValueError(f"action must be 'search_news' or 'decide'; got {action!r}")

    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a dict")

    if action == "search_news":
        query = action_input.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("search_news action_input.query must be a non-empty string")
    else:
        decision = action_input.get("agent_decision")
        reasoning = action_input.get("agent_reasoning")
        if not isinstance(decision, str) or decision not in ALLOWED_VERDICTS:
            raise ValueError(
                f"decide action_input.agent_decision must be one of {sorted(ALLOWED_VERDICTS)}; got {decision!r}"
            )
        if not isinstance(reasoning, str):
            raise ValueError("decide action_input.agent_reasoning must be a string")

    return obj


def _determine_run_status(
    signal_log_df: pd.DataFrame,
    work_list: list[dict],
    run_date: str,
) -> tuple[str, list[dict]]:
    pipeline_rows = signal_log_df[
        (signal_log_df["ticker"] == "PIPELINE")
        & (signal_log_df["date"] == run_date)
    ]

    halt_details = [
        {"actual_action": row["actual_action"], "date": run_date}
        for _, row in pipeline_rows.iterrows()
    ]

    actions_today = set(pipeline_rows["actual_action"])

    if actions_today & LOSS_LIMIT_HALT_CODES:
        status = "LOSS_LIMIT_HALTED"
    elif actions_today & PIPELINE_HALT_CODES:
        status = "PIPELINE_HALTED"
    elif len(work_list) == 0:
        status = "NO_SIGNALS"
    else:
        status = "NORMAL"

    return status, halt_details


def build_work_list(signal_log_df: pd.DataFrame, run_date: str) -> list[dict]:
    """Build the agent's work-list from the day's signal log.

    Filters to row_type=ENTRY AND model_signal IN (BUY, SELL) AND date equals run_date.

    Args:
        signal_log_df: DataFrame loaded from signal_log.csv.
        run_date: ISO date string (e.g. "2026-04-26"). The `date` column is a
            YYYY-MM-DD string and is compared directly.

    Returns:
        List of dicts. Each dict has exactly the keys
        {ticker, model_signal, confidence, shap_values}. shap_values is a list
        of three feature-name strings sourced from columns shap_driver_1,
        shap_driver_2, shap_driver_3.
    """
    df = signal_log_df

    mask = (
        (df["row_type"] == "ENTRY")
        & (df["model_signal"].isin(["BUY", "SELL"]))
        & (df["date"] == run_date)
    )
    filtered = df[mask]

    work_list: list[dict] = []
    for _, row in filtered.iterrows():
        work_list.append({
            "ticker": row["ticker"],
            "model_signal": row["model_signal"],
            "confidence": float(row["confidence"]),
            "shap_values": [
                row["shap_driver_1"],
                row["shap_driver_2"],
                row["shap_driver_3"],
            ],
        })
    return work_list


def _new_decision_row(work_item: dict) -> dict[str, Any]:
    return {
        "ticker": work_item["ticker"],
        "model_signal": work_item["model_signal"],
        "confidence": work_item["confidence"],
        "shap_values": work_item["shap_values"],
        "news_query": None,
        "sources_checked": [],
        "agent_decision": None,
        "agent_reasoning": None,
        "error_category": None,
        "error": None,
    }


def process_ticker(work_item: dict) -> dict:
    """Run the ReAct loop for one ticker and return its decision row.

    The decision row matches the per-decision schema documented in
    docs/AGENT_DESIGN.md §3, with the addition of an `error_category` field.

    Args:
        work_item: dict with keys {ticker, model_signal, confidence, shap_values}.

    Returns:
        dict with keys {ticker, model_signal, confidence, shap_values, news_query,
        sources_checked, agent_decision, agent_reasoning, error_category, error}.
        On success: agent_decision is in ALLOWED_VERDICTS, error_category is None,
        error is None. On failure: agent_decision is None, error_category is in
        ERROR_CATEGORIES, error is a populated string.
    """
    row = _new_decision_row(work_item)

    try:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                ticker=work_item["ticker"],
                model_signal=work_item["model_signal"],
                confidence=work_item["confidence"],
                shap_values=work_item["shap_values"],
            )},
        ]

        parse_retry_budget = PARSE_RETRY_BUDGET
        last_assistant_content: str | None = None
        last_news_query: str | None = None
        last_sources_checked: list[dict] = []

        for _step in range(MAX_AGENT_STEPS):
            try:
                completion = _client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                )
            except Exception as e:
                row["error_category"] = "API_FAILURE"
                row["error"] = f"groq API failed: {type(e).__name__}: {e}"
                row["news_query"] = last_news_query
                row["sources_checked"] = last_sources_checked
                return row

            content = completion.choices[0].message.content
            last_assistant_content = content
            messages.append({"role": "assistant", "content": content})

            try:
                action_obj = _parse_action(content)
            except ValueError as parse_err:
                if parse_retry_budget > 0:
                    parse_retry_budget -= 1
                    messages.append({
                        "role": "user",
                        "content": RETRY_MESSAGE_TEMPLATE.format(parse_error=str(parse_err)),
                    })
                    continue
                row["error_category"] = "PARSE_EXHAUSTED"
                row["agent_reasoning"] = _truncate_reasoning(content)
                row["error"] = f"parse retry exhausted after 2 attempts: {parse_err}"
                row["news_query"] = last_news_query
                row["sources_checked"] = last_sources_checked
                return row

            action = action_obj["action"]
            action_input = action_obj["action_input"]

            if action == "search_news":
                query = action_input["query"]
                try:
                    results = search_news(query)
                except SearchNewsError as e:
                    row["error_category"] = "TOOL_FAILURE"
                    row["agent_reasoning"] = None
                    row["error"] = f"search_news failed: {e}"
                    row["news_query"] = query
                    row["sources_checked"] = []
                    return row
                last_news_query = query
                last_sources_checked = results
                messages.append({
                    "role": "user",
                    "content": json.dumps({"observation": results}),
                })
                continue

            row["agent_decision"] = action_input["agent_decision"]
            row["agent_reasoning"] = action_input["agent_reasoning"]
            row["news_query"] = last_news_query
            row["sources_checked"] = last_sources_checked
            return row

        row["error_category"] = "STEP_CAP"
        row["agent_reasoning"] = (
            _truncate_reasoning(last_assistant_content)
            if last_assistant_content is not None
            else None
        )
        row["error"] = f"step cap reached after {MAX_AGENT_STEPS} steps without verdict"
        row["news_query"] = last_news_query
        row["sources_checked"] = last_sources_checked
        return row

    except Exception as e:
        row["error_category"] = "UNEXPECTED"
        row["agent_reasoning"] = None
        row["error"] = f"unexpected error: {type(e).__name__}: {e}"
        return row


def run_agent(signal_log_path: str, run_date: str) -> dict:
    """Run the agent end-to-end for run_date and return the artifact dict.

    Does NOT write to disk — that's the workflow's job (commit #4).

    Args:
        signal_log_path: path to signal_log.csv
        run_date: ISO date string for filtering and metadata.

    Returns:
        Artifact dict matching docs/AGENT_DESIGN.md §3 (with the error_category
        addition). On success: decisions populated, run_error=None. On
        catastrophic failure: decisions=[], run_error populated.
    """
    run_started = datetime.now(timezone.utc)
    run_started_at = run_started.isoformat().replace("+00:00", "Z")
    agent_version = _git_short_sha()

    try:
        df = pd.read_csv(signal_log_path)
        work_list = build_work_list(df, run_date)
    except Exception as e:
        run_duration = (datetime.now(timezone.utc) - run_started).total_seconds()
        return {
            "run_date": run_date,
            "agent_version": agent_version,
            "llm_model": GROQ_MODEL,
            "run_started_at": run_started_at,
            "run_duration_seconds": float(run_duration),
            "run_status": "AGENT_ERROR",
            "halt_details": [],
            "decisions": [],
            "run_error": f"workspace setup failed: {type(e).__name__}: {e}",
        }

    run_status, halt_details = _determine_run_status(df, work_list, run_date)

    decisions = [process_ticker(item) for item in work_list]

    run_duration = (datetime.now(timezone.utc) - run_started).total_seconds()
    return {
        "run_date": run_date,
        "agent_version": agent_version,
        "llm_model": GROQ_MODEL,
        "run_started_at": run_started_at,
        "run_duration_seconds": float(run_duration),
        "run_status": run_status,
        "halt_details": halt_details,
        "decisions": decisions,
        "run_error": None,
    }
