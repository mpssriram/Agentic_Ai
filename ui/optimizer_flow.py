from __future__ import annotations

from ui.components import safe_float


def build_attempt_chart_rows(attempts: list[dict]) -> list[dict]:
    return [
        {
            "Attempt": f"Attempt {attempt['attempt']}",
            "Open Rate": safe_float(attempt.get("metrics", {}).get("open_rate", 0) or 0),
            "Click Rate": safe_float(attempt.get("metrics", {}).get("click_rate", 0) or 0),
            "Score": safe_float(attempt.get("score", 0) or 0),
        }
        for attempt in attempts
    ]


def build_attempt_summaries(attempts: list[dict]) -> list[str]:
    summaries: list[str] = []
    for attempt in attempts:
        attempt_metrics = attempt.get("metrics", {}) or {}
        rows = [
            f"Campaign ID: {attempt.get('campaign_id', '-')}",
            f"Open Rate: {attempt_metrics.get('open_rate', 0)}%",
            f"Click Rate: {attempt_metrics.get('click_rate', 0)}%",
            f"Report Rows: {attempt_metrics.get('recipient_count', attempt_metrics.get('total_rows', 0))}",
            f"Performance Score: {attempt.get('score', 0)}",
        ]
        summaries.append("\n".join(rows))
    return summaries
