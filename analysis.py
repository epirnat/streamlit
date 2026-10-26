"""
Pandas analysis for the Support Ticket Triage and Escalation Copilot.

All logic lives here as small functions that return DataFrames, so the same code
can be run from the terminal (python analysis.py) and imported by the Streamlit app.
In production, load_data() would read from a database or ticketing-system API
instead of a CSV; nothing else in this file would need to change.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/tickets.csv")

# How repeatable / templated each category is (0-1) and the best-fit automation pattern.
# These are assumptions a real team would validate with agents. Documented here on purpose.
REPEATABILITY = {
    "Order status": 0.95,
    "Account / portal": 0.90,
    "Returns / RMA": 0.75,
    "Invoice / billing": 0.70,
    "Quote request": 0.65,
    "Delivery delay": 0.60,
    "Warranty / defect": 0.40,
    "Technical question": 0.25,
}
SOLUTION_PATTERN = {
    "Order status": "Self-service order tracking + auto-reply with live ERP status",
    "Account / portal": "Self-service password reset and user management",
    "Returns / RMA": "Guided RMA web form with automatic RMA number",
    "Invoice / billing": "Auto-send invoice copies; AI-assisted draft for corrections",
    "Quote request": "Structured quote intake form + AI-drafted quote for review",
    "Delivery delay": "Proactive delay notifications from carrier data",
    "Warranty / defect": "AI triage and evidence checklist; human decision",
    "Technical question": "AI copilot suggests answers from product docs; agent approves",
}


# ----------------------------------------------------------------------------- load
def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["created_at", "closed_at"])
    df["weekday"] = df["created_at"].dt.day_name()
    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    df["hour"] = df["created_at"].dt.hour
    return df


# ----------------------------------------------------------------------------- KPIs
def headline_kpis(df: pd.DataFrame) -> dict:
    return {
        "tickets": len(df),
        "median_first_response_min": float(df["first_response_min"].median()),
        "median_resolution_hours": float(df["resolution_hours"].median()),
        "sla_breach_rate": float(df["sla_breached"].mean()),
        "reopen_rate": float(df["reopened"].mean()),
        "avg_csat": float(df["csat"].mean()),
        "total_handle_hours": float(df["handle_time_min"].sum() / 60),
    }


# ----------------------------------------------------------------------------- breakdowns
def by_category(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("category").agg(
        tickets=("ticket_id", "count"),
        median_handle_min=("handle_time_min", "median"),
        total_handle_hours=("handle_time_min", lambda s: s.sum() / 60),
        median_resolution_h=("resolution_hours", "median"),
        sla_breach_rate=("sla_breached", "mean"),
        reopen_rate=("reopened", "mean"),
        avg_csat=("csat", "mean"),
    )
    g["share_of_volume"] = g["tickets"] / g["tickets"].sum()
    return g.sort_values("tickets", ascending=False).round(3).reset_index()


def by_region(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("region").agg(
        tickets=("ticket_id", "count"),
        median_first_response_min=("first_response_min", "median"),
        sla_breach_rate=("sla_breached", "mean"),
        avg_csat=("csat", "mean"),
    )
    return g.sort_values("median_first_response_min", ascending=False).round(3).reset_index()


def by_channel(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("channel").agg(
        tickets=("ticket_id", "count"),
        median_first_response_min=("first_response_min", "median"),
        sla_breach_rate=("sla_breached", "mean"),
        avg_csat=("csat", "mean"),
    )
    return g.sort_values("median_first_response_min", ascending=False).round(3).reset_index()


def by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    g = df.groupby("weekday").agg(tickets=("ticket_id", "count"), sla_breach_rate=("sla_breached", "mean"))
    return g.reindex(order).round(3).reset_index()


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("month")
        .agg(tickets=("ticket_id", "count"), sla_breach_rate=("sla_breached", "mean"), avg_csat=("csat", "mean"))
        .round(3)
        .reset_index()
    )


def csat_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Average CSAT split by whether the SLA was breached and the ticket reopened."""
    rated = df.dropna(subset=["csat"])
    return (
        rated.groupby(["sla_breached", "reopened"])
        .agg(tickets=("ticket_id", "count"), avg_csat=("csat", "mean"))
        .round(2)
        .reset_index()
    )


# ----------------------------------------------------------------------------- automation scoring
def automation_opportunities(
    df: pd.DataFrame,
    adoption_rate: float = 0.60,
    hourly_cost: float = 35.0,
    annual_volume: int = 60000,
) -> pd.DataFrame:
    """
    Rank ticket categories as automation candidates.

    hours_saved_per_month = monthly handle hours x repeatability x adoption_rate
    priority_score        = hours_saved_per_month x repeatability  (value x feasibility)

    The CSV is a sample, so volumes are scaled to `annual_volume`, the ticket count a
    real commercial support organisation might handle per year (an explicit assumption).
    Repeatability, adoption, hourly cost and volume are all inputs so a stakeholder can
    challenge them, which is the point of the ROI calculator in the app.
    """
    months = max(df["month"].nunique(), 1)
    scale = annual_volume / (len(df) / months * 12)
    g = df.groupby("category").agg(
        tickets=("ticket_id", "count"),
        median_handle_min=("handle_time_min", "median"),
        total_handle_hours=("handle_time_min", lambda s: s.sum() / 60),
    )
    g["monthly_tickets"] = g["tickets"] / months * scale
    g["monthly_handle_hours"] = g["total_handle_hours"] / months * scale
    g["repeatability"] = g.index.map(REPEATABILITY).astype(float)
    g["hours_saved_per_month"] = g["monthly_handle_hours"] * g["repeatability"] * adoption_rate
    g["monthly_saving_eur"] = g["hours_saved_per_month"] * hourly_cost

    # Simple effort proxy: low repeatability = harder to automate. Used for the priority matrix.
    g["effort"] = (1 - g["repeatability"]).round(2) + 0.1
    g["priority_score"] = g["hours_saved_per_month"] * g["repeatability"]
    g["recommended_pattern"] = g.index.map(SOLUTION_PATTERN)

    g = g.sort_values("priority_score", ascending=False)
    g["rank"] = range(1, len(g) + 1)
    return g.round(2).reset_index()


def roi_summary(opps: pd.DataFrame, implementation_cost: float = 25000.0, top_n: int = 3) -> dict:
    top = opps.head(top_n)
    monthly = float(top["monthly_saving_eur"].sum())
    return {
        "top_categories": top["category"].tolist(),
        "hours_saved_per_month": float(top["hours_saved_per_month"].sum()),
        "monthly_saving_eur": monthly,
        "annual_saving_eur": monthly * 12,
        "payback_months": (implementation_cost / monthly) if monthly > 0 else float("inf"),
    }


# ----------------------------------------------------------------------------- terminal report
def main():
    df = load_data()
    k = headline_kpis(df)
    print("=== HEADLINE KPIs ===")
    for key, val in k.items():
        print(f"{key:28s} {val:,.2f}")

    print("\n=== BY CATEGORY ===")
    print(by_category(df).to_string(index=False))
    print("\n=== BY REGION (slowest first response first) ===")
    print(by_region(df).to_string(index=False))
    print("\n=== BY CHANNEL ===")
    print(by_channel(df).to_string(index=False))
    print("\n=== BY WEEKDAY ===")
    print(by_weekday(df).to_string(index=False))

    opps = automation_opportunities(df)
    print("\n=== AUTOMATION OPPORTUNITIES ===")
    print(opps[["rank", "category", "monthly_tickets", "median_handle_min", "repeatability",
                "hours_saved_per_month", "monthly_saving_eur", "priority_score"]].to_string(index=False))

    r = roi_summary(opps)
    print("\n=== ROI (top 3; assumes 60k tickets/yr, 60% adoption, EUR 35/h, EUR 25k implementation) ===")
    print(f"Categories:        {', '.join(r['top_categories'])}")
    print(f"Hours saved/month: {r['hours_saved_per_month']:.0f}")
    print(f"Annual saving:     EUR {r['annual_saving_eur']:,.0f}")
    print(f"Payback:           {r['payback_months']:.1f} months")


if __name__ == "__main__":
    main()
