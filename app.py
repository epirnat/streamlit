"""
Support Ticket Triage & Escalation Copilot: Streamlit app.

Run from the project folder:  streamlit run app.py
All data is synthetic (fictional company: Kelvara Components).
Numbers are shown in European notation (1.234,5 and 1.234 €). The CSV itself keeps the
international format (dot decimals) on purpose, which is best practice for data files.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import analysis as an
import triage as tr

st.set_page_config(page_title="Support Triage Copilot", page_icon="🎧", layout="wide")

TEAL, AMBER, SLATE, LIGHT = "#0F766E", "#F59E0B", "#64748B", "#CBD5E1"
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [TEAL, AMBER, "#6366F1", "#EF4444", "#14B8A6", "#8B5CF6", "#F97316", SLATE]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1250px;}
    h1 {font-weight: 800; letter-spacing: -0.5px;}
    .kpi {background: white; border-radius: 14px; padding: 16px 20px; border: 1px solid #E2E8F0;
          box-shadow: 0 1px 3px rgba(15,23,42,0.06); height: 170px; box-sizing: border-box; overflow: hidden;
          display: flex; flex-direction: column;}
    .kpi .label {color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                 letter-spacing: .4px; line-height: 1.25; min-height: 2.5em;}
    .kpi .value {color: #0F172A; font-size: 2.2rem; font-weight: 800; line-height: 1.1; min-height: 2.9rem;
                 display: flex; align-items: center;}
    .kpi .value.small {font-size: 1.45rem;}
    .kpi .sub {color: #64748B; font-size: 0.85rem; line-height: 1.3; margin-top: auto;}
    .kpi.accent {border-left: 6px solid #0F766E;}
    .kpi.warn {border-left: 6px solid #F59E0B;}
    .finding {background: #ECFDF5; border-radius: 12px; padding: 16px 18px; border: 1px solid #A7F3D0;
              height: 150px; box-sizing: border-box; overflow: hidden;}
    .finding.amber {background: #FFFBEB; border-color: #FDE68A;}
    .finding .big {font-size: 1.5rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;}
    .step {background: white; border-radius: 12px; padding: 14px 16px; border: 1px dashed #94A3B8;
           height: 100px; box-sizing: border-box; overflow: hidden;}
    .tagline {font-size: 1.15rem; color: #475569; margin-top: -0.5rem; margin-bottom: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- European formatting
def eu(x: float, d: int = 0) -> str:
    """1234567.891 -> '1.234.568' (d=0) or '1.234.567,9' (d=1)."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def eur(x: float, d: int = 0) -> str:
    return f"{eu(x, d)} €"


def pct(x: float, d: int = 0) -> str:
    return f"{eu(x * 100, d)} %"


def style(fig, height=360):
    fig.update_layout(separators=",.", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      title_font_size=16, legend_title_text="")
    return fig


def kpi(label, value, sub="", cls="accent"):
    """Fixed-size card so every KPI box looks identical, whatever the text length."""
    size = " small" if len(str(value)) > 11 else ""
    value_html = f'<div class="value{size}">{value}</div>' if value else ""
    st.markdown(
        f'<div class="kpi {cls}"><div class="label">{label}</div>{value_html}<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def euro_table(df: pd.DataFrame, formats: dict, rename: dict):
    """Display a table with European number formatting (values become text)."""
    out = df.copy()
    for col, fmt in formats.items():
        if fmt == "pct":
            out[col] = out[col].map(lambda v: pct(v, 1) if pd.notna(v) else "")
        elif fmt == "eur":
            out[col] = out[col].map(lambda v: eur(v) if pd.notna(v) else "")
        else:
            out[col] = out[col].map(lambda v: eu(v, fmt) if pd.notna(v) else "")
    st.dataframe(out.rename(columns=rename), width="stretch", hide_index=True)


def highlight_colors(values, target):
    return [AMBER if v == target else LIGHT for v in values]


# ----------------------------------------------------------------------------- data
@st.cache_data
def get_data() -> pd.DataFrame:
    return an.load_data()


@st.cache_data
def get_triage_mix(_df: pd.DataFrame) -> dict:
    out = tr.triage_batch(_df.head(3000))
    return out["predicted_automation_level"].value_counts(normalize=True).to_dict()


df = get_data()

st.sidebar.title("🎧 Support Triage Copilot")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Dashboard", "Automation opportunities", "ROI calculator", "Triage copilot", "Rollout plan"],
)
st.sidebar.caption(
    "Synthetic data for a fictional B2B HVAC parts maker (Kelvara Components). "
    "In production, the CSV would be replaced by a ticketing-system API or database."
)

DEFAULTS = dict(adoption=0.6, hourly=35, volume=60_000, impl=25_000)


# ============================================================================ Overview
if page == "Overview":
    st.title("Support Triage Copilot")
    st.markdown(
        '<div class="tagline">Finding which customer-support work can be automated, and proving the business case with data.</div>',
        unsafe_allow_html=True,
    )

    k = an.headline_kpis(df)
    opps = an.automation_opportunities(df, DEFAULTS["adoption"], DEFAULTS["hourly"], DEFAULTS["volume"])
    roi = an.roi_summary(opps, DEFAULTS["impl"], 3)
    mix = get_triage_mix(df)
    automatable = mix.get("Auto-reply", 0) + mix.get("Draft for review", 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Tickets analysed", eu(k["tickets"]), "12 months, 5 regions")
    with c2:
        kpi("Can be automated or pre-drafted", pct(automatable), "of tickets, by the triage copilot")
    with c3:
        kpi("Estimated yearly saving", eur(roi["annual_saving_eur"]), "automating the top 3 ticket types", "warn")
    with c4:
        kpi("Payback time", f"{eu(roi['payback_months'], 1)} months", "at default assumptions", "warn")

    st.write("")
    st.subheader("Three things the data shows")
    reg = an.by_region(df).iloc[0]
    others = an.by_region(df).iloc[1:]["median_first_response_min"].mean()
    cat = an.by_category(df)
    top_cat = cat.iloc[0]
    tech = cat.set_index("category").loc["Technical question"]
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            f'<div class="finding"><div class="big">{pct(top_cat["share_of_volume"])} of tickets</div>'
            f'are "{top_cat["category"]}" and take only about {eu(top_cat["median_handle_min"])} minutes each. '
            f'The easiest thing to automate.</div>',
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            f'<div class="finding amber"><div class="big">{eu(reg["median_first_response_min"] / others, 1)}x slower</div>'
            f'first response in {reg["region"]} than in the other regions, with an SLA breach rate of '
            f'{pct(reg["sla_breach_rate"])}. A coverage gap.</div>',
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            f'<div class="finding"><div class="big">{pct(tech["reopen_rate"])} reopened</div>'
            f'for technical questions (the lowest CSAT of the big categories). '
            f'These need people, not bots.</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("How this project works")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown('<div class="step"><b>1. Data</b><br>Ticket data (synthetic CSV; an API in production)</div>', unsafe_allow_html=True)
    s2.markdown('<div class="step"><b>2. Analysis</b><br>Python + pandas find where time and money go</div>', unsafe_allow_html=True)
    s3.markdown('<div class="step"><b>3. Opportunities</b><br>Each ticket type scored on value and feasibility</div>', unsafe_allow_html=True)
    s4.markdown('<div class="step"><b>4. Triage copilot</b><br>Classifies tickets, suggests priority, drafts replies</div>', unsafe_allow_html=True)
    st.caption(
        "The savings figures are a model built on stated assumptions (ticket volume, adoption, agent cost), "
        "not measured results. Change them on the ROI page."
    )


# ============================================================================ Dashboard
elif page == "Dashboard":
    st.title("Support performance")
    k = an.headline_kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Tickets", eu(k["tickets"]))
    with c2:
        kpi("First response", f"{eu(k['median_first_response_min'])} min", "median")
    with c3:
        kpi("Resolution", f"{eu(k['median_resolution_hours'], 1)} h", "median")
    with c4:
        kpi("SLA breaches", pct(k["sla_breach_rate"], 1), "first response too slow", "warn")
    with c5:
        kpi("CSAT", f"{eu(k['avg_csat'], 2)} / 5", "customer satisfaction, scale 1 to 5")
    st.caption("About 35 % of tickets have no CSAT rating, as in real customer surveys.")

    cat = an.by_category(df)
    reg = an.by_region(df)
    chn = an.by_channel(df)
    wk = an.by_weekday(df)

    left, right = st.columns(2)
    fig = px.bar(cat, x="tickets", y="category", orientation="h", title="Tickets by category")
    fig.update_traces(marker_color=[AMBER if i == 0 else LIGHT for i in range(len(cat))])
    left.plotly_chart(style(fig.update_yaxes(autorange="reversed", title="")), width="stretch")

    fig = px.bar(reg, x="region", y="median_first_response_min", title="Median first response by region (minutes)")
    fig.update_traces(marker_color=highlight_colors(reg["median_first_response_min"], reg["median_first_response_min"].max()))
    right.plotly_chart(style(fig.update_xaxes(title="")), width="stretch")

    left, right = st.columns(2)
    fig = px.bar(chn, x="channel", y="median_first_response_min", title="Median first response by channel (minutes)")
    fig.update_traces(marker_color=highlight_colors(chn["median_first_response_min"], chn["median_first_response_min"].max()))
    left.plotly_chart(style(fig.update_xaxes(title="")), width="stretch")

    fig = px.bar(wk, x="weekday", y="tickets", title="Tickets by weekday")
    fig.update_traces(marker_color=highlight_colors(wk["tickets"], wk["tickets"].max()))
    right.plotly_chart(style(fig.update_xaxes(title="")), width="stretch")

    st.plotly_chart(
        style(px.line(an.monthly_trend(df), x="month", y="tickets", markers=True, title="Tickets per month"), 320),
        width="stretch",
    )

    with st.expander("Category detail table"):
        euro_table(
            cat,
            {"tickets": 0, "median_handle_min": 0, "total_handle_hours": 0, "median_resolution_h": 1,
             "sla_breach_rate": "pct", "reopen_rate": "pct", "avg_csat": 2, "share_of_volume": "pct"},
            {"category": "Category", "tickets": "Tickets", "median_handle_min": "Handle time (min)",
             "total_handle_hours": "Total handle hours", "median_resolution_h": "Resolution (h)",
             "sla_breach_rate": "SLA breaches", "reopen_rate": "Reopened", "avg_csat": "CSAT (1-5)",
             "share_of_volume": "Share of volume"},
        )


# ============================================================================ Opportunities
elif page == "Automation opportunities":
    st.title("Automation opportunities")
    st.caption(
        "Each ticket type is scored on value (hours saved per month) and feasibility (how repeatable the answer is). "
        "Change the assumptions to see how the ranking reacts."
    )
    a, b, c = st.columns(3)
    adoption = a.slider("Adoption rate", 0.1, 1.0, DEFAULTS["adoption"], 0.05, format="%.2f")
    hourly = b.slider("Agent cost (€ / hour)", 15, 80, DEFAULTS["hourly"])
    volume = c.slider("Tickets per year (real organisation)", 10_000, 200_000, DEFAULTS["volume"], 5_000)

    opps = an.automation_opportunities(df, adoption, hourly, volume)
    best = opps.iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Best first target", best["category"], f"priority rank 1 of {len(opps)}")
    with c2:
        kpi("Hours saved per month", eu(best["hours_saved_per_month"]), "for that category alone")
    with c3:
        kpi("Recommended solution", "", best["recommended_pattern"], "warn")

    fig = px.scatter(
        opps, x="repeatability", y="hours_saved_per_month", size="monthly_tickets", color="category",
        title="Value vs. feasibility (bubble size = tickets per month)",
        labels={"repeatability": "Feasibility (how repeatable)", "hours_saved_per_month": "Hours saved per month"},
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 440), width="stretch")
    st.info("Top right = quick wins. Repeatability scores are assumptions to validate with agents, not findings from the data.")

    euro_table(
        opps[["rank", "category", "monthly_tickets", "median_handle_min", "repeatability",
              "hours_saved_per_month", "monthly_saving_eur", "recommended_pattern"]],
        {"monthly_tickets": 0, "median_handle_min": 0, "repeatability": "pct",
         "hours_saved_per_month": 0, "monthly_saving_eur": "eur"},
        {"rank": "Rank", "category": "Category", "monthly_tickets": "Tickets / month",
         "median_handle_min": "Handle time (min)", "repeatability": "Repeatability",
         "hours_saved_per_month": "Hours saved / month", "monthly_saving_eur": "Saving / month",
         "recommended_pattern": "Recommended solution"},
    )


# ============================================================================ ROI
elif page == "ROI calculator":
    st.title("ROI calculator")
    st.caption("Move the sliders. Every number here is an assumption you can challenge.")
    c1, c2 = st.columns(2)
    adoption = c1.slider("Adoption rate", 0.1, 1.0, DEFAULTS["adoption"], 0.05, format="%.2f")
    hourly = c1.slider("Agent cost (€ / hour)", 15, 80, DEFAULTS["hourly"])
    volume = c2.slider("Tickets per year", 10_000, 200_000, DEFAULTS["volume"], 5_000)
    impl = c2.slider("Implementation cost (€)", 5_000, 150_000, DEFAULTS["impl"], 1_000)
    top_n = st.slider("Automate the top N ticket types", 1, 8, 3)

    opps = an.automation_opportunities(df, adoption, hourly, volume)
    r = an.roi_summary(opps, impl, top_n)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi("Hours saved / month", eu(r["hours_saved_per_month"]))
    with m2:
        kpi("Annual saving", eur(r["annual_saving_eur"]))
    with m3:
        kpi("Implementation cost", eur(impl), "", "warn")
    with m4:
        payback = "n/a" if r["payback_months"] == float("inf") else f"{eu(r['payback_months'], 1)} months"
        kpi("Payback", payback, "", "warn")
    st.write("**Ticket types automated:** " + ", ".join(r["top_categories"]))

    months = list(range(0, 25))
    cum = [-impl + r["monthly_saving_eur"] * m for m in months]
    fig = px.line(x=months, y=cum, labels={"x": "Months after launch", "y": "Cumulative net benefit (€)"},
                  title="Cumulative net benefit")
    fig.add_hline(y=0, line_dash="dash", line_color=SLATE)
    st.plotly_chart(style(fig, 380), width="stretch")
    st.warning("This is a model, not a forecast. A pilot would replace these assumptions with measured values.")


# ============================================================================ Triage copilot
elif page == "Triage copilot":
    st.title("Triage copilot")
    st.caption(
        "Paste a raw customer message. The copilot predicts the category, suggests a priority, decides how much "
        "can be automated, recommends escalation, and drafts a reply. Rules-based and fully local."
    )
    tab1, tab2 = st.tabs(["Single ticket", "Batch (CSV)"])

    with tab1:
        examples = {
            "Order status": "Hi, could you tell me where my order PO-483920 is? It was supposed to ship last week.",
            "Late delivery (urgent)": "Order PO-118204 is 9 days late and our installation is on hold. Please escalate.",
            "Technical question": "Is the valve compatible with our existing controller? We are retrofitting a cold room.",
            "Messy / unclear": "Following up on my last mail. Nobody answered. Thanks",
        }
        choice = st.selectbox("Load an example", ["(write your own)"] + list(examples))
        text = st.text_area("Customer message", value=examples.get(choice, ""), height=110)
        segment = st.selectbox("Customer segment", ["Distributor", "Installer", "OEM"])
        if st.button("Triage this ticket", type="primary") and text.strip():
            res = tr.triage_ticket(text, segment)
            a, b, c, d = st.columns(4)
            with a:
                kpi("Category", res["category"])
            with b:
                kpi("Confidence", pct(res["confidence"]))
            with c:
                kpi("Priority", res["priority"], "", "warn" if res["priority"] in ("High", "Critical") else "accent")
            with d:
                kpi("Automation", res["automation_level"])
            st.write("")
            st.write(f"**Escalation:** {res['escalation']}")
            if res["confidence"] == 0:
                st.error("No confident match. Routed to a human agent (safe fallback). No draft is sent.")
            else:
                st.text_area("Draft reply", res["draft_reply"], height=200)
                if res["automation_level"] == "Auto-reply":
                    st.success("Eligible for automatic reply.")
                elif res["automation_level"] == "Draft for review":
                    st.info("An agent reviews and sends this draft.")

    with tab2:
        st.write("Upload a CSV with a `message_text` column (optional: `customer_segment`), or use the sample file.")
        up = st.file_uploader("CSV file", type="csv")
        use_sample = st.checkbox("Use data/incoming_sample.csv", value=up is None)
        inc = pd.read_csv(up) if up is not None else (pd.read_csv("data/incoming_sample.csv") if use_sample else None)
        if inc is not None and "message_text" in inc.columns:
            out = tr.triage_batch(inc)
            dist = out["predicted_automation_level"].value_counts().reset_index()
            dist.columns = ["automation_level", "tickets"]
            l, r_ = st.columns(2)
            l.plotly_chart(style(px.pie(dist, names="automation_level", values="tickets",
                                        title="Tickets by automation level", hole=0.45)), width="stretch")
            catc = out["predicted_category"].value_counts().reset_index()
            r_.plotly_chart(style(px.bar(catc, x="count", y="predicted_category", orientation="h",
                                         title="Predicted categories").update_yaxes(autorange="reversed", title="")),
                            width="stretch")
            cols = ["message_text", "predicted_category", "predicted_confidence", "predicted_priority",
                    "predicted_automation_level", "predicted_escalation"]
            st.dataframe(out[cols], width="stretch", hide_index=True)
            st.download_button("Download triaged CSV", out.to_csv(index=False), "triaged_tickets.csv", "text/csv")
        elif inc is not None:
            st.error("The CSV needs a `message_text` column.")


# ============================================================================ Rollout plan
else:
    st.title("Rollout plan: from prototype to production")
    st.markdown(
        """
### Architecture

**Prototype (this app):** `tickets.csv` → pandas analysis → Streamlit → rules-based triage

**Production:**
`Ticketing system / CRM (API)` → `Scheduled pipeline` → `Data warehouse`
→ `Triage service (rules, later a trained model)` → `Agent workspace (suggested category, priority, draft)`
→ `Monitoring dashboard`

### Phases

**1. Pilot (6 to 8 weeks).** Start with the top automation candidates only (order status, returns). Triage runs in
*suggestion mode*: agents see the proposed category and draft, and accept or edit. Nothing is sent automatically.

**2. Measure.** Compare pilot agents to a control group on handle time, first-response time, SLA breaches, reopen
rate, CSAT, and agent acceptance of suggestions. Replace every assumption in the ROI calculator with a measured value.

**3. Scale.** Enable auto-reply only for categories with high acceptance and no CSAT drop. Extend to the next
categories. Add a feedback loop: agent corrections become training data.

### Guardrails
- Low-confidence tickets always go to a human agent.
- No auto-reply for high-priority or critical tickets, or for warranty and technical cases.
- Every automated reply is logged and sampled for quality review weekly.
- Data privacy review (GDPR) before connecting real customer data.

### How success is measured
| Metric | Target direction |
|---|---|
| Agent adoption (suggestions accepted) | Above 60 % |
| Handle time on automated categories | Down 30 % or more |
| First-response time | Down, especially in the slowest region and channel |
| SLA breach rate | Down |
| CSAT | Stable or up (must not drop) |

### Reusable pattern
The same flow (data analysis, opportunity scoring, assisted triage, measured pilot) applies to sales operations:
quote follow-up, order-intake emails, and lead routing.
        """
    )
