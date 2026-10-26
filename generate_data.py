"""
Generate dummy data i.e. support-ticket data for a fictional B2B HVAC parts maker ("Kelvara Components").

Run:  python generate_data.py
Out:  data/tickets.csv

Insights:
  1. "Order status" is the biggest category and is very quick to handle (automation candidate #1).
  2. Mondays are ~35% busier than other weekdays.
  3. Southern Europe has much slower first responses (staffing gap).
  4. Email is the slowest channel; chat is the fastest.
  5. Technical questions take longest and get reopened most.
  6. Long resolution times drive CSAT down.
  7. Cooling-season months (May-Aug) have more tickets.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_TICKETS = 8000
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- categories
# weight = share of volume; handle = median handle minutes; res = median resolution hours
# repeatable = how templated the answer is (0-1); used later for automation scoring
CATEGORIES = {
    "Order status":        dict(weight=0.28, handle=4,  res=2,  priority=[0.05, 0.60, 0.30, 0.05], repeatable=0.95),
    "Delivery delay":      dict(weight=0.14, handle=12, res=18, priority=[0.02, 0.28, 0.50, 0.20], repeatable=0.60),
    "Quote request":       dict(weight=0.12, handle=15, res=14, priority=[0.01, 0.40, 0.50, 0.09], repeatable=0.65),
    "Invoice / billing":   dict(weight=0.11, handle=10, res=24, priority=[0.02, 0.45, 0.45, 0.08], repeatable=0.70),
    "Returns / RMA":       dict(weight=0.10, handle=14, res=30, priority=[0.01, 0.30, 0.55, 0.14], repeatable=0.75),
    "Technical question":  dict(weight=0.12, handle=28, res=40, priority=[0.00, 0.20, 0.50, 0.30], repeatable=0.25),
    "Warranty / defect":   dict(weight=0.07, handle=25, res=60, priority=[0.00, 0.10, 0.40, 0.50], repeatable=0.40),
    "Account / portal":    dict(weight=0.06, handle=6,  res=4,  priority=[0.10, 0.60, 0.25, 0.05], repeatable=0.90),
}
PRIORITIES = ["Low", "Medium", "High", "Critical"]
SLA_FIRST_RESPONSE_MIN = {"Low": 960, "Medium": 480, "High": 120, "Critical": 30}

# ------------------------------------------------------------- other dims
CHANNELS = {"email": 0.45, "portal": 0.20, "phone": 0.20, "chat": 0.15}
CHANNEL_RESPONSE_MULT = {"email": 1.6, "portal": 1.1, "phone": 0.5, "chat": 0.3}

REGIONS = {"DACH": 0.30, "Nordics": 0.18, "UK & Ireland": 0.17, "Southern Europe": 0.20, "Eastern Europe": 0.15}
REGION_RESPONSE_MULT = {"DACH": 1.0, "Nordics": 0.9, "UK & Ireland": 1.0, "Southern Europe": 2.2, "Eastern Europe": 1.2}

SEGMENTS = {"OEM": 0.25, "Distributor": 0.40, "Installer": 0.35}
PRODUCT_LINES = ["Compressors", "Valves", "Controllers", "Sensors", "Heat exchangers", "Drives"]
AGENTS = [f"AG{str(i).zfill(2)}" for i in range(1, 13)]

# ------------------------------------------------------- message templates
TEMPLATES = {
    "Order status": [
        "Hi, could you tell me where my order {po} is? It was supposed to ship last week.",
        "Please provide tracking information for purchase order {po}.",
        "What is the current status of order {po}? Our customer is waiting.",
        "Can you confirm the expected delivery date for {po}?",
        "Order {po} still shows 'processing' in the portal. Any update?",
    ],
    "Delivery delay": [
        "Order {po} is {d} days late and our installation is on hold. Please escalate.",
        "The shipment for {po} has not arrived and the carrier tracking has not updated in days.",
        "We were promised delivery by Friday for {po}. Still nothing. What is happening?",
        "Delivery of {po} is delayed again. We need a firm new date urgently.",
    ],
    "Quote request": [
        "Please send a quote for 50 units of {prod}, delivery to {region}.",
        "We need pricing and lead time for {prod}. Volume discount possible?",
        "Could you quote {prod} for a new project starting next month?",
        "Request for quotation: {prod}, 200 pcs, framework agreement.",
    ],
    "Invoice / billing": [
        "Invoice for order {po} shows the wrong VAT rate. Please issue a corrected invoice.",
        "We were charged twice for {po}. Please check and refund.",
        "Can you send a copy of the invoice for {po}? Our accounting cannot find it.",
        "The price on the invoice does not match our agreed quote for {po}.",
    ],
    "Returns / RMA": [
        "We need to return {prod} from order {po}, wrong item delivered. Please send an RMA number.",
        "Requesting return authorization for damaged goods in {po}.",
        "How do I return unused {prod}? We ordered too many for {po}.",
    ],
    "Technical question": [
        "What is the maximum operating pressure for {prod} with R290 refrigerant?",
        "Is {prod} compatible with our existing controller? We are retrofitting a cold room.",
        "Need wiring and configuration guidance for {prod}; the unit errors out at startup.",
        "Can you provide the datasheet and selection guidance for {prod} in a low-temperature application?",
        "Our {prod} shows intermittent fault codes after commissioning. What should we check?",
    ],
    "Warranty / defect": [
        "{prod} from order {po} failed after 3 weeks. Requesting warranty replacement.",
        "We have a defective batch of {prod}, several units fail on startup. Please investigate.",
        "{prod} is leaking after installation. Need warranty support and a replacement quickly.",
    ],
    "Account / portal": [
        "I cannot log in to the customer portal, password reset email never arrives.",
        "Please add a new user to our company account on the portal.",
        "The portal does not show our latest orders. Can you check our account settings?",
    ],
}


def weighted_choice(d, size):
    keys = list(d.keys())
    p = np.array(list(d.values()), dtype=float)
    return rng.choice(keys, size=size, p=p / p.sum())


def make_timestamps(n):
    """Business-hours timestamps over 12 months with Monday and cooling-season peaks."""
    days = pd.date_range("2025-10-01", "2026-09-30", freq="D")
    days = days[days.weekday < 5]  # weekdays only
    w = np.ones(len(days))
    w[days.weekday == 0] *= 1.35                                # Monday spike
    w[np.isin(days.month, [5, 6, 7, 8])] *= 1.30                # cooling season
    w[(days.month == 12) & (days.day > 22)] *= 0.4              # holidays
    chosen = rng.choice(len(days), size=n, p=w / w.sum())
    hours = np.clip(rng.normal(11, 2.8, n), 7, 18)              # peak late morning
    ts = days[chosen] + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(rng.integers(0, 60, n), unit="m")
    return ts.floor("min")


def main():
    n = N_TICKETS
    cat_names = list(CATEGORIES.keys())
    cat_p = np.array([CATEGORIES[c]["weight"] for c in cat_names])
    category = rng.choice(cat_names, size=n, p=cat_p / cat_p.sum())

    channel = weighted_choice(CHANNELS, n)
    region = weighted_choice(REGIONS, n)
    segment = weighted_choice(SEGMENTS, n)
    product = rng.choice(PRODUCT_LINES, size=n)
    agent = rng.choice(AGENTS, size=n)
    created = make_timestamps(n)

    priority, handle, resolution, first_resp, messages = [], [], [], [], []
    for i in range(n):
        spec = CATEGORIES[category[i]]
        pr = rng.choice(PRIORITIES, p=spec["priority"])
        priority.append(pr)

        # handle time (minutes): lognormal around category median
        handle.append(max(1, round(rng.lognormal(np.log(spec["handle"]), 0.45))))

        # resolution time (hours): lognormal, slower for higher priority? (slightly faster when critical)
        res_mult = 0.7 if pr == "Critical" else 1.0
        resolution.append(round(rng.lognormal(np.log(spec["res"] * res_mult), 0.6), 1))

        # first response (minutes): depends on priority, channel, region
        base = {"Low": 300, "Medium": 150, "High": 50, "Critical": 12}[pr]
        fr = base * CHANNEL_RESPONSE_MULT[channel[i]] * REGION_RESPONSE_MULT[region[i]]
        first_resp.append(max(1, round(rng.lognormal(np.log(fr), 0.6))))

        tpl = rng.choice(TEMPLATES[category[i]])
        messages.append(tpl.format(
            po=f"PO-{rng.integers(100000, 999999)}",
            prod=product[i].lower(),
            region=region[i],
            d=rng.integers(2, 15),
        ))

    df = pd.DataFrame({
        "ticket_id": [f"T{100000 + i}" for i in range(n)],
        "created_at": created,
        "channel": channel,
        "customer_segment": segment,
        "region": region,
        "product_line": product,
        "category": category,
        "priority": priority,
        "message_text": messages,
        "first_response_min": first_resp,
        "resolution_hours": resolution,
        "handle_time_min": handle,
        "agent_id": agent,
    })

    # SLA breach on first response
    df["sla_breached"] = df.apply(
        lambda r: r["first_response_min"] > SLA_FIRST_RESPONSE_MIN[r["priority"]], axis=1
    )

    # reopened: higher for technical and warranty, and when resolution was slow
    reopen_base = df["category"].map({"Technical question": 0.18, "Warranty / defect": 0.14}).fillna(0.05)
    reopen_p = reopen_base + (df["resolution_hours"] > 48) * 0.06
    df["reopened"] = rng.random(n) < reopen_p

    # CSAT 1-5: falls with resolution time, SLA breach and reopen; some tickets have no rating
    score = 4.6 - 0.012 * df["resolution_hours"] - 0.6 * df["sla_breached"] - 0.8 * df["reopened"] + rng.normal(0, 0.6, n)
    csat = np.clip(np.round(score), 1, 5).astype(float)
    csat[rng.random(n) < 0.35] = np.nan  # 35% did not answer the survey
    df["csat"] = csat

    df["closed_at"] = (df["created_at"] + pd.to_timedelta(df["resolution_hours"], unit="h")).dt.floor("min")
    df = df.sort_values("created_at").reset_index(drop=True)

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/tickets.csv", index=False)
    print(f"Wrote {len(df)} tickets to data/tickets.csv")
    print(df.head())


if __name__ == "__main__":
    main()
