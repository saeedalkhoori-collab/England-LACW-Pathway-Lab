# app.py
# Objective 2 — LACW Pathways Lab (England councils, 2020–2023)
# Focus: treatment pathways & emissions; plus Policy65 scenarios that target the **recycling rate**
# (NOT diversion). “Collected” is treated ONLY as a front-end metric (TotalRecCollecYYYY) and
# is NEVER a pathway route.

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Objective 2 — LACW Pathways Lab (2020–2023)",
    layout="wide",
)

# ============================================================
# CONFIG (match your file)
# ============================================================
XLSX_NAME = "pathway_data.xlsx"
DEFAULT_SHEET = "Pathwaydata"
YEARS = [2020, 2021, 2022, 2023]

FRACTIONS = [
    "Food",
    "Garden",
    "PaperCard",
    "Plastics",
    "Glass",
    "Metals",
    "Textiles",
    "WEEE",
    "Wood",
    "OtherRecyclables",
]

# IMPORTANT:
# - NO "Collected" here (it is NOT a pathway).
# - "Collected for recycling" is captured by TotalRecCollecYYYY columns only.
ROUTES = [
    "Recycled",
    "Reuse",
    "EfW",
    "Landfill",
    "RDF_MHT",
    "IncNoEnergy",   # Incineration without energy (IWE / R2)
    "AD",
    "CompostedIV",
    "CompostedW",
]

# England KPI target (as you asked): 65% **recycling rate** (not diversion)
# Define explicitly what "recycling" means in your model:
# - Strict recycling rate: Recycled only
KPI_ROUTES = {"Recycled"}

# Keep circular routes as a SECONDARY metric (nice to show, but NOT used for Policy65 target)
CIRCULAR_ROUTES = {"Recycled", "Reuse", "AD", "CompostedIV", "CompostedW"}

# Objective 1 residual composition shares (you provided)
RESIDUAL_SHARES = {
    "Food": 0.27,
    "Garden": 0.05,
    "PaperCard": 0.211,
    "Glass": 0.027,
    "Metals": 0.035,
    "Plastics": 0.16,
    "Textiles": 0.055,
    "WEEE": 0.011,
    "Wood": 0.023,
    "NonAvoidable_Hazardous": 0.005,
    "NonAvoidable_Misc": 0.153,
}

# Where diverted residual goes (base mapping)
# (You can tune these as your assumptions evolve.)
DIVERT_TO_BASE = {
    "Food": [("AD", 1.0)],
    "Garden": [("CompostedW", 0.6), ("CompostedIV", 0.4)],
    "PaperCard": [("Recycled", 1.0)],
    "Plastics": [("Recycled", 1.0)],
    "Glass": [("Recycled", 1.0)],
    "Metals": [("Recycled", 1.0)],
    "Textiles": [("Recycled", 1.0)],
    "WEEE": [("Recycled", 1.0)],
    "Wood": [("Recycled", 1.0)],
    "OtherRecyclables": [("Recycled", 1.0)],
}

# "Fast / easy-first" variant: push organics to compost (quick deploy) instead of AD,
# while still sending dry recyclables to Recycled.
DIVERT_TO_FAST = {
    **{k: v for k, v in DIVERT_TO_BASE.items() if k not in ("Food", "Garden")},
    "Food": [("CompostedIV", 1.0)],  # fast operationally (assumption)
    "Garden": [("CompostedW", 1.0)],
}

# "Carbon-smart" variant: keep Food to AD, split Garden, etc. (same as base, but you can tune)
DIVERT_TO_CARBON = DIVERT_TO_BASE

# ============================================================
# FACTORS (kg CO2e per tonne) — your compiled defaults
# NOTE: routes not parameterised are treated as 0 unless you add/edit.
# ============================================================
REF_USER = "User-provided factor set (compiled from CarbonWARM + literature notes)."

DEFAULT_FACTORS = pd.DataFrame([
    # Food
    {"fraction":"Food", "route":"EfW",         "kgCO2e_per_t": -37.00,   "source": REF_USER},
    {"fraction":"Food", "route":"Landfill",    "kgCO2e_per_t": 627.00,   "source": REF_USER},
    {"fraction":"Food", "route":"AD",          "kgCO2e_per_t": -78.00,   "source": REF_USER},
    {"fraction":"Food", "route":"CompostedIV", "kgCO2e_per_t": -55.00,   "source": REF_USER},
    {"fraction":"Food", "route":"CompostedW",  "kgCO2e_per_t": +6.00,    "source": REF_USER},
    {"fraction":"Food", "route":"Recycled",    "kgCO2e_per_t": -3779.40, "source": REF_USER},
    {"fraction":"Food", "route":"Reuse",       "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},

    # Garden
    {"fraction":"Garden", "route":"EfW",         "kgCO2e_per_t": -77.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"Landfill",    "kgCO2e_per_t": 579.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"CompostedIV", "kgCO2e_per_t": -45.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"CompostedW",  "kgCO2e_per_t": +56.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"Recycled",    "kgCO2e_per_t": -184.09, "source": REF_USER},
    {"fraction":"Garden", "route":"Reuse",       "kgCO2e_per_t": 0.0,     "source": "Reuse credit set to 0 unless displacement is modelled."},

    # Paper & Card
    {"fraction":"PaperCard", "route":"EfW",       "kgCO2e_per_t": -217.00,  "source": REF_USER},
    {"fraction":"PaperCard", "route":"Landfill",  "kgCO2e_per_t": 1042.00,  "source": REF_USER},
    {"fraction":"PaperCard", "route":"Recycled",  "kgCO2e_per_t": -1387.47, "source": REF_USER},
    {"fraction":"PaperCard", "route":"Reuse",     "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},

    # Plastics
    {"fraction":"Plastics", "route":"EfW",        "kgCO2e_per_t": 1581.70, "source": REF_USER},
    {"fraction":"Plastics", "route":"Landfill",   "kgCO2e_per_t": 9.00,    "source": REF_USER},
    {"fraction":"Plastics", "route":"Recycled",   "kgCO2e_per_t": -3748.80,"source": REF_USER},
    {"fraction":"Plastics", "route":"Reuse",      "kgCO2e_per_t": 0.0,     "source": "Reuse credit set to 0 unless displacement is modelled."},

    # Glass
    {"fraction":"Glass", "route":"EfW",           "kgCO2e_per_t": 8.00,     "source": REF_USER},
    {"fraction":"Glass", "route":"Landfill",      "kgCO2e_per_t": 9.00,     "source": REF_USER},
    {"fraction":"Glass", "route":"Recycled",      "kgCO2e_per_t": -1728.77, "source": REF_USER},
    {"fraction":"Glass", "route":"Reuse",         "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},

    # Metals
    {"fraction":"Metals", "route":"EfW",          "kgCO2e_per_t": 21.50,    "source": REF_USER},
    {"fraction":"Metals", "route":"Landfill",     "kgCO2e_per_t": 9.00,     "source": REF_USER},
    {"fraction":"Metals", "route":"Recycled",     "kgCO2e_per_t": -9720.39, "source": REF_USER},
    {"fraction":"Metals", "route":"Reuse",        "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},

    # Textiles
    {"fraction":"Textiles", "route":"EfW",        "kgCO2e_per_t": 438.00,   "source": REF_USER},
    {"fraction":"Textiles", "route":"Landfill",   "kgCO2e_per_t": 445.00,   "source": REF_USER},
    {"fraction":"Textiles", "route":"Recycled",   "kgCO2e_per_t": -36625.00,"source": REF_USER},
    {"fraction":"Textiles", "route":"Reuse",      "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},

    # WEEE
    {"fraction":"WEEE", "route":"EfW",            "kgCO2e_per_t": 450.00,   "source": REF_USER},
    {"fraction":"WEEE", "route":"Landfill",       "kgCO2e_per_t": 20.00,    "source": REF_USER},
    {"fraction":"WEEE", "route":"Recycled",       "kgCO2e_per_t": -10535.94,"source": REF_USER},
    {"fraction":"WEEE", "route":"Reuse",          "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},

    # Wood
    {"fraction":"Wood", "route":"EfW",            "kgCO2e_per_t": -318.00,  "source": REF_USER},
    {"fraction":"Wood", "route":"Landfill",       "kgCO2e_per_t": 921.00,   "source": REF_USER},
    {"fraction":"Wood", "route":"Recycled",       "kgCO2e_per_t": -754.50,  "source": REF_USER},
    {"fraction":"Wood", "route":"Reuse",          "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},
])

WILDCARD_DEFAULTS = pd.DataFrame([
    {"fraction":"*", "route":"IncNoEnergy", "kgCO2e_per_t": 360.0, "source":"Placeholder IWE/R2 factor (update if you have a better value)."},
    {"fraction":"*", "route":"RDF_MHT",     "kgCO2e_per_t": 0.0,   "source":"Not parameterised here; set 0 unless you add factor."},
])

# ============================================================
# HELPERS
# ============================================================
def safe_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)

def factor_maps(factors_df: pd.DataFrame):
    exact = {(r["fraction"], r["route"]): float(r["kgCO2e_per_t"])
             for _, r in factors_df.iterrows() if r["fraction"] != "*"}
    wild = {r["route"]: float(r["kgCO2e_per_t"])
            for _, r in factors_df.iterrows() if r["fraction"] == "*"}
    def get(frac, route):
        if (frac, route) in exact:
            return exact[(frac, route)]
        if route in wild:
            return wild[route]
        return np.nan
    return get

@st.cache_data(show_spinner=False)
def load_xlsx(path, sheet):
    return pd.read_excel(path, sheet_name=sheet)

def build_long(df: pd.DataFrame):
    """
    Builds tidy long table from columns like PaperCardRecycled_2023 etc.
    Also loads:
      population_YYYY
      TotalCollectedYYYY
      TotalRecCollecYYYY   (front-end collection metric)
      Residual_YYYY
    """
    if "Council Name" not in df.columns:
        raise ValueError("Missing required column: Council Name")

    long_parts = []
    for frac in FRACTIONS:
        for route in ROUTES:
            for y in YEARS:
                col = f"{frac}{route}_{y}"
                if col in df.columns:
                    long_parts.append(pd.DataFrame({
                        "Council Name": df["Council Name"].astype(str),
                        "year": y,
                        "fraction": frac,
                        "route": route,
                        "tonnes": safe_num(df[col]),
                    }))
    long = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame(
        columns=["Council Name", "year", "fraction", "route", "tonnes"]
    )

    totals_parts, rec_col_parts, residual_parts, pop_parts = [], [], [], []
    for y in YEARS:
        tc = f"TotalCollected{y}"
        trc = f"TotalRecCollec{y}"
        rc = f"Residual_{y}"
        pc = f"population_{y}"

        if tc in df.columns:
            totals_parts.append(pd.DataFrame({
                "Council Name": df["Council Name"].astype(str),
                "year": y,
                "TotalCollected": safe_num(df[tc]),
            }))
        if trc in df.columns:
            rec_col_parts.append(pd.DataFrame({
                "Council Name": df["Council Name"].astype(str),
                "year": y,
                "TotalRecCollec": safe_num(df[trc]),
            }))
        if rc in df.columns:
            residual_parts.append(pd.DataFrame({
                "Council Name": df["Council Name"].astype(str),
                "year": y,
                "Residual": safe_num(df[rc]),
            }))
        if pc in df.columns:
            pop_parts.append(pd.DataFrame({
                "Council Name": df["Council Name"].astype(str),
                "year": y,
                "population": safe_num(df[pc]),
            }))

    totals = pd.concat(totals_parts, ignore_index=True) if totals_parts else None
    rec_col = pd.concat(rec_col_parts, ignore_index=True) if rec_col_parts else None
    residual = pd.concat(residual_parts, ignore_index=True) if residual_parts else None
    pop = pd.concat(pop_parts, ignore_index=True) if pop_parts else None

    if pop is not None and len(long):
        long = long.merge(pop, on=["Council Name", "year"], how="left")
    else:
        long["population"] = np.nan

    return long, totals, rec_col, residual

def apply_factors(long_df: pd.DataFrame, factors_df: pd.DataFrame):
    get = factor_maps(factors_df)
    out = long_df.copy()
    out["kgCO2e_per_t"] = [get(f, r) for f, r in zip(out["fraction"], out["route"])]
    out["factor_missing"] = out["kgCO2e_per_t"].isna().astype(int)
    out["kgCO2e_per_t"] = out["kgCO2e_per_t"].fillna(0.0)
    out["kgCO2e"] = out["tonnes"] * out["kgCO2e_per_t"]
    return out

def kpi_rate(long_y: pd.DataFrame, totals_y: pd.DataFrame, routes_set: set):
    total = float(totals_y["TotalCollected"].sum())
    kpi_t = float(long_y[long_y["route"].isin(routes_set)]["tonnes"].sum())
    rate = kpi_t / total if total > 0 else np.nan
    return total, kpi_t, rate

def build_residual_divertable(residual_y: pd.DataFrame):
    parts = []
    for frac, share in RESIDUAL_SHARES.items():
        if frac in FRACTIONS:
            parts.append(pd.DataFrame({
                "Council Name": residual_y["Council Name"],
                "year": residual_y["year"],
                "fraction": frac,
                "tonnes": residual_y["Residual"] * share
            }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["Council Name", "year", "fraction", "tonnes"]
    )

def baseline_propensity_recycling(long_y: pd.DataFrame):
    """
    Propensity per council×fraction (how "recycling-leaning" the council is for that fraction):
      p = Recycled / (Recycled + EfW + Landfill + RDF_MHT + IncNoEnergy)
    """
    disp = {"EfW", "Landfill", "RDF_MHT", "IncNoEnergy"}
    g = long_y[long_y["route"].isin(disp | {"Recycled"})].copy()
    piv = g.pivot_table(index=["Council Name", "fraction"], columns="route",
                        values="tonnes", aggfunc="sum").fillna(0.0)
    for c in ["Recycled", "EfW", "Landfill", "RDF_MHT", "IncNoEnergy"]:
        if c not in piv.columns:
            piv[c] = 0.0
    denom = (piv["Recycled"] + piv["EfW"] + piv["Landfill"] + piv["RDF_MHT"] + piv["IncNoEnergy"]).replace(0, np.nan)
    p = (piv["Recycled"] / denom).fillna(0.0)
    out = p.reset_index()
    out["p_recycled"] = p.values
    return out

def allocate_diversion(res_div: pd.DataFrame,
                       extra_needed_recycled_t: float,
                       mode: str,
                       long_y: pd.DataFrame,
                       factors_df: pd.DataFrame):
    """
    Allocate extra tonnes from residual pool to increase **Recycled tonnes** (KPI_ROUTES),
    by "capturing" recyclable/avoidable residual.

    We allocate a TAKE (from residual pool), then convert that TAKE to treatment routes
    via a diversion map (fast/carbon/base).

    mode: "fast" | "mixed" | "carbon"
    """
    if extra_needed_recycled_t <= 0:
        return pd.DataFrame(columns=["Council Name", "year", "fraction", "route", "tonnes"])

    pool = res_div.copy()
    pool = pool[pool["fraction"].isin(DIVERT_TO_BASE.keys())].copy()
    if pool["tonnes"].sum() <= 0:
        return pd.DataFrame(columns=["Council Name", "year", "fraction", "route", "tonnes"])

    # How much of a taken tonne becomes "Recycled" depends on diversion map.
    # For KPI targeting, we approximate recycled_yield:
    def recycled_yield(divert_map, frac):
        routes = divert_map.get(frac, [])
        yld = 0.0
        for r, w in routes:
            if r == "Recycled":
                yld += w
        return yld

    # Choose diversion map
    if mode == "fast":
        divert_map = DIVERT_TO_FAST
    elif mode == "carbon":
        divert_map = DIVERT_TO_CARBON
    else:
        divert_map = DIVERT_TO_BASE

    pool["recycled_yield"] = pool["fraction"].apply(lambda f: recycled_yield(divert_map, f))
    pool = pool[pool["recycled_yield"] > 0].copy()
    if pool.empty:
        return pd.DataFrame(columns=["Council Name", "year", "fraction", "route", "tonnes"])

    # Ranking logic
    if mode == "fast":
        # "easy-first": prioritize classic dry recyclables
        priority = {
            "PaperCard": 1, "Metals": 2, "Glass": 3, "Plastics": 4,
            "Wood": 5, "Textiles": 6, "WEEE": 7, "OtherRecyclables": 8,
            "Garden": 9, "Food": 10
        }
        pool["rank"] = pool["fraction"].map(priority).fillna(99)

    elif mode == "mixed":
        # push effort where baseline propensity is LOW (harder councils get more capture)
        prop = baseline_propensity_recycling(long_y)
        pool = pool.merge(prop, on=["Council Name", "fraction"], how="left")
        pool["p_recycled"] = pool["p_recycled"].fillna(0.0)
        pool["rank"] = pool["p_recycled"]  # low first

    elif mode == "carbon":
        # prioritize biggest emissions improvement per tonne captured into recycling routes
        get = factor_maps(factors_df)

        def best_delta(frac):
            # conservative: assume displaced from landfill
            base = get(frac, "Landfill")
            # emissions of diverted mix (weighted)
            mix = 0.0
            for r, w in divert_map.get(frac, []):
                mix += (get(frac, r) if pd.notna(get(frac, r)) else 0.0) * w
            return (base - mix)

        pool["delta"] = pool["fraction"].apply(best_delta)
        pool["rank"] = -pool["delta"].fillna(0.0)

    else:
        pool["rank"] = 0

    pool = pool.sort_values("rank", ascending=True).copy()

    # We need extra_needed_recycled_t of *Recycled*.
    # Each tonne taken contributes recycled_yield * take to recycled KPI.
    remaining_recycled = float(extra_needed_recycled_t)
    pool["take"] = 0.0

    for i, row in pool.iterrows():
        if remaining_recycled <= 0:
            break

        avail = float(row["tonnes"])
        yld = float(row["recycled_yield"])
        if yld <= 0 or avail <= 0:
            continue

        # take enough tonnes so that yld*take meets remaining_recycled
        needed_take = remaining_recycled / yld
        take = min(avail, needed_take)

        if take > 0:
            pool.loc[i, "take"] = take
            remaining_recycled -= yld * take

    chosen = pool[pool["take"] > 0].copy()
    if chosen.empty:
        return pd.DataFrame(columns=["Council Name", "year", "fraction", "route", "tonnes"])

    # Expand chosen takes into treatment routes using divert_map
    out_parts = []
    for frac, routes in divert_map.items():
        sub = chosen[chosen["fraction"] == frac].copy()
        if sub.empty:
            continue
        for route, w in routes:
            tmp = sub[["Council Name", "year", "fraction", "take"]].copy()
            tmp["route"] = route
            tmp["tonnes"] = tmp["take"] * w
            out_parts.append(tmp[["Council Name", "year", "fraction", "route", "tonnes"]])

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(
        columns=["Council Name", "year", "fraction", "route", "tonnes"]
    )

def scenario_df(name: str, base_long_y: pd.DataFrame, add_rows: pd.DataFrame):
    base = base_long_y[["Council Name", "year", "fraction", "route", "tonnes", "population"]].copy()
    base["scenario"] = name

    if add_rows is None or add_rows.empty:
        return base

    add = add_rows.copy()
    add["population"] = np.nan
    add["scenario"] = name
    add = add[["Council Name", "year", "fraction", "route", "tonnes", "population", "scenario"]]
    return pd.concat([base, add], ignore_index=True)

def scenario_stack_for_year(long_y: pd.DataFrame,
                            totals_y: pd.DataFrame,
                            residual_y: pd.DataFrame,
                            target_recycling_rate: float,
                            factors_df: pd.DataFrame):
    # KPI = recycling rate (Recycled / TotalCollected)
    total, recycled_t, base_rate = kpi_rate(long_y, totals_y, KPI_ROUTES)
    extra_recycled_needed = max(0.0, target_recycling_rate * total - recycled_t)

    res_div = build_residual_divertable(residual_y)

    baseline = scenario_df("Baseline (Actual)", long_y, pd.DataFrame())
    pol_fast = scenario_df("Policy65-Fast (Easy-first)", long_y,
                           allocate_diversion(res_div, extra_recycled_needed, "fast", long_y, factors_df))
    pol_mix = scenario_df("Policy65-Mixed (Propensity)", long_y,
                          allocate_diversion(res_div, extra_recycled_needed, "mixed", long_y, factors_df))
    pol_car = scenario_df("Policy65-Circular (Carbon-smart)", long_y,
                          allocate_diversion(res_div, extra_recycled_needed, "carbon", long_y, factors_df))

    # Optimal: capture ALL divertable residual (full take) using base map
    # Convert that into treatment routes. Here we treat “optimal” as the maximum capture from the pool.
    # We approximate by asking for an enormous extra_recycled so it takes everything with recycled_yield>0.
    huge = 1e12
    optimal_add = allocate_diversion(res_div, huge, "carbon", long_y, factors_df)
    optimal = scenario_df("Optimal (Recover all recyclable/avoidable residual)", long_y, optimal_add)

    all_scen = pd.concat([baseline, pol_fast, pol_mix, pol_car, optimal], ignore_index=True)
    return all_scen, base_rate, extra_recycled_needed

def sankey_scenario_fraction_route(df: pd.DataFrame, value_col="tonnes", top_links=80, title="Sankey"):
    """
    Nodes: scenario -> fraction -> route
    Keep top_links for readability.
    """
    d = df.groupby(["scenario", "fraction", "route"], as_index=False).agg(v=(value_col, "sum"))
    d = d[d["v"] > 0].copy()
    if d.empty:
        fig = go.Figure()
        fig.update_layout(title="No data for Sankey.")
        return fig

    d = d.sort_values("v", ascending=False).head(int(top_links)).copy()

    scen = sorted(d["scenario"].unique().tolist())
    frac = sorted(d["fraction"].unique().tolist())
    route = sorted(d["route"].unique().tolist())

    nodes = scen + frac + route
    idx = {n: i for i, n in enumerate(nodes)}

    sf = d.groupby(["scenario", "fraction"], as_index=False).agg(v=("v", "sum"))

    source, target, value = [], [], []

    # scenario -> fraction
    source += [idx[s] for s in sf["scenario"]]
    target += [idx[f] for f in sf["fraction"]]
    value  += sf["v"].tolist()

    # fraction -> route (scenario-specific)
    for s in scen:
        tmp = d[d["scenario"] == s].groupby(["fraction", "route"], as_index=False).agg(v=("v", "sum"))
        source += [idx[f] for f in tmp["fraction"]]
        target += [idx[r] for r in tmp["route"]]
        value  += tmp["v"].tolist()

    fig = go.Figure(go.Sankey(
        node=dict(pad=10, thickness=12, label=nodes),
        link=dict(source=source, target=target, value=value)
    ))
    fig.update_layout(title=title, height=650, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def route_mix(long_df: pd.DataFrame):
    return long_df.groupby(["year", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))

def council_year_base(totals: pd.DataFrame, rec_col: pd.DataFrame, residual: pd.DataFrame, long: pd.DataFrame):
    """
    Build a council-year base table with:
      TotalCollected, TotalRecCollec, Residual, population
      + derived per-capita and capture metrics
      + KPI recycling tonnes and circular tonnes from routes table
    """
    base = totals.merge(rec_col, on=["Council Name", "year"], how="left")
    base = base.merge(residual, on=["Council Name", "year"], how="left")

    pop = long.drop_duplicates(["Council Name", "year"])[["Council Name", "year", "population"]]
    base = base.merge(pop, on=["Council Name", "year"], how="left")

    base["TotalRecCollec"] = base["TotalRecCollec"].fillna(0.0)
    base["Residual"] = base["Residual"].fillna(0.0)

    # KPI tonnes (Recycled only)
    kpi = long[long["route"].isin(KPI_ROUTES)].groupby(["Council Name", "year"], as_index=False).agg(kpi_recycled_t=("tonnes","sum"))
    base = base.merge(kpi, on=["Council Name", "year"], how="left")
    base["kpi_recycled_t"] = base["kpi_recycled_t"].fillna(0.0)

    # Circular tonnes (secondary)
    circ = long[long["route"].isin(CIRCULAR_ROUTES)].groupby(["Council Name", "year"], as_index=False).agg(circular_t=("tonnes","sum"))
    base = base.merge(circ, on=["Council Name", "year"], how="left")
    base["circular_t"] = base["circular_t"].fillna(0.0)

    # Rates
    base["collection_rate"] = np.where(base["TotalCollected"] > 0, base["TotalRecCollec"] / base["TotalCollected"], np.nan)
    base["recycling_rate_kpi"] = np.where(base["TotalCollected"] > 0, base["kpi_recycled_t"] / base["TotalCollected"], np.nan)
    base["circular_rate"] = np.where(base["TotalCollected"] > 0, base["circular_t"] / base["TotalCollected"], np.nan)

    # Post-collection gap vs circular (diagnostic)
    base["post_collection_gap_t"] = np.maximum(0.0, base["TotalRecCollec"] - base["circular_t"])
    base["capture_efficiency_circular"] = np.where(base["TotalRecCollec"] > 0, base["circular_t"] / base["TotalRecCollec"], np.nan)

    # Per-capita
    base["waste_kg_per_cap"] = np.where(base["population"] > 0, base["TotalCollected"] * 1000.0 / base["population"], np.nan)
    base["rec_kg_per_cap"] = np.where(base["population"] > 0, base["TotalRecCollec"] * 1000.0 / base["population"], np.nan)
    base["residual_kg_per_cap"] = np.where(base["population"] > 0, base["Residual"] * 1000.0 / base["population"], np.nan)

    return base

# ============================================================
# SIDEBAR — DATA
# ============================================================
st.sidebar.header("Data")
auto = st.sidebar.checkbox(f"Auto-load {XLSX_NAME} from folder", value=True)

xlsx_path = XLSX_NAME if auto and os.path.exists(XLSX_NAME) else None
if xlsx_path is None:
    up = st.sidebar.file_uploader("Upload pathway_data.xlsx", type=["xlsx"])
    if up is None:
        st.info(f"Put `{XLSX_NAME}` next to `app.py` or upload it.")
        st.stop()
    tmp = "_uploaded_pathway_data.xlsx"
    with open(tmp, "wb") as f:
        f.write(up.getbuffer())
    xlsx_path = tmp

sheet = st.sidebar.text_input("Sheet name", value=DEFAULT_SHEET).strip() or DEFAULT_SHEET

try:
    df = load_xlsx(xlsx_path, sheet)
except Exception as e:
    st.error(f"Could not read XLSX: {e}")
    st.stop()

try:
    long, totals, rec_col, residual = build_long(df)
except Exception as e:
    st.error(f"Could not build dataset: {e}")
    st.stop()

if totals is None or rec_col is None or residual is None:
    st.error("Missing one or more required series: TotalCollectedYYYY, TotalRecCollecYYYY, Residual_YYYY.")
    st.stop()

# ============================================================
# SIDEBAR — CONTROLS
# ============================================================
st.sidebar.header("Controls")
unit = st.sidebar.selectbox("Display unit", ["t CO₂e", "kg CO₂e"], index=0)
to_unit = (lambda kg: kg / 1000.0) if unit == "t CO₂e" else (lambda kg: kg)

# Policy target = recycling rate target (Recycled / TotalCollected)
target_rate = st.sidebar.slider("Policy target recycling rate", 0.20, 0.80, 0.65, 0.01)

st.sidebar.header("Factors")
edit = st.sidebar.checkbox("Enable factor editor (session-only)", value=False)
factors = pd.concat([DEFAULT_FACTORS, WILDCARD_DEFAULTS], ignore_index=True)
if edit:
    factors = st.data_editor(
        factors,
        use_container_width=True,
        num_rows="dynamic",
        column_config={"kgCO2e_per_t": st.column_config.NumberColumn(format="%.2f")},
    )

# Apply factors to observed long (all years)
em_obs = apply_factors(long, factors)
em_obs["CO2e_u"] = to_unit(em_obs["kgCO2e"])

# Council-year base table (includes TotalRecCollec & population)
baseCY = council_year_base(totals, rec_col, residual, long)

# ============================================================
# HEADER
# ============================================================
st.title("Objective 2 — LACW Pathways Lab (All councils, 2020–2023)")
st.caption(
    "Policy KPI here targets **Recycling rate = Recycled / TotalCollected** (England-style, as requested). "
    "Circular rate is shown separately (Recycled + Reuse + AD + Compost IV/W). "
    "TotalRecCollec is a collection metric (front-end) and is NOT a treatment route."
)

# ============================================================
# TABS
# ============================================================
tabTS, tabScen, tabIWE, tabReuse, tabEff, tabCouncil, tabDiag = st.tabs([
    "1) Time series (default)",
    "2) Scenarios & optimal",
    "3) IWE / Incineration without energy (R2)",
    "4) Reuse",
    "5) Collection → Fate efficiency",
    "6) Council explorer",
    "7) Diagnostics",
])

# ============================================================
# 1) TIME SERIES
# ============================================================
with tabTS:
    st.subheader("National time series (2020–2023)")

    rows = []
    for y in YEARS:
        by = baseCY[baseCY["year"] == y].copy()

        total = float(by["TotalCollected"].sum())
        recC = float(by["TotalRecCollec"].sum())
        residualT = float(by["Residual"].sum())
        recycled_kpi = float(by["kpi_recycled_t"].sum())
        circularT = float(by["circular_t"].sum())
        pop = float(by["population"].sum()) if by["population"].notna().sum() else np.nan

        kg = float(em_obs[em_obs["year"] == y]["kgCO2e"].sum())

        rows.append({
            "year": y,
            "TotalCollected_t": total,
            "Residual_t": residualT,
            "TotalRecCollec_t": recC,
            "Recycled_KPI_t": recycled_kpi,
            "Circular_t": circularT,
            "RecyclingRate_KPI": (recycled_kpi / total) if total > 0 else np.nan,
            "CircularRate": (circularT / total) if total > 0 else np.nan,
            "CollectionRate": (recC / total) if total > 0 else np.nan,
            "kgCO2e": kg,
            "CO2e_u": to_unit(kg),
            "waste_kg_per_cap": (total * 1000.0 / pop) if (pop and pop > 0) else np.nan,
            "kgCO2e_per_cap": (kg / pop) if (pop and pop > 0) else np.nan,
        })
    nat = pd.DataFrame(rows)

    y_latest = YEARS[-1]
    nat_latest = nat[nat["year"] == y_latest].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{y_latest} recycling rate (KPI)", f"{nat_latest['RecyclingRate_KPI']*100:.1f}%")
    c2.metric(f"{y_latest} circular rate", f"{nat_latest['CircularRate']*100:.1f}%")
    c3.metric(f"{y_latest} emissions ({unit})", f"{nat_latest['CO2e_u']:,.0f}")
    c4.metric(f"{y_latest} kgCO₂e/cap", f"{nat_latest['kgCO2e_per_cap']:.1f}" if pd.notna(nat_latest["kgCO2e_per_cap"]) else "NA")

    left, right = st.columns(2)
    with left:
        fig_kpi = px.line(nat, x="year", y="RecyclingRate_KPI", markers=True, title="Recycling rate (KPI) over time")
        fig_kpi.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig_kpi, use_container_width=True)

    with right:
        fig_circ = px.line(nat, x="year", y="CircularRate", markers=True, title="Circular rate over time (secondary)")
        fig_circ.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig_circ, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig_em = px.line(nat, x="year", y="CO2e_u", markers=True, title=f"Total emissions over time ({unit})")
        fig_em.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_em, use_container_width=True)

    with right:
        fig_pc = px.line(nat, x="year", y="kgCO2e_per_cap", markers=True, title="Emissions per capita over time (kgCO₂e/person)")
        fig_pc.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown("### Route mix over time (tonnes)")
    rm = route_mix(long)
    fig_mix = px.area(rm, x="year", y="tonnes", color="route", title="Route mix over time (stacked tonnes)")
    fig_mix.update_layout(height=440, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_mix, use_container_width=True)

    st.markdown("### Fraction × route time series")
    frac_sel = st.selectbox("Pick a fraction", options=FRACTIONS, index=0, key="ts_frac")
    fr = long[long["fraction"] == frac_sel].groupby(["year", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))
    fig_fr = px.area(fr, x="year", y="tonnes", color="route", title=f"{frac_sel}: route mix over time")
    fig_fr.update_layout(height=440, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_fr, use_container_width=True)

# ============================================================
# 2) SCENARIOS & OPTIMAL
# ============================================================
with tabScen:
    st.subheader("Scenario engine (Baseline, Policy65 variants, Optimal) — targets recycling rate (KPI)")

    scen_all = []
    base_rates = {}
    extra_needs = {}

    for y in YEARS:
        ly = long[long["year"] == y].copy()
        ty = totals[totals["year"] == y].copy()
        ry = residual[residual["year"] == y].copy()

        sc_y, base_rate_y, extra_y = scenario_stack_for_year(ly, ty, ry, target_rate, factors)
        scen_all.append(sc_y)
        base_rates[y] = base_rate_y
        extra_needs[y] = extra_y

    scen_long = pd.concat(scen_all, ignore_index=True)
    scen_em = apply_factors(scen_long, factors)
    scen_em["CO2e_u"] = to_unit(scen_em["kgCO2e"])

    # Scenario time series summary
    ts_rows = []
    for y in YEARS:
        totalY = float(totals[totals["year"] == y]["TotalCollected"].sum())
        for s in scen_long["scenario"].unique():
            d = scen_long[(scen_long["year"] == y) & (scen_long["scenario"] == s)]
            kpi_t = float(d[d["route"].isin(KPI_ROUTES)]["tonnes"].sum())
            kpi_rate = kpi_t / totalY if totalY > 0 else np.nan

            circ_t = float(d[d["route"].isin(CIRCULAR_ROUTES)]["tonnes"].sum())
            circ_rate = circ_t / totalY if totalY > 0 else np.nan

            kg = float(scen_em[(scen_em["year"] == y) & (scen_em["scenario"] == s)]["kgCO2e"].sum())

            ts_rows.append({
                "year": y,
                "scenario": s,
                "RecyclingRate_KPI": kpi_rate,
                "CircularRate": circ_rate,
                "CO2e_u": to_unit(kg),
            })
    scen_ts = pd.DataFrame(ts_rows)

    st.caption(
        f"Target recycling rate (KPI) = {target_rate:.0%}. "
        f"Extra **Recycled** tonnes needed by year (national): " +
        ", ".join([f"{y}: {extra_needs[y]:,.0f} t" for y in YEARS])
    )

    left, right = st.columns(2)
    with left:
        fig_srate = px.line(scen_ts, x="year", y="RecyclingRate_KPI", color="scenario", markers=True,
                            title="Recycling rate (KPI) by scenario (time series)")
        fig_srate.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig_srate, use_container_width=True)

    with right:
        fig_sem = px.line(scen_ts, x="year", y="CO2e_u", color="scenario", markers=True,
                          title=f"Total emissions by scenario (time series) — {unit}")
        fig_sem.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_sem, use_container_width=True)

    st.markdown("### Single-year deep dive")
    year = st.selectbox("Year", options=YEARS, index=2, key="sc_year")

    # Summary for selected year
    totalY = float(totals[totals["year"] == year]["TotalCollected"].sum())

    dsum = scen_em[scen_em["year"] == year].groupby("scenario", as_index=False).agg(
        tonnes=("tonnes", "sum"),
        kgCO2e=("kgCO2e", "sum"),
        missing=("factor_missing", "sum"),
    )
    dsum["CO2e_u"] = to_unit(dsum["kgCO2e"])

    kpiY = scen_long[(scen_long["year"] == year) & (scen_long["route"].isin(KPI_ROUTES))] \
        .groupby("scenario", as_index=False).agg(Recycled_KPI_t=("tonnes", "sum"))
    circY = scen_long[(scen_long["year"] == year) & (scen_long["route"].isin(CIRCULAR_ROUTES))] \
        .groupby("scenario", as_index=False).agg(Circular_t=("tonnes", "sum"))

    dsum = dsum.merge(kpiY, on="scenario", how="left").merge(circY, on="scenario", how="left")
    dsum[["Recycled_KPI_t", "Circular_t"]] = dsum[["Recycled_KPI_t", "Circular_t"]].fillna(0.0)

    dsum["RecyclingRate_KPI"] = np.where(totalY > 0, dsum["Recycled_KPI_t"] / totalY, np.nan)
    dsum["CircularRate"] = np.where(totalY > 0, dsum["Circular_t"] / totalY, np.nan)

    st.dataframe(dsum.sort_values("RecyclingRate_KPI"), use_container_width=True, height=260)

    st.markdown("### Route mix by scenario (selected year)")
    mix = scen_long[scen_long["year"] == year].groupby(["scenario", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))
    fig_mix2 = px.bar(mix, x="scenario", y="tonnes", color="route", barmode="stack", title="Route mix (stacked tonnes)")
    fig_mix2.update_layout(height=460, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_mix2, use_container_width=True)

    st.markdown("### Recycling rate (KPI) by scenario")
    fig_kpi2 = px.bar(dsum, x="scenario", y="RecyclingRate_KPI", text="RecyclingRate_KPI",
                      title="Recycling rate (KPI) by scenario (selected year)")
    fig_kpi2.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10), yaxis_tickformat=".0%")
    st.plotly_chart(fig_kpi2, use_container_width=True)

    st.markdown("### Emissions waterfall (selected year)")
    order = [
        "Baseline (Actual)",
        "Policy65-Fast (Easy-first)",
        "Policy65-Mixed (Propensity)",
        "Policy65-Circular (Carbon-smart)",
        "Optimal (Recover all recyclable/avoidable residual)",
    ]
    order = [o for o in order if o in dsum["scenario"].values]
    vals = [float(dsum.loc[dsum["scenario"] == s, "CO2e_u"].iloc[0]) for s in order]
    measures = ["absolute"] + ["relative"] * (len(vals) - 1)
    yv = [vals[0]] + [vals[i] - vals[i-1] for i in range(1, len(vals))]
    wf = go.Figure(go.Waterfall(x=order, measure=measures, y=yv))
    wf.update_layout(title=f"Emissions waterfall ({unit}) — selected year", height=420, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(wf, use_container_width=True)

    st.markdown("### Sankey (Scenario → Fraction → Route)")
    top_links = st.slider("Sankey: keep top links", 20, 250, 80, 10, key="sc_sank_top")
    sank_metric = st.radio("Sankey sized by", ["Tonnes", "Emissions magnitude |kgCO₂e|"], horizontal=True, key="sc_sank_metric")

    sank_in = scen_em[scen_em["year"] == year].copy()
    if sank_metric.startswith("Emissions"):
        sank_in["abs_kg"] = np.abs(sank_in["kgCO2e"])
        fig_s = sankey_scenario_fraction_route(sank_in, value_col="abs_kg", top_links=top_links,
                                               title="Sankey sized by |kgCO₂e| (top links)")
    else:
        fig_s = sankey_scenario_fraction_route(sank_in, value_col="tonnes", top_links=top_links,
                                               title="Sankey sized by tonnes (top links)")
    st.plotly_chart(fig_s, use_container_width=True)

# ============================================================
# 3) IWE / Incineration without energy (R2)
# ============================================================
with tabIWE:
    st.subheader("Incineration without energy (IncNoEnergy / IWE / R2) — where it appears, who drives it")

    iwe = long[long["route"] == "IncNoEnergy"].copy()
    if iwe.empty or iwe["tonnes"].sum() == 0:
        st.info("No IncNoEnergy tonnes found in your dataset.")
    else:
        iwe_ts = iwe.groupby("year", as_index=False).agg(tonnes=("tonnes","sum"), councils=("Council Name","nunique"))
        left, right = st.columns(2)
        with left:
            fig_iwe = px.line(iwe_ts, x="year", y="tonnes", markers=True, title="National IncNoEnergy tonnes over time")
            fig_iwe.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(fig_iwe, use_container_width=True)
        with right:
            fig_cnt = px.line(iwe_ts, x="year", y="councils", markers=True, title="Councils reporting IncNoEnergy over time")
            fig_cnt.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(fig_cnt, use_container_width=True)

        iwe_f = iwe.groupby(["year","fraction"], as_index=False).agg(tonnes=("tonnes","sum"))
        fig_iwef = px.bar(iwe_f, x="fraction", y="tonnes", color="year", barmode="group", title="IncNoEnergy by fraction")
        fig_iwef.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_iwef, use_container_width=True)

        y = st.selectbox("Year", YEARS, index=2, key="iwe_year")
        iwe_y = iwe[iwe["year"] == y].groupby("Council Name", as_index=False).agg(tonnes=("tonnes","sum")).sort_values("tonnes", ascending=False)
        fig_top = px.bar(iwe_y.head(30), x="Council Name", y="tonnes", title=f"Top 30 councils by IncNoEnergy — {y}")
        fig_top.update_layout(height=520, margin=dict(l=10,r=10,t=60,b=10), xaxis_tickangle=-45)
        st.plotly_chart(fig_top, use_container_width=True)

# ============================================================
# 4) REUSE
# ============================================================
with tabReuse:
    st.subheader("Reuse — trends, fractions, and leading councils")

    ru = long[long["route"] == "Reuse"].copy()
    if ru.empty or ru["tonnes"].sum() == 0:
        st.info("No Reuse tonnes found in your dataset.")
    else:
        ru_ts = ru.groupby("year", as_index=False).agg(tonnes=("tonnes","sum"), councils=("Council Name","nunique"))
        left, right = st.columns(2)
        with left:
            fig_ru = px.line(ru_ts, x="year", y="tonnes", markers=True, title="National Reuse tonnes over time")
            fig_ru.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(fig_ru, use_container_width=True)
        with right:
            fig_ru_cnt = px.line(ru_ts, x="year", y="councils", markers=True, title="Councils reporting Reuse over time")
            fig_ru_cnt.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(fig_ru_cnt, use_container_width=True)

        ru_f = ru.groupby(["year","fraction"], as_index=False).agg(tonnes=("tonnes","sum"))
        fig_ruf = px.bar(ru_f, x="fraction", y="tonnes", color="year", barmode="group", title="Reuse by fraction")
        fig_ruf.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_ruf, use_container_width=True)

# ============================================================
# 5) COLLECTION → FATE EFFICIENCY
# ============================================================
with tabEff:
    st.subheader("Collection → Fate efficiency (diagnostic, mechanism-focused)")

    y = st.selectbox("Year", YEARS, index=2, key="eff_year")
    d = baseCY[baseCY["year"] == y].copy()

    nat_rec = float(d["TotalRecCollec"].sum())
    nat_circ = float(d["circular_t"].sum())
    nat_gap = float(np.maximum(0.0, d["TotalRecCollec"] - d["circular_t"]).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("National TotalRecCollec (t)", f"{nat_rec:,.0f}")
    c2.metric("National circular treatment (t)", f"{nat_circ:,.0f}")
    c3.metric("National post-collection gap (t)", f"{nat_gap:,.0f}")
    c4.metric("Circular capture efficiency", f"{(nat_circ/nat_rec)*100:.1f}%" if nat_rec > 0 else "NA")

    st.markdown("### Capture efficiency distribution across councils (circular_t / TotalRecCollec)")
    fig_hist = px.histogram(d, x="capture_efficiency_circular", nbins=40,
                            title="Capture efficiency (circular) distribution")
    fig_hist.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10), xaxis_tickformat=".0%")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Biggest gaps (why your TotalRecCollec can exceed ‘actually treated’)")
    d2 = d.copy()
    d2["gap_t"] = np.maximum(0.0, d2["TotalRecCollec"] - d2["circular_t"])
    st.dataframe(
        d2.sort_values("gap_t", ascending=False)[
            ["Council Name", "TotalCollected", "TotalRecCollec", "circular_t", "gap_t",
             "collection_rate", "recycling_rate_kpi", "circular_rate"]
        ].head(40),
        use_container_width=True,
        height=420
    )

# ============================================================
# 6) COUNCIL EXPLORER
# ============================================================
with tabCouncil:
    st.subheader("Council explorer — scale vs pathway mix vs fraction drivers")

    year = st.selectbox("Year", YEARS, index=2, key="cx_year")
    councils = sorted(long[long["year"] == year]["Council Name"].unique().tolist())
    council = st.selectbox("Council", councils, index=0, key="cx_council")

    cy = baseCY[(baseCY["year"] == year) & (baseCY["Council Name"] == council)]
    if cy.empty:
        st.info("No totals for this council/year.")
        st.stop()
    cy = cy.iloc[0]

    d = em_obs[(em_obs["year"] == year) & (em_obs["Council Name"] == council)].copy()
    if d.empty:
        st.info("No pathway data for this council/year.")
        st.stop()

    total_col = float(cy["TotalCollected"])
    pop = float(cy["population"]) if pd.notna(cy["population"]) else np.nan

    recycled_kpi = float(d[d["route"].isin(KPI_ROUTES)]["tonnes"].sum())
    recycling_rate = (recycled_kpi / total_col) if total_col > 0 else np.nan

    circular_t = float(d[d["route"].isin(CIRCULAR_ROUTES)]["tonnes"].sum())
    circular_rate = (circular_t / total_col) if total_col > 0 else np.nan

    kg = float(d["kgCO2e"].sum())
    intensity = (kg / total_col) if total_col > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total collected (t)", f"{total_col:,.0f}")
    c2.metric("Recycling rate (KPI)", f"{recycling_rate*100:.1f}%")
    c3.metric("Circular rate", f"{circular_rate*100:.1f}%")
    c4.metric("Intensity (kgCO₂e/t)", f"{intensity:.1f}")

    mix = d.groupby("route", as_index=False).agg(tonnes=("tonnes","sum"), kgCO2e=("kgCO2e","sum"))
    mix["CO2e_u"] = to_unit(mix["kgCO2e"])

    left, right = st.columns(2)
    with left:
        fig_r = px.bar(mix.sort_values("tonnes", ascending=False), x="route", y="tonnes", title="Route mix (tonnes)")
        fig_r.update_layout(height=380, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_r, use_container_width=True)
    with right:
        fig_re = px.bar(mix.sort_values("CO2e_u", ascending=False), x="route", y="CO2e_u",
                        title=f"Route emissions contribution ({unit})")
        fig_re.update_layout(height=380, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_re, use_container_width=True)

    byf = d.groupby(["fraction","route"], as_index=False).agg(tonnes=("tonnes","sum"), kgCO2e=("kgCO2e","sum"))
    byf["CO2e_u"] = to_unit(byf["kgCO2e"])
    fig_tr = px.treemap(byf, path=["fraction","route"], values="CO2e_u",
                        title=f"Driver treemap: emissions contribution ({unit})")
    fig_tr.update_layout(height=520, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_tr, use_container_width=True)

    st.dataframe(byf.sort_values("CO2e_u", ascending=False), use_container_width=True, height=380)

# ============================================================
# 7) DIAGNOSTICS
# ============================================================
with tabDiag:
    st.subheader("Diagnostics (factors, coverage, and sanity checks)")

    miss = em_obs[em_obs["factor_missing"] == 1].groupby(["fraction","route"], as_index=False).agg(tonnes=("tonnes","sum"))
    if miss.empty:
        st.success("No missing factors for the routes used.")
    else:
        st.warning("Some routes have no factors and are treated as 0. Add factors if you want them to affect results.")
        st.dataframe(miss.sort_values("tonnes", ascending=False), use_container_width=True, height=360)

    st.markdown("### Factor table in use")
    st.dataframe(factors, use_container_width=True, height=360)

    st.markdown("### Quick national checks")
    check_rows = []
    for y in YEARS:
        tot = float(totals[totals["year"] == y]["TotalCollected"].sum())
        recC = float(rec_col[rec_col["year"] == y]["TotalRecCollec"].sum())
        res = float(residual[residual["year"] == y]["Residual"].sum())
        kpi = float(long[(long["year"] == y) & (long["route"].isin(KPI_ROUTES))]["tonnes"].sum())
        circ = float(long[(long["year"] == y) & (long["route"].isin(CIRCULAR_ROUTES))]["tonnes"].sum())
        check_rows.append({
            "year": y,
            "TotalCollected_sum_t": tot,
            "Residual_sum_t": res,
            "TotalRecCollec_sum_t": recC,
            "Recycled_KPI_sum_t": kpi,
            "CircularRoutes_sum_t": circ,
            "CollectionRate": (recC / tot) if tot > 0 else np.nan,
            "RecyclingRate_KPI": (kpi / tot) if tot > 0 else np.nan,
            "CircularRate": (circ / tot) if tot > 0 else np.nan,
        })
    st.dataframe(pd.DataFrame(check_rows), use_container_width=True)

    st.caption(
        "If TotalRecCollec is higher than expected (e.g., ~12 Mt vs ~10 Mt), it usually means boundary differences "
        "(organics included, different council coverage) or reporting semantics. This tab helps you spot it by year."
    )