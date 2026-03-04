# app.py
# Objective 2 — England LACW Pathway Lab (2020–2023)
# Saeed AlKhoori | Supervisor: Prof. Nikolaos Voulvoulis | Imperial College London (CEP)

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Page
# ----------------------------
st.set_page_config(
    page_title="England LACW Pathway Lab (Objective 2)",
    layout="wide",
)

# ----------------------------
# Config (match your file)
# ----------------------------
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

ROUTES = [
    "Collected",
    "Recycled",
    "Reuse",
    "EfW",
    "Landfill",
    "RDF_MHT",
    "IncNoEnergy",   # IWE / R2
    "AD",
    "CompostedIV",
    "CompostedW",
]

TREATMENT_ROUTES = [r for r in ROUTES if r != "Collected"]

# Circular KPI definition you requested
CIRCULAR_ROUTES = {"Recycled", "Reuse", "AD", "CompostedIV", "CompostedW"}

# Obj1-style “dry recycling collected” = excludes Food + Garden
DRY_FRACTIONS = [f for f in FRACTIONS if f not in ("Food", "Garden")]

# Objective 1 residual composition shares applied to Residual_YYYY (given by you)
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

# Where diverted residual goes (explicit assumptions)
DIVERT_TO = {
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

# ----------------------------
# Factors (kg CO2e per tonne) — your defaults
# ----------------------------
REF_USER = "User-provided factor set (compiled from CarbonWARM + literature notes)."

DEFAULT_FACTORS = pd.DataFrame([
    # Food
    {"fraction":"Food", "route":"EfW",         "kgCO2e_per_t": -37.00,   "source": REF_USER},
    {"fraction":"Food", "route":"Landfill",    "kgCO2e_per_t": 627.00,   "source": REF_USER},
    {"fraction":"Food", "route":"AD",          "kgCO2e_per_t": -78.00,   "source": REF_USER},
    {"fraction":"Food", "route":"CompostedIV", "kgCO2e_per_t": -55.00,   "source": REF_USER},
    {"fraction":"Food", "route":"CompostedW",  "kgCO2e_per_t": 6.00,     "source": REF_USER},
    {"fraction":"Food", "route":"Recycled",    "kgCO2e_per_t": -3779.40, "source": REF_USER},
    {"fraction":"Food", "route":"Reuse",       "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},

    # Garden
    {"fraction":"Garden", "route":"EfW",         "kgCO2e_per_t": -77.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"Landfill",    "kgCO2e_per_t": 579.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"CompostedIV", "kgCO2e_per_t": -45.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"CompostedW",  "kgCO2e_per_t": 56.00,   "source": REF_USER},
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
    {"fraction":"Textiles", "route":"EfW",        "kgCO2e_per_t": 438.00,    "source": REF_USER},
    {"fraction":"Textiles", "route":"Landfill",   "kgCO2e_per_t": 445.00,    "source": REF_USER},
    {"fraction":"Textiles", "route":"Recycled",   "kgCO2e_per_t": -36625.00, "source": REF_USER},
    {"fraction":"Textiles", "route":"Reuse",      "kgCO2e_per_t": 0.0,       "source": "Reuse credit set to 0 unless displacement is modelled."},

    # WEEE
    {"fraction":"WEEE", "route":"EfW",            "kgCO2e_per_t": 450.00,    "source": REF_USER},
    {"fraction":"WEEE", "route":"Landfill",       "kgCO2e_per_t": 20.00,     "source": REF_USER},
    {"fraction":"WEEE", "route":"Recycled",       "kgCO2e_per_t": -10535.94, "source": REF_USER},
    {"fraction":"WEEE", "route":"Reuse",          "kgCO2e_per_t": 0.0,       "source": "Reuse credit set to 0 unless displacement is modelled."},

    # Wood
    {"fraction":"Wood", "route":"EfW",            "kgCO2e_per_t": -318.00,  "source": REF_USER},
    {"fraction":"Wood", "route":"Landfill",       "kgCO2e_per_t": 921.00,   "source": REF_USER},
    {"fraction":"Wood", "route":"Recycled",       "kgCO2e_per_t": -754.50,  "source": REF_USER},
    {"fraction":"Wood", "route":"Reuse",          "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},
])

WILDCARD_DEFAULTS = pd.DataFrame([
    {"fraction":"*", "route":"IncNoEnergy", "kgCO2e_per_t": 360.0, "source":"Placeholder IWE/R2 factor (update if you have better)."},
    {"fraction":"*", "route":"RDF_MHT",     "kgCO2e_per_t": 0.0,   "source":"Not parameterised here; set 0 unless you add factor."},
])

# ----------------------------
# Helpers
# ----------------------------
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
    Also loads council-year series:
      population_YYYY
      TotalCollectedYYYY
      TotalRecCollecYYYY
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

def kpi_circular_rate(long_y: pd.DataFrame, totals_y: pd.DataFrame):
    total = float(totals_y["TotalCollected"].sum())
    circ = float(long_y[long_y["route"].isin(CIRCULAR_ROUTES)]["tonnes"].sum())
    rate = circ / total if total > 0 else np.nan
    return total, circ, rate

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

def baseline_propensity(long_y: pd.DataFrame):
    """
    Per council×fraction propensity based on observed disposal split:
      p_recycled = Recycled / (Recycled + EfW + Landfill + RDF_MHT + IncNoEnergy)
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

def carbon_gain_per_tonne(frac: str, factors_df: pd.DataFrame):
    """
    Improvement proxy: moving 1 tonne from landfill to best circular option for that fraction.
    (Conservative baseline.)
    """
    get = factor_maps(factors_df)
    base = get(frac, "Landfill")
    options = [get(frac, r) for r, _w in DIVERT_TO.get(frac, [])]
    if len(options) == 0:
        return 0.0
    best = np.nanmin(options)
    if np.isnan(base) or np.isnan(best):
        return 0.0
    return float(base - best)

def allocate_diversion(res_div: pd.DataFrame, extra_needed: float, mode: str,
                      long_y: pd.DataFrame, factors_df: pd.DataFrame):
    """
    Allocate additional circular tonnes from divertable residual pool.
    Adds “extra capture” flows (does not subtract baseline flows).
    mode: "easy" | "propensity" | "carbon"
    """
    if extra_needed <= 0:
        return pd.DataFrame(columns=["Council Name", "year", "fraction", "route", "tonnes"])

    pool = res_div.copy()
    pool = pool[pool["fraction"].isin(DIVERT_TO.keys())].copy()
    pool = pool[pool["tonnes"] > 0].copy()
    if pool["tonnes"].sum() <= 0:
        return pd.DataFrame(columns=["Council Name", "year", "fraction", "route", "tonnes"])

    # Attach propensity and carbon gain for ranking
    prop = baseline_propensity(long_y)
    pool = pool.merge(prop, on=["Council Name", "fraction"], how="left")
    pool["p_recycled"] = pool["p_recycled"].fillna(0.0)

    pool["carbon_gain"] = pool["fraction"].apply(lambda f: carbon_gain_per_tonne(f, factors_df))

    if mode == "easy":
        # Scale what already works: high propensity first
        pool["rank"] = -pool["p_recycled"]

    elif mode == "propensity":
        # Allocate proportional to propensity (not purely sorted)
        # Implemented as a “soft” ordering: mid/high propensity earlier, but still uses available tonnes.
        pool["rank"] = -(0.7 * pool["p_recycled"] + 0.3 * (pool["tonnes"] / pool["tonnes"].max()))

    elif mode == "carbon":
        # Maximise emissions benefit first
        pool["rank"] = -pool["carbon_gain"]

    else:
        pool["rank"] = 0.0

    pool = pool.sort_values(["rank"]).copy()

    remaining = float(extra_needed)
    pool["take"] = 0.0
    for i in pool.index:
        if remaining <= 0:
            break
        t = float(pool.loc[i, "tonnes"])
        take = min(t, remaining)
        if take > 0:
            pool.loc[i, "take"] = take
            remaining -= take

    chosen = pool[pool["take"] > 0].copy()
    out_parts = []
    for frac, routes in DIVERT_TO.items():
        sub = chosen[chosen["fraction"] == frac].copy()
        if len(sub) == 0:
            continue
        for route, w in routes:
            tmp = sub[["Council Name", "year", "fraction", "take"]].copy()
            tmp["route"] = route
            tmp["tonnes"] = tmp["take"] * float(w)
            out_parts.append(tmp[["Council Name", "year", "fraction", "route", "tonnes"]])

    if not out_parts:
        return pd.DataFrame(columns=["Council Name", "year", "fraction", "route", "tonnes"])
    return pd.concat(out_parts, ignore_index=True)

def scenario_df(name: str, base_long_y: pd.DataFrame, add_rows: pd.DataFrame):
    base = base_long_y[["Council Name", "year", "fraction", "route", "tonnes", "population"]].copy()
    base["scenario"] = name
    if add_rows is None or len(add_rows) == 0:
        return base
    add = add_rows.copy()
    add["population"] = np.nan
    add["scenario"] = name
    add = add[["Council Name", "year", "fraction", "route", "tonnes", "population", "scenario"]]
    return pd.concat([base, add], ignore_index=True)

def scenario_stack_for_year(long_y: pd.DataFrame, totals_y: pd.DataFrame, residual_y: pd.DataFrame,
                           target_rate: float, factors_df: pd.DataFrame):
    total, circ, base_rate = kpi_circular_rate(long_y, totals_y)
    needed = max(0.0, target_rate * total - circ)

    res_div = build_residual_divertable(residual_y)
    capacity = float(res_div["tonnes"].sum())
    achievable = min(needed, capacity)

    baseline = scenario_df("Baseline (Actual)", long_y, None)
    pol_easy = scenario_df("Policy65-Easy (Scale what works)", long_y, allocate_diversion(res_div, achievable, "easy", long_y, factors_df))
    pol_prop = scenario_df("Policy65-Propensity (Weighted)", long_y, allocate_diversion(res_div, achievable, "propensity", long_y, factors_df))
    pol_car  = scenario_df("Policy65-Carbon-smart (Max benefit)", long_y, allocate_diversion(res_div, achievable, "carbon", long_y, factors_df))

    # Optimal = divert ALL divertable residual (recover all avoidable residual proxy)
    optimal_add = allocate_diversion(res_div, capacity, "easy", long_y, factors_df)
    optimal = scenario_df("Optimal (Recover all divertable residual)", long_y, optimal_add)

    all_scen = pd.concat([baseline, pol_easy, pol_prop, pol_car, optimal], ignore_index=True)
    return all_scen, base_rate, needed, capacity

def sankey_scenario_fraction_route(df: pd.DataFrame, value_col="tonnes", top_links=60, title="Sankey"):
    d = df.groupby(["scenario", "fraction", "route"], as_index=False).agg(v=(value_col, "sum"))
    d = d[d["v"] > 0].copy()
    if len(d) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data for Sankey.")
        return fig

    d = d.sort_values("v", ascending=False).head(int(top_links)).copy()

    scen = list(d["scenario"].unique())
    frac = list(d["fraction"].unique())
    route = list(d["route"].unique())

    nodes = scen + frac + route
    idx = {n: i for i, n in enumerate(nodes)}

    sf = d.groupby(["scenario", "fraction"], as_index=False).agg(v=("v", "sum"))

    source, target, value = [], [], []

    source += [idx[s] for s in sf["scenario"]]
    target += [idx[f] for f in sf["fraction"]]
    value  += sf["v"].tolist()

    for s in scen:
        tmp = d[d["scenario"] == s].groupby(["fraction", "route"], as_index=False).agg(v=("v", "sum"))
        source += [idx[f] for f in tmp["fraction"]]
        target += [idx[r] for r in tmp["route"]]
        value  += tmp["v"].tolist()

    fig = go.Figure(go.Sankey(
        node=dict(pad=12, thickness=14, label=nodes),
        link=dict(source=source, target=target, value=value)
    ))
    fig.update_layout(
        title=title,
        height=720,
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(size=14),
    )
    return fig

def council_year_base(totals: pd.DataFrame, rec_col: pd.DataFrame, residual: pd.DataFrame, long: pd.DataFrame):
    base = totals.merge(rec_col, on=["Council Name", "year"], how="left")
    base = base.merge(residual, on=["Council Name", "year"], how="left")

    pop = long.drop_duplicates(["Council Name", "year"])[["Council Name", "year", "population"]]
    base = base.merge(pop, on=["Council Name", "year"], how="left")

    base["TotalRecCollec"] = base["TotalRecCollec"].fillna(0.0)
    base["Residual"] = base["Residual"].fillna(0.0)

    circ = long[long["route"].isin(CIRCULAR_ROUTES)].groupby(["Council Name", "year"], as_index=False).agg(circular_t=("tonnes","sum"))
    base = base.merge(circ, on=["Council Name", "year"], how="left")
    base["circular_t"] = base["circular_t"].fillna(0.0)

    # Diagnostics
    base["collection_rate"] = np.where(base["TotalCollected"] > 0, base["TotalRecCollec"] / base["TotalCollected"], np.nan)
    base["treatment_circular_rate"] = np.where(base["TotalCollected"] > 0, base["circular_t"] / base["TotalCollected"], np.nan)

    base["post_collection_gap_t"] = np.maximum(0.0, base["TotalRecCollec"] - base["circular_t"])
    base["capture_efficiency"] = np.where(base["TotalRecCollec"] > 0, base["circular_t"] / base["TotalRecCollec"], np.nan)

    # Per-capita
    base["waste_kg_per_cap"] = np.where(base["population"] > 0, base["TotalCollected"] * 1000.0 / base["population"], np.nan)
    base["rec_kg_per_cap"] = np.where(base["population"] > 0, base["TotalRecCollec"] * 1000.0 / base["population"], np.nan)
    base["residual_kg_per_cap"] = np.where(base["population"] > 0, base["Residual"] * 1000.0 / base["population"], np.nan)
    base["gap_kg_per_cap"] = np.where(base["population"] > 0, base["post_collection_gap_t"] * 1000.0 / base["population"], np.nan)

    return base

# ----------------------------
# Sidebar — Data
# ----------------------------
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
    long, totals, rec_col, residual = build_long(df)
except Exception as e:
    st.error(f"Data load/build failed: {e}")
    st.stop()

if totals is None or rec_col is None or residual is None:
    st.error("Missing required series columns: TotalCollectedYYYY, TotalRecCollecYYYY, Residual_YYYY (and optionally population_YYYY).")
    st.stop()

# ----------------------------
# Sidebar — Controls
# ----------------------------
st.sidebar.header("Controls")
unit = st.sidebar.selectbox("Display unit", ["t CO₂e", "kg CO₂e"], index=0)
to_unit = (lambda kg: kg / 1000.0) if unit == "t CO₂e" else (lambda kg: kg)

target_rate = st.sidebar.slider("Policy target circular rate", 0.40, 0.80, 0.65, 0.01)

kpi_basis = st.sidebar.radio(
    "KPI basis for “recycling collected” comparisons",
    ["TotalRecCollec (dry + organics)", "Dry-only (Obj1-style)"],
    index=0
)

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

em_obs = apply_factors(long, factors)
em_obs["CO2e_u"] = to_unit(em_obs["kgCO2e"])

baseCY = council_year_base(totals, rec_col, residual, long)

# ----------------------------
# Header branding
# ----------------------------
top = st.container()
with top:
    c1, c2, c3 = st.columns([1.2, 3.6, 1.2], vertical_alignment="center")

    with c1:
        if os.path.exists("assets/imperial.png"):
            st.image("assets/imperial.png", use_container_width=True)
    with c2:
        st.markdown(
            """
            <div style="text-align:center; line-height:1.15;">
              <div style="font-size:34px; font-weight:800;">England LACW Pathway Lab</div>
              <div style="font-size:16px; opacity:0.85;">
                Objective 2 — Treatment pathways & carbon intensities (2020–2023)
              </div>
              <div style="font-size:14px; opacity:0.75; margin-top:6px;">
                Saeed AlKhoori · Supervisor: Prof. Nikolaos Voulvoulis · Imperial College London (CEP)
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        if os.path.exists("assets/cep.png"):
            st.image("assets/cep.png", use_container_width=True)

st.divider()

with st.expander("Method notes (definitions & why totals can differ)"):
    st.write(
        "- **TotalCollected**: your system boundary (residual + collected-for-recycling streams).\n"
        "- **TotalRecCollec**: front-end “collected for recycling” in the dataset. In some datasets this can include **organics**.\n"
        "- **Circular treatment** (KPI numerator): Recycled + Reuse + AD + Compost (IV/W).\n"
        "- **Post-collection gap** is a diagnostic: collected-for-recycling that does not appear in circular treatment routes "
        "(e.g., rejects/contamination, reporting semantics, export accounting differences)."
    )

# ----------------------------
# Tabs
# ----------------------------
tabTS, tabScen, tabSank, tabCouncil, tabEff, tabDiag = st.tabs([
    "1) Time series",
    "2) Scenarios (Policy65 + Optimal)",
    "3) Sankey",
    "4) Council explorer",
    "5) Collection → Fate efficiency",
    "6) Diagnostics",
])

# ----------------------------
# 1) Time series
# ----------------------------
with tabTS:
    st.subheader("National time series (2020–2023)")

    rows = []
    for y in YEARS:
        by = baseCY[baseCY["year"] == y].copy()

        total = float(by["TotalCollected"].sum())
        recC = float(by["TotalRecCollec"].sum())
        residualT = float(by["Residual"].sum())
        circT = float(by["circular_t"].sum())

        kg = float(em_obs[em_obs["year"] == y]["kgCO2e"].sum())

        rows.append({
            "year": y,
            "TotalCollected_t": total,
            "Residual_t": residualT,
            "TotalRecCollec_t": recC,
            "CircularTreatment_t": circT,
            "CircularRate": (circT / total) if total > 0 else np.nan,
            "CollectionRate": (recC / total) if total > 0 else np.nan,
            "CaptureEfficiency_nat": (circT / recC) if recC > 0 else np.nan,
            "CO2e_u": to_unit(kg),
        })

    nat = pd.DataFrame(rows)
    y_latest = YEARS[-1]
    nat_latest = nat[nat["year"] == y_latest].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{y_latest} circular rate", f"{nat_latest['CircularRate']*100:.1f}%")
    c2.metric(f"{y_latest} collection rate", f"{nat_latest['CollectionRate']*100:.1f}%")
    c3.metric(f"{y_latest} emissions ({unit})", f"{nat_latest['CO2e_u']:,.0f}")
    c4.metric(f"{y_latest} capture efficiency", f"{nat_latest['CaptureEfficiency_nat']*100:.1f}%" if pd.notna(nat_latest["CaptureEfficiency_nat"]) else "NA")

    left, right = st.columns(2)
    with left:
        fig_rate = px.line(nat, x="year", y="CircularRate", markers=True, title="Circular treatment rate")
        fig_rate.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig_rate, use_container_width=True)
    with right:
        fig_em = px.line(nat, x="year", y="CO2e_u", markers=True, title=f"Total emissions ({unit})")
        fig_em.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_em, use_container_width=True)

    st.markdown("### Route mix over time (tonnes)")
    rm = long[long["route"].isin(TREATMENT_ROUTES)].groupby(["year","route"], as_index=False).agg(tonnes=("tonnes","sum"))
    fig_mix = px.area(rm, x="year", y="tonnes", color="route", title="Route mix (stacked)")
    fig_mix.update_layout(height=460, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_mix, use_container_width=True)

# ----------------------------
# 2) Scenarios
# ----------------------------
with tabScen:
    st.subheader("Scenario engine (Baseline, Policy65 variants, Optimal)")

    scen_all = []
    meta = []
    for y in YEARS:
        ly = long[long["year"] == y].copy()
        ty = totals[totals["year"] == y].copy()
        ry = residual[residual["year"] == y].copy()

        sc_y, base_rate_y, needed_y, cap_y = scenario_stack_for_year(ly, ty, ry, target_rate, factors)
        scen_all.append(sc_y)
        meta.append({"year": y, "baseline_rate": base_rate_y, "extra_needed_t": needed_y, "divertable_capacity_t": cap_y})

    scen_long = pd.concat(scen_all, ignore_index=True)
    scen_em = apply_factors(scen_long, factors)
    scen_em["CO2e_u"] = to_unit(scen_em["kgCO2e"])

    meta_df = pd.DataFrame(meta)
    st.dataframe(meta_df, use_container_width=True, height=170)

    year = st.selectbox("Year", options=YEARS, index=2, key="sc_year")

    # KPI computation for selected year
    ty = totals[totals["year"] == year].copy()
    totalY = float(ty["TotalCollected"].sum())

    d = scen_long[(scen_long["year"] == year)].copy()

    summ = d.groupby("scenario", as_index=False).agg(
        tonnes=("tonnes","sum")
    )

    circ = d[d["route"].isin(CIRCULAR_ROUTES)].groupby("scenario", as_index=False).agg(Circular_t=("tonnes","sum"))
    summ = summ.merge(circ, on="scenario", how="left")
    summ["Circular_t"] = summ["Circular_t"].fillna(0.0)
    summ["CircularRate"] = np.where(totalY > 0, summ["Circular_t"] / totalY, np.nan)

    # Emissions
    ems = scen_em[scen_em["year"] == year].groupby("scenario", as_index=False).agg(kgCO2e=("kgCO2e","sum"))
    ems["CO2e_u"] = to_unit(ems["kgCO2e"])
    summ = summ.merge(ems[["scenario","CO2e_u"]], on="scenario", how="left")

    # Warn if policy target unreachable
    needed = float(meta_df.loc[meta_df["year"] == year, "extra_needed_t"].iloc[0])
    cap = float(meta_df.loc[meta_df["year"] == year, "divertable_capacity_t"].iloc[0])
    if needed > cap + 1e-9:
        st.warning(f"Policy target {target_rate:.0%} is NOT fully reachable in {year} with current divertable residual capacity. "
                   f"Needed extra: {needed:,.0f} t, capacity: {cap:,.0f} t.")

    # KPI bar
    fig_kpi = px.bar(
        summ.sort_values("CircularRate"),
        x="scenario", y="CircularRate", text="CircularRate",
        title="Recycling rate (KPI) by scenario (selected year)"
    )
    fig_kpi.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10), yaxis_tickformat=".0%")
    fig_kpi.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_kpi, use_container_width=True)

    # Route mix by scenario
    mix = d[d["route"].isin(TREATMENT_ROUTES)].groupby(["scenario","route"], as_index=False).agg(tonnes=("tonnes","sum"))
    fig_mix = px.bar(mix, x="scenario", y="tonnes", color="route", barmode="stack",
                     title="Route mix by scenario (selected year)")
    fig_mix.update_layout(height=520, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_mix, use_container_width=True)

# ----------------------------
# 3) Sankey
# ----------------------------
with tabSank:
    st.subheader("Sankey (Scenario → Fraction → Route)")

    year = st.selectbox("Year", options=YEARS, index=2, key="sank_year")
    top_links = st.slider("Keep top links (readability)", 20, 200, 70, 10)
    metric = st.radio("Size links by", ["Tonnes", "Emissions magnitude |kgCO₂e|"], horizontal=True)

    dd = scen_em[(scen_em["year"] == year) & (scen_em["route"].isin(TREATMENT_ROUTES))].copy()
    if metric.startswith("Emissions"):
        dd["abs_kg"] = np.abs(dd["kgCO2e"])
        fig = sankey_scenario_fraction_route(dd, value_col="abs_kg", top_links=top_links,
                                             title="Sankey sized by |kgCO₂e| (top links)")
    else:
        fig = sankey_scenario_fraction_route(dd, value_col="tonnes", top_links=top_links,
                                             title="Sankey sized by tonnes (top links)")

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 4) Council explorer
# ----------------------------
with tabCouncil:
    st.subheader("Council explorer")

    year = st.selectbox("Year", YEARS, index=2, key="cx_year")
    councils = sorted(long[long["year"] == year]["Council Name"].unique().tolist())
    council = st.selectbox("Council", councils, index=0, key="cx_council")

    cy = baseCY[(baseCY["year"] == year) & (baseCY["Council Name"] == council)]
    if len(cy) == 0:
        st.info("No totals for this council/year.")
        st.stop()
    cy = cy.iloc[0]

    d = em_obs[(em_obs["year"] == year) & (em_obs["Council Name"] == council)].copy()
    if len(d) == 0:
        st.info("No pathway data for this council/year.")
        st.stop()

    total_col = float(cy["TotalCollected"])
    pop = float(cy["population"]) if pd.notna(cy["population"]) else np.nan
    circ_t = float(d[d["route"].isin(CIRCULAR_ROUTES)]["tonnes"].sum())
    circ_rate = (circ_t / total_col) if total_col > 0 else np.nan
    kg = float(d["kgCO2e"].sum())
    intensity = (kg / total_col) if total_col > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total collected (t)", f"{total_col:,.0f}")
    c2.metric("Circular rate", f"{circ_rate*100:.1f}%")
    c3.metric("Intensity (kgCO₂e/t)", f"{intensity:.1f}")
    c4.metric("kgCO₂e per cap", f"{(kg/pop):.1f}" if (pop and pop > 0) else "NA")

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

# ----------------------------
# 5) Collection → Fate efficiency
# ----------------------------
with tabEff:
    st.subheader("Collection → Fate efficiency (diagnostic)")

    y = st.selectbox("Year", YEARS, index=2, key="eff_year")
    d = baseCY[baseCY["year"] == y].copy()

    nat_rec = float(d["TotalRecCollec"].sum())
    nat_circ = float(d["circular_t"].sum())
    nat_gap = float(d["post_collection_gap_t"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("National TotalRecCollec (t)", f"{nat_rec:,.0f}")
    c2.metric("National circular treatment (t)", f"{nat_circ:,.0f}")
    c3.metric("National post-collection gap (t)", f"{nat_gap:,.0f}")
    c4.metric("National capture efficiency", f"{(nat_circ/nat_rec)*100:.1f}%" if nat_rec > 0 else "NA")

    fig_hist = px.histogram(d, x="capture_efficiency", nbins=40, title="Capture efficiency distribution across councils")
    fig_hist.update_layout(height=380, margin=dict(l=10,r=10,t=60,b=10), xaxis_tickformat=".0%")
    st.plotly_chart(fig_hist, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Lowest 25 capture efficiency")
        st.dataframe(
            d.sort_values("capture_efficiency", ascending=True)[
                ["Council Name","TotalCollected","TotalRecCollec","Residual","circular_t",
                 "post_collection_gap_t","capture_efficiency"]
            ].head(25),
            use_container_width=True,
            height=420
        )
    with right:
        st.markdown("### Highest 25 capture efficiency")
        st.dataframe(
            d.sort_values("capture_efficiency", ascending=False)[
                ["Council Name","TotalCollected","TotalRecCollec","Residual","circular_t",
                 "post_collection_gap_t","capture_efficiency"]
            ].head(25),
            use_container_width=True,
            height=420
        )

# ----------------------------
# 6) Diagnostics
# ----------------------------
with tabDiag:
    st.subheader("Diagnostics")

    miss = em_obs[em_obs["factor_missing"] == 1].groupby(["fraction","route"], as_index=False).agg(tonnes=("tonnes","sum"))
    if len(miss) == 0:
        st.success("No missing factors for the routes used (or missing routes have zero tonnes).")
    else:
        st.warning("Some fraction×route pairs have no factor and are treated as 0 in emissions.")
        st.dataframe(miss.sort_values("tonnes", ascending=False), use_container_width=True, height=360)

    st.markdown("### Factor table in use")
    st.dataframe(factors, use_container_width=True, height=360)

    st.markdown("### Quick national checks")
    check_rows = []
    for y in YEARS:
        tot = float(totals[totals["year"] == y]["TotalCollected"].sum())
        rec = float(rec_col[rec_col["year"] == y]["TotalRecCollec"].sum())
        res = float(residual[residual["year"] == y]["Residual"].sum())
        circ = float(long[(long["year"] == y) & (long["route"].isin(CIRCULAR_ROUTES))]["tonnes"].sum())
        check_rows.append({
            "year": y,
            "TotalCollected_sum_t": tot,
            "Residual_sum_t": res,
            "TotalRecCollec_sum_t": rec,
            "CircularRoutes_sum_t": circ,
            "CollectionRate_nat": (rec / tot) if tot > 0 else np.nan,
            "CircularRate_nat": (circ / tot) if tot > 0 else np.nan,
            "CaptureEfficiency_nat": (circ / rec) if rec > 0 else np.nan,
        })
    st.dataframe(pd.DataFrame(check_rows), use_container_width=True)