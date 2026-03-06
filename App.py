# app.py
# Objective 2 — England LACW Pathway Lab (2020–2023)
# Saeed AlKhoori | Supervisor: Prof. Nikolaos Voulvoulis | Imperial College London (CEP)
#
# What this version fixes / adds (vs your previous build):
# 1) No negative tonnes in scenario route-mix charts (clean mass-balance)
#    - We build a synthetic residual-disposal layer for divertable residual fractions
#    - Diversion subtracts ONLY from that layer, never from observed collected-stream flows
# 2) Policy65 sub-scenarios are genuinely distinct in their *allocation logic*
#    - Scale-first (largest residual first)
#    - Propensity (council historic destination mix among recovery routes)
#    - Carbon-smart (max carbon benefit first)
# 3) Sankey uses tCO2e (so you don’t see “G kg”)
# 4) Two-axis plot is interpretable
#    - Labels on bubbles (fraction names)
#    - Axis titles + “bubble size = diverted tonnes”
# 5) Adds an economic value layer
#    - Simple default value table (GBP/t mid, min, max) + sources
#    - Council-level value + carbon summaries + CSV exports
#
# NOTE:
# - RDF_MHT is treated as an INTERMEDIATE transfer category with 0 direct factor (footnote in paper).
# - Residual modelling here covers "divertable residual fractions" (from Objective 1 shares).
#   If you later want full residual (including non-avoidable) emissions, we can extend.

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
# Config (match your CSV)
# ----------------------------
CSV_NAME = "pathway_data.csv"
YEARS = [2020, 2021, 2022, 2023]

# 10 fractions (your 10th is OtherRecyclables)
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

# Routes in your dataset
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

# “Recycling rate” measure used for Policy65 in this tool (DEFRA-style municipal recycling incl. organics)
RECOVERY_ROUTES = {"Recycled", "Reuse", "AD", "CompostedIV", "CompostedW"}

# Disposal / non-recovery routes
DISPOSAL_ROUTES = {"EfW", "Landfill", "RDF_MHT", "IncNoEnergy"}

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

# Divertable fractions (avoidables proxy) – keep only those that are in FRACTIONS list
DIVERTABLE_FRACTIONS = [f for f in RESIDUAL_SHARES.keys() if f in FRACTIONS]

# ----------------------------
# Scenario destination policies
# ----------------------------
# "Easy" (fast rollout): simple deterministic mapping to recovery routes
EASY_DEST = {
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

# Allowed recovery options per fraction for carbon-smart routing
CARBON_ALLOWED = {
    "Food": ["AD", "CompostedIV", "CompostedW", "Recycled", "Reuse"],
    "Garden": ["CompostedIV", "CompostedW", "Recycled", "Reuse"],
    "PaperCard": ["Recycled", "Reuse"],
    "Plastics": ["Recycled", "Reuse"],
    "Glass": ["Recycled", "Reuse"],
    "Metals": ["Recycled", "Reuse"],
    "Textiles": ["Recycled", "Reuse"],
    "WEEE": ["Recycled", "Reuse"],
    "Wood": ["Recycled", "Reuse"],
    "OtherRecyclables": ["Recycled", "Reuse"],
}

# ----------------------------
# Carbon factors (kg CO2e per tonne) — your defaults
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

    # OtherRecyclables (placeholder – adjust when you decide what it really contains)
    {"fraction":"OtherRecyclables", "route":"EfW",      "kgCO2e_per_t": 0.0, "source": "Placeholder (set when composition is clarified)."},
    {"fraction":"OtherRecyclables", "route":"Landfill", "kgCO2e_per_t": 0.0, "source": "Placeholder (set when composition is clarified)."},
    {"fraction":"OtherRecyclables", "route":"Recycled", "kgCO2e_per_t": 0.0, "source": "Placeholder (set when composition is clarified)."},
    {"fraction":"OtherRecyclables", "route":"Reuse",    "kgCO2e_per_t": 0.0, "source": "Placeholder (set when composition is clarified)."},
])

# Wildcards: apply when a route exists but fraction-specific factor is missing
WILDCARD_DEFAULTS = pd.DataFrame([
    {"fraction":"*", "route":"IncNoEnergy", "kgCO2e_per_t": 360.0, "source":"Literature-based placeholder IWE/R2 (update if you have a preferred value)."},
    {"fraction":"*", "route":"RDF_MHT",     "kgCO2e_per_t": 0.0,   "source":"RDF_MHT treated as intermediate transfer; 0 direct factor to avoid double counting."},
])

FACTORS = pd.concat([DEFAULT_FACTORS, WILDCARD_DEFAULTS], ignore_index=True)

# ----------------------------
# Economic value factors (GBP per tonne)
# ----------------------------
# These are DEFAULT placeholders until you finalise a proper UK MPR / WRAP mapping.
# You can replace these with your own validated values and cite the source in "source".
VALUE_FACTORS = pd.DataFrame([
    {"fraction":"Food",            "gbp_per_t_min": 0.0,   "gbp_per_t_mid": 0.0,   "gbp_per_t_max": 0.0,   "source":"Assumed 0 (no commodity revenue modelled for organics)."},
    {"fraction":"Garden",          "gbp_per_t_min": 0.0,   "gbp_per_t_mid": 0.0,   "gbp_per_t_max": 0.0,   "source":"Assumed 0 (no commodity revenue modelled for organics)."},
    {"fraction":"PaperCard",       "gbp_per_t_min": 50.0,  "gbp_per_t_mid": 100.0, "gbp_per_t_max": 150.0, "source":"Placeholder range (replace with UK MPR/WRAP grade price)."},
    {"fraction":"Plastics",        "gbp_per_t_min": 200.0, "gbp_per_t_mid": 400.0, "gbp_per_t_max": 600.0, "source":"Placeholder range (replace with polymer/grade MPR)."},
    {"fraction":"Glass",           "gbp_per_t_min": 0.0,   "gbp_per_t_mid": 20.0,  "gbp_per_t_max": 60.0,  "source":"Placeholder range (replace with cullet price)."},
    {"fraction":"Metals",          "gbp_per_t_min": 300.0, "gbp_per_t_mid": 600.0, "gbp_per_t_max": 1200.0,"source":"Placeholder range (replace with ferrous/non-ferrous blend)."},
    {"fraction":"Textiles",        "gbp_per_t_min": 50.0,  "gbp_per_t_mid": 200.0, "gbp_per_t_max": 500.0, "source":"Placeholder range (replace with reuse/recycling value split)."},
    {"fraction":"WEEE",            "gbp_per_t_min": 200.0, "gbp_per_t_mid": 500.0, "gbp_per_t_max": 1500.0,"source":"Placeholder range (replace with WEEE commodity recovery values)."},
    {"fraction":"Wood",            "gbp_per_t_min": 0.0,   "gbp_per_t_mid": 30.0,  "gbp_per_t_max": 80.0,  "source":"Placeholder range (replace with wood grade value)."},
    {"fraction":"OtherRecyclables","gbp_per_t_min": 0.0,   "gbp_per_t_mid": 50.0,  "gbp_per_t_max": 150.0, "source":"Placeholder range (replace with your definition)."},
])

VALUE_ROUTES = {"Recycled", "Reuse"}  # where commodity value is counted (simple assumption)

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

def value_map(values_df: pd.DataFrame):
    m = {r["fraction"]: float(r["gbp_per_t_mid"]) for _, r in values_df.iterrows()}
    def get(frac):
        return m.get(frac, 0.0)
    return get

@st.cache_data(show_spinner=False)
def load_csv(path: str):
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner=False)
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

def apply_values(long_df: pd.DataFrame, values_df: pd.DataFrame):
    getv = value_map(values_df)
    out = long_df.copy()
    out["gbp_per_t_mid"] = [getv(f) for f in out["fraction"]]
    out["gbp_value"] = np.where(out["route"].isin(VALUE_ROUTES), out["tonnes"] * out["gbp_per_t_mid"], 0.0)
    return out

@st.cache_data(show_spinner=False)
def council_year_base(totals: pd.DataFrame, rec_col: pd.DataFrame, residual: pd.DataFrame, long: pd.DataFrame):
    base = totals.merge(rec_col, on=["Council Name", "year"], how="left")
    base = base.merge(residual, on=["Council Name", "year"], how="left")

    pop = long.drop_duplicates(["Council Name", "year"])[["Council Name", "year", "population"]]
    base = base.merge(pop, on=["Council Name", "year"], how="left")

    base["TotalRecCollec"] = base["TotalRecCollec"].fillna(0.0)
    base["Residual"] = base["Residual"].fillna(0.0)

    # Recovery tonnes (routes counted in the “recycling rate” metric)
    rec_t = (long[long["route"].isin(RECOVERY_ROUTES)]
             .groupby(["Council Name", "year"], as_index=False)
             .agg(recovery_t=("tonnes","sum")))
    base = base.merge(rec_t, on=["Council Name", "year"], how="left")
    base["recovery_t"] = base["recovery_t"].fillna(0.0)

    base["recovery_rate"] = np.where(base["TotalCollected"] > 0, base["recovery_t"] / base["TotalCollected"], np.nan)
    return base

def recovery_rate_national(long_y: pd.DataFrame, totals_y: pd.DataFrame):
    total = float(totals_y["TotalCollected"].sum())
    rec = float(long_y[long_y["route"].isin(RECOVERY_ROUTES)]["tonnes"].sum())
    rate = rec / total if total > 0 else np.nan
    return total, rec, rate

def build_residual_by_fraction(residual_y: pd.DataFrame):
    parts = []
    for frac, share in RESIDUAL_SHARES.items():
        if frac in FRACTIONS:
            parts.append(pd.DataFrame({
                "Council Name": residual_y["Council Name"],
                "year": residual_y["year"],
                "fraction": frac,
                "residual_frac_t": residual_y["Residual"] * float(share)
            }))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["Council Name", "year", "fraction", "residual_frac_t"]
    )
    out = out[out["fraction"].isin(DIVERTABLE_FRACTIONS)].copy()
    return out

def disposal_shares_from_data_by_council(long_y: pd.DataFrame):
    """
    Council-year disposal route shares inferred from observed disposal totals (ALL fractions combined).
    If a council has zero disposal recorded, default EfW=1.
    """
    g = (long_y[long_y["route"].isin(DISPOSAL_ROUTES)]
         .groupby(["Council Name","year","route"], as_index=False)
         .agg(t=("tonnes","sum")))

    if len(g) == 0:
        return pd.DataFrame(columns=["Council Name","year","route","share"])

    piv = g.pivot_table(index=["Council Name","year"], columns="route", values="t", aggfunc="sum").fillna(0.0)
    for r in DISPOSAL_ROUTES:
        if r not in piv.columns:
            piv[r] = 0.0
    s = piv[list(DISPOSAL_ROUTES)].sum(axis=1).replace(0, np.nan)
    shares = piv.div(s, axis=0).fillna(0.0)

    all_zero = (piv[list(DISPOSAL_ROUTES)].sum(axis=1) == 0)
    if all_zero.any():
        shares.loc[all_zero, :] = 0.0
        shares.loc[all_zero, "EfW"] = 1.0

    out = shares.reset_index().melt(id_vars=["Council Name","year"], var_name="route", value_name="share")
    return out

def build_synthetic_residual_disposal(residual_frac: pd.DataFrame, disp_shares: pd.DataFrame):
    """
    Synthetic residual disposal layer:
    residual (by fraction) allocated to disposal routes using council disposal shares.
    This is the ONLY layer we subtract from when we divert.
    """
    m = residual_frac.merge(disp_shares, on=["Council Name","year"], how="left")
    m["share"] = m["share"].fillna(0.0)
    m["tonnes"] = m["residual_frac_t"] * m["share"]
    m = m[m["tonnes"] > 0].copy()
    m["route"] = m["route"].astype(str)
    return m[["Council Name","year","fraction","route","tonnes"]]

def hist_propensity_routes(long_all: pd.DataFrame):
    """
    For propensity scenario:
    - Council×fraction×route shares among RECOVERY_ROUTES, computed across ALL YEARS (2020–2023)
    """
    df = long_all[long_all["route"].isin(RECOVERY_ROUTES)].copy()
    if len(df) == 0:
        shares = pd.DataFrame(columns=["Council Name","fraction","route","w"])
        return shares

    piv = df.pivot_table(index=["Council Name","fraction"], columns="route", values="tonnes", aggfunc="sum").fillna(0.0)
    for r in RECOVERY_ROUTES:
        if r not in piv.columns:
            piv[r] = 0.0

    denom = piv[list(RECOVERY_ROUTES)].sum(axis=1).replace(0, np.nan)
    shares_mat = piv[list(RECOVERY_ROUTES)].div(denom, axis=0).fillna(0.0)

    shares = shares_mat.reset_index().melt(id_vars=["Council Name","fraction"], var_name="route", value_name="w")
    shares = shares[shares["w"] > 0].copy()
    return shares

def carbon_best_route(frac: str, factors_df: pd.DataFrame):
    get = factor_maps(factors_df)
    allowed = CARBON_ALLOWED.get(frac, [])
    vals = []
    for r in allowed:
        v = get(frac, r)
        if not np.isnan(v):
            vals.append((r, float(v)))
    if not vals:
        return EASY_DEST.get(frac, [("Recycled", 1.0)])[0][0]
    vals.sort(key=lambda x: x[1])  # min kgCO2e/t
    return vals[0][0]

def carbon_gain_per_tonne(frac: str, factors_df: pd.DataFrame, base_disposal_mix: dict):
    """
    Carbon benefit of diverting 1 t of fraction from baseline disposal (weighted) to best recovery route.
    base_disposal_mix: dict route->share (national, or council-level – here we use national for ranking)
    """
    get = factor_maps(factors_df)
    best_route = carbon_best_route(frac, factors_df)
    best = get(frac, best_route)
    if np.isnan(best):
        return 0.0

    # weighted baseline disposal factor
    base = 0.0
    any_base = False
    for r, sh in base_disposal_mix.items():
        v = get(frac, r)
        if not np.isnan(v):
            base += float(sh) * float(v)
            any_base = True
    if not any_base:
        # fallback: Landfill then EfW
        vL = get(frac, "Landfill")
        vE = get(frac, "EfW")
        base = vL if not np.isnan(vL) else (vE if not np.isnan(vE) else 0.0)

    return float(base - best)

def choose_destination_weights(mode: str, chosen_rows: pd.DataFrame, shares_hist: pd.DataFrame, factors_df: pd.DataFrame):
    """
    Returns a dict: (Council Name, fraction) -> list[(route, weight)] among RECOVERY_ROUTES
    """
    dest = {}
    for _, r in chosen_rows.iterrows():
        c = r["Council Name"]
        f = r["fraction"]

        if mode == "scale":
            dest[(c, f)] = EASY_DEST.get(f, [("Recycled", 1.0)])

        elif mode == "carbon":
            br = carbon_best_route(f, factors_df)
            dest[(c, f)] = [(br, 1.0)]

        elif mode == "propensity":
            sub = shares_hist[(shares_hist["Council Name"] == c) & (shares_hist["fraction"] == f)].copy()
            if len(sub) == 0:
                dest[(c, f)] = EASY_DEST.get(f, [("Recycled", 1.0)])
            else:
                s = float(sub["w"].sum())
                if s <= 0:
                    dest[(c, f)] = EASY_DEST.get(f, [("Recycled", 1.0)])
                else:
                    tmp = [(rr["route"], float(rr["w"]) / s) for _, rr in sub.iterrows()]
                    dest[(c, f)] = tmp
        else:
            dest[(c, f)] = EASY_DEST.get(f, [("Recycled", 1.0)])

    return dest

def national_disposal_mix(long_y: pd.DataFrame):
    """National disposal shares from observed disposal totals (all councils, all fractions)."""
    g = (long_y[long_y["route"].isin(DISPOSAL_ROUTES)]
         .groupby("route", as_index=False).agg(t=("tonnes","sum")))
    if len(g) == 0:
        return {r: (1.0 if r == "EfW" else 0.0) for r in DISPOSAL_ROUTES}
    total = float(g["t"].sum())
    if total <= 0:
        return {r: (1.0 if r == "EfW" else 0.0) for r in DISPOSAL_ROUTES}
    mix = {r: 0.0 for r in DISPOSAL_ROUTES}
    for _, rr in g.iterrows():
        mix[str(rr["route"])] = float(rr["t"]) / total
    return mix

def allocate_diversion_massbalanced(
    residual_frac: pd.DataFrame,
    residual_disp_synth: pd.DataFrame,
    long_all: pd.DataFrame,
    long_y: pd.DataFrame,
    extra_needed: float,
    mode: str,
    factors_df: pd.DataFrame
):
    """
    Mass-balanced diversion on the SYNTHETIC residual disposal layer only:
    - Take tonnes from residual pool (council×fraction)
    - Allocate to destination recovery routes (policy-dependent)
    - Subtract the same tonnes from synthetic residual disposal routes proportionally
    Returns adjustment rows with +/- tonnes (all within the synthetic residual layer + added recovery).
    """
    if extra_needed <= 0:
        return pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    pool = residual_frac.copy()
    pool = pool[pool["residual_frac_t"] > 0].copy()
    if len(pool) == 0:
        return pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    # propensity destination shares (historic)
    shares_hist = hist_propensity_routes(long_all)

    # national disposal mix for carbon ranking
    nat_disp_mix = national_disposal_mix(long_y)

    # carbon gain proxy
    pool["carbon_gain"] = pool["fraction"].apply(lambda f: carbon_gain_per_tonne(f, factors_df, nat_disp_mix))

    # Ranking who gets diverted first
    if mode == "scale":
        # scale-first: largest residual mass first
        pool["rank"] = -(pool["residual_frac_t"])

    elif mode == "propensity":
        # capability-ish: use historic recovery share among recovery routes as proxy + scale
        # If a council never historically recovered a fraction, it will be deprioritised.
        # Build a simple p_recovery proxy from shares_hist existence (sum w for that council×fraction)
        tmp = shares_hist.groupby(["Council Name","fraction"], as_index=False).agg(p=("w","sum"))
        pool = pool.merge(tmp, on=["Council Name","fraction"], how="left")
        pool["p"] = pool["p"].fillna(0.0).clip(0, 1)
        scale_norm = pool["residual_frac_t"] / (pool["residual_frac_t"].max() if pool["residual_frac_t"].max() > 0 else 1.0)
        pool["rank"] = -(0.7 * pool["p"] + 0.3 * scale_norm)

    elif mode == "carbon":
        # carbon-smart: biggest carbon benefit first (benefit * tonnes)
        pool["rank"] = -(pool["carbon_gain"] * pool["residual_frac_t"])

    else:
        pool["rank"] = 0.0

    pool = pool.sort_values(["rank"]).copy()

    # choose tonnes to divert
    remaining = float(extra_needed)
    pool["take"] = 0.0
    for i in pool.index:
        if remaining <= 0:
            break
        cap = float(pool.loc[i, "residual_frac_t"])
        take = min(cap, remaining)
        if take > 0:
            pool.loc[i, "take"] = take
            remaining -= take

    chosen = pool[pool["take"] > 0].copy()
    if len(chosen) == 0:
        return pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    # destination weights among recovery routes
    dest_map = choose_destination_weights(mode, chosen, shares_hist, factors_df)

    # Positive destination rows (add recovery tonnes)
    pos_parts = []
    for _, r in chosen.iterrows():
        c = r["Council Name"]; y = int(r["year"]); f = r["fraction"]; t = float(r["take"])
        weights = dest_map.get((c, f), EASY_DEST.get(f, [("Recycled", 1.0)]))
        for route, w in weights:
            pos_parts.append({"Council Name": c, "year": y, "fraction": f, "route": route, "tonnes": t * float(w)})

    pos = pd.DataFrame(pos_parts) if pos_parts else pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    # Negative rows: subtract from synthetic residual disposal mix (council×fraction×route)
    avail = (residual_disp_synth
             .groupby(["Council Name","year","fraction","route"], as_index=False)
             .agg(avail_t=("tonnes","sum")))

    neg_parts = []
    for _, r in chosen.iterrows():
        c = r["Council Name"]; y = int(r["year"]); f = r["fraction"]; take = float(r["take"])
        sub = avail[(avail["Council Name"] == c) & (avail["year"] == y) & (avail["fraction"] == f)].copy()
        if len(sub) == 0:
            continue
        tot_av = float(sub["avail_t"].sum())
        if tot_av <= 0:
            continue

        sub["share"] = sub["avail_t"] / tot_av
        sub["sub_t"] = sub["share"] * take
        sub["sub_t"] = np.minimum(sub["sub_t"], sub["avail_t"])

        for _, rr in sub.iterrows():
            if rr["sub_t"] > 0:
                neg_parts.append({
                    "Council Name": c,
                    "year": y,
                    "fraction": f,
                    "route": rr["route"],
                    "tonnes": -float(rr["sub_t"])
                })

        # Update avail to prevent double-subtracting
        for idx in sub.index:
            a = float(avail.loc[idx, "avail_t"])
            s = float(sub.loc[idx, "sub_t"])
            avail.loc[idx, "avail_t"] = max(0.0, a - s)

    neg = pd.DataFrame(neg_parts) if neg_parts else pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    adj = pd.concat([pos, neg], ignore_index=True)
    if len(adj):
        adj["tonnes"] = adj["tonnes"].astype(float)
        adj = adj.groupby(["Council Name","year","fraction","route"], as_index=False).agg(tonnes=("tonnes","sum"))
        adj = adj[np.abs(adj["tonnes"]) > 1e-9].copy()
    return adj

def scenario_df(name: str, collected_stream_long_y: pd.DataFrame, residual_disp_synth: pd.DataFrame, adj_rows: pd.DataFrame):
    """
    Scenario dataset = (observed collected-for-recycling stream) + (synthetic residual disposal layer) + (scenario adjustments)
    This prevents double-counting and prevents negative tonnes in plots.
    """
    base1 = collected_stream_long_y[["Council Name","year","fraction","route","tonnes","population"]].copy()
    base1["scenario"] = name
    base1["stream"] = "Collected_stream"

    base2 = residual_disp_synth.copy()
    if "population" not in base2.columns:
        base2 = base2.merge(
            collected_stream_long_y.drop_duplicates(["Council Name","year"])[["Council Name","year","population"]],
            on=["Council Name","year"],
            how="left"
        )
    base2 = base2[["Council Name","year","fraction","route","tonnes","population"]].copy()
    base2["scenario"] = name
    base2["stream"] = "Residual_synthetic"

    out = pd.concat([base1, base2], ignore_index=True)

    if adj_rows is None or len(adj_rows) == 0:
        return out

    add = adj_rows.copy()
    add["population"] = np.nan
    add["scenario"] = name
    add["stream"] = "Scenario_adjustment"
    add = add[["Council Name","year","fraction","route","tonnes","population","scenario","stream"]]
    out = pd.concat([out, add], ignore_index=True)
    return out

def sankey_scenario_fraction_route(df: pd.DataFrame, value_col="tonnes", top_links=60, title="Sankey"):
    d = df.groupby(["scenario","fraction","route"], as_index=False).agg(v=(value_col, "sum"))
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

    sf = d.groupby(["scenario","fraction"], as_index=False).agg(v=("v","sum"))

    source, target, value = [], [], []
    source += [idx[s] for s in sf["scenario"]]
    target += [idx[f] for f in sf["fraction"]]
    value  += sf["v"].tolist()

    for s in scen:
        tmp = d[d["scenario"] == s].groupby(["fraction","route"], as_index=False).agg(v=("v","sum"))
        source += [idx[f] for f in tmp["fraction"]]
        target += [idx[r] for r in tmp["route"]]
        value  += tmp["v"].tolist()

    fig = go.Figure(go.Sankey(
        node=dict(pad=10, thickness=14, label=nodes),
        link=dict(source=source, target=target, value=value)
    ))
    fig.update_layout(
        title=title,
        height=760,
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(size=14),
    )
    return fig

# ----------------------------
# Sidebar — Data (CSV)
# ----------------------------
st.sidebar.header("Data")
auto = st.sidebar.checkbox(f"Auto-load {CSV_NAME} from folder", value=True)

csv_path = CSV_NAME if auto and os.path.exists(CSV_NAME) else None
if csv_path is None:
    up = st.sidebar.file_uploader("Upload pathway_data.csv", type=["csv"])
    if up is None:
        st.info(f"Put `{CSV_NAME}` next to `app.py` or upload it.")
        st.stop()
    tmp = "_uploaded_pathway_data.csv"
    with open(tmp, "wb") as f:
        f.write(up.getbuffer())
    csv_path = tmp

try:
    df = load_csv(csv_path)
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
unit = st.sidebar.selectbox("Display unit", ["tCO2e", "kgCO2e"], index=0)
to_unit = (lambda kg: kg / 1000.0) if unit == "tCO2e" else (lambda kg: kg)
unit_label = "tCO2e" if unit == "tCO2e" else "kgCO2e"

target_rate = st.sidebar.slider("Policy target recovery rate (Policy65)", 0.40, 0.80, 0.65, 0.01)

# ----------------------------
# Prep emissions + base tables
# ----------------------------
em_obs = apply_factors(long, FACTORS)
em_obs["CO2e_u"] = to_unit(em_obs["kgCO2e"])

val_obs = apply_values(long, VALUE_FACTORS)

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
                Objective 2 — Treatment pathways, carbon intensities, and economic value (2020–2023)
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

# ----------------------------
# Tabs
# ----------------------------
tabTS, tabScen, tabPriority, tabSank, tabCouncil, tabExports, tabDiag = st.tabs([
    "1) Time series",
    "2) Scenarios (Policy65 + Optimal)",
    "3) Value × Carbon priorities",
    "4) Sankey",
    "5) Council explorer",
    "6) Exports",
    "7) Diagnostics",
])

# ----------------------------
# 1) Time series
# ----------------------------
with tabTS:
    st.subheader("National time series (2020–2023) — collected-for-recycling stream only")
    st.caption("This tab reflects the observed pathway dataset for collected materials. Residual-diversion scenarios are in Tab 2.")

    rows = []
    for y in YEARS:
        by = baseCY[baseCY["year"] == y].copy()

        total = float(by["TotalCollected"].sum())
        residualT = float(by["Residual"].sum())
        recT = float(by["recovery_t"].sum())

        kg = float(em_obs[em_obs["year"] == y]["kgCO2e"].sum())
        gbp = float(val_obs[(val_obs["year"] == y) & (val_obs["route"].isin(VALUE_ROUTES))]["gbp_value"].sum())

        rows.append({
            "year": y,
            "TotalCollected_t": total,
            "Residual_t": residualT,
            "Recovery_t": recT,
            "RecoveryRate": (recT / total) if total > 0 else np.nan,
            f"Emissions_{unit_label}": to_unit(kg),
            "RecoveredValue_GBP": gbp,
        })

    nat = pd.DataFrame(rows)
    y_latest = YEARS[-1]
    nat_latest = nat[nat["year"] == y_latest].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{y_latest} recovery rate", f"{nat_latest['RecoveryRate']*100:.1f}%")
    c2.metric(f"{y_latest} emissions ({unit_label})", f"{nat_latest[f'Emissions_{unit_label}']:,.0f}")
    c3.metric(f"{y_latest} recovered value (GBP)", f"{nat_latest['RecoveredValue_GBP']:,.0f}")
    c4.metric(f"{y_latest} total collected (t)", f"{nat_latest['TotalCollected_t']:,.0f}")

    left, right = st.columns(2)
    with left:
        fig_rate = px.line(nat, x="year", y="RecoveryRate", markers=True, title="Recovery rate (RECOVERY routes)")
        fig_rate.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig_rate, use_container_width=True)
    with right:
        fig_em = px.line(nat, x="year", y=f"Emissions_{unit_label}", markers=True, title=f"Total emissions ({unit_label})")
        fig_em.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_em, use_container_width=True)

    st.markdown("### Route mix over time (tonnes) — observed pathways for collected materials")
    rm = long[long["route"].isin(TREATMENT_ROUTES)].groupby(["year","route"], as_index=False).agg(tonnes=("tonnes","sum"))
    fig_mix = px.area(rm, x="year", y="tonnes", color="route", title="Route mix (stacked)")
    fig_mix.update_layout(height=460, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_mix, use_container_width=True)

# ----------------------------
# 2) Scenarios
# ----------------------------
with tabScen:
    st.subheader("Scenario engine (Baseline, Policy65 sub-scenarios, Optimal)")
    st.caption(
        "Key design choice: diversion is applied ONLY against a synthetic residual-disposal layer built from Residual_YYYY × Objective1 shares × observed disposal mix. "
        "This avoids negative tonnes and avoids disturbing observed collected-material pathways."
    )

    # Build scenarios for all years
    scen_all = []
    meta = []

    for y in YEARS:
        collected_y = long[long["year"] == y].copy()
        totals_y = totals[totals["year"] == y].copy()
        residual_y = residual[residual["year"] == y].copy()

        totalY, recY, base_rate_y = recovery_rate_national(collected_y, totals_y)
        needed_y = max(0.0, target_rate * totalY - recY)

        # residual model layer
        residual_frac = build_residual_by_fraction(residual_y)
        disp_sh = disposal_shares_from_data_by_council(collected_y)
        residual_disp_synth = build_synthetic_residual_disposal(residual_frac, disp_sh)

        capacity_y = float(residual_frac["residual_frac_t"].sum())
        achievable_y = min(needed_y, capacity_y)

        # Baseline = observed collected stream + synthetic residual disposal (no diversion)
        baseline = scenario_df("Baseline (Actual)", collected_y, residual_disp_synth, None)

        # Policy65 variants
        adj_scale = allocate_diversion_massbalanced(residual_frac, residual_disp_synth, long_all=long, long_y=collected_y,
                                                   extra_needed=achievable_y, mode="scale", factors_df=FACTORS)
        adj_prop  = allocate_diversion_massbalanced(residual_frac, residual_disp_synth, long_all=long, long_y=collected_y,
                                                   extra_needed=achievable_y, mode="propensity", factors_df=FACTORS)
        adj_car   = allocate_diversion_massbalanced(residual_frac, residual_disp_synth, long_all=long, long_y=collected_y,
                                                   extra_needed=achievable_y, mode="carbon", factors_df=FACTORS)

        pol_scale = scenario_df("Policy65–Scale-first (largest residual first)", collected_y, residual_disp_synth, adj_scale)
        pol_prop  = scenario_df("Policy65–Propensity (council behaviour mix)",   collected_y, residual_disp_synth, adj_prop)
        pol_car   = scenario_df("Policy65–Carbon-smart (max CO2 benefit)",       collected_y, residual_disp_synth, adj_car)

        # Optimal = divert ALL divertable residual capacity (upper bound)
        adj_opt = allocate_diversion_massbalanced(residual_frac, residual_disp_synth, long_all=long, long_y=collected_y,
                                                  extra_needed=capacity_y, mode="carbon", factors_df=FACTORS)
        optimal = scenario_df("Optimal (recover all divertable residual)", collected_y, residual_disp_synth, adj_opt)

        scen_all.append(pd.concat([baseline, pol_scale, pol_prop, pol_car, optimal], ignore_index=True))

        meta.append({
            "year": y,
            "baseline_rate": base_rate_y,
            "extra_needed_t": needed_y,
            "divertable_capacity_t": capacity_y,
            "achievable_diversion_t": achievable_y
        })

    scen_long = pd.concat(scen_all, ignore_index=True)
    meta_df = pd.DataFrame(meta)

    # Apply emissions and values to scenario dataset
    scen_em = apply_factors(scen_long, FACTORS)
    scen_em["CO2e_u"] = to_unit(scen_em["kgCO2e"])
    scen_val = apply_values(scen_long, VALUE_FACTORS)

    st.dataframe(meta_df, use_container_width=True, height=170)

    year = st.selectbox("Year", options=YEARS, index=2, key="sc_year")

    totals_y = totals[totals["year"] == year].copy()
    totalY = float(totals_y["TotalCollected"].sum())

    d = scen_long[(scen_long["year"] == year)].copy()
    d_em = scen_em[(scen_em["year"] == year)].copy()
    d_val = scen_val[(scen_val["year"] == year)].copy()

    # Scenario summary: recovery rate + emissions + value
    rec = (d[d["route"].isin(RECOVERY_ROUTES)]
           .groupby("scenario", as_index=False).agg(Recovery_t=("tonnes","sum")))
    rec["RecoveryRate"] = np.where(totalY > 0, rec["Recovery_t"] / totalY, np.nan)

    ems = (d_em.groupby("scenario", as_index=False)
           .agg(kgCO2e=("kgCO2e","sum")))
    ems[f"Emissions_{unit_label}"] = to_unit(ems["kgCO2e"])

    val = (d_val[d_val["route"].isin(VALUE_ROUTES)]
           .groupby("scenario", as_index=False)
           .agg(RecoveredValue_GBP=("gbp_value","sum")))

    summ = rec.merge(ems[["scenario", f"Emissions_{unit_label}"]], on="scenario", how="left")
    summ = summ.merge(val, on="scenario", how="left")
    summ["RecoveredValue_GBP"] = summ["RecoveredValue_GBP"].fillna(0.0)

    # Warn if target unreachable
    needed = float(meta_df.loc[meta_df["year"] == year, "extra_needed_t"].iloc[0])
    cap = float(meta_df.loc[meta_df["year"] == year, "divertable_capacity_t"].iloc[0])
    if needed > cap + 1e-9:
        st.warning(
            f"Target {target_rate:.0%} is NOT fully reachable in {year} using divertable residual capacity only. "
            f"Needed extra: {needed:,.0f} t, capacity: {cap:,.0f} t."
        )

    # KPI / recovery rate bar
    fig_kpi = px.bar(
        summ.sort_values("RecoveryRate"),
        x="scenario", y="RecoveryRate", text="RecoveryRate",
        title="Recovery rate (RECOVERY routes) by scenario"
    )
    fig_kpi.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10), yaxis_tickformat=".0%")
    fig_kpi.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_kpi, use_container_width=True)

    # Route mix by scenario (TOTAL, non-negative by design)
    mix_total = (d[d["route"].isin(TREATMENT_ROUTES)]
                 .groupby(["scenario","route"], as_index=False).agg(tonnes=("tonnes","sum")))
    fig_mix_total = px.bar(
        mix_total, x="scenario", y="tonnes", color="route", barmode="stack",
        title="Route mix by scenario (TOTAL tonnes) — mass-balanced (no negatives)"
    )
    fig_mix_total.update_layout(height=520, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_mix_total, use_container_width=True)

    # Emissions & Value bars
    c1, c2 = st.columns(2)
    with c1:
        fig_em = px.bar(
            summ.sort_values(f"Emissions_{unit_label}"),
            x="scenario", y=f"Emissions_{unit_label}",
            title=f"Total emissions by scenario ({unit_label})"
        )
        fig_em.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_em, use_container_width=True)

    with c2:
        fig_val = px.bar(
            summ.sort_values("RecoveredValue_GBP"),
            x="scenario", y="RecoveredValue_GBP",
            title="Recovered commodity value by scenario (GBP) — Recycled/Reuse only"
        )
        fig_val.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_val, use_container_width=True)

# ----------------------------
# 3) Value × Carbon priorities (two-axis)
# ----------------------------
with tabPriority:
    st.subheader("Prioritisation map: Value (GBP/t) vs carbon benefit (kgCO2e saved per tonne diverted)")
    st.caption("Bubble size = diverted tonnes (for selected scenario/year). Points are labeled by fraction.")

    year = st.selectbox("Year", options=YEARS, index=2, key="prio_year")

    # pull scenario set for year
    d = scen_long[scen_long["year"] == year].copy()
    d_em = apply_factors(d, FACTORS)
    d_val = apply_values(d, VALUE_FACTORS)

    scenario = st.selectbox(
        "Scenario",
        sorted(d["scenario"].unique().tolist()),
        index=1,
        key="prio_scen"
    )

    dd = d[d["scenario"] == scenario].copy()
    dd_em = d_em[d_em["scenario"] == scenario].copy()
    dd_val = d_val[d_val["scenario"] == scenario].copy()

    # diverted tonnes by fraction = positive tonnes in RECOVERY routes that are in stream=Scenario_adjustment
    adj = dd[(dd.get("stream", "") == "Scenario_adjustment") & (dd["route"].isin(RECOVERY_ROUTES))].copy()
    diverted = adj.groupby("fraction", as_index=False).agg(diverted_t=("tonnes","sum"))
    diverted = diverted[diverted["diverted_t"] > 0].copy()

    if len(diverted) == 0:
        st.info("No diversion in this scenario/year (or diverted tonnes = 0). Pick a Policy65 or Optimal scenario.")
    else:
        # Baseline disposal factor per fraction (national for that year) from synthetic residual layer
        collected_y = long[long["year"] == year].copy()
        nat_mix = national_disposal_mix(collected_y)
        getf = factor_maps(FACTORS)

        # scenario destination factor per fraction = weighted by destination weights in adjustments
        # Compute weighted recovery factor for diverted tonnes by fraction from adj rows
        adj_em = apply_factors(adj, FACTORS)
        rec_fac = (adj_em.groupby(["fraction","route"], as_index=False)
                   .agg(t=("tonnes","sum"), kg=("kgCO2e","sum")))
        # weighted avg recovery factor:
        rec_avg = (rec_fac.groupby("fraction", as_index=False)
                   .apply(lambda g: pd.Series({
                       "recovery_avg_kg_per_t": (g["kg"].sum() / g["t"].sum()) if g["t"].sum() > 0 else np.nan
                   })).reset_index(drop=True))

        # baseline disposal avg factor:
        base_vals = []
        for f in diverted["fraction"].tolist():
            b = 0.0
            ok = False
            for r, sh in nat_mix.items():
                v = getf(f, r)
                if not np.isnan(v):
                    b += float(sh) * float(v)
                    ok = True
            if not ok:
                vL = getf(f, "Landfill")
                vE = getf(f, "EfW")
                b = vL if not np.isnan(vL) else (vE if not np.isnan(vE) else 0.0)
            base_vals.append({"fraction": f, "baseline_disposal_avg_kg_per_t": b})
        base_df = pd.DataFrame(base_vals)

        # value per t
        vdf = VALUE_FACTORS[["fraction","gbp_per_t_mid"]].copy()

        pri = diverted.merge(base_df, on="fraction", how="left").merge(rec_avg, on="fraction", how="left").merge(vdf, on="fraction", how="left")
        pri["gbp_per_t_mid"] = pri["gbp_per_t_mid"].fillna(0.0)
        pri["recovery_avg_kg_per_t"] = pri["recovery_avg_kg_per_t"].fillna(0.0)
        pri["carbon_benefit_kg_per_t"] = pri["baseline_disposal_avg_kg_per_t"] - pri["recovery_avg_kg_per_t"]

        # midlines
        x_mid = float(pri["gbp_per_t_mid"].median())
        y_mid = float(pri["carbon_benefit_kg_per_t"].median())

        fig = px.scatter(
            pri,
            x="gbp_per_t_mid",
            y="carbon_benefit_kg_per_t",
            size="diverted_t",
            text="fraction",
            hover_data={
                "diverted_t": ":,.0f",
                "gbp_per_t_mid": ":,.0f",
                "carbon_benefit_kg_per_t": ":,.0f",
            },
            title="Fraction priorities (Value vs Carbon benefit) — bubble size = diverted tonnes"
        )
        fig.update_traces(textposition="top center")
        fig.add_vline(x=x_mid)
        fig.add_hline(y=y_mid)
        fig.update_layout(
            height=640,
            margin=dict(l=10,r=10,t=60,b=10),
            xaxis_title="Market value (GBP per tonne, mid)",
            yaxis_title="Carbon benefit (kgCO2e saved per tonne diverted)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Priority table (this is the one your advisor will actually read)")
        pri_show = pri.sort_values(["carbon_benefit_kg_per_t","gbp_per_t_mid"], ascending=[False, False]).copy()
        st.dataframe(pri_show, use_container_width=True, height=360)

# ----------------------------
# 4) Sankey
# ----------------------------
with tabSank:
    st.subheader("Sankey (Scenario → Fraction → Route)")
    st.caption("Sankey is sized by tonnes or |tCO2e| (not kg), so the labels don’t show 'G'.")

    year = st.selectbox("Year", options=YEARS, index=2, key="sank_year")

    top_links = st.slider("Keep top links (readability)", 20, 100, 60, 10)

    metric = st.radio(
        "Size links by",
        ["Tonnes", "Emissions magnitude |tCO2e|"],
        horizontal=True
    )

    dd = scen_long[(scen_long["year"] == year) & (scen_long["route"].isin(TREATMENT_ROUTES))].copy()
    dd_em = apply_factors(dd, FACTORS)

    if metric.startswith("Emissions"):
        dd_em["abs_t"] = np.abs(dd_em["kgCO2e"]) / 1000.0
        fig = sankey_scenario_fraction_route(
            dd_em,
            value_col="abs_t",
            top_links=top_links,
            title="Sankey sized by |tCO2e| (top links)"
        )
    else:
        fig = sankey_scenario_fraction_route(
            dd,
            value_col="tonnes",
            top_links=top_links,
            title="Sankey sized by tonnes (top links)"
        )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 5) Council explorer
# ----------------------------
with tabCouncil:
    st.subheader("Council explorer (observed + scenarios)")

    year = st.selectbox("Year", YEARS, index=2, key="cx_year")
    councils = sorted(long[long["year"] == year]["Council Name"].unique().tolist())
    council = st.selectbox("Council", councils, index=0, key="cx_council")

    scenario = st.selectbox(
        "Scenario",
        sorted(scen_long[scen_long["year"] == year]["scenario"].unique().tolist()),
        index=0,
        key="cx_scenario"
    )

    # totals for council
    cy = baseCY[(baseCY["year"] == year) & (baseCY["Council Name"] == council)]
    if len(cy) == 0:
        st.info("No totals for this council/year.")
        st.stop()
    cy = cy.iloc[0]

    dd = scen_long[(scen_long["year"] == year) & (scen_long["Council Name"] == council) & (scen_long["scenario"] == scenario)].copy()
    if len(dd) == 0:
        st.info("No scenario data for this council/year.")
        st.stop()

    dd_em = apply_factors(dd, FACTORS)
    dd_val = apply_values(dd, VALUE_FACTORS)

    total_col = float(cy["TotalCollected"])
    pop = float(cy["population"]) if pd.notna(cy["population"]) else np.nan

    rec_t = float(dd[dd["route"].isin(RECOVERY_ROUTES)]["tonnes"].sum())
    rec_rate = (rec_t / total_col) if total_col > 0 else np.nan

    kg = float(dd_em["kgCO2e"].sum())
    intensity = (kg / total_col) if total_col > 0 else np.nan

    value_gbp = float(dd_val[dd_val["route"].isin(VALUE_ROUTES)]["gbp_value"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total collected (t)", f"{total_col:,.0f}")
    c2.metric("Recovery rate", f"{rec_rate*100:.1f}%")
    c3.metric("Intensity (kgCO2e/t)", f"{intensity:.1f}")
    c4.metric("CO2e per cap", f"{(kg/pop):.1f}" if (pop and pop > 0) else "NA")
    c5.metric("Recovered value (GBP)", f"{value_gbp:,.0f}")

    mix = dd_em.groupby("route", as_index=False).agg(tonnes=("tonnes","sum"), kgCO2e=("kgCO2e","sum"))
    mix["CO2e_u"] = to_unit(mix["kgCO2e"])
    mix_v = dd_val.groupby("route", as_index=False).agg(gbp_value=("gbp_value","sum"))

    left, right = st.columns(2)
    with left:
        fig_r = px.bar(mix.sort_values("tonnes", ascending=False), x="route", y="tonnes", title="Route mix (tonnes)")
        fig_r.update_layout(height=380, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_r, use_container_width=True)
    with right:
        fig_re = px.bar(mix.sort_values("CO2e_u", ascending=False), x="route", y="CO2e_u",
                        title=f"Route emissions contribution ({unit_label})")
        fig_re.update_layout(height=380, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_re, use_container_width=True)

    st.markdown("### Value contribution by route (GBP)")
    fig_v = px.bar(mix_v.sort_values("gbp_value", ascending=False), x="route", y="gbp_value", title="Recovered value (GBP)")
    fig_v.update_layout(height=320, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_v, use_container_width=True)

# ----------------------------
# 6) Exports
# ----------------------------
with tabExports:
    st.subheader("Export tables for your paper (CSV)")

    year = st.selectbox("Year", options=YEARS, index=2, key="exp_year")

    d = scen_long[scen_long["year"] == year].copy()
    d_em = apply_factors(d, FACTORS)
    d_val = apply_values(d, VALUE_FACTORS)

    # Scenario summary table
    totals_y = totals[totals["year"] == year].copy()
    totalY = float(totals_y["TotalCollected"].sum())

    rec = d[d["route"].isin(RECOVERY_ROUTES)].groupby("scenario", as_index=False).agg(Recovery_t=("tonnes","sum"))
    rec["RecoveryRate"] = np.where(totalY > 0, rec["Recovery_t"] / totalY, np.nan)

    ems = d_em.groupby("scenario", as_index=False).agg(kgCO2e=("kgCO2e","sum"))
    ems["tCO2e"] = ems["kgCO2e"] / 1000.0

    val = d_val[d_val["route"].isin(VALUE_ROUTES)].groupby("scenario", as_index=False).agg(RecoveredValue_GBP=("gbp_value","sum"))

    scen_summary = rec.merge(ems[["scenario","kgCO2e","tCO2e"]], on="scenario", how="left").merge(val, on="scenario", how="left")
    scen_summary["RecoveredValue_GBP"] = scen_summary["RecoveredValue_GBP"].fillna(0.0)

    st.markdown("### Scenario summary (national)")
    st.dataframe(scen_summary, use_container_width=True, height=260)

    st.download_button(
        "Download scenario_summary.csv",
        data=scen_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"scenario_summary_{year}.csv",
        mime="text/csv"
    )

    # Council-level table
    council_tbl = (d_em.groupby(["scenario","Council Name"], as_index=False)
                   .agg(kgCO2e=("kgCO2e","sum"),
                        tonnes=("tonnes","sum")))
    council_val = (d_val.groupby(["scenario","Council Name"], as_index=False)
                   .agg(RecoveredValue_GBP=("gbp_value","sum")))
    council_tbl = council_tbl.merge(council_val, on=["scenario","Council Name"], how="left")
    council_tbl["RecoveredValue_GBP"] = council_tbl["RecoveredValue_GBP"].fillna(0.0)
    council_tbl["tCO2e"] = council_tbl["kgCO2e"] / 1000.0

    # add population for per-capita
    pop = long[long["year"] == year].drop_duplicates(["Council Name"])[["Council Name","population"]]
    council_tbl = council_tbl.merge(pop, on="Council Name", how="left")
    council_tbl["kgCO2e_per_cap"] = np.where(council_tbl["population"] > 0, council_tbl["kgCO2e"] / council_tbl["population"], np.nan)
    council_tbl["GBP_per_cap"] = np.where(council_tbl["population"] > 0, council_tbl["RecoveredValue_GBP"] / council_tbl["population"], np.nan)

    st.markdown("### Council results (scenario × council)")
    st.dataframe(council_tbl, use_container_width=True, height=380)

    st.download_button(
        "Download council_results.csv",
        data=council_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"council_results_{year}.csv",
        mime="text/csv"
    )

    # Fraction × route table (to identify highest value / lowest carbon routes)
    fr = (d_em.groupby(["scenario","fraction","route"], as_index=False)
          .agg(tonnes=("tonnes","sum"), kgCO2e=("kgCO2e","sum")))
    fr = fr.merge(
        d_val.groupby(["scenario","fraction","route"], as_index=False).agg(RecoveredValue_GBP=("gbp_value","sum")),
        on=["scenario","fraction","route"],
        how="left"
    )
    fr["RecoveredValue_GBP"] = fr["RecoveredValue_GBP"].fillna(0.0)
    fr["kgCO2e_per_t"] = np.where(fr["tonnes"] > 0, fr["kgCO2e"] / fr["tonnes"], np.nan)

    st.markdown("### Fraction × route (scenario)")
    st.dataframe(fr.sort_values(["scenario","RecoveredValue_GBP"], ascending=[True, False]), use_container_width=True, height=380)

    st.download_button(
        "Download fraction_route_results.csv",
        data=fr.to_csv(index=False).encode("utf-8"),
        file_name=f"fraction_route_results_{year}.csv",
        mime="text/csv"
    )

# ----------------------------
# 7) Diagnostics
# ----------------------------
with tabDiag:
    st.subheader("Diagnostics")

    miss = em_obs[em_obs["factor_missing"] == 1].groupby(["fraction","route"], as_index=False).agg(tonnes=("tonnes","sum"))
    if len(miss) == 0:
        st.success("No missing factors for fraction×route pairs that carry non-zero tonnes in the observed dataset.")
    else:
        st.warning("Some fraction×route pairs have no factor and are treated as 0 in emissions.")
        st.dataframe(miss.sort_values("tonnes", ascending=False), use_container_width=True, height=320)

    st.markdown("### Carbon factor table in use (kgCO2e per tonne)")
    st.dataframe(FACTORS, use_container_width=True, height=320)

    st.markdown("### Value table in use (GBP per tonne)")
    st.dataframe(VALUE_FACTORS, use_container_width=True, height=320)

    st.markdown("### Quick national checks (observed collected stream)")
    check_rows = []
    for y in YEARS:
        tot = float(totals[totals["year"] == y]["TotalCollected"].sum())
        res = float(residual[residual["year"] == y]["Residual"].sum())
        rec = float(long[(long["year"] == y) & (long["route"].isin(RECOVERY_ROUTES))]["tonnes"].sum())
        check_rows.append({
            "year": y,
            "TotalCollected_sum_t": tot,
            "Residual_sum_t": res,
            "Recovery_sum_t": rec,
            "Recovery_rate_nat": (rec / tot) if tot > 0 else np.nan,
        })
    st.dataframe(pd.DataFrame(check_rows), use_container_width=True)