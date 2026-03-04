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
# Config (match your CSV)
# ----------------------------
CSV_NAME = "pathway_data.csv"
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

# “Recycling rate” KPI for Policy65: include organics (UK/DEFRA-style municipal recycling)
KPI_ROUTES = {"Recycled", "Reuse", "AD", "CompostedIV", "CompostedW"}

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

# Divertable residual fractions (avoidables proxy)
DIVERTABLE_FRACTIONS = [f for f in RESIDUAL_SHARES.keys() if f in FRACTIONS]

# ----------------------------
# Scenario destination policies (match what you mean)
# ----------------------------
# Easy = “fast rollout” default destinations (no fancy optimisation)
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

# Allowed circular options per fraction for carbon-smart routing
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

FACTORS = pd.concat([DEFAULT_FACTORS, WILDCARD_DEFAULTS], ignore_index=True)

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
def load_csv(path: str):
    return pd.read_csv(path, low_memory=False)

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

def council_year_base(totals: pd.DataFrame, rec_col: pd.DataFrame, residual: pd.DataFrame, long: pd.DataFrame):
    base = totals.merge(rec_col, on=["Council Name", "year"], how="left")
    base = base.merge(residual, on=["Council Name", "year"], how="left")

    pop = long.drop_duplicates(["Council Name", "year"])[["Council Name", "year", "population"]]
    base = base.merge(pop, on=["Council Name", "year"], how="left")

    base["TotalRecCollec"] = base["TotalRecCollec"].fillna(0.0)
    base["Residual"] = base["Residual"].fillna(0.0)

    kpi_t = long[long["route"].isin(KPI_ROUTES)].groupby(["Council Name", "year"], as_index=False).agg(kpi_t=("tonnes","sum"))
    base = base.merge(kpi_t, on=["Council Name", "year"], how="left")
    base["kpi_t"] = base["kpi_t"].fillna(0.0)

    base["kpi_rate"] = np.where(base["TotalCollected"] > 0, base["kpi_t"] / base["TotalCollected"], np.nan)

    return base

def kpi_rate_national(long_y: pd.DataFrame, totals_y: pd.DataFrame):
    total = float(totals_y["TotalCollected"].sum())
    kpi = float(long_y[long_y["route"].isin(KPI_ROUTES)]["tonnes"].sum())
    rate = kpi / total if total > 0 else np.nan
    return total, kpi, rate

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

def disposal_shares_from_data(long_y: pd.DataFrame):
    """
    Council-year disposal route shares inferred from observed totals in long_y.
    If a council has zero disposal recorded, default to EfW=1.
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

    # default if all zero: EfW=1
    all_zero = (piv[list(DISPOSAL_ROUTES)].sum(axis=1) == 0)
    if all_zero.any():
        shares.loc[all_zero, :] = 0.0
        shares.loc[all_zero, "EfW"] = 1.0

    out = shares.reset_index().melt(id_vars=["Council Name","year"], var_name="route", value_name="share")
    return out

def build_residual_disposal_table(residual_frac: pd.DataFrame, disp_shares: pd.DataFrame):
    """
    Residual (by fraction) allocated to disposal routes using council disposal shares.
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
    1) Council×fraction: historic success at sending stuff into KPI routes vs (KPI + disposal)
    2) Council×fraction×route: historic shares among KPI routes (destination mix)
    computed across ALL YEARS (2020–2023)
    """
    kpi_or_disp = KPI_ROUTES | DISPOSAL_ROUTES
    df = long_all[long_all["route"].isin(kpi_or_disp)].copy()

    if len(df) == 0:
        prop = pd.DataFrame(columns=["Council Name","fraction","p_kpi"])
        shares = pd.DataFrame(columns=["Council Name","fraction","route","w"])
        return prop, shares

    piv = df.pivot_table(index=["Council Name","fraction"], columns="route", values="tonnes", aggfunc="sum").fillna(0.0)
    for r in KPI_ROUTES:
        if r not in piv.columns:
            piv[r] = 0.0
    for r in DISPOSAL_ROUTES:
        if r not in piv.columns:
            piv[r] = 0.0

    kpi_sum = piv[list(KPI_ROUTES)].sum(axis=1)
    disp_sum = piv[list(DISPOSAL_ROUTES)].sum(axis=1)
    denom = (kpi_sum + disp_sum).replace(0, np.nan)
    p_kpi = (kpi_sum / denom).fillna(0.0).clip(0, 1)

    prop = p_kpi.reset_index().rename(columns={0: "p_kpi"})
    prop["p_kpi"] = p_kpi.values

    # KPI destination shares among KPI routes
    denom2 = kpi_sum.replace(0, np.nan)
    shares_mat = piv[list(KPI_ROUTES)].div(denom2, axis=0).fillna(0.0)
    shares = shares_mat.reset_index().melt(id_vars=["Council Name","fraction"], var_name="route", value_name="w")
    shares = shares[shares["w"] > 0].copy()

    return prop, shares

def carbon_best_route(frac: str, factors_df: pd.DataFrame):
    get = factor_maps(factors_df)
    allowed = CARBON_ALLOWED.get(frac, [])
    vals = []
    for r in allowed:
        v = get(frac, r)
        if not np.isnan(v):
            vals.append((r, float(v)))
    if not vals:
        # fallback: easy mapping first route
        fallback = EASY_DEST.get(frac, [("Recycled", 1.0)])[0][0]
        return fallback
    # choose minimum kgCO2e/t
    vals.sort(key=lambda x: x[1])
    return vals[0][0]

def carbon_gain_per_tonne(frac: str, factors_df: pd.DataFrame):
    """
    Approx benefit of moving 1 t from weighted disposal baseline to best carbon circular route.
    Baseline disposal reference here is EfW+Landfill+RDF+IWE average is handled later
    by subtracting from actual residual disposal mix; this function is just a ranking proxy.
    """
    get = factor_maps(factors_df)
    best_route = carbon_best_route(frac, factors_df)
    best = get(frac, best_route)
    # Use landfill as conservative proxy for ranking if present, else efw
    base = get(frac, "Landfill")
    if np.isnan(base):
        base = get(frac, "EfW")
    if np.isnan(base) or np.isnan(best):
        return 0.0
    return float(base - best)

def choose_destination_weights(mode: str, chosen_rows: pd.DataFrame, shares_hist: pd.DataFrame, factors_df: pd.DataFrame):
    """
    Returns a dict: (Council Name, fraction) -> list[(route, weight)]
    """
    dest = {}
    for _, r in chosen_rows.iterrows():
        c = r["Council Name"]
        f = r["fraction"]

        if mode == "easy":
            dest[(c, f)] = EASY_DEST.get(f, [("Recycled", 1.0)])

        elif mode == "carbon":
            br = carbon_best_route(f, factors_df)
            dest[(c, f)] = [(br, 1.0)]

        elif mode == "propensity":
            sub = shares_hist[(shares_hist["Council Name"] == c) & (shares_hist["fraction"] == f)].copy()
            if len(sub) == 0:
                dest[(c, f)] = EASY_DEST.get(f, [("Recycled", 1.0)])
            else:
                # normalise
                s = float(sub["w"].sum())
                if s <= 0:
                    dest[(c, f)] = EASY_DEST.get(f, [("Recycled", 1.0)])
                else:
                    tmp = [(rr["route"], float(rr["w"]) / s) for _, rr in sub.iterrows()]
                    dest[(c, f)] = tmp
        else:
            dest[(c, f)] = EASY_DEST.get(f, [("Recycled", 1.0)])

    return dest

def allocate_diversion_massbalanced(
    residual_frac: pd.DataFrame,
    residual_disp: pd.DataFrame,
    long_all: pd.DataFrame,
    extra_needed: float,
    mode: str,
    factors_df: pd.DataFrame
):
    """
    Mass-balanced diversion:
    - Take tonnes from residual pool (by council×fraction)
    - Allocate to destination KPI routes (policy-dependent)
    - Subtract the same tonnes from residual disposal routes proportionally to baseline disposal split
    Returns a dataframe of adjustment rows with +/- tonnes.
    """
    if extra_needed <= 0:
        return pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    pool = residual_frac.copy()
    pool = pool[pool["residual_frac_t"] > 0].copy()
    if len(pool) == 0:
        return pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    # Historic capability + destination shares (2020–2023)
    prop_hist, shares_hist = hist_propensity_routes(long_all)

    pool = pool.merge(prop_hist, on=["Council Name","fraction"], how="left")
    pool["p_kpi"] = pool["p_kpi"].fillna(0.0)

    # Carbon gain proxy
    pool["carbon_gain"] = pool["fraction"].apply(lambda f: carbon_gain_per_tonne(f, factors_df))

    # Ranking (who gets diverted first)
    if mode == "easy":
        # “Fast” = move biggest residual volumes first (scale effect)
        pool["rank"] = -(pool["residual_frac_t"])

    elif mode == "propensity":
        # “Capability-based” = do councils/fractions that already perform well, but also consider scale
        pool["rank"] = -(0.7 * pool["p_kpi"] + 0.3 * (pool["residual_frac_t"] / pool["residual_frac_t"].max()))

    elif mode == "carbon":
        # “Carbon-smart” = target highest benefit first (benefit * tonnes)
        pool["rank"] = -(pool["carbon_gain"] * pool["residual_frac_t"])

    else:
        pool["rank"] = 0.0

    pool = pool.sort_values(["rank"]).copy()

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

    # Destination weights per chosen council×fraction
    dest_map = choose_destination_weights(mode, chosen, shares_hist, factors_df)

    # Positive destination rows
    pos_parts = []
    for _, r in chosen.iterrows():
        c = r["Council Name"]; y = int(r["year"]); f = r["fraction"]; t = float(r["take"])
        weights = dest_map.get((c, f), EASY_DEST.get(f, [("Recycled", 1.0)]))
        for route, w in weights:
            pos_parts.append({"Council Name": c, "year": y, "fraction": f, "route": route, "tonnes": t * float(w)})

    pos = pd.DataFrame(pos_parts) if pos_parts else pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    # Negative disposal rows: subtract from residual disposal mix proportionally, capped at available
    # Build available residual disposal by council×fraction×route
    avail = residual_disp.copy()
    avail = avail.groupby(["Council Name","year","fraction","route"], as_index=False).agg(avail_t=("tonnes","sum"))

    neg_parts = []
    for _, r in chosen.iterrows():
        c = r["Council Name"]; y = int(r["year"]); f = r["fraction"]; take = float(r["take"])
        sub = avail[(avail["Council Name"] == c) & (avail["year"] == y) & (avail["fraction"] == f)].copy()
        if len(sub) == 0:
            continue
        tot_av = float(sub["avail_t"].sum())
        if tot_av <= 0:
            continue

        # proportional subtraction
        sub["share"] = sub["avail_t"] / tot_av
        sub["sub_t"] = sub["share"] * take

        # cap to avail_t (numerical safety)
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

        # Update avail in-memory to avoid double-subtracting if same council×fraction appears again (it can)
        for idx in sub.index:
            a = float(avail.loc[idx, "avail_t"])
            s = float(sub.loc[idx, "sub_t"])
            avail.loc[idx, "avail_t"] = max(0.0, a - s)

    neg = pd.DataFrame(neg_parts) if neg_parts else pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    # Combine adjustments (and tidy tiny floats)
    adj = pd.concat([pos, neg], ignore_index=True)
    if len(adj):
        adj["tonnes"] = adj["tonnes"].astype(float)
        adj = adj.groupby(["Council Name","year","fraction","route"], as_index=False).agg(tonnes=("tonnes","sum"))
        adj = adj[np.abs(adj["tonnes"]) > 1e-9].copy()
    return adj

def scenario_df(name: str, base_long_y: pd.DataFrame, adj_rows: pd.DataFrame):
    base = base_long_y[["Council Name","year","fraction","route","tonnes","population"]].copy()
    base["scenario"] = name
    if adj_rows is None or len(adj_rows) == 0:
        return base
    add = adj_rows.copy()
    add["population"] = np.nan
    add["scenario"] = name
    add = add[["Council Name","year","fraction","route","tonnes","population","scenario"]]
    return pd.concat([base, add], ignore_index=True)

def scenario_stack_for_year(
    long_all: pd.DataFrame,
    long_y: pd.DataFrame,
    totals_y: pd.DataFrame,
    residual_y: pd.DataFrame,
    target_rate: float,
    factors_df: pd.DataFrame
):
    total, kpi, base_rate = kpi_rate_national(long_y, totals_y)
    needed = max(0.0, target_rate * total - kpi)

    residual_frac = build_residual_by_fraction(residual_y)
    disp_sh = disposal_shares_from_data(long_y)
    residual_disp = build_residual_disposal_table(residual_frac, disp_sh)

    # capacity: divertable residual total
    capacity = float(residual_frac["residual_frac_t"].sum())
    achievable = min(needed, capacity)

    baseline = scenario_df("Baseline (Actual)", long_y, None)

    # Policy65 sub-scenarios (all aim at same 65% KPI, but implementation differs)
    adj_easy = allocate_diversion_massbalanced(residual_frac, residual_disp, long_all, achievable, "easy", factors_df)
    adj_prop = allocate_diversion_massbalanced(residual_frac, residual_disp, long_all, achievable, "propensity", factors_df)
    adj_car  = allocate_diversion_massbalanced(residual_frac, residual_disp, long_all, achievable, "carbon", factors_df)

    pol_easy = scenario_df("Policy65–Easy (fast rollout)", long_y, adj_easy)
    pol_prop = scenario_df("Policy65–Propensity (capability-based)", long_y, adj_prop)
    pol_car  = scenario_df("Policy65–Carbon-smart (min kgCO₂e/t)", long_y, adj_car)

    # Optimal = recycle all divertable residual (recover all avoidable residual proxy)
    adj_opt  = allocate_diversion_massbalanced(residual_frac, residual_disp, long_all, capacity, "carbon", factors_df)
    optimal  = scenario_df("Optimal (recover all avoidable residual)", long_y, adj_opt)

    all_scen = pd.concat([baseline, pol_easy, pol_prop, pol_car, optimal], ignore_index=True)
    return all_scen, base_rate, needed, capacity

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
unit = st.sidebar.selectbox("Display unit", ["t CO₂e", "kg CO₂e"], index=0)
to_unit = (lambda kg: kg / 1000.0) if unit == "t CO₂e" else (lambda kg: kg)

target_rate = st.sidebar.slider("Policy target recycling rate", 0.40, 0.80, 0.65, 0.01)

# Fixed factor table (no editor)
st.sidebar.header("Factors")
st.sidebar.caption("Fixed factor set (session editing removed).")

# ----------------------------
# Prep emissions + base tables
# ----------------------------
em_obs = apply_factors(long, FACTORS)
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

# ----------------------------
# Tabs
# ----------------------------
tabTS, tabScen, tabSank, tabCouncil, tabDiag = st.tabs([
    "1) Time series",
    "2) Scenarios (Policy65 + Optimal)",
    "3) Sankey",
    "4) Council explorer",
    "5) Diagnostics",
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
        residualT = float(by["Residual"].sum())
        kpiT = float(by["kpi_t"].sum())

        kg = float(em_obs[em_obs["year"] == y]["kgCO2e"].sum())

        rows.append({
            "year": y,
            "TotalCollected_t": total,
            "Residual_t": residualT,
            "KPI_Recycling_t": kpiT,
            "RecyclingRate": (kpiT / total) if total > 0 else np.nan,
            "CO2e_u": to_unit(kg),
        })

    nat = pd.DataFrame(rows)
    y_latest = YEARS[-1]
    nat_latest = nat[nat["year"] == y_latest].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{y_latest} recycling rate", f"{nat_latest['RecyclingRate']*100:.1f}%")
    c2.metric(f"{y_latest} emissions ({unit})", f"{nat_latest['CO2e_u']:,.0f}")
    c3.metric(f"{y_latest} total collected (t)", f"{nat_latest['TotalCollected_t']:,.0f}")

    left, right = st.columns(2)
    with left:
        fig_rate = px.line(nat, x="year", y="RecyclingRate", markers=True, title="Recycling rate (KPI routes)")
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
    st.subheader("Scenario engine (Baseline, Policy65 sub-scenarios, Optimal)")

    scen_all = []
    meta = []

    for y in YEARS:
        ly = long[long["year"] == y].copy()
        ty = totals[totals["year"] == y].copy()
        ry = residual[residual["year"] == y].copy()

        sc_y, base_rate_y, needed_y, cap_y = scenario_stack_for_year(
            long_all=long,
            long_y=ly,
            totals_y=ty,
            residual_y=ry,
            target_rate=target_rate,
            factors_df=FACTORS
        )

        scen_all.append(sc_y)
        meta.append({
            "year": y,
            "baseline_rate": base_rate_y,
            "extra_needed_t": needed_y,
            "divertable_capacity_t": cap_y
        })

    scen_long = pd.concat(scen_all, ignore_index=True)
    scen_em = apply_factors(scen_long, FACTORS)
    scen_em["CO2e_u"] = to_unit(scen_em["kgCO2e"])

    meta_df = pd.DataFrame(meta)
    st.dataframe(meta_df, use_container_width=True, height=170)

    year = st.selectbox("Year", options=YEARS, index=2, key="sc_year")

    # KPI computation for selected year
    ty = totals[totals["year"] == year].copy()
    totalY = float(ty["TotalCollected"].sum())

    d = scen_long[(scen_long["year"] == year)].copy()

    # Scenario summary
    kpi = d[d["route"].isin(KPI_ROUTES)].groupby("scenario", as_index=False).agg(KPI_t=("tonnes","sum"))
    summ = kpi.copy()
    summ["RecyclingRate"] = np.where(totalY > 0, summ["KPI_t"] / totalY, np.nan)

    # Emissions
    ems = scen_em[scen_em["year"] == year].groupby("scenario", as_index=False).agg(kgCO2e=("kgCO2e","sum"))
    ems["CO2e_u"] = to_unit(ems["kgCO2e"])
    summ = summ.merge(ems[["scenario","CO2e_u"]], on="scenario", how="left")

    # Warn if policy target unreachable
    needed = float(meta_df.loc[meta_df["year"] == year, "extra_needed_t"].iloc[0])
    cap = float(meta_df.loc[meta_df["year"] == year, "divertable_capacity_t"].iloc[0])
    if needed > cap + 1e-9:
        st.warning(
            f"Policy target {target_rate:.0%} is NOT fully reachable in {year} with current divertable residual capacity. "
            f"Needed extra: {needed:,.0f} t, capacity: {cap:,.0f} t."
        )

    # KPI bar
    fig_kpi = px.bar(
        summ.sort_values("RecyclingRate"),
        x="scenario", y="RecyclingRate", text="RecyclingRate",
        title="Recycling rate (KPI) by scenario (selected year)"
    )
    fig_kpi.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10), yaxis_tickformat=".0%")
    fig_kpi.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_kpi, use_container_width=True)

    # Route mix by scenario (TOTAL)
    mix_total = d[d["route"].isin(TREATMENT_ROUTES)].groupby(["scenario","route"], as_index=False).agg(tonnes=("tonnes","sum"))
    fig_mix_total = px.bar(
        mix_total, x="scenario", y="tonnes", color="route", barmode="stack",
        title="Route mix by scenario (TOTAL tonnes) — now mass-balanced so sub-scenarios differ"
    )
    fig_mix_total.update_layout(height=520, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_mix_total, use_container_width=True)

    # Route mix change vs baseline (DELTA)
    base_mix = (
        mix_total[mix_total["scenario"] == "Baseline (Actual)"]
        .groupby("route", as_index=False).agg(base_t=("tonnes","sum"))
    )
    mix_delta = mix_total.merge(base_mix, on="route", how="left")
    mix_delta["base_t"] = mix_delta["base_t"].fillna(0.0)
    mix_delta["delta_t"] = mix_delta["tonnes"] - mix_delta["base_t"]
    mix_delta = mix_delta[mix_delta["scenario"] != "Baseline (Actual)"].copy()

    fig_mix_delta = px.bar(
        mix_delta, x="scenario", y="delta_t", color="route", barmode="stack",
        title="Route mix (DELTA vs Baseline) — shows where each Policy65 variant pushes the system"
    )
    fig_mix_delta.update_layout(height=520, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_mix_delta, use_container_width=True)

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
    kpi_t = float(d[d["route"].isin(KPI_ROUTES)]["tonnes"].sum())
    kpi_rate = (kpi_t / total_col) if total_col > 0 else np.nan
    kg = float(d["kgCO2e"].sum())
    intensity = (kg / total_col) if total_col > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total collected (t)", f"{total_col:,.0f}")
    c2.metric("Recycling rate (KPI)", f"{kpi_rate*100:.1f}%")
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
# 5) Diagnostics
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
    st.dataframe(FACTORS, use_container_width=True, height=360)

    st.markdown("### Quick national checks")
    check_rows = []
    for y in YEARS:
        tot = float(totals[totals["year"] == y]["TotalCollected"].sum())
        res = float(residual[residual["year"] == y]["Residual"].sum())
        kpi = float(long[(long["year"] == y) & (long["route"].isin(KPI_ROUTES))]["tonnes"].sum())
        check_rows.append({
            "year": y,
            "TotalCollected_sum_t": tot,
            "Residual_sum_t": res,
            "KPI_sum_t": kpi,
            "KPI_rate_nat": (kpi / tot) if tot > 0 else np.nan,
        })
    st.dataframe(pd.DataFrame(check_rows), use_container_width=True)