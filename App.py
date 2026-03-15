# -*- coding: utf-8 -*-
# app.py
# Objective 2 — England LACW Pathway Lab (2020–2023)
# Saeed AlKhoori | Supervisor: Prof. Nikolaos Voulvoulis | Imperial College London (CEP)

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import Tuple

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="England LACW Pathway Lab (Objective 2)",
    layout="wide",
)

# ============================================================
# GLOBAL PLOT STYLE (Times New Roman)
# ============================================================
pio.templates["paper_style"] = pio.templates["plotly_white"]
pio.templates["paper_style"].layout.font.family = "Times New Roman"
pio.templates["paper_style"].layout.font.size = 16
pio.templates["paper_style"].layout.title.font.size = 18
pio.templates["paper_style"].layout.xaxis.title.font.size = 16
pio.templates["paper_style"].layout.yaxis.title.font.size = 16
pio.templates.default = "paper_style"

# ============================================================
# CONFIG
# ============================================================
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
    "IncNoEnergy",
    "AD",
    "CompostedIV",
    "CompostedW",
]

TREATMENT_ROUTES = [r for r in ROUTES if r != "Collected"]
RECOVERY_ROUTES = {"Recycled", "Reuse", "AD", "CompostedIV", "CompostedW"}
DISPOSAL_ROUTES = {"EfW", "Landfill", "RDF_MHT", "IncNoEnergy"}
SCENARIO_ORDER = ["Baseline (Actual)", "Policy65 - Behaviour", "Policy65 - Cap", "Policy65 - Carbon", "Optimal"]

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
DIVERTABLE_FRACTIONS = [f for f in RESIDUAL_SHARES.keys() if f in FRACTIONS]

EASY_DEST = {
    "Food": [("AD", 1.0)],
    "Garden": [("AD", 1.0)],
    "PaperCard": [("Recycled", 1.0)],
    "Plastics": [("Recycled", 1.0)],
    "Glass": [("Recycled", 1.0)],
    "Metals": [("Recycled", 1.0)],
    "Textiles": [("Recycled", 1.0)],
    "WEEE": [("Recycled", 1.0)],
    "Wood": [("Recycled", 1.0)],
    "OtherRecyclables": [("Recycled", 1.0)],
}

CARBON_ALLOWED = {
    "Food": ["AD", "CompostedIV", "CompostedW", "Reuse"],
    "Garden": ["AD", "CompostedIV", "CompostedW", "Reuse"],
    "PaperCard": ["Recycled", "Reuse"],
    "Plastics": ["Recycled", "Reuse"],
    "Glass": ["Recycled", "Reuse"],
    "Metals": ["Recycled", "Reuse"],
    "Textiles": ["Recycled", "Reuse"],
    "WEEE": ["Recycled", "Reuse"],
    "Wood": ["Recycled", "Reuse"],
    "OtherRecyclables": ["Recycled", "Reuse"],
}

# ============================================================
# CARBON FACTORS (kg CO2e per tonne)
# ============================================================
REF_USER = "User-provided factor set (compiled from CarbonWARM + literature notes)."

DEFAULT_FACTORS = pd.DataFrame([
    {"fraction":"Food", "route":"EfW",         "kgCO2e_per_t": -37.00,   "source": REF_USER},
    {"fraction":"Food", "route":"Landfill",    "kgCO2e_per_t": 627.00,   "source": REF_USER},
    {"fraction":"Food", "route":"AD",          "kgCO2e_per_t": -78.00,   "source": REF_USER},
    {"fraction":"Food", "route":"CompostedIV", "kgCO2e_per_t": -55.00,   "source": REF_USER},
    {"fraction":"Food", "route":"CompostedW",  "kgCO2e_per_t": 6.00,     "source": REF_USER},
    {"fraction":"Food", "route":"Reuse",       "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},

    {"fraction":"Garden", "route":"EfW",         "kgCO2e_per_t": -77.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"Landfill",    "kgCO2e_per_t": 579.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"AD",          "kgCO2e_per_t": -184.09, "source": REF_USER},
    {"fraction":"Garden", "route":"CompostedIV", "kgCO2e_per_t": -45.00,  "source": REF_USER},
    {"fraction":"Garden", "route":"CompostedW",  "kgCO2e_per_t": 56.00,   "source": REF_USER},
    {"fraction":"Garden", "route":"Reuse",       "kgCO2e_per_t": 0.0,     "source": "Reuse credit set to 0 unless displacement is modelled."},

    {"fraction":"PaperCard", "route":"EfW",       "kgCO2e_per_t": -217.00, "source": REF_USER},
    {"fraction":"PaperCard", "route":"Landfill",  "kgCO2e_per_t": 1042.00, "source": REF_USER},
    {"fraction":"PaperCard", "route":"Recycled",  "kgCO2e_per_t": -109.70, "source": REF_USER},
    {"fraction":"PaperCard", "route":"Reuse",     "kgCO2e_per_t": 0.0,     "source": "Reuse credit set to 0 unless displacement is modelled."},

    {"fraction":"Plastics", "route":"EfW",        "kgCO2e_per_t": 1581.70, "source": REF_USER},
    {"fraction":"Plastics", "route":"Landfill",   "kgCO2e_per_t": 9.00,    "source": REF_USER},
    {"fraction":"Plastics", "route":"Recycled",   "kgCO2e_per_t": -576.30, "source": REF_USER},
    {"fraction":"Plastics", "route":"Reuse",      "kgCO2e_per_t": 0.0,     "source": "Reuse credit set to 0 unless displacement is modelled."},

    {"fraction":"Glass", "route":"EfW",           "kgCO2e_per_t": 8.00,    "source": REF_USER},
    {"fraction":"Glass", "route":"Landfill",      "kgCO2e_per_t": 9.00,    "source": REF_USER},
    {"fraction":"Glass", "route":"Recycled",      "kgCO2e_per_t": -326.00, "source": REF_USER},
    {"fraction":"Glass", "route":"Reuse",         "kgCO2e_per_t": 0.0,     "source": "Reuse credit set to 0 unless displacement is modelled."},

    {"fraction":"Metals", "route":"EfW",          "kgCO2e_per_t": 21.50,   "source": REF_USER},
    {"fraction":"Metals", "route":"Landfill",     "kgCO2e_per_t": 9.00,    "source": REF_USER},
    {"fraction":"Metals", "route":"Recycled",     "kgCO2e_per_t": -4578.50,"source": REF_USER},
    {"fraction":"Metals", "route":"Reuse",        "kgCO2e_per_t": 0.0,     "source": "Reuse credit set to 0 unless displacement is modelled."},

    {"fraction":"Textiles", "route":"EfW",        "kgCO2e_per_t": 438.00,   "source": REF_USER},
    {"fraction":"Textiles", "route":"Landfill",   "kgCO2e_per_t": 445.00,   "source": REF_USER},
    {"fraction":"Textiles", "route":"Recycled",   "kgCO2e_per_t": -14315.0, "source": REF_USER},
    {"fraction":"Textiles", "route":"Reuse",      "kgCO2e_per_t": 0.0,      "source": "Reuse credit set to 0 unless displacement is modelled."},

    {"fraction":"WEEE", "route":"EfW",            "kgCO2e_per_t": 450.00,  "source": REF_USER},
    {"fraction":"WEEE", "route":"Landfill",       "kgCO2e_per_t": 20.00,   "source": REF_USER},
    {"fraction":"WEEE", "route":"Recycled",       "kgCO2e_per_t": -1000.0, "source": REF_USER},
    {"fraction":"WEEE", "route":"Reuse",          "kgCO2e_per_t": 0.0,     "source": "Reuse credit set to 0 unless displacement is modelled."},

    {"fraction":"Wood", "route":"EfW",            "kgCO2e_per_t": -318.00, "source": REF_USER},
    {"fraction":"Wood", "route":"Landfill",       "kgCO2e_per_t": 921.00,  "source": REF_USER},
    {"fraction":"Wood", "route":"Recycled",       "kgCO2e_per_t": -754.50, "source": REF_USER},
    {"fraction":"Wood", "route":"Reuse",          "kgCO2e_per_t": 0.0,     "source": "Reuse credit set to 0 unless displacement is modelled."},

    {"fraction":"OtherRecyclables", "route":"EfW",      "kgCO2e_per_t": 0.0, "source":"Placeholder"},
    {"fraction":"OtherRecyclables", "route":"Landfill", "kgCO2e_per_t": 0.0, "source":"Placeholder"},
    {"fraction":"OtherRecyclables", "route":"Recycled", "kgCO2e_per_t": 0.0, "source":"Placeholder"},
    {"fraction":"OtherRecyclables", "route":"Reuse",    "kgCO2e_per_t": 0.0, "source":"Placeholder"},
])

WILDCARD_DEFAULTS = pd.DataFrame([
    {"fraction":"*", "route":"IncNoEnergy", "kgCO2e_per_t": 360.0, "source":"Incineration without energy recovery proxy."},
    {"fraction":"*", "route":"RDF_MHT",     "kgCO2e_per_t": 0.0,   "source":"Intermediate transfer category; no direct factor applied."},
])

FACTORS = pd.concat([DEFAULT_FACTORS, WILDCARD_DEFAULTS], ignore_index=True)

# ============================================================
# ECONOMIC VALUE FACTORS (from your Table 2 logic)
# IMPORTANT: Food = 0, Garden = 0; WEEE uses £200–250/t midpoint £225/t
# ============================================================
VALUE_FACTORS = pd.DataFrame([
    {"fraction":"Food",            "gbp_per_t_min": 0.0,   "gbp_per_t_mid": 0.0,   "gbp_per_t_max": 0.0,   "source":"No direct commodity value assumed"},
    {"fraction":"Garden",          "gbp_per_t_min": 0.0,   "gbp_per_t_mid": 0.0,   "gbp_per_t_max": 0.0,   "source":"No direct commodity value assumed"},
    {"fraction":"PaperCard",       "gbp_per_t_min": 80.0,  "gbp_per_t_mid": 105.0, "gbp_per_t_max": 130.0, "source":"MPR proxy"},
    {"fraction":"Plastics",        "gbp_per_t_min": 135.0, "gbp_per_t_mid": 427.5, "gbp_per_t_max": 720.0, "source":"MPR proxy"},
    {"fraction":"Glass",           "gbp_per_t_min": -5.0,  "gbp_per_t_mid": 5.0,   "gbp_per_t_max": 15.0,  "source":"MPR proxy"},
    {"fraction":"Metals",          "gbp_per_t_min": 80.0,  "gbp_per_t_mid": 600.0, "gbp_per_t_max": 1120.0, "source":"MPR proxy"},
    {"fraction":"Textiles",        "gbp_per_t_min": 65.0,  "gbp_per_t_mid": 207.5, "gbp_per_t_max": 350.0, "source":"MPR proxy"},
    {"fraction":"WEEE",            "gbp_per_t_min": 200.0, "gbp_per_t_mid": 225.0, "gbp_per_t_max": 250.0, "source":"Recovered WEEE price estimate"},
    {"fraction":"Wood",            "gbp_per_t_min": -45.0, "gbp_per_t_mid": -10.0, "gbp_per_t_max": 25.0,  "source":"MPR proxy"},
    {"fraction":"OtherRecyclables","gbp_per_t_min": 0.0,   "gbp_per_t_mid": 0.0,   "gbp_per_t_max": 0.0,   "source":"Placeholder"},
])

VALUE_ROUTES = {"Recycled", "Reuse"}

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

def value_map(values_df: pd.DataFrame):
    m = {r["fraction"]: float(r["gbp_per_t_mid"]) for _, r in values_df.iterrows()}
    def get(frac):
        return m.get(frac, 0.0)
    return get

def apply_paper_style(fig, title=None, x_title=None, y_title=None, tickangle=28):
    if title is not None:
        fig.update_layout(
            title=dict(
                text=title,
                x=0.02,
                xanchor="left",
                y=0.97,
                yanchor="top",
                font=dict(size=20, color="black")
            )
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman", size=16, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=70, b=90),
        legend=dict(
            title=None,
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(0,0,0,0)"
        ),
        bargap=0.18,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Times New Roman"
        ),
    )

    fig.update_xaxes(
        title_text=x_title if x_title is not None else None,
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=1.2,
        linecolor="black",
        mirror=False,
        ticks="outside",
        tickwidth=1.0,
        tickcolor="black",
        ticklen=5,
        tickangle=tickangle,
        automargin=True
    )

    fig.update_yaxes(
        title_text=y_title if y_title is not None else None,
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=1.2,
        linecolor="black",
        mirror=False,
        ticks="outside",
        tickwidth=1.0,
        tickcolor="black",
        ticklen=5,
        automargin=True
    )
    return fig

def figure_download_buttons(fig, base_filename):
    st.caption("Use the camera icon on the chart to download a PNG quickly.")
@st.cache_data(show_spinner=False)
def load_csv(path: str):
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner=False)
def build_long(df: pd.DataFrame):
    if "Council Name" not in df.columns:
        raise ValueError("Missing required column: Council Name")

    out_frames = []
    council = df[["Council Name"]].copy()
    council["Council Name"] = council["Council Name"].astype(str)

    for y in YEARS:
        year_cols = [c for c in df.columns if c.endswith(f"_{y}")]
        route_cols = []
        for frac in FRACTIONS:
            for route in ROUTES:
                col = f"{frac}{route}_{y}"
                if col in year_cols:
                    route_cols.append(col)
        if route_cols:
            tmp = pd.concat([council, df[route_cols]], axis=1).melt(
                id_vars=["Council Name"],
                value_vars=route_cols,
                var_name="key",
                value_name="tonnes",
            )
            tmp["year"] = y
            tmp["tonnes"] = safe_num(tmp["tonnes"])
            tmp = tmp[tmp["tonnes"] != 0].copy()
            tmp["key"] = tmp["key"].str.replace(f"_{y}", "", regex=False)
            parsed = []
            for frac in FRACTIONS:
                for route in ROUTES:
                    parsed.append((f"{frac}{route}", frac, route))
            map_df = pd.DataFrame(parsed, columns=["key", "fraction", "route"])
            tmp = tmp.merge(map_df, on="key", how="left").drop(columns="key")
            # Harmonise reported garden "recycling" with biological recovery for the carbon model
            tmp.loc[(tmp["fraction"] == "Garden") & (tmp["route"] == "Recycled"), "route"] = "AD"
            out_frames.append(tmp[["Council Name", "year", "fraction", "route", "tonnes"]])

    long = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(
        columns=["Council Name", "year", "fraction", "route", "tonnes"]
    )

    def _collect_metric(prefix: str, rename_to: str, suffix_sep: str = ""):
        frames = []
        for y in YEARS:
            col = f"{prefix}{suffix_sep}{y}"
            if col in df.columns:
                frames.append(pd.DataFrame({
                    "Council Name": council["Council Name"],
                    "year": y,
                    rename_to: safe_num(df[col]),
                }))
        return pd.concat(frames, ignore_index=True) if frames else None

    totals = _collect_metric("TotalCollected", "TotalCollected")
    rec_col = _collect_metric("TotalRecCollec", "TotalRecCollec")
    residual = _collect_metric("Residual", "Residual", suffix_sep="_")
    pop = _collect_metric("population", "population", suffix_sep="_")

    if pop is not None and len(long):
        long = long.merge(pop, on=["Council Name", "year"], how="left")
    else:
        long["population"] = np.nan

    return long, totals, rec_col, residual

def apply_factors(long_df: pd.DataFrame, factors_df: pd.DataFrame):
    """Apply carbon factors with exact fraction-route matches first, then route-level defaults.

    Expected factor table columns: fraction, route, kgCO2e_per_t.
    Route-level defaults are stored with fraction == "*".
    """
    out = long_df.copy()

    if not {"fraction", "route", "kgCO2e_per_t"}.issubset(factors_df.columns):
        missing = {"fraction", "route", "kgCO2e_per_t"} - set(factors_df.columns)
        raise KeyError(f"FACTORS is missing required columns: {sorted(missing)}")

    exact = (
        factors_df[factors_df["fraction"].astype(str) != "*"]
        [["fraction", "route", "kgCO2e_per_t"]]
        .drop_duplicates(subset=["fraction", "route"])
        .copy()
    )

    wild = (
        factors_df[factors_df["fraction"].astype(str) == "*"]
        [["route", "kgCO2e_per_t"]]
        .drop_duplicates(subset=["route"])
        .copy()
    )

    out = out.merge(exact, on=["fraction", "route"], how="left")

    if len(wild):
        out = out.merge(
            wild.rename(columns={"kgCO2e_per_t": "kgCO2e_per_t_wild"}),
            on="route",
            how="left",
        )
        out["kgCO2e_per_t"] = out["kgCO2e_per_t"].fillna(out["kgCO2e_per_t_wild"])
        out = out.drop(columns=["kgCO2e_per_t_wild"])

    out["factor_missing"] = out["kgCO2e_per_t"].isna().astype(int)
    out["kgCO2e_per_t"] = out["kgCO2e_per_t"].fillna(0.0)
    out["kgCO2e"] = out["tonnes"] * out["kgCO2e_per_t"]
    return out

def apply_values(long_df: pd.DataFrame, values_df: pd.DataFrame):
    out = long_df.copy()
    out = out.merge(values_df[["fraction", "gbp_per_t_mid"]], on="fraction", how="left")
    out["gbp_per_t_mid"] = out["gbp_per_t_mid"].fillna(0.0)
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

    rec_t = (long[long["route"].isin(RECOVERY_ROUTES)]
             .groupby(["Council Name", "year"], as_index=False)
             .agg(recovery_t=("tonnes", "sum")))
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
    g = (long_y[long_y["route"].isin(DISPOSAL_ROUTES)]
         .groupby(["Council Name", "year", "route"], as_index=False)
         .agg(t=("tonnes", "sum")))

    if len(g) == 0:
        return pd.DataFrame(columns=["Council Name", "year", "route", "share"])

    piv = g.pivot_table(index=["Council Name", "year"], columns="route", values="t", aggfunc="sum").fillna(0.0)
    for r in DISPOSAL_ROUTES:
        if r not in piv.columns:
            piv[r] = 0.0
    s = piv[list(DISPOSAL_ROUTES)].sum(axis=1).replace(0, np.nan)
    shares = piv.div(s, axis=0).fillna(0.0)

    all_zero = (piv[list(DISPOSAL_ROUTES)].sum(axis=1) == 0)
    if all_zero.any():
        shares.loc[all_zero, :] = 0.0
        shares.loc[all_zero, "EfW"] = 1.0

    out = shares.reset_index().melt(id_vars=["Council Name", "year"], var_name="route", value_name="share")
    return out

def build_synthetic_residual_disposal(residual_frac: pd.DataFrame, disp_shares: pd.DataFrame):
    m = residual_frac.merge(disp_shares, on=["Council Name", "year"], how="left")
    m["share"] = m["share"].fillna(0.0)
    m["tonnes"] = m["residual_frac_t"] * m["share"]
    m = m[m["tonnes"] > 0].copy()
    m["route"] = m["route"].astype(str)
    return m[["Council Name", "year", "fraction", "route", "tonnes"]]

def hist_propensity_routes(long_all: pd.DataFrame):
    df = long_all[long_all["route"].isin(RECOVERY_ROUTES)].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["Council Name", "fraction", "route", "w"])

    piv = df.pivot_table(index=["Council Name", "fraction"], columns="route", values="tonnes", aggfunc="sum").fillna(0.0)
    for r in RECOVERY_ROUTES:
        if r not in piv.columns:
            piv[r] = 0.0

    denom = piv[list(RECOVERY_ROUTES)].sum(axis=1).replace(0, np.nan)
    shares_mat = piv[list(RECOVERY_ROUTES)].div(denom, axis=0).fillna(0.0)

    shares = shares_mat.reset_index().melt(id_vars=["Council Name", "fraction"], var_name="route", value_name="w")
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
    vals.sort(key=lambda x: x[1])
    return vals[0][0]

def national_disposal_mix(long_y: pd.DataFrame):
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

def carbon_gain_per_tonne(frac: str, factors_df: pd.DataFrame, base_disposal_mix: dict):
    get = factor_maps(factors_df)
    best_route = carbon_best_route(frac, factors_df)
    best = get(frac, best_route)
    if np.isnan(best):
        return 0.0

    base = 0.0
    any_base = False
    for r, sh in base_disposal_mix.items():
        v = get(frac, r)
        if not np.isnan(v):
            base += float(sh) * float(v)
            any_base = True
    if not any_base:
        vL = get(frac, "Landfill")
        vE = get(frac, "EfW")
        base = vL if not np.isnan(vL) else (vE if not np.isnan(vE) else 0.0)
    return float(base - best)

def choose_destination_weights(mode: str, chosen_rows: pd.DataFrame, shares_hist: pd.DataFrame, factors_df: pd.DataFrame):
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
                    dest[(c, f)] = [(rr["route"], float(rr["w"]) / s) for _, rr in sub.iterrows()]
        else:
            dest[(c, f)] = EASY_DEST.get(f, [("Recycled", 1.0)])

    return dest

def allocate_diversion_massbalanced(
    residual_frac: pd.DataFrame,
    residual_disp_synth: pd.DataFrame,
    long_all: pd.DataFrame,
    long_y: pd.DataFrame,
    extra_needed: float,
    mode: str,
    factors_df: pd.DataFrame
):
    if extra_needed <= 0:
        return pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    pool = residual_frac.copy()
    pool = pool[pool["residual_frac_t"] > 0].copy()
    if len(pool) == 0:
        return pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

    shares_hist = hist_propensity_routes(long_all)
    nat_disp_mix = national_disposal_mix(long_y)
    pool["carbon_gain"] = pool["fraction"].apply(lambda f: carbon_gain_per_tonne(f, factors_df, nat_disp_mix))

    if mode == "scale":
        pool["rank"] = -(pool["residual_frac_t"])
    elif mode == "propensity":
        tmp = shares_hist.groupby(["Council Name","fraction"], as_index=False).agg(p=("w","sum"))
        pool = pool.merge(tmp, on=["Council Name","fraction"], how="left")
        pool["p"] = pool["p"].fillna(0.0).clip(0, 1)
        scale_norm = pool["residual_frac_t"] / (pool["residual_frac_t"].max() if pool["residual_frac_t"].max() > 0 else 1.0)
        pool["rank"] = -(0.7 * pool["p"] + 0.3 * scale_norm)
    elif mode == "carbon":
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

    dest_map = choose_destination_weights(mode, chosen, shares_hist, factors_df)

    pos_parts = []
    for _, r in chosen.iterrows():
        c = r["Council Name"]; y = int(r["year"]); f = r["fraction"]; t = float(r["take"])
        weights = dest_map.get((c, f), EASY_DEST.get(f, [("Recycled", 1.0)]))
        for route, w in weights:
            pos_parts.append({"Council Name": c, "year": y, "fraction": f, "route": route, "tonnes": t * float(w)})
    pos = pd.DataFrame(pos_parts) if pos_parts else pd.DataFrame(columns=["Council Name","year","fraction","route","tonnes"])

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
        sub["sub_t"] = np.minimum(sub["share"] * take, sub["avail_t"])

        for _, rr in sub.iterrows():
            if rr["sub_t"] > 0:
                neg_parts.append({
                    "Council Name": c,
                    "year": y,
                    "fraction": f,
                    "route": rr["route"],
                    "tonnes": -float(rr["sub_t"])
                })

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
    d = df.groupby(["scenario","fraction","route"], as_index=False).agg(v=(value_col,"sum"))
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
    value += sf["v"].tolist()

    for s in scen:
        tmp = d[d["scenario"] == s].groupby(["fraction","route"], as_index=False).agg(v=("v","sum"))
        source += [idx[f] for f in tmp["fraction"]]
        target += [idx[r] for r in tmp["route"]]
        value += tmp["v"].tolist()

    fig = go.Figure(go.Sankey(
        node=dict(pad=10, thickness=14, label=nodes),
        link=dict(source=source, target=target, value=value)
    ))
    fig.update_layout(title=title, height=760)
    return fig

# ============================================================
# SIDEBAR — DATA
# ============================================================
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
    st.error("Missing required columns: TotalCollectedYYYY, TotalRecCollecYYYY, Residual_YYYY, and ideally population_YYYY.")
    st.stop()

# ============================================================
# SIDEBAR — CONTROLS
# ============================================================
st.sidebar.header("Controls")
unit = st.sidebar.selectbox("Display unit", ["tCO2e", "kgCO2e"], index=0)
to_unit = (lambda kg: kg / 1000.0) if unit == "tCO2e" else (lambda kg: kg)
unit_label = unit

target_rate = st.sidebar.slider("Policy target recovery rate", 0.40, 0.80, 0.65, 0.01)

# ============================================================
# PREP BASE TABLES
# ============================================================
em_obs = apply_factors(long, FACTORS)
em_obs["CO2e_u"] = to_unit(em_obs["kgCO2e"])
val_obs = apply_values(long, VALUE_FACTORS)
baseCY = council_year_base(totals, rec_col, residual, long)

long.to_csv("raw_council_route_data.csv", index=False)

council_route_summary = (
    long.groupby(["Council Name", "year", "route"], as_index=False)
    .agg(tonnes=("tonnes", "sum"))
)

council_route_summary.to_csv("council_route_summary.csv", index=False)

# ============================================================
# HEADER
# ============================================================
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

@st.cache_data(show_spinner="Building scenario tables…")
def build_all_scenarios(long: pd.DataFrame, totals: pd.DataFrame, residual: pd.DataFrame, target_rate: float):
    scen_all = []
    meta = []

    for y in YEARS:
        collected_y = long[long["year"] == y].copy()
        totals_y = totals[totals["year"] == y].copy()
        residual_y = residual[residual["year"] == y].copy()

        totalY, recY, base_rate_y = recovery_rate_national(collected_y, totals_y)
        needed_y = max(0.0, target_rate * totalY - recY)

        residual_frac = build_residual_by_fraction(residual_y)
        disp_sh = disposal_shares_from_data_by_council(collected_y)
        residual_disp_synth = build_synthetic_residual_disposal(residual_frac, disp_sh)

        capacity_y = float(residual_frac["residual_frac_t"].sum())
        achievable_y = min(needed_y, capacity_y)

        baseline = scenario_df("Baseline (Actual)", collected_y, residual_disp_synth, None)

        adj_scale = allocate_diversion_massbalanced(
            residual_frac, residual_disp_synth, long_all=long, long_y=collected_y,
            extra_needed=achievable_y, mode="scale", factors_df=FACTORS
        )
        adj_prop = allocate_diversion_massbalanced(
            residual_frac, residual_disp_synth, long_all=long, long_y=collected_y,
            extra_needed=achievable_y, mode="propensity", factors_df=FACTORS
        )
        adj_car = allocate_diversion_massbalanced(
            residual_frac, residual_disp_synth, long_all=long, long_y=collected_y,
            extra_needed=achievable_y, mode="carbon", factors_df=FACTORS
        )
        adj_opt = allocate_diversion_massbalanced(
            residual_frac, residual_disp_synth, long_all=long, long_y=collected_y,
            extra_needed=capacity_y, mode="carbon", factors_df=FACTORS
        )

        pol_scale = scenario_df("Policy65 - Cap", collected_y, residual_disp_synth, adj_scale)
        pol_prop = scenario_df("Policy65 - Behaviour", collected_y, residual_disp_synth, adj_prop)
        pol_car = scenario_df("Policy65 - Carbon", collected_y, residual_disp_synth, adj_car)
        optimal = scenario_df("Optimal", collected_y, residual_disp_synth, adj_opt)

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
    scen_em = apply_factors(scen_long, FACTORS)
    scen_val = apply_values(scen_long, VALUE_FACTORS)
    return scen_long, meta_df, scen_em, scen_val

# ============================================================
# BUILD SCENARIOS
# ============================================================
scen_long, meta_df, scen_em, scen_val = build_all_scenarios(long, totals, residual, target_rate)
scen_em["CO2e_u"] = to_unit(scen_em["kgCO2e"])

# ============================================================
# TABS
# ============================================================
tabTS, tabScen, tabPriority, tabSank, tabCouncil, tabExports, tabDiag = st.tabs([
    "1) Time series",
    "2) Scenarios",
    "3) Value × Carbon priorities",
    "4) Sankey",
    "5) Council explorer",
    "6) Exports",
    "7) Diagnostics",
])

# ============================================================
# 1) TIME SERIES
# ============================================================
with tabTS:
    st.subheader("National time series (observed collected-material pathways)")

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

    left, right = st.columns(2)
    with left:
        fig_rate = px.line(nat, x="year", y="RecoveryRate", markers=True)
        fig_rate = apply_paper_style(
            fig_rate,
            title="Recovery rate of the observed system",
            x_title="Year",
            y_title="Recovery rate"
        )
        fig_rate.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_rate, use_container_width=True)
        figure_download_buttons(fig_rate, "fig_observed_recovery_rate")

    with right:
        fig_em = px.line(nat, x="year", y=f"Emissions_{unit_label}", markers=True)
        fig_em = apply_paper_style(
            fig_em,
            title=f"Total emissions of the observed system ({unit_label})",
            x_title="Year",
            y_title=f"Emissions ({unit_label})"
        )
        st.plotly_chart(fig_em, use_container_width=True)
        figure_download_buttons(fig_em, "fig_observed_total_emissions")

    st.markdown("### Recovery pathways over time (tonnes)")
    rec_mix = (
        long[long["route"].isin(RECOVERY_ROUTES)]
        .groupby(["year", "route"], as_index=False)
        .agg(tonnes=("tonnes", "sum"))
    )
    fig_rec_mix = px.area(rec_mix, x="year", y="tonnes", color="route")
    fig_rec_mix = apply_paper_style(
        fig_rec_mix,
        title="Recovery pathways over time in the observed system",
        x_title="Year",
        y_title="Tonnes"
    )
    st.plotly_chart(fig_rec_mix, use_container_width=True)
    figure_download_buttons(fig_rec_mix, "fig_recovery_pathways_over_time")

    st.markdown("### Average treatment pathway structure (2020–2023)")

    # =========================
    # ALL TREATMENT PATHWAYS
    # =========================
    route_mix_avg = (
        long[long["route"].isin(TREATMENT_ROUTES)]
        .groupby(["year", "route"], as_index=False)
        .agg(tonnes=("tonnes", "sum"))
    )

    route_mix_avg = (
        route_mix_avg
        .groupby("route", as_index=False)
        .agg(avg_tonnes=("tonnes", "mean"))
    )

    route_mix_avg["pct"] = route_mix_avg["avg_tonnes"] / route_mix_avg["avg_tonnes"].sum()
    route_mix_avg["Mt"] = route_mix_avg["avg_tonnes"] / 1_000_000

    route_mix_avg.to_csv("average_treatment_pathways_2020_2023.csv", index=False)

    fig_all_routes = px.pie(
        route_mix_avg,
        values="avg_tonnes",
        names="route",
        hole=0.45
    )

    fig_all_routes.update_traces(
        texttemplate="%{label}<br>%{percent:.1%}<br>%{value:.2s}",
        hovertemplate="Route: %{label}<br>Average tonnes: %{value:,.0f}<extra></extra>"
    )

    fig_all_routes.update_layout(
        title="Average distribution of LACW across all treatment pathways (England, 2020–2023)",
        font=dict(family="Times New Roman", size=16),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_all_routes, use_container_width=True)
    figure_download_buttons(fig_all_routes, "fig_avg_all_treatment_pathways_2020_2023")

    # =========================
    # RECYCLING / RECOVERY ONLY
    # =========================
    recovery_routes = ["Recycled", "Reuse", "AD", "CompostedIV", "CompostedW"]

    rec_mix_avg = (
        long[long["route"].isin(recovery_routes)]
        .groupby(["year", "route"], as_index=False)
        .agg(tonnes=("tonnes", "sum"))
    )

    rec_mix_avg = (
        rec_mix_avg
        .groupby("route", as_index=False)
        .agg(avg_tonnes=("tonnes", "mean"))
    )

    rec_mix_avg["pct"] = rec_mix_avg["avg_tonnes"] / rec_mix_avg["avg_tonnes"].sum()
    rec_mix_avg["Mt"] = rec_mix_avg["avg_tonnes"] / 1_000_000

    rec_mix_avg.to_csv("average_recycling_pathways_2020_2023.csv", index=False)

    fig_recovery = px.pie(
        rec_mix_avg,
        values="avg_tonnes",
        names="route",
        hole=0.45
    )

    fig_recovery.update_traces(
        texttemplate="%{label}<br>%{percent:.1%}<br>%{value:.2s}",
        hovertemplate="Route: %{label}<br>Average tonnes: %{value:,.0f}<extra></extra>"
    )

    fig_recovery.update_layout(
        title="Average distribution of recovered LACW across recycling pathways (England, 2020–2023)",
        font=dict(family="Times New Roman", size=16),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_recovery, use_container_width=True)
    figure_download_buttons(fig_recovery, "fig_avg_recycling_pathways_2020_2023")

    st.markdown("### Emissions by treatment pathway over time")

    route_em = (
        em_obs[em_obs["route"].isin(TREATMENT_ROUTES)]
        .groupby(["year", "route"], as_index=False)
        .agg(
            kgCO2e=("kgCO2e", "sum"),
            tonnes=("tonnes", "sum")
        )
    )

    route_em[f"Emissions_{unit_label}"] = to_unit(route_em["kgCO2e"])

    fig_route_em = px.bar(
        route_em,
        x="year",
        y=f"Emissions_{unit_label}",
        color="route",
        barmode="stack",
        custom_data=["tonnes", "kgCO2e"]
    )

    fig_route_em = apply_paper_style(
        fig_route_em,
        title=f"Emissions by treatment pathway in the observed system ({unit_label})",
        x_title="Year",
        y_title=f"Emissions ({unit_label})",
        tickangle=0
    )

    fig_route_em.update_traces(
        hovertemplate=(
            "Year: %{x}<br>"
            "Route: %{fullData.name}<br>"
            f"Emissions: %{{y:,.0f}} {unit_label}<br>"
            "Tonnes: %{customdata[0]:,.0f}<br>"
            "kgCO2e: %{customdata[1]:,.0f}<extra></extra>"
        ),
        marker_line_width=0
    )

    st.plotly_chart(fig_route_em, use_container_width=True)
    figure_download_buttons(fig_route_em, "fig_emissions_by_treatment_pathway_over_time")

    st.markdown("### Exact emissions by treatment pathway and year")
    route_em_table = (
        route_em
        .pivot(index="route", columns="year", values=f"Emissions_{unit_label}")
        .round(2)
    )
    st.dataframe(route_em_table, use_container_width=True)

    csv_em = route_em_table.to_csv().encode("utf-8")
    st.download_button(
        "Download emissions by pathway table (CSV)",
        csv_em,
        "emissions_by_pathway_2020_2023.csv",
        "text/csv"
    )

# ============================================================
# 2) SCENARIOS
# ============================================================
with tabScen:
    st.subheader("Baseline, Policy65, and Optimal scenario comparison")
    st.dataframe(meta_df, use_container_width=True, height=170)

    year = st.selectbox("Year", YEARS, index=2, key="sc_year")
    totals_y = totals[totals["year"] == year].copy()
    totalY = float(totals_y["TotalCollected"].sum())

    d = scen_long[scen_long["year"] == year].copy()
    d_em = scen_em[(scen_em["year"] == year) & (scen_em["route"].isin(RECOVERY_ROUTES))].copy()
    d_val = scen_val[scen_val["year"] == year].copy()

    rec = d[d["route"].isin(RECOVERY_ROUTES)].groupby("scenario", as_index=False).agg(Recovery_t=("tonnes","sum"))
    rec["RecoveryRate"] = np.where(totalY > 0, rec["Recovery_t"] / totalY, np.nan)

    ems = d_em.groupby("scenario", as_index=False).agg(kgCO2e=("kgCO2e","sum"))
    ems[f"Emissions_{unit_label}"] = to_unit(ems["kgCO2e"])

    # Figure 6a should reflect the 2020–2023 average for recovery pathways only,
    # not the currently selected year and not disposal/residual pathways.
    scen_em_recovery = scen_em[scen_em["route"].isin(RECOVERY_ROUTES)].copy()
    ems_fig6a = (
        scen_em_recovery.groupby(["scenario", "year"], as_index=False)
        .agg(kgCO2e=("kgCO2e", "sum"))
        .groupby("scenario", as_index=False)
        .agg(kgCO2e_avg=("kgCO2e", "mean"))
    )
    ems_fig6a[f"Emissions_{unit_label}"] = to_unit(ems_fig6a["kgCO2e_avg"])

    val = d_val[d_val["route"].isin(VALUE_ROUTES)].groupby("scenario", as_index=False).agg(RecoveredValue_GBP=("gbp_value","sum"))

    summ = rec.merge(ems[["scenario", f"Emissions_{unit_label}"]], on="scenario", how="left").merge(val, on="scenario", how="left")
    summ["RecoveredValue_GBP"] = summ["RecoveredValue_GBP"].fillna(0.0)
    for _df in [summ, ems_fig6a]:
        _df["scenario"] = pd.Categorical(_df["scenario"], categories=SCENARIO_ORDER, ordered=True)


    fig1 = px.bar(
        summ.sort_values("scenario"),
        x="scenario",
        y="RecoveryRate",
        text="RecoveryRate"
    )
    fig1 = apply_paper_style(
        fig1,
        title="Recovery rate under baseline and scenario conditions",
        x_title="Scenario",
        y_title="Recovery rate (%)"
    )
    fig1.update_yaxes(
        tickformat=".0%",
        range=[0, max(1.0, float(summ["RecoveryRate"].max()) * 1.10)]
    )
    fig1.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate="Scenario: %{x}<br>Recovery rate: %{y:.1%}<extra></extra>"
    )
    st.plotly_chart(fig1, use_container_width=True)
    figure_download_buttons(fig1, "fig_scenario_recovery_rate")

    mix_total = d[d["route"].isin(RECOVERY_ROUTES)].groupby(["scenario","route"], as_index=False).agg(tonnes=("tonnes","sum"))
    mix_total["pct"] = mix_total["tonnes"] / mix_total.groupby("scenario")["tonnes"].transform("sum")
    fig2 = px.bar(
        mix_total,
        x="scenario",
        y="pct",
        color="route",
        barmode="stack",
        custom_data=["tonnes", "pct"]
    )
    fig2 = apply_paper_style(
        fig2,
        title="Changes in recovery pathway distribution across scenarios",
        x_title="Scenario",
        y_title="Share of recovered pathways (%)"
    )
    fig2.update_yaxes(tickformat=".0%", range=[0, 1])
    fig2.update_traces(
        hovertemplate=(
            "Scenario: %{x}<br>"
            "Route: %{fullData.name}<br>"
            "Share: %{customdata[1]:.1%}<br>"
            "Tonnes: %{customdata[0]:,.0f}<extra></extra>"
        ),
        marker_line_width=0
    )
    st.plotly_chart(fig2, use_container_width=True)
    figure_download_buttons(fig2, "fig_route_mix_by_scenario_pct")

    c1, c2 = st.columns(2)
    with c1:
        fig3_df = ems_fig6a.sort_values("scenario")
        fig3 = px.bar(
            fig3_df,
            x="scenario",
            y=f"Emissions_{unit_label}",
            custom_data=[f"Emissions_{unit_label}"]
        )
        fig3 = apply_paper_style(
            fig3,
            title=f"Average greenhouse-gas emissions from recovery pathways under baseline and scenario conditions (2020–2023, {unit_label})",
            x_title="Scenario",
            y_title=f"Emissions from recovery pathways ({unit_label})"
        )
        emin = float(fig3_df[f"Emissions_{unit_label}"].min())
        emax = float(fig3_df[f"Emissions_{unit_label}"].max())
        fig3.update_yaxes(range=[emin * 1.10 if emin < 0 else 0, emax * 1.10 if emax > 0 else 0])
        fig3.update_traces(
            marker_line_width=0,
            hovertemplate=(
                "Scenario: %{x}<br>"
                f"Emissions: %{{customdata[0]:,.0f}} {unit_label}<extra></extra>"
            )
        )
        st.plotly_chart(fig3, use_container_width=True)
        figure_download_buttons(fig3, "fig_scenario_total_emissions")

    with c2:
        fig4 = px.bar(
            summ.sort_values("scenario"),
            x="scenario",
            y="RecoveredValue_GBP",
            custom_data=["RecoveredValue_GBP"]
        )
        fig4 = apply_paper_style(
            fig4,
            title="Estimated recovered commodity value under baseline and scenario conditions",
            x_title="Scenario",
            y_title="Recovered value (GBP)"
        )
        fig4.update_yaxes(tickprefix="£", separatethousands=True)
        fig4.update_traces(
            marker_line_width=0,
            hovertemplate=(
                "Scenario: %{x}<br>"
                "Recovered value: £%{customdata[0]:,.0f}<extra></extra>"
            )
        )
        st.plotly_chart(fig4, use_container_width=True)
        figure_download_buttons(fig4, "fig_scenario_recovered_value")

    st.markdown("### Emission intensity by scenario")

    scenario_tonnes = (
        scen_long[scen_long["route"].isin(RECOVERY_ROUTES)]
        .groupby(["scenario", "year"], as_index=False)
        .agg(RecoveredTonnes=("tonnes", "sum"))
        .groupby("scenario", as_index=False)
        .agg(RecoveredTonnes=("RecoveredTonnes", "mean"))
    )
    scenario_tonnes["scenario"] = pd.Categorical(scenario_tonnes["scenario"], categories=SCENARIO_ORDER, ordered=True)

    summ_int = ems_fig6a.merge(scenario_tonnes, on="scenario", how="left")
    summ_int["EmissionIntensity_kgCO2e_per_t"] = np.where(
        summ_int["RecoveredTonnes"] > 0,
        summ_int["kgCO2e_avg"] / summ_int["RecoveredTonnes"],
        np.nan,
    )

    st.dataframe(
        summ_int[["scenario", "RecoveredTonnes", "EmissionIntensity_kgCO2e_per_t"]].round(2),
        use_container_width=True,
    )

    fig_intensity = px.bar(
        summ_int.sort_values("scenario"),
        x="scenario",
        y="EmissionIntensity_kgCO2e_per_t",
        custom_data=["RecoveredTonnes"],
    )
    fig_intensity = apply_paper_style(
        fig_intensity,
        title="Average emission intensity by scenario (2020–2023, recovery pathways only)",
        x_title="Scenario",
        y_title="kgCO2e per tonne of recovered material",
    )
    fig_intensity.update_traces(
        marker_line_width=0,
        hovertemplate=(
            "Scenario: %{x}<br>"
            "Emission intensity: %{y:,.2f} kgCO2e/t<br>"
            "Recovered tonnes: %{customdata[0]:,.0f}<extra></extra>"
        )
    )
    st.plotly_chart(fig_intensity, use_container_width=True)
    figure_download_buttons(fig_intensity, "fig_emission_intensity_by_scenario")

# ============================================================
# 3) VALUE × CARBON PRIORITIES
# ============================================================
with tabPriority:
    st.subheader("Carbon–value prioritisation of diverted fractions")

    year = st.selectbox("Year", YEARS, index=2, key="prio_year")
    scenario = st.selectbox(
        "Scenario",
        sorted(scen_long[scen_long["year"] == year]["scenario"].unique().tolist()),
        index=1,
        key="prio_scen"
    )

    d = scen_long[scen_long["year"] == year].copy()
    d_em = scen_em[scen_em["year"] == year].copy()
    adj = d[(d["scenario"] == scenario) & (d["stream"] == "Scenario_adjustment") & (d["route"].isin(RECOVERY_ROUTES))].copy()

    if len(adj) == 0:
        st.info("No diversion in this scenario/year.")
    else:
        adj_em = d_em[(d_em["scenario"] == scenario) & (d_em["stream"] == "Scenario_adjustment") & (d_em["route"].isin(RECOVERY_ROUTES))].copy()
        diverted = adj.groupby("fraction", as_index=False).agg(diverted_t=("tonnes","sum"))
        diverted = diverted[diverted["diverted_t"] > 0].copy()

        collected_y = long[long["year"] == year].copy()
        nat_mix = national_disposal_mix(collected_y)
        getf = factor_maps(FACTORS)

        rec_fac = adj_em.groupby(["fraction","route"], as_index=False).agg(t=("tonnes","sum"), kg=("kgCO2e","sum"))
        rec_avg = (rec_fac.groupby("fraction", as_index=False)
                   .apply(lambda g: pd.Series({
                       "recovery_avg_kg_per_t": (g["kg"].sum() / g["t"].sum()) if g["t"].sum() > 0 else np.nan
                   })).reset_index(drop=True))

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

        vdf = VALUE_FACTORS[["fraction","gbp_per_t_mid"]].copy()
        pri = diverted.merge(base_df, on="fraction", how="left").merge(rec_avg, on="fraction", how="left").merge(vdf, on="fraction", how="left")
        pri["gbp_per_t_mid"] = pri["gbp_per_t_mid"].fillna(0.0)
        pri["recovery_avg_kg_per_t"] = pri["recovery_avg_kg_per_t"].fillna(0.0)
        pri["carbon_benefit_kg_per_t"] = pri["baseline_disposal_avg_kg_per_t"] - pri["recovery_avg_kg_per_t"]

        # filter toggle
        show_zero_value = st.checkbox("Show fractions with zero direct commodity value", value=True, key="show_zero_value")
        plot_df = pri.copy() if show_zero_value else pri[pri["gbp_per_t_mid"] > 0].copy()

        x_mid = float(plot_df["gbp_per_t_mid"].median()) if len(plot_df) else 0.0
        y_mid = float(plot_df["carbon_benefit_kg_per_t"].median()) if len(plot_df) else 0.0

        fig5 = px.scatter(
            plot_df,
            x="gbp_per_t_mid",
            y="carbon_benefit_kg_per_t",
            size="diverted_t",
            text="fraction",
            hover_data={
                "diverted_t": ":,.0f",
                "gbp_per_t_mid": ":,.0f",
                "carbon_benefit_kg_per_t": ":,.0f",
            }
        )
        fig5.add_vline(x=x_mid)
        fig5.add_hline(y=y_mid)
        fig5.update_traces(textposition="top center")
        fig5 = apply_paper_style(
            fig5,
            title="Carbon–value prioritisation of diverted fractions (bubble size = diverted tonnes)",
            x_title="Market value of recovered material (GBP per tonne)",
            y_title="Carbon benefit of diversion (kgCO2e saved per tonne diverted)"
        )
        st.plotly_chart(fig5, use_container_width=True)
        figure_download_buttons(fig5, "fig_carbon_value_prioritisation")

        st.markdown("### Matrix input table")
        st.dataframe(
            pri[["fraction", "gbp_per_t_mid", "carbon_benefit_kg_per_t", "diverted_t"]].sort_values(
                ["gbp_per_t_mid", "carbon_benefit_kg_per_t"], ascending=[False, False]
            ),
            use_container_width=True,
            height=320
        )

# ============================================================
# 4) SANKEY
# ============================================================
with tabSank:
    st.subheader("Sankey representation of scenario flows")

    year = st.selectbox("Year", YEARS, index=2, key="sank_year")
    top_links = st.slider("Keep top links (readability)", 20, 100, 60, 10)
    metric = st.radio("Size links by", ["Tonnes", "Emissions magnitude |tCO2e|"], horizontal=True)

    dd = scen_long[(scen_long["year"] == year) & (scen_long["route"].isin(TREATMENT_ROUTES))].copy()
    dd_em = scen_em[(scen_em["year"] == year) & (scen_em["route"].isin(RECOVERY_ROUTES))].copy()

    if metric.startswith("Emissions"):
        dd_em["abs_tco2e"] = np.abs(dd_em["kgCO2e"]) / 1000.0
        fig6 = sankey_scenario_fraction_route(
            dd_em, value_col="abs_tco2e", top_links=top_links,
            title="Scenario–fraction–route recovery links sized by absolute emissions (tCO2e)"
        )
    else:
        fig6 = sankey_scenario_fraction_route(
            dd, value_col="tonnes", top_links=top_links,
            title="Scenario–fraction–route links sized by tonnes"
        )

    fig6 = apply_paper_style(fig6)
    st.plotly_chart(fig6, use_container_width=True)
    figure_download_buttons(fig6, "fig_sankey")

# ============================================================
# 5) COUNCIL EXPLORER
# ============================================================
with tabCouncil:
    st.subheader("Council-level results")

    year = st.selectbox("Year", YEARS, index=2, key="cx_year")
    councils = sorted(long[long["year"] == year]["Council Name"].unique().tolist())
    council = st.selectbox("Council", councils, index=0, key="cx_council")
    scenario = st.selectbox(
        "Scenario",
        sorted(scen_long[scen_long["year"] == year]["scenario"].unique().tolist()),
        index=0,
        key="cx_scenario"
    )

    cy = baseCY[(baseCY["year"] == year) & (baseCY["Council Name"] == council)]
    if len(cy) == 0:
        st.info("No totals for this council/year.")
        st.stop()
    cy = cy.iloc[0]

    dd = scen_long[(scen_long["year"] == year) & (scen_long["Council Name"] == council) & (scen_long["scenario"] == scenario)].copy()
    if len(dd) == 0:
        st.info("No scenario data for this council/year.")
        st.stop()

    dd_em = scen_em[(scen_em["year"] == year) & (scen_em["Council Name"] == council) & (scen_em["scenario"] == scenario) & (scen_em["route"].isin(RECOVERY_ROUTES))].copy()
    dd_val = scen_val[(scen_val["year"] == year) & (scen_val["Council Name"] == council) & (scen_val["scenario"] == scenario)].copy()

    total_col = float(cy["TotalCollected"])
    pop = float(cy["population"]) if pd.notna(cy["population"]) else np.nan
    rec_t = float(dd[dd["route"].isin(RECOVERY_ROUTES)]["tonnes"].sum())
    rec_rate = (rec_t / total_col) if total_col > 0 else np.nan
    kg = float(dd_em["kgCO2e"].sum())
    intensity = (kg / rec_t) if rec_t > 0 else np.nan
    value_gbp = float(dd_val[dd_val["route"].isin(VALUE_ROUTES)]["gbp_value"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total collected (t)", f"{total_col:,.0f}")
    c2.metric("Recovery rate", f"{rec_rate*100:.1f}%")
    c3.metric("Intensity (kgCO2e/recovered t)", f"{intensity:.1f}")
    c4.metric("CO2e per cap", f"{(kg/pop):.1f}" if (pop and pop > 0) else "NA")
    c5.metric("Recovered value (GBP)", f"{value_gbp:,.0f}")

    mix = dd_em.groupby("route", as_index=False).agg(tonnes=("tonnes","sum"), kgCO2e=("kgCO2e","sum"))
    mix["CO2e_u"] = to_unit(mix["kgCO2e"])

    mix_v = dd_val.groupby(["fraction","route"], as_index=False).agg(gbp_value=("gbp_value","sum"))
    mix_v = mix_v.sort_values("gbp_value", ascending=False)

    left, right = st.columns(2)
    with left:
        fig7 = px.bar(mix.sort_values("tonnes", ascending=False), x="route", y="tonnes")
        fig7 = apply_paper_style(
            fig7,
            title="Route mix at council level",
            x_title="Route",
            y_title="Tonnes"
        )
        st.plotly_chart(fig7, use_container_width=True)
        figure_download_buttons(fig7, "fig_council_route_mix")

    with right:
        fig8 = px.bar(mix_v.head(15), x="fraction", y="gbp_value", color="route")
        fig8 = apply_paper_style(
            fig8,
            title="Highest recovered economic values by fraction and route",
            x_title="Fraction",
            y_title="Recovered value (GBP)"
        )
        st.plotly_chart(fig8, use_container_width=True)
        figure_download_buttons(fig8, "fig_council_value_by_fraction_route")

# ============================================================
# 6) EXPORTS
# ============================================================
with tabExports:
    st.subheader("CSV exports for the paper")

    year = st.selectbox("Year", YEARS, index=2, key="exp_year")
    d = scen_long[scen_long["year"] == year].copy()
    d_em = scen_em[(scen_em["year"] == year) & (scen_em["route"].isin(RECOVERY_ROUTES))].copy()
    d_val = scen_val[scen_val["year"] == year].copy()

    totals_y = totals[totals["year"] == year].copy()
    totalY = float(totals_y["TotalCollected"].sum())

    rec = d[d["route"].isin(RECOVERY_ROUTES)].groupby("scenario", as_index=False).agg(Recovery_t=("tonnes","sum"))
    rec["RecoveryRate"] = np.where(totalY > 0, rec["Recovery_t"] / totalY, np.nan)

    ems = d_em.groupby("scenario", as_index=False).agg(kgCO2e=("kgCO2e","sum"))
    ems["tCO2e"] = ems["kgCO2e"] / 1000.0

    val = d_val[d_val["route"].isin(VALUE_ROUTES)].groupby("scenario", as_index=False).agg(RecoveredValue_GBP=("gbp_value","sum"))
    scen_summary = rec.merge(ems[["scenario","kgCO2e","tCO2e"]], on="scenario", how="left").merge(val, on="scenario", how="left")
    scen_summary["RecoveredValue_GBP"] = scen_summary["RecoveredValue_GBP"].fillna(0.0)

    st.dataframe(scen_summary, use_container_width=True, height=260)
    st.download_button(
        "Download scenario_summary.csv",
        data=scen_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"scenario_summary_{year}.csv",
        mime="text/csv"
    )

    council_tbl = (d_em.groupby(["scenario","Council Name"], as_index=False)
                   .agg(kgCO2e=("kgCO2e","sum"), tonnes=("tonnes","sum")))
    council_val = (d_val.groupby(["scenario","Council Name"], as_index=False)
                   .agg(RecoveredValue_GBP=("gbp_value","sum")))
    council_tbl = council_tbl.merge(council_val, on=["scenario","Council Name"], how="left")
    council_tbl["RecoveredValue_GBP"] = council_tbl["RecoveredValue_GBP"].fillna(0.0)
    council_tbl["tCO2e"] = council_tbl["kgCO2e"] / 1000.0

    pop = long[long["year"] == year].drop_duplicates(["Council Name"])[["Council Name","population"]]
    council_tbl = council_tbl.merge(pop, on="Council Name", how="left")
    council_tbl["kgCO2e_per_cap"] = np.where(council_tbl["population"] > 0, council_tbl["kgCO2e"] / council_tbl["population"], np.nan)
    council_tbl["GBP_per_cap"] = np.where(council_tbl["population"] > 0, council_tbl["RecoveredValue_GBP"] / council_tbl["population"], np.nan)

    st.dataframe(council_tbl, use_container_width=True, height=380)
    st.download_button(
        "Download council_results.csv",
        data=council_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"council_results_{year}.csv",
        mime="text/csv"
    )

    fr = (d_em.groupby(["scenario","fraction","route"], as_index=False)
          .agg(tonnes=("tonnes","sum"), kgCO2e=("kgCO2e","sum")))
    fr = fr.merge(
        d_val.groupby(["scenario","fraction","route"], as_index=False).agg(RecoveredValue_GBP=("gbp_value","sum")),
        on=["scenario","fraction","route"],
        how="left"
    )
    fr["RecoveredValue_GBP"] = fr["RecoveredValue_GBP"].fillna(0.0)
    fr["kgCO2e_per_t"] = np.where(fr["tonnes"] > 0, fr["kgCO2e"] / fr["tonnes"], np.nan)

    st.dataframe(fr.sort_values(["scenario","RecoveredValue_GBP"], ascending=[True, False]), use_container_width=True, height=380)
    st.download_button(
        "Download fraction_route_results.csv",
        data=fr.to_csv(index=False).encode("utf-8"),
        file_name=f"fraction_route_results_{year}.csv",
        mime="text/csv"
    )

# ============================================================
# 7) DIAGNOSTICS
# ============================================================
with tabDiag:
    st.subheader("Diagnostics")

    miss = em_obs[em_obs["factor_missing"] == 1].groupby(["fraction","route"], as_index=False).agg(tonnes=("tonnes","sum"))
    if len(miss) == 0:
        st.success("No missing factors for fraction × route pairs with non-zero observed tonnes.")
    else:
        st.warning("Some fraction × route pairs have no factor and are treated as 0 in emissions.")
        st.dataframe(miss.sort_values("tonnes", ascending=False), use_container_width=True, height=320)

    st.markdown("### Carbon factor table in use")
    st.dataframe(FACTORS, use_container_width=True, height=320)

    st.markdown("### Economic value table in use (check WEEE = 0 here)")
    st.dataframe(VALUE_FACTORS, use_container_width=True, height=320)

    st.markdown("### Quick national checks")
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