import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="Cash Sales Velocity Predictor", layout="centered")


# =========================
# Local Excel path (file in same repo)
# =========================
DATA_PATH = "Cash Sales - AI Stats.xlsx"

# Features used by model (must match training)
num_cols = ["Total Purchase Price", "Acres", "marketing_score", "purchase_month", "sale_month"]
cat_cols = ["state", "county", "city"]


# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_excel_local(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(
            f"❌ Excel file not found: `{path}`\n\n"
            f"Fix: Upload it to the GitHub repo root (same folder as app.py) and ensure the filename matches exactly."
        )
        st.stop()
    return pd.read_excel(path, engine="openpyxl")


def build_preprocess():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features and targets; returns cleaned DF used for training + location dropdowns."""
    df = df.copy()

    # ---------- Dates / targets ----------
    df["PURCHASE DATE"] = pd.to_datetime(df.get("PURCHASE DATE"), errors="coerce")
    df["SALE DATE - start"] = pd.to_datetime(df.get("SALE DATE - start"), errors="coerce")
    df["Days_to_sell_cash"] = (df["SALE DATE - start"] - df["PURCHASE DATE"]).dt.days

    df = df[df["Days_to_sell_cash"].notna()].copy()
    df = df[df["Days_to_sell_cash"] >= 0].copy()

    df["sell_30d"] = (df["Days_to_sell_cash"] <= 30).astype(int)
    df["sell_60d_excl"] = ((df["Days_to_sell_cash"] > 30) & (df["Days_to_sell_cash"] <= 60)).astype(int)

    # ---------- Months ----------
    df["purchase_month"] = df["PURCHASE DATE"].dt.month
    df["sale_month"] = df["SALE DATE - start"].dt.month

    # ---------- Numerics ----------
    df["Total Purchase Price"] = pd.to_numeric(df.get("Total Purchase Price"), errors="coerce")
    df["Acres"] = pd.to_numeric(df.get("Acres"), errors="coerce")

    # ---------- Marketing Score ----------
    listing_col = None
    for c in df.columns:
        if str(c).strip().lower() in ["listing", "land id link listing", "link listing"]:
            listing_col = c
            break

    df["promo_confirmed"] = (
        df["Promo Price Status"].astype(str).str.lower().str.contains("confirmed").astype(int)
        if "Promo Price Status" in df.columns else 0
    )
    df["listed_yes"] = (
        df[listing_col].astype(str).str.lower().str.contains("yes").astype(int)
        if listing_col else 0
    )

    if "Photographer/Inspector Status" in df.columns:
        pis = df["Photographer/Inspector Status"].astype(str).str.lower()
        df["has_photos"] = pis.str.contains("photo").astype(int)
        df["has_drone"] = pis.str.contains("drone").astype(int)
    else:
        df["has_photos"] = 0
        df["has_drone"] = 0

    df["marketing_score"] = df[["promo_confirmed", "listed_yes", "has_photos", "has_drone"]].sum(axis=1)

    # ---------- Location ----------
    if "County, State" in df.columns:
        df["state"] = df["County, State"].astype(str).str.split(",").str[-1].str.strip()
        df["county"] = df["County, State"].astype(str).str.split(",").str[0].str.strip()
    else:
        df["state"] = "Unknown"
        df["county"] = "Unknown"

    df["city"] = (
        df["Property Location or City"].astype(str).str.strip()
        if "Property Location or City" in df.columns else "Unknown"
    )

    # Keep only what's needed (avoid surprises)
    keep_cols = list(set(num_cols + cat_cols + ["sell_30d", "sell_60d_excl"]))
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Fill missing location as Unknown to keep dropdown consistent
    for c in ["state", "county", "city"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str).str.strip()

    return df


def make_location_maps(df_feat: pd.DataFrame):
    """Build State -> Counties and (State, County) -> Cities mappings."""
    # ensure strings
    df_loc = df_feat[["state", "county", "city"]].copy()
    df_loc["state"] = df_loc["state"].astype(str)
    df_loc["county"] = df_loc["county"].astype(str)
    df_loc["city"] = df_loc["city"].astype(str)

    state_to_counties = (
        df_loc.groupby("state")["county"]
        .apply(lambda s: sorted(set([x for x in s if x and x != "nan"])))
        .to_dict()
    )

    sc_to_cities = (
        df_loc.groupby(["state", "county"])["city"]
        .apply(lambda s: sorted(set([x for x in s if x and x != "nan"])))
        .to_dict()
    )

    states = sorted(state_to_counties.keys())
    return states, state_to_counties, sc_to_cities


@st.cache_resource(show_spinner=False)
def train_models_and_meta():
    raw = load_excel_local(DATA_PATH)
    df_feat = enrich_features(raw)

    # Model A: sell_30d
    X = df_feat[num_cols + cat_cols].copy()
    y30 = df_feat["sell_30d"].astype(int)

    pipe30 = Pipeline(steps=[
        ("preprocess", build_preprocess()),
        ("model", LogisticRegression(max_iter=20000, class_weight="balanced", solver="saga")),
    ])
    pipe30.fit(X, y30)

    # Model B: sell_60d_excl trained only on NOT sell_30d
    df60 = df_feat[df_feat["sell_30d"] == 0].copy()
    X60 = df60[num_cols + cat_cols].copy()
    y60 = df60["sell_60d_excl"].astype(int)

    pipe60 = Pipeline(steps=[
        ("preprocess", build_preprocess()),
        ("model", LogisticRegression(max_iter=20000, class_weight="balanced", solver="saga")),
    ])
    pipe60.fit(X60, y60)

    # Location dropdown mappings
    states, state_to_counties, sc_to_cities = make_location_maps(df_feat)

    meta = {
        "states": states,
        "state_to_counties": state_to_counties,
        "sc_to_cities": sc_to_cities,
    }

    return pipe30, pipe60, meta


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# =========================
# UI
# =========================
st.title("Cash Sales Velocity Predictor")
st.caption("Client inputs → instant probability + decision (no model metrics shown).")

with st.expander("Data source", expanded=False):
    st.write("This app reads the Excel file directly from the repository:")
    st.code(DATA_PATH, language="text")

with st.spinner("Loading & training model..."):
    pipe30, pipe60, meta = train_models_and_meta()

st.divider()
st.subheader("Deal Inputs")

c1, c2 = st.columns(2)
with c1:
    total_purchase_price = st.number_input("Total Purchase Price", min_value=0.0, value=15000.0, step=500.0)
    acres = st.number_input("Acres", min_value=0.0, value=1.0, step=0.01)

with c2:
    purchase_month = st.selectbox("Purchase Month", list(range(1, 13)), index=0)
    sale_month = st.selectbox("Expected Sale Month", list(range(1, 13)), index=0)

st.markdown("### Marketing Signals")
m1, m2, m3, m4 = st.columns(4)
with m1:
    promo_confirmed = st.checkbox("Promo Confirmed", value=False)
with m2:
    listed_yes = st.checkbox("Listed", value=False)
with m3:
    has_photos = st.checkbox("Photos", value=False)
with m4:
    has_drone = st.checkbox("Drone", value=False)

marketing_score = int(promo_confirmed) + int(listed_yes) + int(has_photos) + int(has_drone)

st.markdown("### Location")

states = meta["states"] if meta["states"] else ["Unknown"]
default_state_idx = 0
state = st.selectbox("State", states, index=default_state_idx)

counties = meta["state_to_counties"].get(state, ["Unknown"])
county = st.selectbox("County", counties, index=0)

cities = meta["sc_to_cities"].get((state, county), ["Unknown"])
city = st.selectbox("City", cities, index=0)

st.divider()

# Optional: simple explanation line (removes confusion)
st.caption("Note: ≤60 days = (0–30 days) + (31–60 days). The 31–60 estimate is approximate (conditional model).")

if st.button("Predict", type="primary"):
    X_in = pd.DataFrame([{
        "Total Purchase Price": total_purchase_price,
        "Acres": acres,
        "marketing_score": marketing_score,
        "purchase_month": int(purchase_month),
        "sale_month": int(sale_month),
        "state": state,
        "county": county,
        "city": city,
    }])

    # P(sell <= 30d)
    p30 = clamp01(float(pipe30.predict_proba(X_in)[0, 1]))

    # Conditional P(31-60 | not <=30)
    p60_cond = clamp01(float(pipe60.predict_proba(X_in)[0, 1]))

    # Convert to unconditional-ish:
    p60 = clamp01((1.0 - p30) * p60_cond)
    p_le_60 = clamp01(p30 + p60)

    # Decision logic (edit thresholds if needed)
    decision = "Pass / Re-check pricing"
    if p30 >= 0.60:
        decision = "Strong Buy (fast flip likely 0–30 days)"
    elif p_le_60 >= 0.60:
        decision = "Buy (reasonable chance within 60 days)"
    elif p30 >= 0.45:
        decision = "Maybe (improve price/marketing or location)"

    st.subheader("Prediction")
    st.metric("Chance to sell in 0–30 days", f"{p30*100:.1f}%")
    st.metric("Additional chance in 31–60 days", f"{p60*100:.1f}%")
    st.metric("Total chance within 60 days (approx.)", f"{p_le_60*100:.1f}%")

    st.subheader("Decision")
    st.success(decision)
