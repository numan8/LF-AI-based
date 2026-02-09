import os
import io
import requests
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
# Data URL (GitHub)
# =========================
# Put your file raw/blob link here OR set env var DATA_URL on Streamlit Cloud
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/<USER>/<REPO>/main/Cash%20Sales%20-%20AI%20Stats.xlsx"
DATA_URL = os.getenv("DATA_URL", DEFAULT_DATA_URL)


# =========================
# Feature columns (must match training)
# =========================
num_cols = ["Total Purchase Price", "Acres", "marketing_score", "purchase_month", "sale_month"]
cat_cols = ["state", "county", "city"]


# =========================
# Helpers
# =========================
def to_github_raw(url: str) -> str:
    # Convert GitHub blob link -> raw link
    # https://github.com/user/repo/blob/main/file.xlsx
    # -> https://raw.githubusercontent.com/user/repo/main/file.xlsx
    if "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        url = url.replace("/blob/", "/")
    return url


@st.cache_data(show_spinner=False)
def load_excel_from_url(url: str) -> pd.DataFrame:
    url = to_github_raw(url)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/octet-stream",
    }

    r = requests.get(url, headers=headers, timeout=45)
    if r.status_code != 200:
        st.error(
            f"❌ Could not download Excel file.\n\n"
            f"HTTP {r.status_code}\n\n"
            f"URL used:\n{url}\n\n"
            f"Fix:\n"
            f"1) Ensure file is public\n"
            f"2) Ensure URL points to the file (raw)\n"
            f"3) Ensure correct branch/path\n"
        )
        st.stop()

    data = io.BytesIO(r.content)
    df = pd.read_excel(data, engine="openpyxl")
    return df


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


def prepare_training_frame(df: pd.DataFrame):
    df = df.copy()

    # ---------- Targets ----------
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

    pis = df["Photographer/Inspector Status"].astype(str).str.lower() if "Photographer/Inspector Status" in df.columns else ""
    df["has_photos"] = pis.str.contains("photo").astype(int) if "Photographer/Inspector Status" in df.columns else 0
    df["has_drone"]  = pis.str.contains("drone").astype(int) if "Photographer/Inspector Status" in df.columns else 0

    df["marketing_score"] = df[["promo_confirmed", "listed_yes", "has_photos", "has_drone"]].sum(axis=1)

    # ---------- Location ----------
    if "County, State" in df.columns:
        df["state"] = df["County, State"].astype(str).str.split(",").str[-1].str.strip()
        df["county"] = df["County, State"].astype(str).str.split(",").str[0].str.strip()
    else:
        df["state"] = "Unknown"
        df["county"] = "Unknown"

    df["city"] = df["Property Location or City"].astype(str).str.strip() if "Property Location or City" in df.columns else "Unknown"

    # ---------- Train frames ----------
    X = df[num_cols + cat_cols].copy()
    y30 = df["sell_30d"].astype(int)

    df60 = df[df["sell_30d"] == 0].copy()
    X60 = df60[num_cols + cat_cols].copy()
    y60 = df60["sell_60d_excl"].astype(int)

    meta = {
        "listing_col_used": listing_col,
        "states": sorted(df["state"].dropna().unique().tolist()),
        "counties": sorted(df["county"].dropna().unique().tolist()),
        "cities": sorted(df["city"].dropna().unique().tolist()),
    }

    return X, y30, X60, y60, meta


@st.cache_resource(show_spinner=False)
def train_models(data_url: str):
    df = load_excel_from_url(data_url)
    X, y30, X60, y60, meta = prepare_training_frame(df)

    pipe30 = Pipeline(steps=[
        ("preprocess", build_preprocess()),
        ("model", LogisticRegression(max_iter=20000, class_weight="balanced", solver="saga")),
    ])
    pipe30.fit(X, y30)

    pipe60 = Pipeline(steps=[
        ("preprocess", build_preprocess()),
        ("model", LogisticRegression(max_iter=20000, class_weight="balanced", solver="saga")),
    ])
    pipe60.fit(X60, y60)

    return pipe30, pipe60, meta


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# =========================
# UI
# =========================
st.title("Cash Sales Velocity Predictor")
st.caption("Client inputs → instant probability + decision (no model metrics shown).")

with st.expander("Data source (GitHub)", expanded=False):
    st.write("Set an env var `DATA_URL` in Streamlit Cloud OR edit DEFAULT_DATA_URL in app.py.")
    st.code(DATA_URL, language="text")

with st.spinner("Loading & training model..."):
    pipe30, pipe60, meta = train_models(DATA_URL)

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
counties = meta["counties"] if meta["counties"] else ["Unknown"]
cities = meta["cities"] if meta["cities"] else ["Unknown"]

state = st.selectbox("State", states, index=0)
county = st.selectbox("County", counties, index=0)
city = st.selectbox("City", cities, index=0)

st.divider()

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

    p30 = clamp01(float(pipe30.predict_proba(X_in)[0, 1]))
    p60_cond = clamp01(float(pipe60.predict_proba(X_in)[0, 1]))

    # Unconditional approximation for 31-60
    p60 = clamp01((1.0 - p30) * p60_cond)
    p_le_60 = clamp01(p30 + p60)

    # Decision logic (edit thresholds as you like)
    decision = "Pass / Re-check pricing"
    if p30 >= 0.60:
        decision = "Strong Buy (fast flip likely ≤30 days)"
    elif p_le_60 >= 0.60:
        decision = "Buy (reasonable chance ≤60 days)"
    elif p30 >= 0.45:
        decision = "Maybe (improve price/marketing or location)"

    st.subheader("Prediction")
    st.metric("Probability: Sell ≤ 30 Days", f"{p30*100:.1f}%")
    st.metric("Probability: Sell in 31–60 Days", f"{p60*100:.1f}%")
    st.metric("Probability: Sell ≤ 60 Days (approx.)", f"{p_le_60*100:.1f}%")

    st.subheader("Decision")
    st.success(decision)

    st.caption(
        "31–60 day probability is estimated from a conditional model trained only on deals that did NOT sell within 30 days."
    )
