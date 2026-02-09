import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Cash Sales Velocity Predictor", layout="wide")

DATA_PATH = "data/Cash Sales - AI Stats.xlsx"

NUM_COLS = ["Total Purchase Price", "Acres", "marketing_score", "purchase_month", "sale_month"]
CAT_COLS = ["state", "county", "city"]

# -------------------------
# Utilities
# -------------------------
def find_listing_col(df: pd.DataFrame):
    for c in df.columns:
        c_norm = c.strip().lower()
        if c_norm in ["listing", "land id link listing", "link listing"]:
            return c
    return None


def prepare_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Required cols
    required = ["PURCHASE DATE", "SALE DATE - start", "Total Purchase Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sheet: {missing}")

    # Dates + target
    df["PURCHASE DATE"] = pd.to_datetime(df["PURCHASE DATE"], errors="coerce")
    df["SALE DATE - start"] = pd.to_datetime(df["SALE DATE - start"], errors="coerce")
    df["Days_to_sell_cash"] = (df["SALE DATE - start"] - df["PURCHASE DATE"]).dt.days

    df = df[df["Days_to_sell_cash"].notna()].copy()
    df = df[df["Days_to_sell_cash"] >= 0].copy()

    df["sell_30d"] = (df["Days_to_sell_cash"] <= 30).astype(int)
    df["sell_60d_excl"] = ((df["Days_to_sell_cash"] > 30) & (df["Days_to_sell_cash"] <= 60)).astype(int)

    df["purchase_month"] = df["PURCHASE DATE"].dt.month
    df["sale_month"] = df["SALE DATE - start"].dt.month

    # Numeric
    df["Total Purchase Price"] = pd.to_numeric(df["Total Purchase Price"], errors="coerce")
    df["Acres"] = pd.to_numeric(df["Acres"], errors="coerce") if "Acres" in df.columns else np.nan

    # Marketing score
    listing_col = find_listing_col(df)

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

    # Geography
    if "County, State" in df.columns:
        cs = df["County, State"].astype(str)
        df["state"] = cs.str.split(",").str[-1].str.strip().replace({"": "Unknown"})
        df["county"] = cs.str.split(",").str[0].str.strip().replace({"": "Unknown"})
    else:
        df["state"] = "Unknown"
        df["county"] = "Unknown"

    if "Property Location or City" in df.columns:
        df["city"] = df["Property Location or City"].astype(str).str.strip().replace({"": "Unknown"})
    else:
        df["city"] = "Unknown"

    # Ensure cols exist
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "Unknown"

    return df


def build_pipeline():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS)
    ])

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", LogisticRegression(max_iter=20000, class_weight="balanced", solver="saga"))
    ])
    return pipe


@st.cache_data
def load_raw():
    return pd.read_excel(DATA_PATH, engine="openpyxl")


@st.cache_resource
def train_models():
    df_raw = load_raw()
    df = prepare_data(df_raw)

    # Model A
    X = df[NUM_COLS + CAT_COLS]
    y30 = df["sell_30d"].astype(int)

    pipe30 = build_pipeline()
    pipe30.fit(X, y30)

    # Model B: train only on not sell_30d
    df60 = df[df["sell_30d"] == 0].copy()
    X60 = df60[NUM_COLS + CAT_COLS]
    y60 = df60["sell_60d_excl"].astype(int)

    pipe60 = build_pipeline()
    pipe60.fit(X60, y60)

    # defaults for UI
    defaults = {
        "median_cost": float(np.nanmedian(df["Total Purchase Price"])),
        "median_acres": float(np.nanmedian(df["Acres"])) if df["Acres"].notna().any() else 1.0,
        "median_mkt": int(np.nanmedian(df["marketing_score"])) if df["marketing_score"].notna().any() else 0,
        "mode_state": df["state"].mode().iloc[0] if df["state"].notna().any() else "Unknown",
        "mode_county": df["county"].mode().iloc[0] if df["county"].notna().any() else "Unknown",
        "mode_city": df["city"].mode().iloc[0] if df["city"].notna().any() else "Unknown",
    }

    return pipe30, pipe60, defaults


def predict(pipe30, pipe60, row_df: pd.DataFrame):
    p30 = float(pipe30.predict_proba(row_df)[0, 1])
    p60_cond = float(pipe60.predict_proba(row_df)[0, 1])
    p31_60 = (1 - p30) * p60_cond
    p60 = p30 + p31_60
    return p30, p60


# -------------------------
# App UI (Client-facing)
# -------------------------
st.title("Cash Sales Velocity Predictor")
st.caption("Enter a deal scenario → get probability of selling within 30 or 60 days (cash).")

pipe30, pipe60, defaults = train_models()

with st.form("deal_form"):
    st.subheader("Deal inputs")

    c1, c2, c3 = st.columns(3)

    with c1:
        total_cost = st.number_input(
            "Total Purchase Price (All-in Cost)",
            min_value=0.0,
            value=defaults["median_cost"],
            step=1000.0
        )
        acres = st.number_input(
            "Acres",
            min_value=0.0,
            value=defaults["median_acres"],
            step=0.25
        )

    with c2:
        marketing_score = st.slider("Marketing Score (0–4)", 0, 4, defaults["median_mkt"])
        purchase_month = st.slider("Purchase Month", 1, 12, int(pd.Timestamp.today().month))
        sale_month = st.slider("Expected Sale Month", 1, 12, int(pd.Timestamp.today().month))

    with c3:
        state = st.text_input("State", value=str(defaults["mode_state"]))
        county = st.text_input("County", value=str(defaults["mode_county"]))
        city = st.text_input("City", value=str(defaults["mode_city"]))

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "Total Purchase Price": total_cost,
        "Acres": acres,
        "marketing_score": marketing_score,
        "purchase_month": purchase_month,
        "sale_month": sale_month,
        "state": state,
        "county": county,
        "city": city,
    }])

    p30, p60 = predict(pipe30, pipe60, row)

    st.subheader("Prediction")
    k1, k2 = st.columns(2)
    k1.metric("Probability of selling within 30 days", f"{p30:.1%}")
    k2.metric("Probability of selling within 60 days", f"{p60:.1%}")

    st.info("Tip: Lower Total Purchase Price and stronger marketing score typically increases sell-fast probability.")


# Optional advanced panel (hidden by default)
with st.expander("Advanced (optional)"):
    st.write("This app trains 2 logistic models on historical cash sales:")
    st.write("- Model A: sell within 30 days")
    st.write("- Model B: sell within 31–60 days, trained only on deals that did not sell in 30 days")
