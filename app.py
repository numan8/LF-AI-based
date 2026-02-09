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
st.set_page_config(page_title="Cash Sales Velocity Predictor", layout="wide")


# =========================
# Premium + Compact CSS
# =========================
st.markdown("""
<style>
/* Background */
.stApp{
  background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.14), transparent 60%),
              radial-gradient(1200px 600px at 90% 0%, rgba(14,165,233,0.12), transparent 55%),
              linear-gradient(180deg, #ffffff 0%, #f7f8ff 55%, #ffffff 100%);
}

/* Compact spacing */
.block-container { padding-top: 0.8rem; padding-bottom: 1.0rem; max-width: 1300px; }
div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }
h1, h2, h3 { margin: 0.15rem 0 0.35rem 0; letter-spacing: -0.02em; }

/* Cards */
.card {
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 12px 26px rgba(2,6,23,0.06);
  border-radius: 18px;
  padding: 14px 14px;
  backdrop-filter: blur(7px);
}
.card-title {
  font-size: 0.9rem;
  color: rgba(15,23,42,0.70);
  margin-bottom: 0.2rem;
}
.big-number {
  font-size: 2.05rem;
  font-weight: 850;
  margin: 0.05rem 0 0.45rem 0;
}
.badge {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  border: 1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.72);
}
.subtle {
  color: rgba(15,23,42,0.65);
  font-size: 0.9rem;
}

/* Top input panel */
.top-panel {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 10px 22px rgba(2,6,23,0.05);
  border-radius: 18px;
  padding: 14px 14px;
  backdrop-filter: blur(7px);
}

/* Make primary button more prominent */
div.stButton > button[kind="primary"]{
  width: 100%;
  border-radius: 14px;
  padding: 0.7rem 1rem;
  font-weight: 700;
  font-size: 1.05rem;
}

/* Reduce extra padding around widgets */
label { font-size: 0.92rem !important; }

/* Hide top header whitespace */
header { height: 0px !important; }
</style>
""", unsafe_allow_html=True)


# =========================
# Local Excel path
# =========================
DATA_PATH = "Cash Sales - AI Stats.xlsx"

num_cols = ["Total Purchase Price", "Acres", "marketing_score", "purchase_month", "sale_month"]
cat_cols = ["state", "county", "city"]


# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_excel_local(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"❌ Excel file not found: `{path}`. Upload it in repo root (same folder as app.py).")
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
    df = df.copy()

    df["PURCHASE DATE"] = pd.to_datetime(df.get("PURCHASE DATE"), errors="coerce")
    df["SALE DATE - start"] = pd.to_datetime(df.get("SALE DATE - start"), errors="coerce")
    df["Days_to_sell_cash"] = (df["SALE DATE - start"] - df["PURCHASE DATE"]).dt.days

    df = df[df["Days_to_sell_cash"].notna()].copy()
    df = df[df["Days_to_sell_cash"] >= 0].copy()

    df["sell_30d"] = (df["Days_to_sell_cash"] <= 30).astype(int)
    df["sell_60d_excl"] = ((df["Days_to_sell_cash"] > 30) & (df["Days_to_sell_cash"] <= 60)).astype(int)

    df["purchase_month"] = df["PURCHASE DATE"].dt.month
    df["sale_month"] = df["SALE DATE - start"].dt.month

    df["Total Purchase Price"] = pd.to_numeric(df.get("Total Purchase Price"), errors="coerce")
    df["Acres"] = pd.to_numeric(df.get("Acres"), errors="coerce")

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

    keep = list(set(num_cols + cat_cols + ["sell_30d", "sell_60d_excl"]))
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    for c in ["state", "county", "city"]:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()

    return df


def make_location_maps(df_feat: pd.DataFrame):
    df_loc = df_feat[["state", "county", "city"]].copy()
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

    X = df_feat[num_cols + cat_cols].copy()
    y30 = df_feat["sell_30d"].astype(int)

    pipe30 = Pipeline(steps=[
        ("preprocess", build_preprocess()),
        ("model", LogisticRegression(max_iter=20000, class_weight="balanced", solver="saga")),
    ])
    pipe30.fit(X, y30)

    df60 = df_feat[df_feat["sell_30d"] == 0].copy()
    X60 = df60[num_cols + cat_cols].copy()
    y60 = df60["sell_60d_excl"].astype(int)

    pipe60 = Pipeline(steps=[
        ("preprocess", build_preprocess()),
        ("model", LogisticRegression(max_iter=20000, class_weight="balanced", solver="saga")),
    ])
    pipe60.fit(X60, y60)

    states, state_to_counties, sc_to_cities = make_location_maps(df_feat)
    meta = {"states": states, "state_to_counties": state_to_counties, "sc_to_cities": sc_to_cities}
    return pipe30, pipe60, meta


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def decision_text(p30: float, p60_total: float):
    if p30 >= 0.60:
        return ("Strong Buy", "Fast flip likely (0–30 days).", "success")
    if p60_total >= 0.60:
        return ("Buy", "Reasonable chance within 60 days.", "success")
    if p30 >= 0.45:
        return ("Maybe", "Improve price/marketing or location.", "warning")
    return ("Pass", "Low probability of fast sale. Re-check pricing/strategy.", "error")


# =========================
# Header
# =========================
st.markdown("## Cash Sales Velocity Predictor")
st.markdown('<span class="badge">Inputs on top • One-click prediction • Clean output</span>', unsafe_allow_html=True)

with st.spinner("Loading & training models..."):
    pipe30, pipe60, meta = train_models_and_meta()


# =========================
# TOP INPUT BAR (instead of left sidebar)
# =========================
st.markdown('<div class="top-panel">', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns([1.2, 0.9, 0.9, 1.2, 1.2, 1.1], gap="medium")

with c1:
    total_purchase_price = st.number_input("Total Purchase Price", min_value=0.0, value=15000.0, step=500.0)

with c2:
    acres = st.number_input("Acres", min_value=0.0, value=1.0, step=0.01)

with c3:
    purchase_month = st.selectbox("Purchase Month", list(range(1, 13)), index=0)

with c4:
    sale_month = st.selectbox("Expected Sale Month", list(range(1, 13)), index=0)

# Location dependent dropdowns
states = meta["states"] if meta["states"] else ["Unknown"]
with c5:
    state = st.selectbox("State", states, index=0)
counties = meta["state_to_counties"].get(state, ["Unknown"])
with c6:
    county = st.selectbox("County", counties, index=0)

# City row + marketing + predict
c7, c8, c9, c10, c11, c12 = st.columns([1.4, 0.9, 0.9, 0.9, 0.9, 1.1], gap="medium")

cities = meta["sc_to_cities"].get((state, county), ["Unknown"])
with c7:
    city = st.selectbox("City", cities, index=0)

with c8:
    promo_confirmed = st.checkbox("Promo", value=False)
with c9:
    listed_yes = st.checkbox("Listed", value=False)
with c10:
    has_photos = st.checkbox("Photos", value=False)
with c11:
    has_drone = st.checkbox("Drone", value=False)

marketing_score = int(promo_confirmed) + int(listed_yes) + int(has_photos) + int(has_drone)

with c12:
    predict_btn = st.button("🚀 Predict Now", type="primary")

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# OUTPUT AREA (under button)
# =========================
if predict_btn:
    X_in = pd.DataFrame([{
        "Total Purchase Price": float(total_purchase_price),
        "Acres": float(acres),
        "marketing_score": int(marketing_score),
        "purchase_month": int(purchase_month),
        "sale_month": int(sale_month),
        "state": str(state),
        "county": str(county),
        "city": str(city),
    }])

    p30 = clamp01(float(pipe30.predict_proba(X_in)[0, 1]))
    p60_cond = clamp01(float(pipe60.predict_proba(X_in)[0, 1]))
    p31_60 = clamp01((1.0 - p30) * p60_cond)
    p_le_60 = clamp01(p30 + p31_60)

    tag, msg, level = decision_text(p30, p_le_60)

    st.write("")
    st.markdown("### Prediction")

    a, b, c = st.columns(3, gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Chance to sell in 0–30 days</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{p30*100:.1f}%</div>", unsafe_allow_html=True)
        st.progress(p30)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Additional chance in 31–60 days</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{p31_60*100:.1f}%</div>", unsafe_allow_html=True)
        st.progress(p31_60)
        st.markdown("</div>", unsafe_allow_html=True)

    with c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Total chance within 60 days (approx.)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{p_le_60*100:.1f}%</div>", unsafe_allow_html=True)
        st.progress(p_le_60)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### Decision")
    if level == "success":
        st.success(f"{tag} — {msg}")
    elif level == "warning":
        st.warning(f"{tag} — {msg}")
    else:
        st.error(f"{tag} — {msg}")

    st.caption("Note: ≤60 = (0–30) + (31–60). 31–60 is estimated from a conditional model (approx.).")

else:
    st.info("Fill inputs above and click **🚀 Predict Now**.")
