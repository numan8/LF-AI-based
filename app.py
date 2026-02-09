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
.block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1200px; }
div[data-testid="stVerticalBlock"] > div { gap: 0.55rem; }
h1, h2, h3 { margin: 0.2rem 0 0.4rem 0; letter-spacing: -0.02em; }

/* Cards */
.card {
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 10px 22px rgba(2,6,23,0.06);
  border-radius: 18px;
  padding: 14px 14px;
  backdrop-filter: blur(6px);
}
.card-title {
  font-size: 0.9rem;
  color: rgba(15,23,42,0.70);
  margin-bottom: 0.2rem;
}
.big-number {
  font-size: 2.0rem;
  font-weight: 800;
  margin: 0.1rem 0 0.4rem 0;
}
.badge {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  border: 1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.70);
}
.mini { color: rgba(15,23,42,0.70); font-size: 0.9rem; }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
  border-right: 1px solid rgba(15,23,42,0.08);
}

/* Make metrics tighter */
div[data-testid="stMetric"] > div { padding: 0.2rem 0; }

/* Hide "hamburger" extra spacing sometimes */
header { height: 0px !important; }

/* Reduce expander padding if any */
div[data-testid="stExpander"] details { padding: 0.2rem 0.2rem; }

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

    # keep only necessary columns
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

    meta = {
        "states": states,
        "state_to_counties": state_to_counties,
        "sc_to_cities": sc_to_cities,
    }
    return pipe30, pipe60, meta


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def decision_label(p30: float, p60_total: float) -> str:
    if p30 >= 0.60:
        return "Strong Buy"
    if p60_total >= 0.60:
        return "Buy"
    if p30 >= 0.45:
        return "Maybe"
    return "Pass"


# =========================
# Load models
# =========================
st.markdown("## Cash Sales Velocity Predictor")
st.markdown('<span class="badge">Prompt-free • Client inputs → prediction</span>', unsafe_allow_html=True)

with st.spinner("Loading & training models..."):
    pipe30, pipe60, meta = train_models_and_meta()


# =========================
# Sidebar inputs
# =========================
st.sidebar.markdown("## Inputs")

total_purchase_price = st.sidebar.number_input("Total Purchase Price", min_value=0.0, value=15000.0, step=500.0)
acres = st.sidebar.number_input("Acres", min_value=0.0, value=1.0, step=0.01)

purchase_month = st.sidebar.selectbox("Purchase Month", list(range(1, 13)), index=0)
sale_month = st.sidebar.selectbox("Expected Sale Month", list(range(1, 13)), index=0)

st.sidebar.markdown("### Marketing")
promo_confirmed = st.sidebar.checkbox("Promo Confirmed", value=False)
listed_yes = st.sidebar.checkbox("Listed", value=False)
has_photos = st.sidebar.checkbox("Photos", value=False)
has_drone = st.sidebar.checkbox("Drone", value=False)
marketing_score = int(promo_confirmed) + int(listed_yes) + int(has_photos) + int(has_drone)

st.sidebar.markdown("### Location")
states = meta["states"] if meta["states"] else ["Unknown"]
state = st.sidebar.selectbox("State", states, index=0)

counties = meta["state_to_counties"].get(state, ["Unknown"])
county = st.sidebar.selectbox("County", counties, index=0)

cities = meta["sc_to_cities"].get((state, county), ["Unknown"])
city = st.sidebar.selectbox("City", cities, index=0)

predict_btn = st.sidebar.button("Predict", type="primary")


# =========================
# Main output (one frame)
# =========================
# Calculate immediately after click
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

    tag = decision_label(p30, p_le_60)

    st.markdown("### Prediction")
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='card-title'>0–30 days</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{p30*100:.1f}%</div>", unsafe_allow_html=True)
        st.progress(p30)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='card-title'>31–60 days (additional)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{p31_60*100:.1f}%</div>", unsafe_allow_html=True)
        st.progress(p31_60)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='card-title'>≤ 60 days (total)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{p_le_60*100:.1f}%</div>", unsafe_allow_html=True)
        st.progress(p_le_60)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Decision")
    if tag == "Strong Buy":
        st.success("Strong Buy — Fast flip likely (0–30 days).")
    elif tag == "Buy":
        st.success("Buy — Reasonable chance to sell within 60 days.")
    elif tag == "Maybe":
        st.warning("Maybe — Improve price/marketing or location.")
    else:
        st.error("Pass — Low probability of fast sale. Re-check pricing/strategy.")

else:
    st.info("Set inputs in the sidebar and click **Predict**.")
