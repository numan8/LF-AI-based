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
# PREMIUM UI CSS
# =========================
st.markdown("""
<style>
/* App background */
.stApp{
  background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.14), transparent 60%),
              radial-gradient(1200px 600px at 90% 0%, rgba(14,165,233,0.12), transparent 55%),
              linear-gradient(180deg, #ffffff 0%, #f7f8ff 55%, #ffffff 100%);
}

/* Reduce top padding */
.block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }

/* Titles */
h1, h2, h3 { letter-spacing: -0.02em; }

/* Cards */
.card {
  background: rgba(255,255,255,0.80);
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 10px 25px rgba(2,6,23,0.06);
  border-radius: 18px;
  padding: 18px 18px;
  backdrop-filter: blur(6px);
}

.card-title {
  font-size: 0.95rem;
  color: rgba(15,23,42,0.75);
  margin-bottom: 6px;
}

.big-number {
  font-size: 2.1rem;
  font-weight: 800;
  margin-top: -6px;
}

.badge {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  border: 1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.65);
}

.mini {
  color: rgba(15,23,42,0.68);
  font-size: 0.9rem;
}

.hr {
  height: 1px;
  background: rgba(15,23,42,0.10);
  margin: 12px 0;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
  border-right: 1px solid rgba(15,23,42,0.08);
}
</style>
""", unsafe_allow_html=True)


# =========================
# Local Excel path (file in same repo)
# =========================
DATA_PATH = "Cash Sales - AI Stats.xlsx"

num_cols = ["Total Purchase Price", "Acres", "marketing_score", "purchase_month", "sale_month"]
cat_cols = ["state", "county", "city"]


# =========================
# Data + Model Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_excel_local(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(
            f"❌ Excel file not found: `{path}`\n\n"
            f"Upload it in the same GitHub folder as app.py (repo root) and match the filename exactly."
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

    keep_cols = list(set(num_cols + cat_cols + ["sell_30d", "sell_60d_excl"]))
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

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


def decision_label(p30: float, p60_total: float) -> tuple[str, str]:
    # Returns (badge, message)
    if p30 >= 0.60:
        return ("Strong Buy", "Fast flip likely (0–30 days).")
    if p60_total >= 0.60:
        return ("Buy", "Reasonable chance to sell within 60 days.")
    if p30 >= 0.45:
        return ("Maybe", "Improve price/marketing or location to increase speed.")
    return ("Pass", "Low probability of fast sale. Re-check pricing and strategy.")


def improvement_tips(marketing_score: int, purchase_month: int, sale_month: int) -> list[str]:
    tips = []
    if marketing_score <= 1:
        tips.append("Increase marketing: confirm promo + ensure listing + add photos/drone.")
    if purchase_month != sale_month:
        tips.append("Timing mismatch: expected sale month differs from purchase month (seasonality may matter).")
    if marketing_score <= 2:
        tips.append("Try adding at least 3/4 marketing signals for better conversion.")
    return tips


# =========================
# APP HEADER
# =========================
st.markdown("## Cash Sales Velocity Predictor")
st.markdown('<span class="badge">Client inputs → instant probability + decision</span>', unsafe_allow_html=True)
st.write("")


# =========================
# Load models
# =========================
with st.spinner("Loading data & training models..."):
    pipe30, pipe60, meta = train_models_and_meta()


# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.markdown("## Inputs")
st.sidebar.caption("Fill these fields to get prediction & decision.")

total_purchase_price = st.sidebar.number_input("Total Purchase Price", min_value=0.0, value=15000.0, step=500.0)
acres = st.sidebar.number_input("Acres", min_value=0.0, value=1.0, step=0.01)

st.sidebar.markdown("### Timing")
purchase_month = st.sidebar.selectbox("Purchase Month", list(range(1, 13)), index=0)
sale_month = st.sidebar.selectbox("Expected Sale Month", list(range(1, 13)), index=0)

st.sidebar.markdown("### Marketing Signals")
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
# MAIN AREA
# =========================
left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Summary")
    st.markdown(
        f"<div class='mini'>"
        f"<b>Price:</b> {total_purchase_price:,.0f}  |  "
        f"<b>Acres:</b> {acres:.2f}  |  "
        f"<b>Marketing score:</b> {marketing_score}/4"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='mini'><b>Location:</b> {city}, {county}, {state}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='mini'><b>Timing:</b> purchase month {purchase_month} → expected sale month {sale_month}</div>",
        unsafe_allow_html=True
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.caption("Note: Total ≤60 = (0–30) + (31–60). The 31–60 estimate is approximate (conditional model).")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Quick Help")
    tips = improvement_tips(marketing_score, int(purchase_month), int(sale_month))
    if tips:
        for t in tips:
            st.markdown(f"- {t}")
    else:
        st.markdown("- Inputs look strong. Run prediction.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# PREDICTION
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

    badge, msg = decision_label(p30, p_le_60)

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
    st.markdown(f"<span class='badge'><b>{badge}</b></span>  <span class='mini'>{msg}</span>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Recommended next actions")
    recs = []
    if marketing_score < 3:
        recs.append("Add at least 3 marketing signals (listing + photos + promo/drone) to improve sale speed.")
    if p30 < 0.60 and p_le_60 < 0.60:
        recs.append("Re-check pricing / markup strategy for this location.")
    if not recs:
        recs.append("Proceed with this deal strategy and monitor weekly.")
    for r in recs:
        st.markdown(f"- {r}")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Use the sidebar to set inputs, then click **Predict**.")


# =========================
# Footer
# =========================
with st.expander("Data source", expanded=False):
    st.write("Excel file loaded locally from repository:")
    st.code(DATA_PATH, language="text")
