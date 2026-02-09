import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# =========================
# Config
# =========================
st.set_page_config(page_title="Cash Sales Sweet Spot (30/60 days)", layout="wide")

NUM_COLS = ["Total Purchase Price", "Acres", "marketing_score", "purchase_month", "sale_month"]
CAT_COLS = ["state", "county", "city"]


# =========================
# Helpers
# =========================
def find_listing_col(df: pd.DataFrame):
    for c in df.columns:
        c_norm = c.strip().lower()
        if c_norm in ["listing", "land id link listing", "link listing"]:
            return c
    return None


def to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def compute_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # --- Dates / Targets ---
    if "PURCHASE DATE" not in df.columns or "SALE DATE - start" not in df.columns:
        raise ValueError("Missing required columns: 'PURCHASE DATE' and/or 'SALE DATE - start'.")

    df["PURCHASE DATE"] = to_dt(df["PURCHASE DATE"])
    df["SALE DATE - start"] = to_dt(df["SALE DATE - start"])
    df["Days_to_sell_cash"] = (df["SALE DATE - start"] - df["PURCHASE DATE"]).dt.days

    df = df[df["Days_to_sell_cash"].notna()].copy()
    df = df[df["Days_to_sell_cash"] >= 0].copy()

    df["sell_30d"] = (df["Days_to_sell_cash"] <= 30).astype(int)
    df["sell_60d_excl"] = ((df["Days_to_sell_cash"] > 30) & (df["Days_to_sell_cash"] <= 60)).astype(int)

    df["purchase_month"] = df["PURCHASE DATE"].dt.month
    df["sale_month"] = df["SALE DATE - start"].dt.month

    # --- Numeric columns ---
    if "Total Purchase Price" not in df.columns:
        raise ValueError("Missing required column: 'Total Purchase Price'.")
    df["Total Purchase Price"] = pd.to_numeric(df["Total Purchase Price"], errors="coerce")

    if "Acres" in df.columns:
        df["Acres"] = pd.to_numeric(df["Acres"], errors="coerce")
    else:
        df["Acres"] = np.nan

    # --- marketing_score ---
    listing_col = find_listing_col(df)

    if "Promo Price Status" in df.columns:
        df["promo_confirmed"] = (
            df["Promo Price Status"].astype(str).str.lower().str.contains("confirmed")
        ).astype(int)
    else:
        df["promo_confirmed"] = 0

    if listing_col:
        df["listed_yes"] = df[listing_col].astype(str).str.lower().str.contains("yes").astype(int)
    else:
        df["listed_yes"] = 0

    if "Photographer/Inspector Status" in df.columns:
        pis = df["Photographer/Inspector Status"].astype(str).str.lower()
        df["has_photos"] = pis.str.contains("photo").astype(int)
        df["has_drone"] = pis.str.contains("drone").astype(int)
    else:
        df["has_photos"] = 0
        df["has_drone"] = 0

    df["marketing_score"] = df[["promo_confirmed", "listed_yes", "has_photos", "has_drone"]].sum(axis=1)

    # --- Geography ---
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

    # Ensure required engineered columns exist
    for c in (NUM_COLS + CAT_COLS):
        if c not in df.columns:
            df[c] = np.nan if c in NUM_COLS else "Unknown"

    return df


def build_preprocess():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )


def train_model(df_feat: pd.DataFrame, target_col: str, conditional_not_30=False, seed=42):
    if conditional_not_30:
        df_train = df_feat[df_feat["sell_30d"] == 0].copy()
    else:
        df_train = df_feat.copy()

    X = df_train[NUM_COLS + CAT_COLS].copy()
    y = df_train[target_col].astype(int)

    # Guard: need at least 2 classes
    if y.nunique() < 2:
        raise ValueError(f"Target '{target_col}' has only one class after filtering; cannot train.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    preprocess = build_preprocess()

    clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=20000, class_weight="balanced", solver="saga")),
        ]
    )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, digits=3, output_dict=False)

    # Feature importance (coefficients)
    ohe = clf.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(CAT_COLS)
    feature_names = np.array(NUM_COLS + list(cat_feature_names))
    coefs = clf.named_steps["model"].coef_.ravel()

    imp = (
        pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)})
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "pipe": clf,
        "auc": auc,
        "cm": cm,
        "report": report,
        "importance": imp,
        "train_rows": len(df_train),
    }


def predict_one(pipe30, pipe60, row_df: pd.DataFrame):
    # P(sell<=30)
    p30 = float(pipe30.predict_proba(row_df)[0, 1])

    # P(31-60 | not<=30) from conditional model
    p60_cond = float(pipe60.predict_proba(row_df)[0, 1])

    # Unconditional probability of selling in 31-60:
    # P(31-60) = P(not<=30) * P(31-60 | not<=30)
    p_31_60 = (1.0 - p30) * p60_cond

    # Unconditional probability of selling within 60:
    # P(<=60) = P(<=30) + P(31-60)
    p60 = p30 + p_31_60

    return p30, p60_cond, p_31_60, p60


# =========================
# UI
# =========================
st.title("Cash Sales Sweet Spot — Logistic Regression (≤30d and 31–60d)")

with st.sidebar:
    st.header("Upload data")
    uploaded = st.file_uploader("Upload AI Stats (XLSX or CSV)", type=["xlsx", "xls", "csv"])
    seed = st.number_input("Random seed", value=42, min_value=0, step=1)
    st.caption("Models: Logistic Regression (balanced classes)")

if not uploaded:
    st.info("Upload your AI Stats XLSX/CSV to train the models.")
    st.stop()

# Load file
try:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# Feature engineering
try:
    df = compute_features(df_raw)
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Data preview")
st.write(f"Rows after cleaning (valid dates, non-negative days_to_sell): **{len(df):,}**")
st.dataframe(df.head(20), use_container_width=True)

# Train
st.subheader("Train models")
colA, colB = st.columns(2)

try:
    with st.spinner("Training Model A (sell_30d)..."):
        res30 = train_model(df, target_col="sell_30d", conditional_not_30=False, seed=int(seed))

    with st.spinner("Training Model B (sell_60d_excl | not sell_30d)..."):
        res60 = train_model(df, target_col="sell_60d_excl", conditional_not_30=True, seed=int(seed))

except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

with colA:
    st.markdown("### Model A: **Sell ≤ 30 days**")
    st.metric("AUC", f"{res30['auc']:.3f}")
    st.write("Confusion Matrix (threshold=0.50):")
    st.write(pd.DataFrame(res30["cm"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
    st.text(res30["report"])
    st.caption(f"Training rows used: {res30['train_rows']:,}")

with colB:
    st.markdown("### Model B: **Sell 31–60 days** (trained only where NOT ≤30)")
    st.metric("AUC", f"{res60['auc']:.3f}")
    st.write("Confusion Matrix (threshold=0.50):")
    st.write(pd.DataFrame(res60["cm"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
    st.text(res60["report"])
    st.caption(f"Training rows used: {res60['train_rows']:,}")

# Feature importance
st.subheader("Top coefficients (feature importance)")
top_n = st.slider("Show top N", 10, 50, 25)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Model A: ≤30 days")
    st.dataframe(res30["importance"].head(top_n), use_container_width=True)
with c2:
    st.markdown("#### Model B: 31–60 (conditional)")
    st.dataframe(res60["importance"].head(top_n), use_container_width=True)

# What-if
st.subheader("What-if predictor (single deal)")

# Defaults from data
default_state = df["state"].mode().iloc[0] if df["state"].notna().any() else "Unknown"
default_county = df["county"].mode().iloc[0] if df["county"].notna().any() else "Unknown"
default_city = df["city"].mode().iloc[0] if df["city"].notna().any() else "Unknown"

w1, w2, w3 = st.columns(3)
with w1:
    total_cost = st.number_input("Total Purchase Price (all-in cost)", value=float(np.nanmedian(df["Total Purchase Price"])), min_value=0.0, step=1000.0)
    acres = st.number_input("Acres", value=float(np.nanmedian(df["Acres"])) if df["Acres"].notna().any() else 1.0, min_value=0.0, step=0.25)
with w2:
    marketing_score = st.slider("Marketing score (0–4)", 0, 4, int(np.nanmedian(df["marketing_score"])) if "marketing_score" in df.columns else 0)
    purchase_month = st.slider("Purchase month", 1, 12, int(pd.Timestamp.today().month))
with w3:
    sale_month = st.slider("Sale month", 1, 12, int(pd.Timestamp.today().month))
    state = st.text_input("State", value=str(default_state))
    county = st.text_input("County", value=str(default_county))
    city = st.text_input("City", value=str(default_city))

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

pipe30 = res30["pipe"]
pipe60 = res60["pipe"]

p30, p60_cond, p31_60, p60 = predict_one(pipe30, pipe60, row)

m1, m2, m3, m4 = st.columns(4)
m1.metric("P(sell ≤ 30d)", f"{p30:.2%}")
m2.metric("P(31–60d | not ≤30)", f"{p60_cond:.2%}")
m3.metric("P(sell in 31–60d)", f"{p31_60:.2%}")
m4.metric("P(sell ≤ 60d)", f"{p60:.2%}")

st.caption(
    "Note: P(sell ≤60d) is computed as P(≤30) + (1−P(≤30))×P(31–60 | not ≤30)."
)
