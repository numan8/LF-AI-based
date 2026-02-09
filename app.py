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

st.set_page_config(page_title="Cash Sales Velocity Model", layout="wide")

DATA_PATH = "Cash Sales - AI Stats.xlsx"

NUM_COLS = ["Total Purchase Price", "Acres", "marketing_score", "purchase_month", "sale_month"]
CAT_COLS = ["state", "county", "city"]

# -------------------------
# Feature Engineering
# -------------------------
def prepare_data(df):
    df["PURCHASE DATE"] = pd.to_datetime(df["PURCHASE DATE"], errors="coerce")
    df["SALE DATE - start"] = pd.to_datetime(df["SALE DATE - start"], errors="coerce")

    df["Days_to_sell_cash"] = (df["SALE DATE - start"] - df["PURCHASE DATE"]).dt.days
    df = df[df["Days_to_sell_cash"].notna()]
    df = df[df["Days_to_sell_cash"] >= 0]

    df["sell_30d"] = (df["Days_to_sell_cash"] <= 30).astype(int)
    df["sell_60d_excl"] = ((df["Days_to_sell_cash"] > 30) & (df["Days_to_sell_cash"] <= 60)).astype(int)

    df["purchase_month"] = df["PURCHASE DATE"].dt.month
    df["sale_month"] = df["SALE DATE - start"].dt.month

    df["Total Purchase Price"] = pd.to_numeric(df["Total Purchase Price"], errors="coerce")
    df["Acres"] = pd.to_numeric(df["Acres"], errors="coerce")

    df["promo_confirmed"] = df["Promo Price Status"].astype(str).str.contains("confirmed", case=False, na=False).astype(int)

    df["has_photos"] = df["Photographer/Inspector Status"].astype(str).str.contains("photo", case=False, na=False).astype(int)
    df["has_drone"] = df["Photographer/Inspector Status"].astype(str).str.contains("drone", case=False, na=False).astype(int)

    df["marketing_score"] = df[["promo_confirmed", "has_photos", "has_drone"]].sum(axis=1)

    df["state"] = df["County, State"].astype(str).str.split(",").str[-1].str.strip()
    df["county"] = df["County, State"].astype(str).str.split(",").str[0].str.strip()
    df["city"] = df["Property Location or City"].astype(str)

    return df


# -------------------------
# Preprocessing Pipeline
# -------------------------
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


# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH, engine="openpyxl")

df_raw = load_data()
df = prepare_data(df_raw)

st.title("Cash Sales Velocity AI Model")

st.write("Rows after cleaning:", len(df))

# -------------------------
# Model A (≤30 days)
# -------------------------
X = df[NUM_COLS + CAT_COLS]
y30 = df["sell_30d"]

pipe30 = build_pipeline()
X_train, X_test, y_train, y_test = train_test_split(
    X, y30, test_size=0.2, random_state=42, stratify=y30
)

pipe30.fit(X_train, y_train)

pred30 = pipe30.predict(X_test)
proba30 = pipe30.predict_proba(X_test)[:, 1]

st.subheader("Model A — Sell ≤ 30 Days")
st.write("AUC:", roc_auc_score(y_test, proba30))
st.write(confusion_matrix(y_test, pred30))
st.text(classification_report(y_test, pred30))

# -------------------------
# Model B (31–60 days)
# -------------------------
df60 = df[df["sell_30d"] == 0]

X60 = df60[NUM_COLS + CAT_COLS]
y60 = df60["sell_60d_excl"]

pipe60 = build_pipeline()

X_train, X_test, y_train, y_test = train_test_split(
    X60, y60, test_size=0.2, random_state=42, stratify=y60
)

pipe60.fit(X_train, y_train)

pred60 = pipe60.predict(X_test)
proba60 = pipe60.predict_proba(X_test)[:, 1]

st.subheader("Model B — Sell 31–60 Days")
st.write("AUC:", roc_auc_score(y_test, proba60))
st.write(confusion_matrix(y_test, pred60))
st.text(classification_report(y_test, pred60))
