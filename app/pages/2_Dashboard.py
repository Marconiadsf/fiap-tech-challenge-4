import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Dashboard | Obesity Predictor", page_icon="📊", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] {background: #0f172a;}
    [data-testid="stSidebar"] * {color: #e2e8f0 !important;}
    .kpi-card {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .kpi-label {color: #64748b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;}
    .kpi-value {color: #f1f5f9; font-size: 2rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent.parent
DATA_PATH = BASE / "data" / "obesity.csv"
MODEL_PATH = BASE / "model" / "model_pipeline.pkl"

# ── Carregar dados e modelo ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().drop(columns=["Weight", "Height"]).reset_index(drop=True)
    ordinal_cols = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
    df[ordinal_cols] = df[ordinal_cols].round(0).astype(int)
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()

try:
    artifacts = load_model()
    model_name = artifacts.get("model_name", "XGBoost")
    acc        = artifacts.get("accuracy_test", 0.787)
except Exception:
    model_name, acc = "XGBoost", 0.787

TARGET_ORDER = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]
CLASS_COLORS = {
    "Insufficient_Weight": "#06b6d4",
    "Normal_Weight":        "#22c55e",
    "Overweight_Level_I":   "#eab308",
    "Overweight_Level_II":  "#f97316",
    "Obesity_Type_I":       "#ef4444",
    "Obesity_Type_II":      "#dc2626",
    "Obesity_Type_III":     "#991b1b",
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 Dashboard Analítico — Padrões de Obesidade")
st.markdown("Insights sobre a distribuição e os fatores associados ao nível de obesidade no dataset.")
st.markdown("---")

# ── KPIs ───────────────────────────────────────────────────────────────────────
obese_pct = df["Obesity"].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]).mean() * 100
overw_pct  = df["Obesity"].isin(["Overweight_Level_I", "Overweight_Level_II"]).mean() * 100
normal_pct = (df["Obesity"] == "Normal_Weight").mean() * 100
fh_pct     = (df["family_history"] == "yes").mean() * 100

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("Pacientes", f"{len(df):,}"),
    ("Obesos", f"{obese_pct:.1f}%"),
    ("Sobrepeso", f"{overw_pct:.1f}%"),
    ("Peso Normal", f"{normal_pct:.1f}%"),
    (f"Acurácia ({model_name})", f"{acc*100:.1f}%"),
]
for col, (label, value) in zip([c1, c2, c3, c4, c5], metrics):
    with col:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Linha 1: Distribuição do target | Pizza ───────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Distribuição por Nível de Obesidade")
    counts = df["Obesity"].value_counts().reindex(TARGET_ORDER)
    fig = px.bar(
        x=counts.index, y=counts.values,
        color=counts.index,
        color_discrete_map=CLASS_COLORS,
        labels={"x": "", "y": "Pacientes", "color": "Nível"},
    )
    fig.update_layout(
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        xaxis_tickangle=-30, height=320,
    )
    st.plotly_chart(fig, width="stretch")

with col_right:
    st.subheader("Proporção das Classes")
    fig = px.pie(
        names=counts.index, values=counts.values,
        color=counts.index, color_discrete_map=CLASS_COLORS,
        hole=0.45,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=320,
    )
    st.plotly_chart(fig, width="stretch")

# ── Linha 2: Histórico familiar | Atividade física ────────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🧬 Histórico Familiar × Nível de Obesidade")
    cross = pd.crosstab(df["Obesity"], df["family_history"])
    cross = cross.reindex(TARGET_ORDER)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    fig = go.Figure()
    colors_fh = {"yes": "#ef4444", "no": "#22c55e"}
    for col_ in ["yes", "no"]:
        if col_ in cross_pct.columns:
            fig.add_trace(go.Bar(
                name="Histórico: " + col_,
                x=cross_pct.index, y=cross_pct[col_],
                marker_color=colors_fh[col_],
            ))
    fig.update_layout(
        barmode="stack", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        xaxis_tickangle=-30, yaxis_title="%", height=350, legend_title="Histórico familiar",
    )
    st.plotly_chart(fig, width="stretch")

with col_b:
    st.subheader("🏃 Atividade Física (FAF) × Nível de Obesidade")
    fig = px.box(
        df, x="Obesity", y="FAF",
        category_orders={"Obesity": TARGET_ORDER},
        color="Obesity", color_discrete_map=CLASS_COLORS,
        labels={"FAF": "FAF (0=nenhuma · 3=intensa)", "Obesity": ""},
    )
    fig.update_layout(
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        xaxis_tickangle=-30, height=350,
    )
    st.plotly_chart(fig, width="stretch")

# ── Linha 3: Transporte | Comida calórica ─────────────────────────────────────
st.markdown("---")
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🚗 Meio de Transporte × Nível de Obesidade")
    cross_mt = pd.crosstab(df["Obesity"], df["MTRANS"]).reindex(TARGET_ORDER)
    cross_mt_pct = cross_mt.div(cross_mt.sum(axis=1), axis=0) * 100
    fig = px.bar(
        cross_mt_pct.reset_index().melt(id_vars="Obesity"),
        x="Obesity", y="value", color="MTRANS", barmode="stack",
        labels={"value": "%", "Obesity": "", "MTRANS": "Transporte"},
        category_orders={"Obesity": TARGET_ORDER},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", xaxis_tickangle=-30, height=380,
    )
    st.plotly_chart(fig, width="stretch")

with col_d:
    st.subheader("🍔 Alimentos Calóricos (FAVC) × Nível de Obesidade")
    cross_fv = pd.crosstab(df["Obesity"], df["FAVC"]).reindex(TARGET_ORDER)
    cross_fv_pct = cross_fv.div(cross_fv.sum(axis=1), axis=0) * 100
    colors_fv = {"yes": "#f97316", "no": "#22c55e"}
    fig = go.Figure()
    for col_ in ["yes", "no"]:
        if col_ in cross_fv_pct.columns:
            fig.add_trace(go.Bar(
                name="FAVC: " + col_,
                x=cross_fv_pct.index, y=cross_fv_pct[col_],
                marker_color=colors_fv[col_],
            ))
    fig.update_layout(
        barmode="stack", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        xaxis_tickangle=-30, yaxis_title="%", height=380,
    )
    st.plotly_chart(fig, width="stretch")

# ── Linha 4: Feature Importance ───────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Importância das Features no Modelo")

try:
    pipe     = artifacts["pipeline"]
    le       = artifacts.get("label_encoder")
    features = artifacts.get("feature_names", [])

    model_step = pipe.named_steps["model"]
    ohe_feats  = (
        pipe.named_steps["prep"]
        .named_transformers_["ohe"]
        .get_feature_names_out(["MTRANS"])
        .tolist()
    )
    numeric_cols = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    binary_cols  = ["Gender", "family_history", "FAVC", "SMOKE", "SCC"]
    ordered_cols = ["CAEC", "CALC"]
    all_feat_names = numeric_cols + binary_cols + ordered_cols + ohe_feats

    if hasattr(model_step, "feature_importances_"):
        imp = pd.Series(model_step.feature_importances_, index=all_feat_names).sort_values(ascending=True)
        fig = px.bar(
            x=imp.values, y=imp.index, orientation="h",
            labels={"x": "Importância", "y": "Feature"},
            color=imp.values, color_continuous_scale="Viridis",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=450, showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Modelo: {model_name} · Acurácia no teste: {acc*100:.1f}% · Trained without Weight/Height (leakage-free)")
except Exception as e:
    st.warning(f"Feature importance não disponível: {e}")

st.markdown("---")
st.caption("Fonte: Dataset de Estimativa de Obesidade — UCI ML Repository · Processado pelo Tech Challenge Fase 4 POSTECH/FIAP")
