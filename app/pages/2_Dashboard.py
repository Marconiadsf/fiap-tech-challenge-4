import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Dashboard | Obesity Predictor",
    page_icon="📊",
    layout="wide"
)

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

# ── Modelo carregado pelo app.py e disponível via session_state ───────────────
if "pipeline" not in st.session_state:
    st.error("⚠️ Sessão expirada ou modelo não carregado. Volte à página inicial.")
    st.page_link("app.py", label="← Voltar ao início")
    st.stop()

pipeline      = st.session_state["pipeline"]
model_name    = st.session_state["model_name"]
acc           = st.session_state["accuracy"]
feature_names = st.session_state["feature_names"]
n_samples     = st.session_state["n_samples"]

# ── Dados para o dashboard ────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "obesity.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Sem drop_duplicates — consistente com o notebook
    df = df.drop(columns=["Weight", "Height"])
    return df

df = load_data()

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

# ── KPIs — lidos do session_state, sem hardcode ───────────────────────────────
obese_pct  = df["Obesity"].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]).mean() * 100
overw_pct  = df["Obesity"].isin(["Overweight_Level_I", "Overweight_Level_II"]).mean() * 100
normal_pct = (df["Obesity"] == "Normal_Weight").mean() * 100

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("Pacientes",              f"{n_samples:,}".replace(",", ".")),
    ("Obesos",                 f"{obese_pct:.1f}%"),
    ("Sobrepeso",              f"{overw_pct:.1f}%"),
    ("Peso Normal",            f"{normal_pct:.1f}%"),
    (f"Acurácia ({model_name})", f"{acc*100:.1f}%"),
]
for col, (label, value) in zip([c1, c2, c3, c4, c5], metrics):
    with col:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Distribuição do target ────────────────────────────────────────────────────
col_left, col_right = st.columns(2)
counts = df["Obesity"].value_counts().reindex(TARGET_ORDER)

with col_left:
    st.subheader("Distribuição por Nível de Obesidade")
    
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map=CLASS_COLORS,
        text=counts.values,
        labels={"x": "", "y": "Pacientes", "color": "Nível"},
    )

    fig.update_traces(
        width=0.7,
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis_tickangle=-30,
        height=420,
        bargap=0.18,
        yaxis_title="Quantidade de pacientes",
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Proporção das Classes")
    fig = px.pie(
        names=counts.index, values=counts.values,
        color=counts.index, color_discrete_map=CLASS_COLORS, hole=0.45,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=320)
    st.plotly_chart(fig, use_container_width=True)

# ── Histórico familiar | Atividade física ─────────────────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🧬 Histórico Familiar × Nível de Obesidade")
    cross     = pd.crosstab(df["Obesity"], df["family_history"]).reindex(TARGET_ORDER)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for col_, color in [("yes", "#ef4444"), ("no", "#22c55e")]:
        if col_ in cross_pct.columns:
            fig.add_trace(go.Bar(
                name="Histórico: " + col_,
                x=cross_pct.index, y=cross_pct[col_],
                marker_color=color,
            ))
    fig.update_layout(
        barmode="stack", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        xaxis_tickangle=-30, yaxis_title="%", height=350,
        legend_title="Histórico familiar",
    )
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)

# ── Transporte | Comida calórica ──────────────────────────────────────────────
st.markdown("---")
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🚗 Meio de Transporte × Nível de Obesidade")
    cross_mt     = pd.crosstab(df["Obesity"], df["MTRANS"]).reindex(TARGET_ORDER)
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
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    st.subheader("🍔 Alimentos Calóricos (FAVC) × Nível de Obesidade")
    cross_fv     = pd.crosstab(df["Obesity"], df["FAVC"]).reindex(TARGET_ORDER)
    cross_fv_pct = cross_fv.div(cross_fv.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for col_, color in [("yes", "#f97316"), ("no", "#22c55e")]:
        if col_ in cross_fv_pct.columns:
            fig.add_trace(go.Bar(
                name="FAVC: " + col_,
                x=cross_fv_pct.index, y=cross_fv_pct[col_],
                marker_color=color,
            ))
    fig.update_layout(
        barmode="stack", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        xaxis_tickangle=-30, yaxis_title="%", height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Importância das Features no Modelo")

try:
    # Step names do notebook: "preprocessor" e "classifier"
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]

    # Reconstrói os nomes de todas as features após o preprocessor
    num_cols = pipeline.named_steps["preprocessor"].transformers_[0][2]
    cat_cols = pipeline.named_steps["preprocessor"].transformers_[1][2]
    ohe_feat_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    all_feat_names = list(num_cols) + ohe_feat_names

    if hasattr(classifier, "feature_importances_"):
        imp = pd.Series(
            classifier.feature_importances_,
            index=all_feat_names
        ).sort_values(ascending=True).tail(15)

        fig = px.bar(
            x=imp.values, y=imp.index, orientation="h",
            labels={"x": "Importância", "y": "Feature"},
            color=imp.values, color_continuous_scale="Viridis",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=450, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Modelo: {model_name} · "
            f"Acurácia CV K=10: {acc*100:.1f}% · "
            f"Treinado sem Weight/Height (leakage-free)"
        )

except Exception as e:
    st.warning(f"Feature importance não disponível: {e}")

st.markdown("---")
st.caption(
    "Fonte: [Dataset de Estimativa de Obesidade — UCI ML Repository]"
    "(https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)"
    " · Tech Challenge Fase 4 POSTECH/FIAP"
)
