import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Obesity Predictor | Tech Challenge 4",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {background: #0f172a;}
    [data-testid="stSidebar"] * {color: #e2e8f0 !important;}
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #22d3ee);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;}
    .card {
        background: #1e293b; border-radius: 16px; padding: 1.5rem;
        border: 1px solid #334155; margin-bottom: 1rem;
    }
    .card-title {color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;}
    .card-value {color: #f1f5f9; font-size: 1.8rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# ── Carregamento do modelo — ocorre uma vez ao iniciar a aplicação ─────────────
# O modelo é armazenado em st.session_state para ficar disponível
# em todas as páginas sem precisar ser recarregado a cada navegação.


MODEL_PATH = Path(__file__).parent.parent / "model" / "model_pipeline.pkl"

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)

try:
    artifacts = load_artifacts()
    st.session_state["artifacts"] = artifacts
except FileNotFoundError:
    st.error("⚠️ model_pipeline.pkl não encontrado. Execute o notebook de treinamento primeiro.")
    st.stop()

# Atalhos diretos para as páginas usarem sem repetir a lógica de extração
st.session_state["pipeline"]       = artifacts["pipeline"]
st.session_state["label_encoder"]  = artifacts["label_encoder"]
st.session_state["target_classes"] = artifacts["target_classes"]
st.session_state["model_name"]     = artifacts.get("model_name", "Random Forest")
st.session_state["accuracy"]       = artifacts.get("accuracy_test", 0.0)
st.session_state["feature_names"]  = artifacts.get("feature_names", [])
st.session_state["n_samples"]      = artifacts.get("n_samples", 0)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏥 Obesity Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Sistema preditivo de risco de obesidade baseado em hábitos de vida '
    '— Tech Challenge Fase 4 · POSTECH/FIAP</p>',
    unsafe_allow_html=True
)

# ── KPIs — lidos do pkl, sem hardcode ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

kpis = [
    ("Pacientes analisados", f"{st.session_state['n_samples']:,}".replace(",", ".")),
    ("Acurácia do modelo",   f"{st.session_state['accuracy']*100:.1f}%"),
    ("Classes previstas",    "7"),
    ("Modelo",               st.session_state["model_name"]),
]

for col, (title, value) in zip([c1, c2, c3, c4], kpis):
    with col:
        st.markdown(f"""<div class="card">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🔮 Predição de Risco")
    st.write(
        "Insira os hábitos de vida do paciente e obtenha uma previsão "
        "do nível de obesidade com probabilidades por classe."
    )
    st.page_link("pages/1_Predicao.py", label="Ir para Predição →", icon="🔮")

with col_b:
    st.markdown("### 📊 Dashboard Analítico")
    st.write(
        "Explore insights sobre a distribuição de obesidade, padrões de hábitos "
        "e os fatores mais relevantes para o modelo."
    )
    st.page_link("pages/2_Dashboard.py", label="Ir para Dashboard →", icon="📊")

st.markdown("---")
st.markdown("""
<div style="color:#475569; font-size:0.85rem;">
⚠️ <strong>Nota metodológica:</strong> Este modelo foi treinado <strong>sem Peso e Altura</strong>
para evitar data leakage — o label de obesidade é matematicamente derivado do BMI (Peso/Altura²).
O modelo aprende padrões genuínos de hábitos de vida, com valor prático para triagem clínica.
</div>
""", unsafe_allow_html=True)
