import streamlit as st

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
    .badge {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px;
        font-size: 0.8rem; font-weight: 600;
    }
    .badge-green {background: #064e3b; color: #34d399;}
    .badge-blue  {background: #1e3a5f; color: #60a5fa;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏥 Obesity Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistema preditivo de risco de obesidade baseado em hábitos de vida — Tech Challenge Fase 4 · POSTECH/FIAP</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="card">
        <div class="card-title">Pacientes analisados</div>
        <div class="card-value">2.087</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="card">
        <div class="card-title">Acurácia do modelo</div>
        <div class="card-value">78,7%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="card">
        <div class="card-title">Classes previstas</div>
        <div class="card-value">7</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="card">
        <div class="card-title">Modelo</div>
        <div class="card-value" style="font-size:1.2rem;">XGBoost</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("### 🔮 Predição de Risco")
    st.write("Insira os hábitos de vida do paciente e obtenha uma previsão do nível de obesidade com probabilidades por classe.")
    st.page_link("pages/1_Predicao.py", label="Ir para Predição →", icon="🔮")

with col_b:
    st.markdown("### 📊 Dashboard Analítico")
    st.write("Explore insights sobre a distribuição de obesidade, padrões de hábitos e os fatores mais relevantes para o modelo.")
    st.page_link("pages/2_Dashboard.py", label="Ir para Dashboard →", icon="📊")

st.markdown("---")
st.markdown("""
<div style="color:#475569; font-size:0.85rem;">
⚠️ <strong>Nota metodológica:</strong> Este modelo foi treinado <strong>sem Peso e Altura</strong> para evitar data leakage
— o label de obesidade é matematicamente derivado do BMI (Peso/Altura²).
O modelo aprende padrões genuínos de hábitos de vida, com valor prático para triagem clínica.
</div>
""", unsafe_allow_html=True)
