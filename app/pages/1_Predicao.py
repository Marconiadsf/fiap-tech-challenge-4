import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Predição | Obesity Predictor", page_icon="🔮", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {background: #0f172a;}
    [data-testid="stSidebar"] * {color: #e2e8f0 !important;}
    .pred-badge {
        padding: 1rem 2rem; border-radius: 12px; text-align: center;
        font-size: 1.5rem; font-weight: 700; margin: 1rem 0;
    }
    .section-title {color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# ── Carregar modelo ────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "model" / "model_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    artifacts = load_model()
    pipeline  = artifacts["pipeline"]
    le        = artifacts["label_encoder"]
    classes   = artifacts["target_classes"]
except FileNotFoundError:
    st.error("⚠️ Modelo não encontrado. Execute o notebook de treinamento primeiro.")
    st.stop()

# ── Cores por classe ───────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Insufficient_Weight": "#06b6d4",
    "Normal_Weight":        "#22c55e",
    "Overweight_Level_I":   "#eab308",
    "Overweight_Level_II":  "#f97316",
    "Obesity_Type_I":       "#ef4444",
    "Obesity_Type_II":      "#dc2626",
    "Obesity_Type_III":     "#991b1b",
}
CLASS_LABELS = {
    "Insufficient_Weight": "⬇ Abaixo do Peso",
    "Normal_Weight":        "✅ Peso Normal",
    "Overweight_Level_I":   "⚠ Sobrepeso Grau I",
    "Overweight_Level_II":  "⚠ Sobrepeso Grau II",
    "Obesity_Type_I":       "🚨 Obesidade Tipo I",
    "Obesity_Type_II":      "🚨 Obesidade Tipo II",
    "Obesity_Type_III":     "🚨 Obesidade Tipo III",
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔮 Predição de Nível de Obesidade")
st.markdown("Preencha os dados de hábitos de vida do paciente para obter a previsão do modelo.")
st.markdown("---")

# ── Formulário ─────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.markdown("### 👤 Dados pessoais")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Sexo", ["Male", "Female"], help="Sexo biológico do paciente")
    with c2:
        age = st.number_input("Idade", min_value=10, max_value=80, value=25, step=1)
    with c3:
        family_history = st.selectbox("Histórico familiar de sobrepeso?", ["yes", "no"])

    st.markdown("### 🍽️ Hábitos alimentares")
    c4, c5, c6 = st.columns(3)
    with c4:
        favc = st.selectbox("Consome alimentos calóricos frequentemente? (FAVC)", ["yes", "no"])
    with c5:
        fcvc = st.slider("Frequência de consumo de vegetais (FCVC)", 1, 3, 2,
                         help="1 = raramente · 2 = às vezes · 3 = sempre")
    with c6:
        ncp = st.slider("Número de refeições principais por dia (NCP)", 1, 4, 3)

    c7, c8 = st.columns(2)
    with c7:
        caec = st.selectbox("Come entre refeições? (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
    with c8:
        ch2o = st.slider("Consumo de água por dia em litros (CH2O)", 1, 3, 2,
                         help="1 = < 1L · 2 = 1–2L · 3 = > 2L")

    st.markdown("### 🏃 Estilo de vida")
    c9, c10, c11 = st.columns(3)
    with c9:
        faf = st.slider("Frequência de atividade física por semana (FAF)", 0, 3, 1,
                        help="0 = nenhuma · 1 = 1–2x · 2 = 3–4x · 3 = 4–5x")
    with c10:
        tue = st.slider("Tempo diário em telas (TUE)", 0, 2, 1,
                        help="0 = 0–2h · 1 = 3–5h · 2 = > 5h")
    with c11:
        calc = st.selectbox("Consumo de álcool (CALC)", ["no", "Sometimes", "Frequently", "Always"])

    c12, c13, c14 = st.columns(3)
    with c12:
        smoke = st.selectbox("Fuma? (SMOKE)", ["no", "yes"])
    with c13:
        scc = st.selectbox("Monitora as calorias? (SCC)", ["no", "yes"])
    with c14:
        mtrans = st.selectbox("Meio de transporte habitual (MTRANS)",
                              ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    submitted = st.form_submit_button("🔍 Analisar Paciente", use_container_width=True, type="primary")

# ── Resultado ──────────────────────────────────────────────────────────────────
if submitted:
    input_data = pd.DataFrame([{
        "Age": age, "FCVC": fcvc, "NCP": ncp, "CH2O": ch2o, "FAF": faf, "TUE": tue,
        "Gender": gender, "family_history": family_history, "FAVC": favc,
        "SMOKE": smoke, "SCC": scc, "CAEC": caec, "CALC": calc, "MTRANS": mtrans,
    }])

    pred_enc  = pipeline.predict(input_data)[0]
    pred_prob = pipeline.predict_proba(input_data)[0]
    pred_class = le.inverse_transform([pred_enc])[0]

    st.markdown("---")
    st.markdown("## 📋 Resultado")

    col_res, col_prob = st.columns([1, 2])

    with col_res:
        color = CLASS_COLORS.get(pred_class, "#6366f1")
        label = CLASS_LABELS.get(pred_class, pred_class)
        confidence = pred_prob[pred_enc] * 100
        st.markdown(f"""
        <div class="pred-badge" style="background:{color}22; border:2px solid {color}; color:{color};">
            {label}<br>
            <span style="font-size:1rem; font-weight:400;">Confiança: {confidence:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col_prob:
        prob_df = pd.DataFrame({
            "Classe": le.classes_,
            "Probabilidade (%)": pred_prob * 100
        }).sort_values("Probabilidade (%)", ascending=True)

        fig = px.bar(
            prob_df, x="Probabilidade (%)", y="Classe", orientation="h",
            title="Probabilidade por Classe",
            color="Classe",
            color_discrete_map={c: CLASS_COLORS.get(c, "#6366f1") for c in le.classes_},
        )
        fig.update_layout(
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            yaxis_title="", xaxis_title="Probabilidade (%)",
            height=320,
        )
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(fig, width="stretch")

    # Interpretação clínica
    st.markdown("---")
    st.markdown("### 💡 Fatores de risco identificados no perfil")

    risk_factors = []
    if family_history == "yes":
        risk_factors.append("🧬 **Histórico familiar:** predisposição genética/epigenética ao sobrepeso")
    if favc == "yes":
        risk_factors.append("🍔 **Alimentação:** consumo frequente de alimentos calóricos")
    if faf == 0:
        risk_factors.append("🛋️ **Sedentarismo:** ausência de atividade física regular")
    if caec in ["Frequently", "Always"]:
        risk_factors.append("🍪 **Petiscos:** alimentação frequente entre refeições")
    if mtrans == "Automobile":
        risk_factors.append("🚗 **Transporte:** uso habitual de automóvel (baixo gasto calórico)")
    if calc in ["Frequently", "Always"]:
        risk_factors.append("🍺 **Álcool:** consumo frequente de bebidas alcoólicas")
    if ch2o == 1:
        risk_factors.append("💧 **Hidratação:** consumo de água abaixo do recomendado")

    if risk_factors:
        for f in risk_factors:
            st.markdown(f"- {f}")
    else:
        st.success("✅ Nenhum fator de risco evidente identificado no perfil de hábitos.")

    st.caption("⚠️ Esta análise é auxiliar e não substitui avaliação médica profissional.")
