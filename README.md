# 🏥 Obesity Predictor — Tech Challenge Fase 4

> Sistema preditivo de nível de obesidade baseado em hábitos de vida e dados físicos, desenvolvido como entregável do Tech Challenge Fase 4 da POSTECH/FIAP.

---

## 📌 Sobre o Projeto

Um hospital contratou um cientista de dados para desenvolver um modelo de Machine Learning que auxilie médicos a **prever se uma pessoa pode ter obesidade**, usando dados de hábitos de vida e características físicas.

**Problema:** Classificação multiclasse — prever o nível de obesidade de um paciente entre 7 categorias possíveis.

**Dataset:** `obesity.csv` — 2.111 registros, 14 features (após remoção de Weight/Height), 7 classes balanceadas.

> **Nota sobre Data Leakage:** Para garantir um modelo com valor clínico real, as variáveis de Peso (Weight) e Altura (Height) foram removidas das features de treinamento, pois o label de Obesidade é derivado diretamente do IMC (Peso/Altura²). O modelo foca em **hábitos de vida**.

---

## 🎯 Entregáveis

- [x] Pipeline de Machine Learning com feature engineering e treinamento
- [x] Modelo com acurácia > 75%
- [x] App preditivo em Streamlit (Concluído)
- [x] Dashboard analítico com insights para equipe médica (Concluído)
- [ ] Vídeo de apresentação (4–10 min)

---

## 🗂️ Estrutura do Repositório

```
tc4-obesity-predictor/
├── notebooks/
│   └── 01_EDA_e_Treinamento.ipynb   # Pipeline ML completa
├── app/
│   ├── app.py                        # Entrada do Streamlit
│   └── pages/
│       ├── 1_Predicao.py             # Formulário preditivo
│       └── 2_Dashboard.py            # Painel analítico
├── model/
│   └── model_pipeline.pkl            # Pipeline sklearn serializada
├── data/
│   └── obesity.csv                   # Dataset
├── tests/
│   └── test_model.py                 # Testes de qualidade do modelo
└── requirements.txt
```

---

## 🛠️ Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.14 |
| EDA & ML | pandas · numpy · scikit-learn · xgboost |
| Serialização | joblib |
| Visualização | plotly · matplotlib · seaborn |
| App | Streamlit |
| Deploy | Streamlit Community Cloud |

---

## 🚀 Como Executar Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/Marconiadsf/fiap-tech-challenge-4.git
cd fiap-tech-challenge-4

# 2. Crie o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o app
streamlit run app/app.py
```

---

## 📊 Variáveis do Dataset

| Feature | Descrição |
|---|---|
| Gender | Sexo (Female/Male) |
| Age | Idade (14–61 anos) |
| Height | Altura em metros |
| Weight | Peso em kg |
| family_history | Histórico familiar de sobrepeso |
| FAVC | Consumo frequente de alimentos calóricos |
| FCVC | Frequência de consumo de vegetais (1–3) |
| NCP | Número de refeições principais/dia (1–4) |
| CAEC | Come entre refeições? |
| SMOKE | Fuma? |
| CH2O | Consumo de água/dia (1–3) |
| SCC | Monitora calorias? |
| FAF | Frequência de atividade física (0–3) |
| TUE | Tempo em dispositivos eletrônicos (0–2) |
| CALC | Consumo de álcool |
| MTRANS | Meio de transporte habitual |
| **Obesity** | **TARGET** — 7 classes de nível de obesidade |

---

## 🔗 Links

- **App Streamlit:** [fiap-tech-challenge-4-obesity-predictor.streamlit.app](https://fiap-tech-challenge-4-obesity-predictor.streamlit.app/)
- **Dashboard:** [fiap-tech-challenge-4-obesity-predictor.streamlit.app/Dashboard](https://fiap-tech-challenge-4-obesity-predictor.streamlit.app/Dashboard)
- **Repositório:** [github.com/Marconiadsf/fiap-tech-challenge-4](https://github.com/Marconiadsf/fiap-tech-challenge-4)

---

*Desenvolvido para o Tech Challenge Fase 4 — Data Analytics · POSTECH/FIAP · 2026*
