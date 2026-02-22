# ============================================================
# Streamlit App — Calculadora de Prêmio Puro
# Seguro Auto SUSEP (2019-2021)
# Autor: Arthur Pontes Motta
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import statsmodels.api as sm

st.set_page_config(
    page_title="Precificação de Seguro Auto",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Calculadora de Prêmio Puro — Seguro Auto")
st.markdown("""
Modelo atuarial baseado em dados reais da **SUSEP AUTOSEG (2019-2021)**.  
Utiliza **GLM Poisson × Gamma** para frequência e severidade de **colisão e roubo** separadamente.  
Comparação com **XGBoost Tweedie** (Gini Colisão = 0.241 | Gini Roubo = 0.402).
""")

st.divider()

# ============================================================
# REGIÕES E FAIXAS
# ============================================================

regioes = {
    "01 - RS: Met. Porto Alegre e Caxias do Sul": "01",
    "02 - RS: Demais regiões": "02",
    "03 - SC: Met. Florianópolis e Sul": "03",
    "04 - SC: Oeste": "04",
    "05 - SC: Blumenau e demais regiões": "05",
    "06 - PR: Foz do Iguaçu / Cascavel": "06",
    "07 - PR: Met. Curitiba": "07",
    "08 - PR: Demais regiões": "08",
    "09 - SP: Vale do Paraíba e Ribeira": "09",
    "10 - SP: Litoral Norte e Baixada Santista": "10",
    "11 - SP: Met. de São Paulo": "11",
    "12 - SP: Grande Campinas": "12",
    "13 - SP: Ribeirão Preto e demais": "13",
    "14 - MG: Triângulo Mineiro": "14",
    "15 - MG: Sul": "15",
    "16 - MG: Met. BH e Centro-Oeste": "16",
    "17 - MG: Vale do Aço e Norte": "17",
    "18 - RJ: Met. do Rio de Janeiro": "18",
    "19 - RJ: Interior": "19",
    "20 - ES: Espírito Santo": "20",
    "21 - BA: Bahia": "21",
    "22 - SE: Sergipe": "22",
    "23 - PE: Pernambuco": "23",
    "24 - PB: Paraíba": "24",
    "25 - RN: Rio Grande do Norte": "25",
    "26 - AL: Alagoas": "26",
    "27 - CE: Ceará": "27",
    "28 - PI: Piauí": "28",
    "29 - MA: Maranhão": "29",
    "30 - PA: Pará": "30",
    "31 - AM: Amazonas": "31",
    "32 - AP: Amapá": "32",
    "33 - RO: Rondônia": "33",
    "34 - RR: Roraima": "34",
    "35 - AC: Acre": "35",
    "36 - MT: Mato Grosso": "36",
    "37 - MS: Mato Grosso do Sul": "37",
    "38 - DF: Brasília": "38",
    "39 - GO: Goiás": "39",
    "40 - TO: Tocantins": "40",
    "41 - GO: Sudeste de Goiás": "41",
}

faixas_etarias = {
    "18 a 25 anos": 1,
    "26 a 35 anos": 2,
    "36 a 45 anos": 3,
    "46 a 55 anos": 4,
    "Maior que 55 anos": 5,
}

# ============================================================
# CARREGAR MODELOS
# ============================================================

MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_models():
    def load(name):
        with open(f'{MODELS_PATH}/{name}', 'rb') as f:
            return pickle.load(f)

    return {
        'glm_freq_col':     load('glm_freq_col.pkl'),
        'glm_sev_col':      load('glm_sev_col.pkl'),
        'glm_sev_col_cols': load('glm_sev_col_cols.pkl'),
        'glm_freq_rou':     load('glm_freq_rou.pkl'),
        'glm_sev_rou':      load('glm_sev_rou.pkl'),
        'glm_sev_rou_cols': load('glm_sev_rou_cols.pkl'),
        'xgb_col':          load('xgb_col.pkl'),
        'xgb_rou':          load('xgb_rou.pkl'),
        'idade_median':     load('idade_veiculo_median.pkl'),
        'meta':             load('model_meta.pkl'),
    }

models = load_models()

features_base = (
    ['sexo_bin', 'faixa_etaria', 'idade_veiculo', 'log_is_media'] +
    [f'regiao_{str(i).zfill(2)}' for i in range(2, 42)]
)

# ============================================================
# INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Perfil do Condutor")
    sexo         = st.selectbox("Sexo", ["Masculino", "Feminino"])
    faixa_label  = st.selectbox("Faixa Etária", list(faixas_etarias.keys()))
    regiao_label = st.selectbox("Região de Circulação", list(regioes.keys()))

with col2:
    st.subheader("🚘 Dados do Veículo")
    ano_modelo = st.slider("Ano do Veículo", 1990, 2021, 2018)
    is_media   = st.number_input("Importância Segurada (R$)",
                                  min_value=5_000, max_value=500_000,
                                  value=50_000, step=5_000, format="%d")

with col3:
    st.subheader("📋 Resumo")
    st.metric("Sexo", sexo)
    st.metric("Faixa Etária", faixa_label)
    st.metric("IS", f"R$ {is_media:,.0f}")
    st.metric("Idade do Veículo", f"{2021 - ano_modelo} anos")

st.divider()

# ============================================================
# FUNÇÕES DE CÁLCULO
# ============================================================

def build_row(sexo, faixa_label, regiao_label, ano_modelo, is_media):
    row = {f: 0.0 for f in features_base}
    row['sexo_bin']      = 1.0 if sexo == "Masculino" else 0.0
    row['faixa_etaria']  = float(faixas_etarias[faixa_label])
    row['idade_veiculo'] = float(np.clip(2021 - ano_modelo, 0, 30))
    row['log_is_media']  = float(np.log1p(is_media))
    regiao_cod = regioes[regiao_label]
    if regiao_cod != '01':
        col = f'regiao_{regiao_cod}'
        if col in row:
            row[col] = 1.0
    return row

def calcular_glm_cobertura(row, glm_freq, glm_sev, sev_cols):
    X_freq = sm.add_constant(pd.DataFrame([row]), has_constant='add')
    offset = np.log(np.array([1.0]))
    freq   = float(glm_freq.predict(X_freq, offset=offset)[0])

    X_sev = pd.DataFrame([row])[sev_cols].astype(np.float64)
    sev   = float(glm_sev.predict(X_sev)[0])
    return freq, sev, freq * sev

def calcular_xgb_cobertura(row, xgb_model):
    X = pd.DataFrame([row])[xgb_model.feature_names_in_]
    return float(max(xgb_model.predict(X)[0], 0))

# ============================================================
# CÁLCULO
# ============================================================

if st.button("🧮 Calcular Prêmio Puro", type="primary", use_container_width=True):

    row = build_row(sexo, faixa_label, regiao_label, ano_modelo, is_media)

    # GLM — Colisão
    freq_col, sev_col, pp_col = calcular_glm_cobertura(
        row,
        models['glm_freq_col'],
        models['glm_sev_col'],
        models['glm_sev_col_cols']
    )

    # GLM — Roubo
    freq_rou, sev_rou, pp_rou = calcular_glm_cobertura(
        row,
        models['glm_freq_rou'],
        models['glm_sev_rou'],
        models['glm_sev_rou_cols']
    )

    # XGBoost
    xgb_col_pred = calcular_xgb_cobertura(row, models['xgb_col'])
    xgb_rou_pred = calcular_xgb_cobertura(row, models['xgb_rou'])

    pp_total_glm = pp_col + pp_rou
    pp_total_xgb = xgb_col_pred * sev_col + xgb_rou_pred * sev_rou

    # ============================================================
    # RESULTADOS
    # ============================================================

    st.subheader("📊 Prêmio Puro por Cobertura — GLM")

    c1, c2, c3 = st.columns(3)
    c1.metric("Freq. Colisão", f"{freq_col:.4f}",
              help="Probabilidade de sinistro por veículo-ano")
    c2.metric("Severidade Colisão", f"R$ {sev_col:,.2f}",
              help="Custo médio por sinistro de colisão")
    c3.metric("💰 PP Colisão (GLM)", f"R$ {pp_col:,.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Freq. Roubo", f"{freq_rou:.4f}",
              help="Probabilidade de roubo por veículo-ano")
    c5.metric("Severidade Roubo", f"R$ {sev_rou:,.2f}",
              help="Custo médio por sinistro de roubo")
    c6.metric("💰 PP Roubo (GLM)", f"R$ {pp_rou:,.2f}")

    st.divider()
    st.subheader("📊 Comparação GLM vs XGBoost")

    c7, c8, c9 = st.columns(3)
    c7.metric("PP Total GLM", f"R$ {pp_total_glm:,.2f}",
              help="Colisão + Roubo")
    c8.metric("PP Total XGBoost", f"R$ {pp_total_xgb:,.2f}",
              help="Usando freq XGBoost × sev GLM")
    c9.metric("Diferença", f"R$ {(pp_total_xgb - pp_total_glm):+,.2f}")

    st.divider()
    st.subheader("📊 XGBoost — Frequências")

    c10, c11 = st.columns(2)
    c10.metric("Freq. Colisão (XGBoost)",  f"{xgb_col_pred:.4f}",
               delta=f"{(xgb_col_pred - freq_col):+.4f} vs GLM")
    c11.metric("Freq. Roubo (XGBoost)", f"{xgb_rou_pred:.4f}",
               delta=f"{(xgb_rou_pred - freq_rou):+.4f} vs GLM")

    st.info(f"""
    **Interpretação:** Para este perfil, o GLM estima **{freq_col:.2%}** de chance de colisão 
    e **{freq_rou:.2%}** de chance de roubo por ano.  
    Custos médios: colisão **R$ {sev_col:,.2f}** | roubo **R$ {sev_rou:,.2f}**.  
    **Prêmio puro total (GLM): R$ {pp_total_glm:,.2f} por veículo-ano.**
    """)

st.divider()
st.caption(
    "Fonte: SUSEP AUTOSEG 2019-2021 | "
    "GLM Poisson × Gamma | "
    "XGBoost Tweedie (Gini Colisão=0.241 | Gini Roubo=0.402) | "
    "Autor: Arthur Pontes Motta"
)