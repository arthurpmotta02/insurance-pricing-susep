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

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Precificação de Seguro Auto",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Calculadora de Prêmio Puro — Seguro Auto")
st.markdown("""
Modelo atuarial baseado em dados reais da **SUSEP AUTOSEG (2019-2021)**.  
Utiliza **GLM Poisson** (frequência) × **GLM Gamma** (severidade) para estimar o prêmio puro de colisão.  
Comparação com **XGBoost Tweedie** (Gini = 0.233, Early Stopping round 341).
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
    with open(f'{MODELS_PATH}/glm_freq.pkl', 'rb') as f:
        glm_freq = pickle.load(f)
    with open(f'{MODELS_PATH}/glm_sev.pkl', 'rb') as f:
        glm_sev = pickle.load(f)
    with open(f'{MODELS_PATH}/glm_sev_cols.pkl', 'rb') as f:
        glm_sev_cols = pickle.load(f)
    with open(f'{MODELS_PATH}/idade_veiculo_median.pkl', 'rb') as f:
        idade_median = pickle.load(f)
    with open(f'{MODELS_PATH}/xgb_freq.pkl', 'rb') as f:
        xgb_freq = pickle.load(f)
    return glm_freq, glm_sev, glm_sev_cols, idade_median, xgb_freq

glm_freq, glm_sev, glm_sev_cols, idade_median, xgb_freq = load_models()

# Features base
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
    st.subheader("📋 Resumo da Apólice")
    st.metric("Sexo", sexo)
    st.metric("Faixa Etária", faixa_label)
    st.metric("IS", f"R$ {is_media:,.0f}")

st.divider()

# ============================================================
# FUNÇÕES DE CÁLCULO
# ============================================================

def build_feature_row(sexo, faixa_label, regiao_label, ano_modelo, is_media):
    sexo_bin   = 1 if sexo == "Masculino" else 0
    faixa      = faixas_etarias[faixa_label]
    regiao_cod = regioes[regiao_label]
    idade_veic = float(np.clip(2021 - ano_modelo, 0, 30))
    log_is     = float(np.log1p(is_media))

    row = {f: 0.0 for f in features_base}
    row['sexo_bin']      = sexo_bin
    row['faixa_etaria']  = faixa
    row['idade_veiculo'] = idade_veic if not np.isnan(idade_veic) else idade_median
    row['log_is_media']  = log_is

    if regiao_cod != '01':
        col = f'regiao_{regiao_cod}'
        if col in row:
            row[col] = 1.0

    return row, regiao_cod


def calcular_glm(sexo, faixa_label, regiao_label, ano_modelo, is_media):
    row, _ = build_feature_row(sexo, faixa_label, regiao_label, ano_modelo, is_media)

    # Frequência
    X_freq = pd.DataFrame([row])
    X_freq = sm.add_constant(X_freq, has_constant='add')
    offset = np.log(np.array([1.0]))  # exposição = 1 veículo-ano
    freq = float(glm_freq.predict(X_freq, offset=offset)[0])

    # Severidade
    X_sev = pd.DataFrame([row])[glm_sev_cols].astype(np.float64)
    sev = float(glm_sev.predict(X_sev)[0])

    return freq, sev, freq * sev


def calcular_xgb(sexo, faixa_label, regiao_label, ano_modelo, is_media):
    row, _ = build_feature_row(sexo, faixa_label, regiao_label, ano_modelo, is_media)
    X = pd.DataFrame([row])[xgb_freq.feature_names_in_]
    return float(max(xgb_freq.predict(X)[0], 0))

# ============================================================
# CÁLCULO
# ============================================================

if st.button("🧮 Calcular Prêmio Puro", type="primary", use_container_width=True):

    freq_glm, sev_glm, premio_glm = calcular_glm(
        sexo, faixa_label, regiao_label, ano_modelo, is_media
    )
    freq_xgb = calcular_xgb(
        sexo, faixa_label, regiao_label, ano_modelo, is_media
    )

    st.subheader("📊 Resultado — GLM Poisson × Gamma")
    c1, c2, c3 = st.columns(3)
    c1.metric("Frequência (GLM)", f"{freq_glm:.4f}",
              help="Probabilidade de sinistro por veículo-ano")
    c2.metric("Severidade Média (GLM)", f"R$ {sev_glm:,.2f}",
              help="Custo médio por sinistro")
    c3.metric("💰 Prêmio Puro (GLM)", f"R$ {premio_glm:,.2f}",
              help="Frequência × Severidade")

    st.subheader("📊 Resultado — XGBoost Tweedie (Gini = 0.233)")
    c4, c5 = st.columns(2)
    c4.metric("Frequência (XGBoost)", f"{freq_xgb:.4f}")
    c5.metric("Diferença vs GLM", f"{(freq_xgb - freq_glm):+.4f}")

    st.info(f"""
    **Interpretação:** Para este perfil, o GLM estima **{freq_glm:.2%}** de chance de sinistro 
    por ano, com custo médio de **R$ {sev_glm:,.2f}** por ocorrência.  
    O prêmio puro estimado é de **R$ {premio_glm:,.2f}** por veículo-ano.  
    O XGBoost estima frequência de **{freq_xgb:.2%}**.
    """)

st.divider()
st.caption("Fonte: SUSEP AUTOSEG 2019-2021 | GLM Poisson × Gamma | XGBoost Tweedie (Gini=0.233) | Autor: Arthur Pontes Motta")