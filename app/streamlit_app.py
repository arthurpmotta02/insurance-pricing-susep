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
""")

st.divider()

# ============================================================
# REGIÕES
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
# INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Perfil do Condutor")
    sexo        = st.selectbox("Sexo", ["Masculino", "Feminino"])
    faixa_label = st.selectbox("Faixa Etária", list(faixas_etarias.keys()))
    regiao_label = st.selectbox("Região de Circulação", list(regioes.keys()))

with col2:
    st.subheader("🚘 Dados do Veículo")
    ano_modelo  = st.slider("Ano do Veículo", 1990, 2021, 2018)
    is_media    = st.number_input("Importância Segurada (R$)", 
                                   min_value=5_000, max_value=500_000,
                                   value=50_000, step=5_000,
                                   format="%d")

with col3:
    st.subheader("📋 Resumo da Apólice")
    st.metric("Sexo", sexo)
    st.metric("Faixa Etária", faixa_label)
    st.metric("IS", f"R$ {is_media:,.0f}")

st.divider()

# ============================================================
# CÁLCULO DO PRÊMIO PURO
# ============================================================

# Coeficientes do GLM Poisson (frequência)
# Extraídos do notebook 03
coef_freq = {
    'const':         -9.2306,
    'sexo_bin':       0.0470,
    'faixa_etaria':  -0.1321,
    'idade_veiculo':  0.0195,
    'log_is_media':   0.4631,
    'regiao': {
        '01': 0.0000,  # referência
        '02': 0.1317, '03': 0.4494, '04': 0.6522, '05': 0.1889,
        '06': 0.7434, '07': 0.0504, '08': 0.2385, '09': -0.0775,
        '10': 0.5969, '11': -1.0797, '12': 0.0079, '13': -0.6326,
        '14': 0.5957, '15': 0.3992, '16': -0.1285, '17': 1.0261,
        '18': -0.3607, '19': 0.1742, '20': 0.3260, '21': 0.0169,
        '22': 0.8123, '23': 0.0115, '24': 0.7252, '25': 0.4874,
        '26': 0.8682, '27': 0.4304, '28': 1.0174, '29': 0.9296,
        '30': 0.6582, '31': 1.1057, '32': 0.9837, '33': 0.9347,
        '34': 0.8665, '35': 1.1563, '36': 0.8512, '37': 0.9120,
        '38': 0.5005, '39': 0.4564, '40': 1.3410, '41': 1.0340,
    }
}

# Coeficientes do GLM Gamma (severidade)
coef_sev = {
    'sexo_bin':       0.3684,
    'faixa_etaria':   0.1513,
    'idade_veiculo':  0.1075,
    'log_is_media':   0.6645,
    'regiao': {
        '01': 0.0000,  # referência
        '02': 0.6780, '03': 0.5944, '04': 0.7225, '05': 0.7187,
        '06': 0.6609, '07': 0.7066, '08': 0.6826, '09': 0.6363,
        '10': 0.6362, '11': 0.6933, '12': 0.6264, '13': 0.5730,
        '14': 1.2785, '15': 0.5827, '16': 0.6506, '17': 0.5924,
        '18': 0.7760, '19': 0.8084, '20': 1.7557, '21': 1.0770,
        '22': 0.3564, '23': 0.6584, '24': 0.6049, '25': 0.4978,
        '26': 0.6034, '27': 0.6493, '28': 0.8312, '29': 0.7410,
        '30': 0.8104, '31': 0.7082, '32': 0.8651, '33': 0.5103,
        '34': 1.1415, '35': 0.7650, '36': 0.7616, '37': 0.6240,
        '38': 0.5731, '39': 0.7304, '40': 0.8211, '41': 0.6449,
    }
}

def calcular_premio(sexo, faixa_label, regiao_label, ano_modelo, is_media):
    sexo_bin      = 1 if sexo == "Masculino" else 0
    faixa_etaria  = faixas_etarias[faixa_label]
    regiao_cod    = regioes[regiao_label]
    idade_veiculo = np.clip(2021 - ano_modelo, 0, 30)
    log_is        = np.log1p(is_media)

    # Frequência (Poisson com link log)
    eta_freq = (
        coef_freq['const'] +
        coef_freq['sexo_bin']      * sexo_bin +
        coef_freq['faixa_etaria']  * faixa_etaria +
        coef_freq['idade_veiculo'] * idade_veiculo +
        coef_freq['log_is_media']  * log_is +
        coef_freq['regiao'].get(regiao_cod, 0)
    )
    freq = np.exp(eta_freq)

    # Severidade (Gamma com link log)
    eta_sev = (
        coef_sev['sexo_bin']      * sexo_bin +
        coef_sev['faixa_etaria']  * faixa_etaria +
        coef_sev['idade_veiculo'] * idade_veiculo +
        coef_sev['log_is_media']  * log_is +
        coef_sev['regiao'].get(regiao_cod, 0)
    )
    sev = np.exp(eta_sev)

    return freq, sev, freq * sev

if st.button("🧮 Calcular Prêmio Puro", type="primary", use_container_width=True):
    freq, sev, premio = calcular_premio(
        sexo, faixa_label, regiao_label, ano_modelo, is_media
    )

    st.subheader("📊 Resultado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Frequência de Sinistro", f"{freq:.4f}", help="Probabilidade de sinistro por veículo-ano")
    c2.metric("Severidade Média", f"R$ {sev:,.2f}", help="Custo médio por sinistro")
    c3.metric("💰 Prêmio Puro", f"R$ {premio:,.2f}", help="Frequência × Severidade")

    st.info(f"""
    **Interpretação:** Para um veículo com este perfil, espera-se **{freq:.2%}** de chance 
    de sinistro de colisão por ano, com custo médio de **R$ {sev:,.2f}** por ocorrência.  
    O prêmio puro estimado é de **R$ {premio:,.2f}** por veículo-ano.
    """)

st.divider()
st.caption("Fonte: SUSEP AUTOSEG 2019-2021 | Modelo: GLM Poisson × Gamma | Autor: Arthur Pontes Motta")