# 🚗 Insurance Pricing Model — SUSEP AUTOSEG

> Actuarial pricing model for Brazilian auto insurance using real regulatory data from SUSEP (2019–2021).  
> Modelo atuarial de precificação de seguro auto com dados reais da SUSEP (2019–2021).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-SUSEP%20AUTOSEG-orange)

---

## 📌 Overview | Visão Geral

This project builds an end-to-end auto insurance pricing model using **12.6 million policy records** from Brazil's insurance regulator (SUSEP). It combines classical actuarial methods (GLM) with modern machine learning (XGBoost + SHAP) and deploys an interactive pricing calculator via Streamlit.

Este projeto constrói um modelo completo de precificação de seguro auto utilizando **12,6 milhões de registros de apólices** da SUSEP. Combina métodos atuariais clássicos (GLM) com machine learning moderno (XGBoost + SHAP) e disponibiliza uma calculadora interativa via Streamlit.

---

## 🎯 Business Problem | Problema de Negócio

**EN:** How should an insurer price auto collision coverage given the policyholder's profile (age, gender, region) and vehicle characteristics (model year, insured value)?

**PT:** Como uma seguradora deve precificar a cobertura de colisão dado o perfil do segurado (idade, sexo, região) e as características do veículo (ano do modelo, importância segurada)?

---

## 📊 Data | Dados

| Source | Description | Records |
|--------|-------------|---------|
| SUSEP AUTOSEG 2019B | 2nd semester 2019 | 3,128,606 |
| SUSEP AUTOSEG 2020A | 1st semester 2020 | 3,210,981 |
| SUSEP AUTOSEG 2020B | 2nd semester 2020 | 2,941,865 |
| SUSEP AUTOSEG 2021A | 1st semester 2021 | 3,390,758 |
| **Total** | | **12,672,210** |

Data available at: https://www2.susep.gov.br/menuestatistica/autoseg/principal.aspx

---

## 🔬 Methodology | Metodologia

### Frequency Model (GLM Poisson)
- **Target:** Collision claim frequency (claims / exposure)
- **Link function:** Log with exposure offset
- **Key finding:** Young drivers (18–25) have ~2× higher frequency than drivers over 55

### Severity Model (GLM Gamma)
- **Target:** Average claim cost (indemnity / claims)
- **Link function:** Log
- **Key finding:** Vehicles in Espírito Santo (region 20) show highest severity

### Pure Premium
```
Pure Premium = Frequency × Severity
```

### ML Comparison (XGBoost + SHAP)
- Tweedie objective for zero-inflated data (90.6% zeros)
- SHAP explainability to meet regulatory transparency requirements
- XGBoost outperforms GLM: MAE 0.053 vs 0.057

---

## 📈 Key Insights

- **COVID-19 impact:** Collision frequency dropped sharply in 2020-S1 due to lockdowns, recovering in 2021
- **Age effect:** Each older age band reduces collision frequency by ~12% (GLM coefficient: -0.132)
- **Gender gap:** Male drivers show ~4.8% higher collision frequency
- **Regional risk:** Tocantins (region 40) has the highest frequency; São Paulo metro (region 11) the lowest
- **Vehicle value:** Higher insured amounts correlate with both higher frequency and higher severity

---

## 🗂️ Project Structure
```
insurance-pricing-susep/
├── data/
│   ├── raw/              # SUSEP AUTOSEG data (not versioned)
│   │   └── README.md     # Download instructions
│   └── processed/        # Cleaned datasets (not versioned)
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature Engineering
│   ├── 03_glm_modeling.ipynb      # GLM Poisson + Gamma
│   └── 04_ml_comparison.ipynb     # XGBoost + SHAP
├── app/
│   └── streamlit_app.py           # Interactive pricing calculator
├── reports/
│   └── figures/                   # Generated visualizations
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── modeling.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run | Como Executar
```bash
# Clone the repository
git clone https://github.com/arthurpmotta02/insurance-pricing-susep.git
cd insurance-pricing-susep

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download SUSEP data (see data/raw/README.md)

# Run notebooks in order
jupyter notebook

# Run Streamlit app
streamlit run app/streamlit_app.py
```

---

## 📦 Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
streamlit
statsmodels
pyarrow
jupyter
ipykernel
```

---

## 📸 Screenshots

### EDA — SUSEP AUTOSEG (2019–2021)
![EDA](reports/figures/01_eda_overview.png)

### GLM Evaluation
![GLM](reports/figures/03_glm_avaliacao.png)

### SHAP Feature Importance
![SHAP](reports/figures/04_shap_summary.png)

---

## 👤 Author | Autor

**Arthur Pontes Motta**  
Statistics & Actuarial Science — UFRJ  
[GitHub](https://github.com/arthurpmotta02) · [LinkedIn](https://linkedin.com/in/arthurpmotta)

---

## 📄 License

MIT License — feel free to use and adapt with attribution.