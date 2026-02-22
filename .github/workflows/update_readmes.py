import re

# ============================================================
# Lê o README do projeto
# ============================================================
with open('README.md', 'r', encoding='utf-8') as f:
    project_readme = f.read()

# Extrai seção de Key Findings
findings_match = re.search(
    r'### Key Findings\n(.*?)(?=\n---|\n##)', 
    project_readme, re.DOTALL
)
findings = findings_match.group(1).strip() if findings_match else ""

# Extrai métricas da tabela de resultados
gini_match = re.search(r'Gini.*?\*\*([\d.]+)\*\*', project_readme)
gini = gini_match.group(1) if gini_match else "0.233"

# ============================================================
# Atualiza profile README (arthurpmotta02/README.md)
# ============================================================
profile_readme = f"""# Arthur Pontes Motta

Undergraduate student in **Actuarial Science & Statistics** at UFRJ (Federal University of Rio de Janeiro).  
Focused on actuarial modeling and data science applied to the insurance industry.

[![Portfolio](https://img.shields.io/badge/Portfolio-arthurpmotta02.github.io-blue?style=flat&logo=github)](https://arthurpmotta02.github.io)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-arthurpmotta-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/arthurpmotta)

---

## 🚗 Featured Project

### [Insurance Pricing Model — SUSEP AUTOSEG](https://github.com/arthurpmotta02/insurance-pricing-susep)

End-to-end auto insurance pricing model using real policy records from Brazil's insurance regulator (SUSEP, 2019–2021).

- **GLM Poisson** for claim frequency + **GLM Gamma** for claim severity
- **XGBoost** with Tweedie objective for zero-inflated data (Gini = {gini})
- **SHAP** explainability for regulatory transparency
- Interactive **Streamlit app** — pure premium calculator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://insurance-pricing-susep.streamlit.app)

---

## 🛠️ Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
"""

with open('profile-repo/README.md', 'w', encoding='utf-8') as f:
    f.write(profile_readme)

print("✓ Profile README atualizado")

# ============================================================
# Atualiza portfolio index.md
# ============================================================
portfolio_index = f"""---
layout: default
title: Portfolio
---

## Portfolio

### Actuarial Science & Insurance

<div class="project-card">
<h4><a href="https://github.com/arthurpmotta02/insurance-pricing-susep">🚗 Insurance Pricing Model — SUSEP AUTOSEG</a></h4>
<p>End-to-end auto insurance pricing model built on real policy records from Brazil's insurance regulator (SUSEP, 2019–2021). Combines classical actuarial methods with modern machine learning.</p>

<ul>
<li><strong>GLM Poisson</strong> for claim frequency with log(exposure) offset — standard actuarial assumption for rare, independent events</li>
<li><strong>GLM Gamma</strong> for claim severity — canonical choice for strictly positive, right-skewed cost distributions</li>
<li><strong>XGBoost Tweedie</strong> (variance_power=1.5) handling 90.6% zero-inflated data with early stopping at round 341</li>
<li><strong>SHAP</strong> explainability for regulatory transparency — theoretically grounded feature attributions</li>
<li><strong>Gini = {gini}</strong> — within the 0.20–0.35 range typical for auto insurance frequency models</li>
<li>Key finding: COVID-19 lockdowns caused a ~25% drop in collision frequency in 2020-S1</li>
</ul>

<div class="img-grid">
<img src="https://raw.githubusercontent.com/arthurpmotta02/insurance-pricing-susep/main/reports/figures/01_eda_overview.png" alt="EDA">
<img src="https://raw.githubusercontent.com/arthurpmotta02/insurance-pricing-susep/main/reports/figures/04_shap_summary.png" alt="SHAP">
<img src="https://raw.githubusercontent.com/arthurpmotta02/insurance-pricing-susep/main/reports/figures/04_gini_lift.png" alt="Gini & Lift">
</div>

<div class="badges">
<span class="badge">Python</span>
<span class="badge">GLM Poisson</span>
<span class="badge">GLM Gamma</span>
<span class="badge">XGBoost Tweedie</span>
<span class="badge">SHAP</span>
<span class="badge">Gini {gini}</span>
<span class="badge">Streamlit</span>
<span class="badge">SUSEP</span>
</div>

<div class="project-links">
<a href="https://github.com/arthurpmotta02/insurance-pricing-susep">View on GitHub →</a>
<a href="https://insurance-pricing-susep.streamlit.app">Live Demo →</a>
</div>
</div>
"""

with open('portfolio-repo/index.md', 'w', encoding='utf-8') as f:
    f.write(portfolio_index)

print("✓ Portfolio index.md atualizado")