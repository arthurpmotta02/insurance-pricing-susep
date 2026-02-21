# ============================================================
# src/modeling.py
# Funções de modelagem GLM e métricas atuariais
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, Gamma
from statsmodels.genmod.families.links import Log
from sklearn.metrics import mean_absolute_error, mean_squared_error


def fit_glm_poisson(X: pd.DataFrame, y: pd.Series, offset: pd.Series,
                     sample_size: int = 500_000, random_state: int = 42):
    """
    Ajusta GLM Poisson com offset de exposição.
    Usa amostragem para evitar MemoryError com datasets grandes.
    """
    if len(X) > sample_size:
        idx = X.sample(n=sample_size, random_state=random_state).index
        X, y, offset = X.loc[idx], y.loc[idx], offset.loc[idx]

    # Corrigir NaN em idade_veiculo
    X = X.copy()
    X['idade_veiculo'] = X['idade_veiculo'].fillna(X['idade_veiculo'].median())
    X = sm.add_constant(X)

    model = sm.GLM(
        y, X,
        family=Poisson(link=Log()),
        offset=np.log(offset)
    ).fit()

    return model


def fit_glm_gamma(X: pd.DataFrame, y: pd.Series,
                   sample_size: int = 200_000, random_state: int = 42):
    """
    Ajusta GLM Gamma com link log para severidade.
    Requer y > 0. Usa start_params e método Newton para convergência.
    """
    if len(X) > sample_size:
        idx = X.sample(n=sample_size, random_state=random_state).index
        X, y = X.loc[idx], y.loc[idx]

    X = X.astype(np.float64).copy()
    X['idade_veiculo'] = X['idade_veiculo'].fillna(X['idade_veiculo'].median())

    y = y.clip(lower=1, upper=y.quantile(0.99)).astype(np.float64)

    # Remover colunas de variância zero
    cols_zero_var = X.var()[X.var() < 1e-10].index.tolist()
    if cols_zero_var:
        X = X.drop(columns=cols_zero_var)

    start_params = np.zeros(X.shape[1])
    start_params[0] = np.log(y.mean())

    model = sm.GLM(
        y, X,
        family=Gamma(link=Log())
    ).fit(start_params=start_params, method='newton')

    return model


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de Gini normalizado — métrica padrão em precificação atuarial.
    Mede o poder discriminante do modelo.
    Valores: 0 = sem poder preditivo, 1 = discriminação perfeita.
    """
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df = df.sort_values('pred', ascending=False).reset_index(drop=True)

    n = len(df)
    cum_true = df['true'].cumsum() / df['true'].sum()
    cum_pop  = (df.index + 1) / n

    gini = 2 * (cum_true - cum_pop).mean()
    return round(gini, 4)


def double_lift_chart(y_true: pd.Series, pred_glm: pd.Series,
                       pred_ml: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """
    Lift chart por decil de risco.
    Ordena por pred_ml e calcula lift = obs_mean / mean_global.
    """
    df = pd.DataFrame({
        'true': y_true.values,
        'glm':  pred_glm.values,
        'ml':   pred_ml.values,
    }).sort_values('ml', ascending=False).reset_index(drop=True)

    df['decil'] = pd.cut(df.index, bins=n_bins, labels=range(1, n_bins + 1))

    result = df.groupby('decil', observed=True).agg(
        obs_mean=('true', 'mean'),
        glm_mean=('glm',  'mean'),
        ml_mean=('ml',   'mean'),
        n=('true', 'count')
    ).reset_index()

    mean_global = df['true'].mean()
    result['lift'] = result['obs_mean'] / mean_global

    return result


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str = 'Model') -> dict:
    """Métricas completas: MAE, RMSE, Correlação, Gini."""
    metrics = {
        'model':       model_name,
        'mae':         mean_absolute_error(y_true, y_pred),
        'rmse':        np.sqrt(mean_squared_error(y_true, y_pred)),
        'correlation': np.corrcoef(y_true, y_pred)[0, 1],
        'gini':        gini_coefficient(y_true, y_pred)
    }
    return metrics