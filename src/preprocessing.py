# ============================================================
# src/preprocessing.py
# Funções de limpeza e feature engineering
# ============================================================

import pandas as pd
import numpy as np


def load_autoseg(paths: list) -> pd.DataFrame:
    """Carrega e concatena arquivos SUSEP AUTOSEG."""
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=';', encoding='latin1', low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def clean_autoseg(df: pd.DataFrame, df_reg: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros e limpeza no dataset AUTOSEG.
    - Remove pessoa jurídica e não informado
    - Remove regiões inválidas
    - Remove faixa etária inválida
    """
    # Apenas pessoa física
    df = df[df['sexo'].isin(['M', 'F'])].copy()

    # Faixa etária válida
    df = df[df['faixa_etaria'] > 0].copy()

    # Padronizar código de região
    df['regiao_cod'] = df['regiao_cod'].astype(str).str.strip().str.zfill(2)

    # Regiões válidas
    regioes_validas = df_reg['codigo'].astype(str).str.zfill(2).unique()
    df = df[df['regiao_cod'].isin(regioes_validas)].copy()

    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame, ano_ref: int = 2021) -> pd.DataFrame:
    """
    Constrói features para modelagem GLM/ML.
    """
    df = df.copy()

    # Variáveis resposta
    df['freq_colisao_rel'] = df['freq_colisao'] / df['exposicao']
    df['severidade_colisao'] = np.where(
        df['freq_colisao'] > 0,
        df['indeniz_colisao'] / df['freq_colisao'],
        np.nan
    )

    # Features
    df['sexo_bin'] = (df['sexo'] == 'M').astype(int)
    df['idade_veiculo'] = (ano_ref - df['ano_modelo']).clip(0, 30)
    df['log_is_media'] = np.log1p(df['is_media'])

    # Outlier removal (p99 em freq_colisao_rel)
    p99 = df['freq_colisao_rel'].quantile(0.99)
    df = df[df['freq_colisao_rel'] <= p99].copy()

    # Dummies de região
    df = pd.get_dummies(df, columns=['regiao_cod'], prefix='regiao', drop_first=True)

    return df.reset_index(drop=True)


def train_test_split_temporal(df: pd.DataFrame, test_year: int = 2021):
    """Split temporal sem data leakage."""
    df_train = df[df['ano'] < test_year].copy()
    df_test  = df[df['ano'] >= test_year].copy()
    return df_train, df_test