# ============================================================
# src/data_loader.py
# Utilitários de carregamento de dados
# ============================================================

import pandas as pd
from pathlib import Path


def load_parquet(path: str) -> pd.DataFrame:
    """Carrega arquivo parquet."""
    return pd.read_parquet(path)


def load_autoseg_raw(data_dir: str) -> pd.DataFrame:
    """
    Carrega todos os arquivos CSV da SUSEP AUTOSEG de um diretório.
    Espera arquivos arq_casco_comp.csv em subpastas por período.
    """
    data_path = Path(data_dir)
    dfs = []

    for csv_file in sorted(data_path.rglob('arq_casco_comp.csv')):
        periodo = csv_file.parent.name
        df = pd.read_csv(
            csv_file,
            sep=';',
            encoding='latin1',
            low_memory=False
        )
        df['periodo'] = periodo
        df['ano'] = int(periodo[:4])
        dfs.append(df)
        print(f"  Carregado: {periodo} — {len(df):,} linhas")

    return pd.concat(dfs, ignore_index=True)