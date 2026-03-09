"""
Helpers de leitura e escrita em Parquet (pyarrow/pandas).

Todas as funções operam sobre a pasta de dados configurada em config.py.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path | str, compression: str = "snappy") -> Path:
    """
    Salva um DataFrame como arquivo Parquet.

    Parameters
    ----------
    df : DataFrame a salvar
    path : caminho completo do arquivo (incluindo .parquet)
    compression : 'snappy' (padrão), 'gzip' ou 'none'

    Returns
    -------
    Path do arquivo salvo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=compression, engine="pyarrow")
    return path


def load_parquet(path: Path | str) -> pd.DataFrame:
    """
    Lê um arquivo Parquet e retorna um DataFrame.

    Raises FileNotFoundError se o arquivo não existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo Parquet não encontrado: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def parquet_exists(path: Path | str) -> bool:
    """Retorna True se o arquivo Parquet existir."""
    return Path(path).exists()


def file_size_mb(path: Path | str) -> float:
    """Retorna o tamanho do arquivo em MB (0.0 se não existir)."""
    p = Path(path)
    if p.exists():
        return p.stat().st_size / (1024 * 1024)
    return 0.0


def folder_size_mb(folder: Path | str) -> float:
    """Retorna o tamanho total de todos os arquivos em uma pasta (recursivo), em MB."""
    folder = Path(folder)
    if not folder.exists():
        return 0.0
    total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def free_disk_gb(path: Path | str) -> float:
    """Retorna o espaço livre em disco para o caminho informado, em GB."""
    import shutil
    usage = shutil.disk_usage(Path(path).anchor)
    return usage.free / (1024 ** 3)
