"""
sg_versioning — Utilitários de versionamento para Segmentação SG.

Cada "projeto" é uma pasta dentro de {data}/processed/sinergia/:

    sinergia/
      {project_slug}/
        v000.parquet      ← startpoint (upload original)
        v000.json
        v001.parquet
        v001.json
        ...

Cada JSON de versão tem o esquema:
    {
      "version":       int,
      "label":         str,          # descrição da versão
      "created_at":    str,          # ISO 8601 UTC
      "source_file":   str,          # nome do arquivo original
      "rows":          int,
      "rows_changed":  int,          # em relação à versão anterior
      "col_map":       dict,         # ex: {"cnpj_col": "CNPJ"}
      "changes":       list | null   # preenchido pelo rule-engine futuramente
    }
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.config import get_data_folder

# Coluna target criada/gerenciada por este módulo
SEG_COL = "Segmentação SG"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de caminho
# ─────────────────────────────────────────────────────────────────────────────

def sg_root() -> Path:
    """Pasta raiz onde todos os projetos são armazenados."""
    p = get_data_folder() / "processed" / "sinergia"
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(name: str) -> str:
    """Converte um nome de arquivo/texto em slug seguro para nome de pasta."""
    stem = Path(name).stem
    s = re.sub(r"[^\w\-]", "_", stem)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:60] or "projeto").lower()


def version_paths(project: str, version: int) -> tuple[Path, Path]:
    """Retorna (parquet_path, json_path) para uma versão específica."""
    base = sg_root() / project / f"v{version:03d}"
    return base.with_suffix(".parquet"), base.with_suffix(".json")


# ─────────────────────────────────────────────────────────────────────────────
# Leitura
# ─────────────────────────────────────────────────────────────────────────────

def list_projects() -> list[dict]:
    """
    Lista todos os projetos existentes em disco.
    Retorna lista de dicts ordenada por data de criação (mais recente primeiro).
    """
    root = sg_root()
    projects = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        metas = sorted(d.glob("v*.json"))
        if not metas:
            continue
        try:
            with open(metas[0], encoding="utf-8") as f:
                v0 = json.load(f)
            with open(metas[-1], encoding="utf-8") as f:
                vn = json.load(f)
            projects.append({
                "slug":           d.name,
                "source_file":    v0.get("source_file", d.name),
                "n_versions":     len(metas),
                "latest_version": vn["version"],
                "latest_label":   vn["label"],
                "latest_at":      vn["created_at"],
                "created_at":     v0["created_at"],
            })
        except Exception:
            continue
    # Mais recentes primeiro
    return sorted(projects, key=lambda p: p["created_at"], reverse=True)


def list_versions(project: str) -> list[dict]:
    """Lista todas as versões de um projeto em ordem crescente (v0 primeiro)."""
    folder = sg_root() / project
    versions = []
    for meta_path in sorted(folder.glob("v*.json")):
        try:
            with open(meta_path, encoding="utf-8") as f:
                versions.append(json.load(f))
        except Exception:
            continue
    return versions


@st.cache_data(show_spinner=False)
def _load_parquet_cached(pq_path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(pq_path_str)


def load_version(project: str, version: int) -> pd.DataFrame:
    """Carrega o DataFrame de uma versão específica (com cache por mtime)."""
    pq_path, _ = version_paths(project, version)
    mtime = pq_path.stat().st_mtime if pq_path.exists() else 0.0
    return _load_parquet_cached(str(pq_path), mtime).copy()


def load_meta(project: str, version: int) -> dict:
    """Carrega o JSON de metadados de uma versão."""
    _, json_path = version_paths(project, version)
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Escrita
# ─────────────────────────────────────────────────────────────────────────────

def save_version(project: str, df: pd.DataFrame, meta: dict) -> None:
    """
    Persiste um novo snapshot de versão:
      1. Salva parquet (SEG_COL sempre como última coluna se presente).
      2. Salva JSON de metadados.
      3. Invalida o cache de leitura.
    """
    folder = sg_root() / project
    folder.mkdir(parents=True, exist_ok=True)
    pq_path, json_path = version_paths(project, meta["version"])

    # Ordena colunas: SEG_COL por último
    other_cols = [c for c in df.columns if c != SEG_COL]
    ordered_cols = other_cols + ([SEG_COL] if SEG_COL in df.columns else [])
    df[ordered_cols].to_parquet(pq_path, index=False, compression="snappy")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    _load_parquet_cached.clear()


def next_version_number(project: str) -> int:
    """Retorna o número da próxima versão disponível para um projeto."""
    existing = list_versions(project)
    return (existing[-1]["version"] + 1) if existing else 0


def build_meta(
    project: str,
    version: int,
    label: str,
    df: pd.DataFrame,
    source_file: str,
    col_map: dict,
    prev_df: pd.DataFrame | None = None,
    changes: list | None = None,
) -> dict:
    """Monta o dict de metadados para uma nova versão."""
    rows_changed = 0
    if prev_df is not None and SEG_COL in df.columns and SEG_COL in prev_df.columns:
        try:
            rows_changed = int(
                (df[SEG_COL].fillna("__null__") != prev_df[SEG_COL].fillna("__null__")).sum()
            )
        except Exception:
            rows_changed = 0

    return {
        "version":      version,
        "label":        label,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "source_file":  source_file,
        "rows":         len(df),
        "rows_changed": rows_changed,
        "col_map":      col_map,
        "changes":      changes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_dt(iso: str) -> str:
    """ISO timestamp → string legível em horário local (sem conversão de fuso)."""
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return iso


def unique_slug(base: str, existing: list[str]) -> str:
    """Garante que o slug seja único adicionando sufixo numérico se necessário."""
    slug = base
    counter = 1
    while slug in existing:
        slug = f"{base}_{counter}"
        counter += 1
    return slug
