"""
Módulo 3 — CNAEs IBGE

Constrói banco hierárquico de CNAEs via API REST do IBGE.
Seção → Divisão → Grupo → Classe → Subclasse.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from utils.config import get_subfolders
from utils.storage import parquet_exists, file_size_mb
from utils.build_cnae_hierarquia import fetch_subclasses, parse_row
from utils.sidebar import render_sidebar

st.set_page_config(page_title="CNAEs IBGE · SGGeoData", page_icon="📚", layout="wide")
render_sidebar()
st.title("📚 Módulo 3 — CNAEs IBGE")
st.caption("Banco hierárquico de CNAEs via API REST do IBGE · Seção → Divisão → Grupo → Classe → Subclasse")

subs = get_subfolders()
OUTPUT_PATH = subs["cnae_ibge"] / "cnae_hierarquia.parquet"

# ---------------------------------------------------------------------------
# 1. Fonte de dados
# ---------------------------------------------------------------------------

st.subheader("1 · Fonte de dados")
st.markdown(
    "**API REST IBGE:** `https://servicodados.ibge.gov.br/api/v2/cnae/subclasses`  \n"
    "Retorna a hierarquia CNAE completa (1.332 subclasses) em uma única requisição, "
    "sem necessidade de download de PDFs."
)

# ---------------------------------------------------------------------------
# 2. Status do banco gerado
# ---------------------------------------------------------------------------

st.subheader("2 · Status do banco de CNAEs")

if parquet_exists(OUTPUT_PATH):
    size_mb = file_size_mb(OUTPUT_PATH)
    df_cnae = pd.read_parquet(OUTPUT_PATH)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Subclasses", f"{len(df_cnae):,}")
    col2.metric("Seções", df_cnae["secao"].nunique())
    col3.metric("Divisões", df_cnae["divisao"].nunique())
    col4.metric("Tamanho (Parquet)", f"{size_mb:.2f} MB")
    st.success("✅ Banco de CNAEs disponível.")
else:
    st.info("Banco de CNAEs ainda não gerado. Clique em **Construir banco** abaixo.")
    df_cnae = None

# ---------------------------------------------------------------------------
# 3. Construir banco
# ---------------------------------------------------------------------------

st.subheader("3 · Construir banco de CNAEs")

force_rebuild = st.checkbox("Forçar reconstrução mesmo se já existir", value=False)
already_built = parquet_exists(OUTPUT_PATH)

if st.button(
    "🔨 Construir banco de CNAEs",
    disabled=(already_built and not force_rebuild),
    type="primary",
    width="stretch",
):
    log_lines = []
    log_area = st.empty()

    def _log(msg: str) -> None:
        log_lines.append(msg)
        log_area.text_area("Log", value="\n".join(log_lines[-30:]), height=200, disabled=True)

    with st.spinner("Consultando API IBGE..."):
        try:
            _log("Conectando à API IBGE…")
            data = fetch_subclasses()
            _log(f"  {len(data)} subclasses recebidas.")
            _log("Estruturando hierarquia…")
            rows = [parse_row(item) for item in data]
            df_result = pd.DataFrame(rows)
            df_result = df_result.sort_values(
                ["secao", "divisao", "grupo", "classe", "subclasse"]
            ).reset_index(drop=True)
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            df_result.to_parquet(OUTPUT_PATH, index=False)
            _log(f"  Salvo em {OUTPUT_PATH}")
            _log(f"  Seções:    {df_result['secao'].nunique()}")
            _log(f"  Divisões:  {df_result['divisao'].nunique()}")
            _log(f"  Grupos:    {df_result['grupo'].nunique()}")
            _log(f"  Classes:   {df_result['classe'].nunique()}")
            st.success(f"✅ Banco gerado com {len(df_result):,} subclasses!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro: {e}")
            raise

# ---------------------------------------------------------------------------
# 4. Visualizar banco
# ---------------------------------------------------------------------------

if df_cnae is not None and len(df_cnae) > 0:
    st.subheader("4 · Explorar hierarquia de CNAEs")

    col1, col2 = st.columns(2)
    with col1:
        secoes = ["(todas)"] + sorted(df_cnae["secao"].dropna().unique().tolist())
        secao_sel = st.selectbox("Filtrar por Seção:", secoes)

    df_view = df_cnae.copy()
    if secao_sel != "(todas)":
        df_view = df_view[df_view["secao"] == secao_sel]

    with col2:
        divisoes = ["(todas)"] + sorted(df_view["divisao"].dropna().unique().tolist())
        div_sel = st.selectbox("Filtrar por Divisão:", divisoes)

    if div_sel != "(todas)":
        df_view = df_view[df_view["divisao"] == div_sel]

    display_cols = [c for c in [
        "secao", "secao_desc",
        "divisao", "divisao_desc",
        "grupo", "grupo_desc",
        "classe", "classe_desc",
        "subclasse", "subclasse_cod", "subclasse_desc",
    ] if c in df_view.columns]

    st.dataframe(df_view[display_cols].head(200), width="stretch", hide_index=True)

    # Resumo por seção
    st.subheader("5 · Resumo por Seção")
    resumo = (
        df_cnae.groupby(["secao", "secao_desc"])
        .agg(
            divisoes=("divisao", "nunique"),
            grupos=("grupo", "nunique"),
            classes=("classe", "nunique"),
            subclasses=("subclasse", "count"),
        )
        .reset_index()
        .sort_values("secao")
    )
    st.dataframe(resumo, width="stretch", hide_index=True)

