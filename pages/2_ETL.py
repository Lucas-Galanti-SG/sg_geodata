"""
Módulo 2 — ETL e Base Unificada

Lê os CSVs brutos de Empresas, Estabelecimentos, Sócios e Simples Nacional,
aplica filtros e limpeza, une todas as tabelas e salva como Parquet enriquecido.

Parquets gerados:
  processed/empresas.parquet          – empresas deduplicadas + desc. natureza/porte
  processed/estabelecimentos.parquet  – estabelecimentos limpos (sem joins)
  processed/socios.parquet            – sócios limpos + desc. qualificação (opcional)
  processed/base_unificada.parquet    – estabelecimentos × empresas × auxiliares × simples
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import threading
import time

import streamlit as st
import pandas as pd

from utils.config import get_subfolders
from utils.storage import parquet_exists, load_parquet, file_size_mb, folder_size_mb
from utils.rfb_etl import run_etl
from utils.sidebar import render_sidebar

st.set_page_config(page_title="ETL · SGGeoData", page_icon="🔧", layout="wide")
render_sidebar()
st.title("🔧 Módulo 2 — ETL e Base Unificada")
st.caption("Limpeza e consolidação dos dados da Receita Federal")

subs = get_subfolders()
output_folder = subs["processed"]

# Caminhos dos Parquets gerados
PATHS = {
    "empresas":         output_folder / "empresas.parquet",
    "estabelecimentos": output_folder / "estabelecimentos.parquet",
    "socios":           output_folder / "socios.parquet",
    "unified":          output_folder / "base_unificada.parquet",
}

# ---------------------------------------------------------------------------
# 1. Status dos dados de entrada
# ---------------------------------------------------------------------------

st.subheader("1 · Status das pastas de entrada")

raw_folders = {
    "Empresas":          subs["raw_empresas"],
    "Estabelecimentos":  subs["raw_estab"],
    "Sócios":            subs["raw_socios"],
    "Auxiliares":        subs["raw_aux"],
}
status_rows = []
for label, folder in raw_folders.items():
    n = len(list(folder.glob("*.csv"))) if folder.exists() else 0
    status_rows.append({
        "Pasta":    label,
        "Arquivos": n,
        "Tamanho":  f"{folder_size_mb(folder):.0f} MB" if folder.exists() else "—",
        "Caminho":  str(folder),
    })
st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 2. Status dos Parquets processados
# ---------------------------------------------------------------------------

st.subheader("2 · Parquets processados")

parquet_rows = []
for name, path in PATHS.items():
    exists = parquet_exists(path)
    parquet_rows.append({
        "Arquivo":  path.name,
        "Existe":   "✅" if exists else "❌",
        "Tamanho":  f"{file_size_mb(path):.1f} MB" if exists else "—",
        "Caminho":  str(path),
    })
st.dataframe(pd.DataFrame(parquet_rows), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 3. Configurações do ETL
# ---------------------------------------------------------------------------

st.subheader("3 · Configurações")

col1, col2 = st.columns(2)
with col1:
    only_active = st.checkbox(
        "Apenas estabelecimentos ativos (Situação Cadastral = 2)",
        value=True,
        help="Filtra somente estabelecimentos com situação cadastral ativa.",
    )
    include_socios = st.checkbox(
        "Incluir Sócios (socios.parquet)",
        value=True,
        help="Gera parquet separado de sócios com qualificação decodificada.",
    )
with col2:
    include_simples = st.checkbox(
        "Incluir Simples Nacional / MEI",
        value=True,
        help="Faz join com o arquivo de Simples (~2,8 GB). Pode demorar vários minutos.",
    )
    force_rerun = st.checkbox(
        "Reprocessar mesmo se Parquets já existirem",
        value=False,
    )

st.info(
    "**Colunas do Parquet base_unificada:**  \n"
    "Identidade (cnpj, cnpj_basico, matriz_filial) · Empresa (razao_social, capital_social, porte, natureza) · "
    "Situação (situacao_cadastral, motivo, datas) · Atividade (cnae_principal + desc, cnaes_secundarios) · "
    "Endereço (logradouro, bairro, cep, uf, municipio + nome, pais + nome) · "
    "Contato (ddd1, telefone1, email) · Simples/MEI (opcao_simples, opcao_mei, datas)"
)

# Verifica se já existem parquets principais
main_exist = parquet_exists(PATHS["empresas"]) and parquet_exists(PATHS["unified"])
if main_exist and not force_rerun:
    st.info("✅ Parquets principais já existem. Marque 'Reprocessar' para executar novamente.")

# ---------------------------------------------------------------------------
# 4. Executar ETL
# ---------------------------------------------------------------------------

missing_input = (
    len(list(subs["raw_empresas"].glob("*.csv"))) == 0 or
    len(list(subs["raw_estab"].glob("*.csv"))) == 0
)

if missing_input:
    st.warning("⚠️ Pastas de entrada vazias. Execute o **Módulo 1** primeiro para baixar os dados.")

run_disabled = missing_input or (main_exist and not force_rerun)

if st.button("▶️ Executar ETL", disabled=run_disabled, type="primary", width="stretch"):

    log_lines: list[str] = []
    holder: dict = {"result": None, "error": None, "done": False}

    def _log(msg: str) -> None:
        log_lines.append(msg)

    def _run_etl() -> None:
        try:
            holder["result"] = run_etl(
                empresas_folder=subs["raw_empresas"],
                estab_folder=subs["raw_estab"],
                socios_folder=subs["raw_socios"],
                aux_folder=subs["raw_aux"],
                output_folder=output_folder,
                only_active=only_active,
                include_socios=include_socios,
                include_simples=include_simples,
                log=_log,
            )
        except Exception as exc:
            holder["error"] = exc
        finally:
            holder["done"] = True

    thread = threading.Thread(target=_run_etl, daemon=True)
    thread.start()

    progress  = st.progress(0, text="ETL em andamento…")
    timer_slot = st.empty()
    log_slot   = st.empty()

    t0 = time.time()
    while not holder["done"]:
        elapsed = int(time.time() - t0)
        mins, secs = divmod(elapsed, 60)
        timer_slot.info(
            f"⏱ Processando… **{mins:02d}:{secs:02d}** — o processo está ativo, aguarde."
        )
        log_slot.text_area(
            "Log de execução",
            value="\n".join(log_lines[-50:]),
            height=300,
            disabled=True,
        )
        time.sleep(1)

    elapsed = int(time.time() - t0)
    mins, secs = divmod(elapsed, 60)
    timer_slot.empty()

    if holder["error"]:
        progress.progress(0, text="Erro!")
        st.error(f"Erro durante o ETL: {holder['error']}")
    else:
        result_paths = holder["result"]
        progress.progress(1.0, text=f"ETL concluído em {mins:02d}:{secs:02d}!")
        st.success(f"✅ ETL finalizado em {mins:02d}:{secs:02d}! {len(result_paths)} Parquet(s) gerado(s).")
        for name, path in result_paths.items():
            _log(f"  → {name}: {file_size_mb(path):.1f} MB")

    log_slot.text_area(
        "Log de execução",
        value="\n".join(log_lines[-50:]),
        height=300,
        disabled=True,
    )

# ---------------------------------------------------------------------------
# 5. Preview dos Parquets
# ---------------------------------------------------------------------------

st.subheader("4 · Preview — Base Unificada")

if parquet_exists(PATHS["unified"]):
    @st.cache_data(show_spinner="Carregando amostra da base unificada…")
    def _load_sample() -> tuple:
        import duckdb
        p = str(PATHS["unified"]).replace("\\", "/")
        con = duckdb.connect()
        n_rows = con.execute(f"SELECT count(*) FROM '{p}'").fetchone()[0]
        cols   = con.execute(f"DESCRIBE SELECT * FROM '{p}' LIMIT 1").fetchdf()["column_name"].tolist()
        sample = con.execute(f"SELECT * FROM '{p}' LIMIT 500").fetchdf()
        con.close()
        return sample, (n_rows, len(cols)), cols

    df_sample, shape, cols = _load_sample()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de registros", f"{shape[0]:,}")
    c2.metric("Colunas", shape[1])
    c3.metric("Tamanho (Parquet)", f"{file_size_mb(PATHS['unified']):.1f} MB")

    with st.expander("Colunas disponíveis"):
        st.write(cols)

    st.dataframe(df_sample, width="stretch", hide_index=True)
else:
    st.info("Execute o ETL para ver a base unificada.")

# Preview de Sócios
if parquet_exists(PATHS["socios"]):
    with st.expander("Preview — Sócios"):
        @st.cache_data(show_spinner="Carregando amostra de sócios…")
        def _load_socios_sample() -> tuple:
            df = load_parquet(PATHS["socios"])
            return df.head(200), df.shape

        df_soc, soc_shape = _load_socios_sample()
        st.caption(f"{soc_shape[0]:,} sócios · {soc_shape[1]} colunas · {file_size_mb(PATHS['socios']):.1f} MB")
        st.dataframe(df_soc, width="stretch", hide_index=True)
