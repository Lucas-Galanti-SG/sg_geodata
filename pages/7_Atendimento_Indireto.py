"""
Módulo 7 — Atendimento Indireto

Analisa a cobertura de clientes finais a partir da rede de distribuidores
presentes no parquet foco gerado no Módulo 5. As três primeiras seções
replicam o Módulo 6; a seção "Avaliação via Distribuição" acrescenta:
  • Seleção do campo/valor que identifica distribuidores
  • BallTree haversine: distância cliente final→distribuidor mais próximo
  • Mapa de calor dual (azul=dentro do raio · vermelho=fora)
  • KPIs, barras por distribuidor e exportação filtrável
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pyarrow.parquet as pq
import streamlit as st
from sklearn.neighbors import BallTree

from utils.config import get_subfolders
from utils.sidebar import render_sidebar

st.set_page_config(
    page_title="Atendimento Indireto · SGGeoData",
    page_icon="🏭",
    layout="wide",
)
render_sidebar()
st.title("🏭 Módulo 7 — Atendimento Indireto")
st.caption(
    "Selecione um parquet foco, defina o contexto da carteira e avalie "
    "a cobertura da sua rede de distribuidores sobre os clientes finais."
)

# ---------------------------------------------------------------------------
# Constants (espelhados do Módulo 6)
# ---------------------------------------------------------------------------

_KNOWN_COLS: frozenset[str] = frozenset([
    "cnpj", "cnpj_basico", "cnpj_ordem", "cnpj_dv", "matriz_filial",
    "razao_social", "nome_fantasia", "capital_social", "porte", "porte_desc",
    "natureza_juridica", "natureza_juridica_desc",
    "qualificacao_responsavel", "qualificacao_responsavel_desc",
    "situacao_cadastral", "data_situacao_cadastral",
    "motivo_situacao", "motivo_situacao_desc",
    "situacao_especial", "data_situacao_especial",
    "cnae_principal", "cnae_principal_desc", "cnaes_secundarios",
    "data_inicio_atividade",
    "secao", "secao_desc", "divisao", "divisao_desc",
    "grupo", "grupo_desc", "classe", "classe_desc",
    "tipo_logradouro", "logradouro", "numero", "complemento", "bairro",
    "cep", "uf", "municipio", "municipio_nome",
    "nome_cidade_exterior", "pais", "pais_nome",
    "ddd1", "telefone1", "email",
    "opcao_simples", "data_opcao_simples", "data_exclusao_simples",
    "opcao_mei", "data_opcao_mei", "data_exclusao_mei",
    "cliente_atendido", "empresa_base_atendida", "relevancia_valor",
    "lat", "lon", "cd_mun", "CD_MUN",
])

_PT_STOPWORDS: frozenset[str] = frozenset([
    "de", "da", "do", "das", "dos", "e", "em", "a", "o", "os", "as",
    "um", "uma", "com", "para", "por", "que", "se", "na", "no", "nas",
    "nos", "me", "ltda", "eireli", "epp", "sa", "s/a",
    "ss", "sc", "srp", "cia", "industrial", "comercio", "comercial",
    "industria", "servicos", "servico",
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", t).strip()


@st.cache_data(show_spinner=False, ttl=600)
def _parquet_schema(path: str):
    return pq.read_schema(path)


@st.cache_data(show_spinner=False, ttl=600)
def _col_values(path: str, col: str) -> list:
    con = duckdb.connect()
    rows = con.execute(
        f'SELECT DISTINCT "{col}" FROM \'{path}\' WHERE "{col}" IS NOT NULL ORDER BY 1'
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


@st.cache_data(show_spinner=False, ttl=600)
def _load_distributors(path: str, field: str, vals: tuple) -> pd.DataFrame:
    """Carrega TODOS os clientes atendidos cujo `field` esteja em `vals`,
    sem qualquer restrição de CNAE. Usado para carregar distribuidores
    que podem ter CNAEs completamente diferentes dos clientes finais."""
    if not vals:
        return pd.DataFrame()
    ph = ", ".join(f"'{v}'" for v in vals)
    con = duckdb.connect()
    df = con.execute(
        f"SELECT * FROM '{path}' WHERE cliente_atendido = true AND \"{field}\" IN ({ph})"
    ).df()
    con.close()
    return df


@st.cache_data(show_spinner=False)
def _cnae_pareto(path: str, filters: tuple) -> pd.DataFrame:
    where_parts = ["cliente_atendido = true"]
    for col, vals in filters:
        ph = ", ".join(f"'{v}'" for v in vals)
        where_parts.append(f'"{col}" IN ({ph})')
    where = " AND ".join(where_parts)
    has_val = "relevancia_valor" in pq.read_schema(path).names
    val_expr = "SUM(relevancia_valor)" if has_val else "0"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            cnae_principal,
            cnae_principal_desc,
            COUNT(DISTINCT cnpj)   AS n_cnpjs,
            {val_expr}             AS soma_valor
        FROM '{path}'
        WHERE {where}
        GROUP BY cnae_principal, cnae_principal_desc
        ORDER BY n_cnpjs DESC
    """).df()
    con.close()
    return df


@st.cache_data(show_spinner=False)
def _all_subclasses(path: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            cnae_principal       AS subclasse,
            cnae_principal_desc  AS desc,
            COUNT(DISTINCT cnpj) AS n_total,
            COUNT(DISTINCT CASE WHEN cliente_atendido THEN cnpj END) AS n_atend
        FROM '{path}'
        WHERE cnae_principal IS NOT NULL
        GROUP BY cnae_principal, cnae_principal_desc
        ORDER BY n_total DESC
    """).df()
    con.close()
    return df


@st.cache_data(show_spinner=False)
def _market_data(
    path: str,
    subclasses: tuple,
    filters: tuple,
    has_empresa_base: bool,
    expand_empresa: bool,
) -> pd.DataFrame:
    ph = ", ".join(f"'{s}'" for s in subclasses)
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{path}' WHERE LEFT(cnae_principal, 7) IN ({ph})").df()
    con.close()
    if filters:
        attended = df["cliente_atendido"] == True
        ctx_mask = pd.Series(True, index=df.index)
        for col, vals in filters:
            if col in df.columns:
                ctx_mask = ctx_mask & df[col].isin(vals)
        df = df[~attended | (attended & ctx_mask)]
    if has_empresa_base and expand_empresa and "empresa_base_atendida" in df.columns:
        df = df.copy()
        df["cliente_atendido"] = df["cliente_atendido"] | df["empresa_base_atendida"]
    return df


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

foco_path = st.session_state.get("_sb_foco_path")

if not foco_path or not Path(foco_path).exists():
    st.info(
        "👈 Selecione um parquet foco na barra lateral "
        "(expander **📊 Base Foco**) para iniciar a análise."
    )
    st.stop()

FOCO_P = str(foco_path)

try:
    _schema = _parquet_schema(FOCO_P)
    _schema_names = set(_schema.names)
except Exception as e:
    st.error(f"Erro ao ler o parquet: {e}")
    st.stop()

CLASS_COLS = sorted(_schema_names - _KNOWN_COLS)
HAS_LATLON  = "lat" in _schema_names and "lon" in _schema_names
HAS_EMPRESA_BASE = "empresa_base_atendida" in _schema_names

ss = st.session_state

# ---------------------------------------------------------------------------
# Seleção de contexto
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Seleção de contexto")
st.caption(
    "Escolha os campos que definem o contexto da carteira a analisar. "
    "Os filtros identificam os clientes **atendidos** nesse contexto."
)

ss.setdefault("ai_filters", [{"field": CLASS_COLS[0] if CLASS_COLS else "", "values": []}])
ss.setdefault("ai_ctx_ready", False)

if not CLASS_COLS:
    st.warning(
        "O parquet selecionado não possui colunas de classificação de carteira. "
        "Regere o parquet no Módulo 5 selecionando colunas de filtro (canal, segmento, etc.)."
    )
    st.stop()

_to_remove = None
for _idx, _flt in enumerate(ss["ai_filters"]):
    _c1, _c2, _c3 = st.columns([2, 4, 0.5])
    with _c1:
        _chosen_field = st.selectbox(
            "Campo",
            options=CLASS_COLS,
            index=CLASS_COLS.index(_flt["field"]) if _flt["field"] in CLASS_COLS else 0,
            key=f"ai_flt_field_{_idx}",
            label_visibility="collapsed",
        )
    with _c2:
        _vals_available = _col_values(FOCO_P, _chosen_field)
        _chosen_vals = st.multiselect(
            "Valores",
            options=_vals_available,
            default=[v for v in _flt["values"] if v in _vals_available],
            key=f"ai_flt_vals_{_idx}",
            label_visibility="collapsed",
            placeholder=f"Todos os valores de «{_chosen_field}»",
        )
    with _c3:
        if st.button("✕", key=f"ai_flt_rm_{_idx}", help="Remover filtro",
                     disabled=len(ss["ai_filters"]) == 1):
            _to_remove = _idx
    ss["ai_filters"][_idx] = {"field": _chosen_field, "values": _chosen_vals}

if _to_remove is not None:
    ss["ai_filters"].pop(_to_remove)
    ss["ai_ctx_ready"] = False
    st.rerun()

_btn1, _btn2, _btn3 = st.columns([1, 1, 4])
with _btn1:
    if st.button("➕ Adicionar filtro", key="ai_add_filter"):
        ss["ai_filters"].append({"field": CLASS_COLS[0], "values": []})
        ss["ai_ctx_ready"] = False
        st.rerun()
with _btn2:
    if st.button("🔍 Analisar CNAEs de clientes", type="primary", key="ai_run_ctx"):
        ss["ai_ctx_ready"] = True
        ss.pop("ai_cnae_result", None)
        ss.pop("ai_subclasses_sel", None)
        ss.pop("ai_market_df", None)
        ss.pop("ai_dist_calc_ready", None)

# ---------------------------------------------------------------------------
# CNAEs da carteira atendida
# ---------------------------------------------------------------------------

if ss.get("ai_ctx_ready"):
    st.markdown("---")
    st.subheader("CNAEs da carteira atendida")

    _active_filters = [
        (flt["field"], tuple(flt["values"]))
        for flt in ss["ai_filters"]
        if flt["field"] and flt["values"]
    ]
    _filters_tuple = tuple(_active_filters)

    with st.spinner("Calculando CNAEs da carteira…"):
        _cnae_df = _cnae_pareto(FOCO_P, _filters_tuple)

    if _cnae_df.empty:
        st.warning(
            "Nenhum cliente atendido encontrado com esses filtros. "
            "Verifique os critérios acima ou tente sem filtros de valor."
        )
    else:
        ss["ai_cnae_result"] = _cnae_df

        _top_n = 10
        _total_cnpjs = _cnae_df["n_cnpjs"].sum()
        _df10_cnt = _cnae_df.nlargest(_top_n, "n_cnpjs").copy()
        _df10_val = _cnae_df.nlargest(_top_n, "soma_valor").copy()

        def _pareto_fig(df, x_col, title, x_label):
            df = df.sort_values(x_col, ascending=False).reset_index(drop=True)
            total = df[x_col].sum()
            if total == 0:
                return go.Figure()
            df["pct"] = df[x_col] / total * 100
            df["pct_acc"] = df["pct"].cumsum()

            def _abc_color(acc):
                if acc <= 80:
                    return "#C8A214"   # dourado — A
                if acc <= 95:
                    return "#1B3D8F"   # safira  — B
                return "#9E9E9E"       # cinza   — C

            df["color"] = df["pct_acc"].apply(_abc_color)
            df["label"] = (df["cnae_principal"].astype(str) + " – "
                           + df["cnae_principal_desc"].fillna("").str[:40])
            df["text_lbl"] = df.apply(
                lambda r: f"  {r['pct']:.1f}%  |  acc {r['pct_acc']:.1f}%", axis=1
            )
            df = df.iloc[::-1].reset_index(drop=True)

            fig = go.Figure(go.Bar(
                y=df["label"],
                x=df[x_col],
                orientation="h",
                marker_color=list(df["color"]),
                text=list(df["text_lbl"]),
                textposition="outside",
                cliponaxis=False,
                name=x_label,
            ))
            for _lc, _ln in [("#C8A214", "A ≤ 80%"), ("#1B3D8F", "B 80–95%"), ("#9E9E9E", "C > 95%")]:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=11, color=_lc, symbol="square"),
                    name=_ln, showlegend=True,
                ))
            fig.update_layout(
                title=title,
                height=max(380, len(df) * 42),
                margin=dict(l=10, r=230, t=40, b=10),
                legend=dict(orientation="h", y=1.08, x=0),
                xaxis=dict(showgrid=True, autorange=True),
            )
            return fig

        _ca, _cb = st.columns(2)
        with _ca:
            st.plotly_chart(
                _pareto_fig(_df10_cnt, "n_cnpjs",
                            f"Top {_top_n} CNAEs por nº de clientes (total: {_total_cnpjs:,})",
                            "Clientes"),
                use_container_width=True, key="ai_fig_pareto_cnt")
        with _cb:
            st.plotly_chart(
                _pareto_fig(_df10_val, "soma_valor",
                            "Top 10 CNAEs por valor agregado (relevancia_valor)",
                            "Valor"),
                use_container_width=True, key="ai_fig_pareto_val")

# ---------------------------------------------------------------------------
# Avaliação do mercado similar
# ---------------------------------------------------------------------------

if ss.get("ai_cnae_result") is not None:
    st.markdown("---")
    st.subheader("Avaliação do mercado similar")
    st.caption("Selecione as subclasses de interesse para analisar o mercado completo.")

    with st.spinner("Carregando subclasses…"):
        _all_sub = _all_subclasses(FOCO_P)

    _cnae_res = ss["ai_cnae_result"]
    _rel_order = list(_cnae_res["cnae_principal"].astype(str))
    _all_sub["_rank"] = _all_sub["subclasse"].apply(
        lambda c: _rel_order.index(c) if c in _rel_order else len(_rel_order) + 1
    )
    _all_sub = _all_sub.sort_values("_rank")
    _total_atend_g = max(_cnae_res["n_cnpjs"].sum(), 1)

    def _sub_label(row):
        pct = _cnae_res.loc[_cnae_res["cnae_principal"] == row["subclasse"], "n_cnpjs"].sum()
        pct = pct / _total_atend_g * 100
        return (f"{row['subclasse']} – {str(row['desc'])[:50]}"
                f"  ({row['n_total']:,} CNPJs · {pct:.1f}% da carteira)")

    _opts_map = {_sub_label(r): r["subclasse"] for _, r in _all_sub.iterrows()}
    _opts_list = list(_opts_map.keys())
    _presel = [lbl for lbl, sub in _opts_map.items() if sub in _rel_order[:10]]

    _sel_labels = st.multiselect(
        "Subclasses de interesse",
        options=_opts_list,
        default=_presel,
        key="ai_sub_multisel",
        placeholder="Digite ou selecione…",
    )

    if st.button("📊 Analisar mercado", type="primary", key="ai_run_market",
                 disabled=not _sel_labels):
        ss["ai_subclasses_sel"] = tuple(_opts_map[lbl] for lbl in _sel_labels)
        ss.pop("ai_market_df", None)
        ss.pop("ai_dist_calc_ready", None)

    if ss.get("ai_subclasses_sel"):
        _filters_1c = tuple(
            (flt["field"], tuple(flt["values"]))
            for flt in ss["ai_filters"]
            if flt["field"] and flt["values"]
        )
        with st.spinner("Carregando dados do mercado…"):
            _mkt_df = _market_data(
                FOCO_P, ss["ai_subclasses_sel"], _filters_1c,
                HAS_EMPRESA_BASE, expand_empresa=False,
            )
        ss["ai_market_df"] = _mkt_df

        if _mkt_df.empty:
            st.warning("Nenhum registro encontrado para as subclasses selecionadas.")
        else:
            _atend_m  = _mkt_df[_mkt_df["cliente_atendido"] == True]
            _natend_m = _mkt_df[_mkt_df["cliente_atendido"] != True]

            # Stacked bars por subclasse
            _sagg = (_mkt_df.groupby(["cnae_principal", "cnae_principal_desc", "cliente_atendido"])
                     ["cnpj"].nunique().reset_index(name="n"))
            _top10_sub = _sagg.groupby("cnae_principal")["n"].sum().nlargest(10).index
            _sagg10 = _sagg[_sagg["cnae_principal"].isin(_top10_sub)].copy()
            _sagg10["lbl"] = (_sagg10["cnae_principal"].astype(str) + " – "
                              + _sagg10["cnae_principal_desc"].fillna("").str[:35])
            _fig_sub = go.Figure()
            for _flag, _col_c, _lbl in [(True, "#0079c1", "Atendido"), (False, "#ff6b35", "Não atendido")]:
                _g = _sagg10[_sagg10["cliente_atendido"] == _flag]
                _fig_sub.add_trace(go.Bar(y=_g["lbl"], x=_g["n"], orientation="h",
                                          name=_lbl, marker_color=_col_c))
            _fig_sub.update_layout(barmode="stack",
                                   title="CNPJs atendidos vs não atendidos por subclasse",
                                   height=400, margin=dict(l=10, r=10, t=40, b=10),
                                   legend=dict(orientation="h", y=1.06))
            st.plotly_chart(_fig_sub, use_container_width=True, key="ai_fig_sub_stack")

            # CNPJs por UF
            _uf_agg = (_mkt_df.groupby("uf")["cnpj"].nunique()
                       .reset_index(name="n_cnpjs").sort_values("n_cnpjs", ascending=False))
            _fig_uf = px.bar(_uf_agg, x="uf", y="n_cnpjs",
                             title="CNPJs por UF", labels={"uf": "UF", "n_cnpjs": "CNPJs"},
                             color_discrete_sequence=["#0079c1"])
            _fig_uf.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(_fig_uf, use_container_width=True, key="ai_fig_uf_total")

            if "municipio_nome" in _mkt_df.columns:
                _ctop = (_mkt_df.groupby(["municipio_nome", "uf"])["cnpj"].nunique()
                         .reset_index(name="n").sort_values("n", ascending=False).head(10))
                st.markdown("**Top 10 cidades por concentração de CNPJs**")
                st.dataframe(_ctop.rename(columns={"municipio_nome": "Cidade", "uf": "UF",
                                                   "n": "CNPJs"}),
                             use_container_width=True, hide_index=True)

            # Top 20 palavras (não atendidos)
            if "nome_fantasia" in _natend_m.columns:
                _words: dict[str, int] = {}
                for _nf in _natend_m["nome_fantasia"].dropna():
                    for w in _norm(_nf).split():
                        if len(w) > 2 and w not in _PT_STOPWORDS:
                            _words[w] = _words.get(w, 0) + 1
                if _words:
                    _wdf = pd.Series(_words).sort_values(ascending=False).head(20).reset_index()
                    _wdf.columns = ["palavra", "freq"]
                    _wdf = _wdf.sort_values("freq", ascending=True)
                    _fig_w = px.bar(_wdf, x="freq", y="palavra", orientation="h",
                                    title="Top 20 palavras em Nome Fantasia (não atendidos)",
                                    labels={"freq": "Frequência", "palavra": ""},
                                    color_discrete_sequence=["#ff6b35"])
                    _fig_w.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(_fig_w, use_container_width=True, key="ai_fig_words")

# ---------------------------------------------------------------------------
# Avaliação via Distribuição
# ---------------------------------------------------------------------------

if ss.get("ai_subclasses_sel") and ss.get("ai_market_df") is not None:
    st.markdown("---")
    st.subheader("Avaliação via Distribuição")
    st.caption(
        "Identifique os distribuidores na carteira atendida, defina o raio de cobertura "
        "e veja quais clientes finais estão cobertos pela rede."
    )

    if not HAS_LATLON:
        st.warning(
            "O parquet foco não possui colunas **lat** e **lon**. "
            "Regere o parquet no Módulo 5 com enriquecimento de coordenadas para usar esta seção."
        )
        st.stop()

    # -- Configuração do distribuidor --
    st.markdown("##### Identificação dos distribuidores")
    _col_d1, _col_d2, _col_d3 = st.columns([2, 4, 2])
    with _col_d1:
        _dist_field = st.selectbox(
            "Campo que identifica o distribuidor",
            options=CLASS_COLS,
            key="ai_dist_field_sel",
        )
    with _col_d2:
        _dist_vals_avail = _col_values(FOCO_P, _dist_field)
        _dist_vals = st.multiselect(
            "Valores que indicam um distribuidor",
            options=_dist_vals_avail,
            key="ai_dist_vals_sel",
            placeholder="Selecione os valores que correspondem a distribuidores…",
        )
    with _col_d3:
        _raio_km = st.number_input(
            "Raio de cobertura (km)",
            min_value=1.0,
            max_value=5000.0,
            value=150.0,
            step=10.0,
            key="ai_raio_km",
        )

    # -- Checkboxes e keywords --
    st.markdown("##### Opções de filtragem")
    _cc1, _cc2, _cc3 = st.columns(3)
    with _cc1:
        _skip_blank = st.checkbox(
            "Ignorar CNPJs com Nome Fantasia em branco",
            key="ai_skip_blank_nf", value=True,
        )
    with _cc2:
        _expand_empr = st.checkbox(
            "Considerar empresa_base_atendida como atendida",
            key="ai_expand_empresa",
            value=True,
            disabled=not HAS_EMPRESA_BASE,
            help="Sinaliza como atendido qualquer CNPJ de empresa que já tenha outro estabelecimento atendido.",
        )
    with _cc3:
        _use_excl_kw = st.checkbox(
            "Excluir clientes finais por palavras-chave",
            key="ai_excl_kw_on", value=True,
        )

    ss.setdefault("ai_excl_kw", [])
    if _use_excl_kw:
        _kw_c1, _kw_c2 = st.columns([4, 1])
        with _kw_c1:
            _new_kw = st.text_input(
                "Palavra-chave (Enter para adicionar)", key="ai_excl_kw_input",
                placeholder="Ex: matriz, holding…",
            )
        with _kw_c2:
            st.write(""); st.write("")
            if st.button("➕ Incluir", key="ai_excl_kw_add") and _new_kw.strip():
                _nkw = _norm(_new_kw.strip())
                if _nkw and _nkw not in ss["ai_excl_kw"]:
                    ss["ai_excl_kw"].append(_nkw)
        if ss["ai_excl_kw"]:
            st.caption("Excluindo: " + " · ".join(f"`{k}`" for k in ss["ai_excl_kw"]))
            if st.button("🗑️ Limpar keywords", key="ai_excl_kw_clear"):
                ss["ai_excl_kw"] = []
                st.rerun()

    # -- Botão Calcular --
    ss.setdefault("ai_dist_calc_ready", False)
    if st.button("🔄 Calcular distribuição", type="primary", key="ai_dist_calc_btn",
                 disabled=not _dist_vals):
        ss["ai_dist_calc_ready"]   = True
        ss["_ai_dist_field"]       = _dist_field
        ss["_ai_dist_vals"]        = list(_dist_vals)
        ss["_ai_raio_km"]          = float(_raio_km)
        ss["_ai_skip_blank"]       = _skip_blank
        ss["_ai_expand_empr"]      = _expand_empr
        ss["_ai_use_excl_kw"]      = _use_excl_kw
        ss["_ai_excl_kw_snap"]     = list(ss["ai_excl_kw"])

    if not ss.get("ai_dist_calc_ready"):
        st.info(
            "Configure o campo do distribuidor, o raio e as opções acima, "
            "depois clique em **🔄 Calcular distribuição** para gerar o relatório."
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Cálculo
    # -----------------------------------------------------------------------

    _snap_field   = ss["_ai_dist_field"]
    _snap_vals    = ss["_ai_dist_vals"]
    _snap_raio    = ss["_ai_raio_km"]
    _snap_skip    = ss["_ai_skip_blank"]
    _snap_expand  = ss["_ai_expand_empr"]
    _snap_excl_kw = ss.get("_ai_excl_kw_snap", [])

    # ── Distribuidores ─────────────────────────────────────────────────────
    # Carregados DIRETAMENTE do parquet, sem restrição de CNAE, pois atacadistas
    # têm CNAEs completamente diferentes dos clientes finais (varejo, revendas).
    with st.spinner("Carregando distribuidores da base atendida…"):
        _df_distrib = _load_distributors(
            FOCO_P, _snap_field, tuple(_snap_vals)
        ).copy()

    # ── Clientes finais + diretos ────────────────────────────────────────────
    # Carregados via _market_data (restrito às CNAEs selecionadas na seção anterior).
    with st.spinner("Carregando clientes finais do mercado selecionado…"):
        _df_raw = _market_data(
            FOCO_P, ss["ai_subclasses_sel"], (),
            HAS_EMPRESA_BASE, expand_empresa=_snap_expand,
        ).copy()

    # Aplicar filtros de limpeza apenas sobre o universo de clientes (não distribuidores)
    if _snap_skip and "nome_fantasia" in _df_raw.columns:
        _is_atend_raw = _df_raw["cliente_atendido"] == True
        _df_raw = _df_raw[
            _is_atend_raw
            | (_df_raw["nome_fantasia"].notna() & (_df_raw["nome_fantasia"].str.strip() != ""))
        ]

    if _snap_excl_kw and "nome_fantasia" in _df_raw.columns:
        _norm_nf = _df_raw["nome_fantasia"].fillna("").apply(_norm)
        _excl_mask = _norm_nf.apply(lambda nf: any(kw in nf for kw in _snap_excl_kw))
        _is_atend = _df_raw["cliente_atendido"] == True
        _df_raw = _df_raw[_is_atend | (~_is_atend & ~_excl_mask)]

    # Clientes finais = não atendidos dentro das CNAEs selecionadas
    _df_finais = _df_raw[_df_raw["cliente_atendido"] != True].copy()
    # Diretos = atendidos cujo campo NÃO é distribuidor (pode ser empty se CNAEs divergem)
    _df_diretos = _df_raw[
        (_df_raw["cliente_atendido"] == True)
        & (~_df_raw.get(_snap_field, pd.Series(dtype=str)).isin(_snap_vals))
    ].copy()

    if _df_distrib.empty:
        st.warning(
            f"Nenhum distribuidor encontrado com **{_snap_field}** ∈ "
            + ", ".join(f"`{v}`" for v in _snap_vals)
            + ". Verifique o campo e os valores selecionados."
        )
        st.stop()

    if not HAS_LATLON or "lat" not in _df_distrib.columns:
        st.warning("Distribuidores sem coordenadas lat/lon — impossível calcular distâncias.")
        st.stop()

    # Deduplica distribuidores por CNPJ (um ponto por CNPJ, media de lat/lon por municipio)
    _dist_geo = (
        _df_distrib.dropna(subset=["lat", "lon"])
        .groupby("cnpj", as_index=False)
        .agg(
            cnpj_basico=("cnpj_basico", "first"),
            nome_fantasia=("nome_fantasia", "first"),
            razao_social=("razao_social", "first"),
            uf=("uf", "first"),
            municipio_nome=("municipio_nome", "first") if "municipio_nome" in _df_distrib.columns else ("uf", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
    )

    _df_finais_geo = _df_finais.dropna(subset=["lat", "lon"]).copy()
    _df_finais_nogeo = _df_finais[_df_finais["lat"].isna() | _df_finais["lon"].isna()].copy()

    # BallTree haversine
    _EARTH_R = 6371.0
    with st.spinner("Calculando distâncias (BallTree haversine)…"):
        _coords_dist_rad = np.radians(
            _dist_geo[["lat", "lon"]].to_numpy(dtype=float)
        )
        _tree = BallTree(_coords_dist_rad, metric="haversine")

        _coords_fin_rad = np.radians(
            _df_finais_geo[["lat", "lon"]].to_numpy(dtype=float)
        )
        _dist_rad, _idx = _tree.query(_coords_fin_rad, k=1)
        _dist_km = _dist_rad[:, 0] * _EARTH_R
        _nearest_idx = _idx[:, 0]

        _df_finais_geo = _df_finais_geo.copy()
        _df_finais_geo["dist_km_min"]          = _dist_km
        _df_finais_geo["cnpj_distrib_proximo"] = _dist_geo.iloc[_nearest_idx]["cnpj"].values
        _df_finais_geo["cnpj_basico_distrib"]  = _dist_geo.iloc[_nearest_idx]["cnpj_basico"].values
        _df_finais_geo["nome_distrib"]         = _dist_geo.iloc[_nearest_idx]["nome_fantasia"].values
        _df_finais_geo["razao_distrib"]        = _dist_geo.iloc[_nearest_idx]["razao_social"].values

    # Classificação
    _df_finais_geo["atendimento_via"] = np.where(
        _df_finais_geo["dist_km_min"] <= _snap_raio,
        "Dentro do raio",
        "Fora do raio",
    )
    if not _df_finais_nogeo.empty:
        _df_finais_nogeo = _df_finais_nogeo.copy()
        for _col_new in ["dist_km_min","cnpj_distrib_proximo","cnpj_basico_distrib",
                         "nome_distrib","razao_distrib","atendimento_via"]:
            _df_finais_nogeo[_col_new] = None
        _df_finais_nogeo["atendimento_via"] = "Fora do raio"

    _df_diretos = _df_diretos.copy()
    for _col_new in ["dist_km_min","cnpj_distrib_proximo","cnpj_basico_distrib",
                     "nome_distrib","razao_distrib"]:
        _df_diretos[_col_new] = None
    _df_diretos["atendimento_via"] = "Atendido direto"

    _df_result = pd.concat(
        [_df_diretos, _df_finais_geo, _df_finais_nogeo], ignore_index=True
    )

    # -----------------------------------------------------------------------
    # Mapa
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("**Mapa de cobertura via distribuidores**")

    _MAP_GRID = 0.05
    _fin_dentro = _df_finais_geo[_df_finais_geo["atendimento_via"] == "Dentro do raio"]
    _fin_fora   = _df_finais_geo[_df_finais_geo["atendimento_via"] == "Fora do raio"]

    def _grid_agg(df_pts):
        if df_pts.empty:
            return pd.DataFrame(columns=["lat", "lon", "n"])
        _r = df_pts[["lat", "lon"]].copy()
        _r["_lg"] = (_r["lat"] / _MAP_GRID).round() * _MAP_GRID
        _r["_og"] = (_r["lon"] / _MAP_GRID).round() * _MAP_GRID
        return (
            _r.groupby(["_lg", "_og"], sort=False).size()
            .reset_index(name="n").rename(columns={"_lg": "lat", "_og": "lon"})
        )

    _grid_dentro = _grid_agg(_fin_dentro)
    _grid_fora   = _grid_agg(_fin_fora)

    # ── Seleção de distribuidores visíveis no mapa ────────────────────────
    _has_relval_dist = (
        "relevancia_valor" in _df_distrib.columns
        and _df_distrib["relevancia_valor"].notna().any()
    )
    if _has_relval_dist:
        _valor_por_empresa = (
            _df_distrib.groupby("cnpj_basico")["relevancia_valor"]
            .sum().reset_index()
            .rename(columns={"relevancia_valor": "soma_valor"})
        )
    else:
        _valor_por_empresa = pd.DataFrame({
            "cnpj_basico": _dist_geo["cnpj_basico"].unique(),
            "soma_valor": 0.0,
        })

    _dist_empresas = (
        _dist_geo.groupby("cnpj_basico", as_index=False)
        .agg(razao_social=("razao_social", "first"),
             nome_fantasia=("nome_fantasia", "first"))
        .merge(_valor_por_empresa, on="cnpj_basico", how="left")
        .sort_values("soma_valor", ascending=False, na_position="last")
    )
    _dist_empresas["_opt"] = (
        _dist_empresas["razao_social"].fillna("").str[:55]
        + "  [" + _dist_empresas["cnpj_basico"].astype(str) + "]"
    )

    _TODAS_OPT = "◉ Ver todos"
    _mapa_dist_sel = st.multiselect(
        "Distribuidores visíveis no mapa (ordenados por Valor decrescente):",
        options=[_TODAS_OPT] + _dist_empresas["_opt"].tolist(),
        default=[_TODAS_OPT],
        key="ai_mapa_dist_sel",
    )

    if not _mapa_dist_sel or _TODAS_OPT in _mapa_dist_sel:
        _dist_geo_vis = _dist_geo
    else:
        _sel_cb = set(
            _dist_empresas.loc[
                _dist_empresas["_opt"].isin(_mapa_dist_sel), "cnpj_basico"
            ]
        )
        _dist_geo_vis = _dist_geo[_dist_geo["cnpj_basico"].isin(_sel_cb)].copy()

    _vis_suffix = (
        f"{len(_dist_geo_vis):,} de {len(_dist_geo):,} distribuidores visíveis"
        if len(_dist_geo_vis) < len(_dist_geo)
        else f"{len(_dist_geo):,} distribuidores"
    )
    st.caption(
        f"ℹ️ {len(_fin_fora):,} clientes fora do raio · "
        f"{len(_fin_dentro):,} dentro do raio · "
        f"{_vis_suffix} · raio fixo = {_snap_raio:.0f} km"
    )

    _fig_mapa = go.Figure()

    if not _grid_fora.empty:
        _zmax_fora = float(_grid_fora["n"].quantile(0.90)) or 1.0
        _fig_mapa.add_trace(go.Densitymapbox(
            lat=_grid_fora["lat"], lon=_grid_fora["lon"], z=_grid_fora["n"],
            radius=18, colorscale="Reds", zmin=0, zmax=_zmax_fora,
            showscale=False, name="Fora do raio",
            hovertemplate="Fora do raio<br>CNPJs: %{z:.0f}<extra></extra>",
        ))

    if not _grid_dentro.empty:
        _zmax_dentro = float(_grid_dentro["n"].quantile(0.90)) or 1.0
        _fig_mapa.add_trace(go.Densitymapbox(
            lat=_grid_dentro["lat"], lon=_grid_dentro["lon"], z=_grid_dentro["n"],
            radius=18, colorscale="Blues", zmin=0, zmax=_zmax_dentro,
            showscale=False, name="Dentro do raio",
            hovertemplate="Dentro do raio<br>CNPJs: %{z:.0f}<extra></extra>",
        ))

    # Pontos dos distribuidores (apenas os selecionados no multiselect)
    _fig_mapa.add_trace(go.Scattermapbox(
        lat=_dist_geo_vis["lat"],
        lon=_dist_geo_vis["lon"],
        mode="markers",
        marker=dict(size=12, color="#0079c1", symbol="circle"),
        name="Distribuidores",
        text=_dist_geo_vis.apply(
            lambda r: (
                f"<b>{r.get('nome_fantasia') or r.get('razao_social', r['cnpj'])}</b><br>"
                f"CNPJ: {r['cnpj']}<br>"
                f"Raio: {_snap_raio:.0f} km"
            ),
            axis=1,
        ),
        hovertemplate="%{text}<extra></extra>",
    ))

    _fig_mapa.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": -14.2, "lon": -51.9},
            zoom=3,
        ),
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h", bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            y=0.01, x=0.01,
        ),
    )
    st.plotly_chart(_fig_mapa, use_container_width=True, key="ai_fig_mapa")

    # -----------------------------------------------------------------------
    # KPIs
    # -----------------------------------------------------------------------
    st.markdown("---")
    _tot_fin   = _df_result["cnpj"].nunique()
    _n_direto  = _df_result.loc[_df_result["atendimento_via"] == "Atendido direto", "cnpj"].nunique()
    _n_dentro  = _df_result.loc[_df_result["atendimento_via"] == "Dentro do raio",  "cnpj"].nunique()
    _n_fora    = _df_result.loc[_df_result["atendimento_via"] == "Fora do raio",    "cnpj"].nunique()

    _pct_dir = _n_direto / _tot_fin * 100 if _tot_fin else 0
    _pct_den = _n_dentro / _tot_fin * 100 if _tot_fin else 0
    _pct_for = _n_fora   / _tot_fin * 100 if _tot_fin else 0

    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.metric("Clientes finais (CNAEs)", f"{_tot_fin:,}")
    _k2.metric("Atendidos direto",        f"{_n_direto:,}", f"{_pct_dir:.1f}%")
    _k3.metric("Dentro do raio",          f"{_n_dentro:,}", f"{_pct_den:.1f}%")
    _k4.metric("Fora do raio",            f"{_n_fora:,}",   f"{_pct_for:.1f}%")

    # -----------------------------------------------------------------------
    # Gráfico de barras horizontais por distribuidor
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("**Clientes finais mais próximos de cada distribuidor**")

    _uf_opts = sorted(_df_result["uf"].dropna().unique().tolist())
    _uf_sel = st.multiselect(
        "Filtrar por UF (deixe vazio para todas)",
        options=_uf_opts,
        key="ai_uf_distrib_filter",
        placeholder="Todas as UFs",
    )

    _df_plot = _df_result.copy()
    if _uf_sel:
        _df_plot = _df_plot[_df_plot["uf"].isin(_uf_sel)]

    _df_finais_plot = _df_plot[
        _df_plot["atendimento_via"].isin(["Dentro do raio", "Fora do raio"])
        & _df_plot["cnpj_basico_distrib"].notna()
    ].copy()

    if not _df_finais_plot.empty:
        # Label de TODOS os distribuidores (inclusive os sem clientes próximos)
        _all_dist_lbl = _dist_geo[["cnpj_basico", "razao_social"]].copy()
        _all_dist_lbl["dist_label"] = (
            _all_dist_lbl["razao_social"].fillna("").str[:40]
            + " ["
            + _all_dist_lbl["cnpj_basico"].astype(str)
            + "]"
        )

        _df_finais_plot = _df_finais_plot.merge(
            _all_dist_lbl[["cnpj_basico", "dist_label"]].rename(
                columns={"cnpj_basico": "cnpj_basico_distrib"}
            ),
            on="cnpj_basico_distrib", how="left",
        )

        _dist_agg = (
            _df_finais_plot
            .groupby(["dist_label", "atendimento_via"])["cnpj"]
            .nunique()
            .reset_index(name="n")
        )

        # Ordenar por "Dentro do raio" decrescente — maior no topo do gráfico horizontal
        _all_labels = _all_dist_lbl["dist_label"].tolist()
        _dentro_cnt = (
            _dist_agg[_dist_agg["atendimento_via"] == "Dentro do raio"]
            .set_index("dist_label")["n"]
            .reindex(_all_labels, fill_value=0)
        )
        _ordered_labels = list(_dentro_cnt.sort_values(ascending=True).index)

        _fig_distrib = go.Figure()
        for _av, _colour in [("Dentro do raio", "#0079c1"), ("Fora do raio", "#ff6b35")]:
            _g = (
                _dist_agg[_dist_agg["atendimento_via"] == _av]
                .set_index("dist_label")["n"]
                .reindex(_ordered_labels, fill_value=0)
            )
            _fig_distrib.add_trace(go.Bar(
                y=_ordered_labels, x=list(_g.values),
                orientation="h", name=_av, marker_color=_colour,
            ))
        _n_bars = len(_ordered_labels)
        _fig_distrib.update_layout(
            barmode="stack",
            title=f"Todos os {_n_bars} distribuidores — clientes finais mais próximos (top 10 visíveis; scroll/zoom para ver todos)",
            height=400,
            yaxis=dict(
                range=[_n_bars - 10.5, _n_bars - 0.5],  # janela inicial: top 10 no raio
                autorange=False,
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(_fig_distrib, use_container_width=True, key="ai_fig_distrib_bars",
                        config={"scrollZoom": True})
    else:
        st.info("Nenhum cliente final com distribuidor associado para exibir.")

    # -----------------------------------------------------------------------
    # Tabela de exportação
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Exportação")

    _export_mode = st.radio(
        "Filtrar exportação por:",
        options=["Apenas fora do raio", "Distribuidor específico", "Todos os clientes finais"],
        horizontal=True,
        key="ai_export_mode",
    )

    _df_exp = _df_result[
        _df_result["atendimento_via"].isin(["Dentro do raio", "Fora do raio"])
    ].copy()

    if _export_mode == "Apenas fora do raio":
        _df_exp = _df_exp[_df_exp["atendimento_via"] == "Fora do raio"]
    elif _export_mode == "Distribuidor específico":
        _dist_options = (
            _df_result[_df_result["cnpj_basico_distrib"].notna()]
            .groupby("cnpj_basico_distrib")
            .agg(razao=("razao_distrib", "first"))
            .reset_index()
        )
        _dist_options["label"] = (
            _dist_options["razao"].fillna("").str[:50]
            + " [" + _dist_options["cnpj_basico_distrib"].astype(str) + "]"
        )
        _sel_dist = st.multiselect(
            "Selecione o(s) distribuidor(es) (CNPJ base)",
            options=_dist_options["label"].tolist(),
            key="ai_export_dist_sel",
            placeholder="Todos os distribuidores",
        )
        if _sel_dist:
            _sel_cnpj_b = _dist_options.loc[
                _dist_options["label"].isin(_sel_dist), "cnpj_basico_distrib"
            ].tolist()
            _df_exp = _df_exp[_df_exp["cnpj_basico_distrib"].isin(_sel_cnpj_b)]

    # Ordenar por capital social decrescente
    if "capital_social" in _df_exp.columns:
        _df_exp["_cap_n"] = pd.to_numeric(
            _df_exp["capital_social"].astype(str).str.replace(",", "."), errors="coerce"
        ).fillna(0)
        _df_exp = _df_exp.sort_values("_cap_n", ascending=False).drop(columns=["_cap_n"])

    _export_cols = [c for c in [
        "cnpj", "razao_social", "nome_fantasia",
        "cnae_principal", "cnae_principal_desc",
        "capital_social", "porte_desc",
        "uf", "municipio_nome", "cep",
        "atendimento_via",
        "cnpj_distrib_proximo", "nome_distrib",
        "dist_km_min",
    ] if c in _df_exp.columns]

    st.caption(
        f"**{len(_df_exp):,}** CNPJs selecionados · "
        "preview: top 100 por capital social"
    )
    st.dataframe(_df_exp[_export_cols].head(100), use_container_width=True, hide_index=True)

    _csv = _df_exp.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        label="⬇️ Baixar CSV completo",
        data=_csv,
        file_name="atendimento_indireto.csv",
        mime="text/csv",
        key="ai_download_csv",
    )
