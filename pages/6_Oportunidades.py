"""
Módulo 6 — Atendimento Direto

Analisa a presença comercial em uma população de clientes usando o parquet
foco gerado no Módulo 5. Seções progressivas:
  1-a  Seleção de contexto (filtros por campo de classificação)
  1-b  CNAEs da carteira atendida (Pareto por clientes e por valor)
  1-c  Avaliação do mercado similar (mercado total vs atendido)
  1-d  Avaliação de atendimento direto (KPIs, mapas, export)
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from plotly.subplots import make_subplots

from utils.config import get_subfolders
from utils.sidebar import render_sidebar

st.set_page_config(
    page_title="Atendimento Direto · SGGeoData",
    page_icon="🔭",
    layout="wide",
)
render_sidebar()
st.title("🔭 Módulo 6 — Atendimento Direto")
st.caption(
    "Selecione um parquet foco na barra lateral, defina o contexto da sua carteira "
    "e analise o nível de cobertura do seu mercado."
)

# ---------------------------------------------------------------------------
# Constants
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

_COLORS = {"atendido": "#0079c1", "nao_atendido": "#ff6b35"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Normaliza string: remove acentos, lower, sem caracteres especiais."""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", t).strip()


def _safe_key(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", _norm(s))


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
            COUNT(DISTINCT cnpj)          AS n_cnpjs,
            {val_expr}                    AS soma_valor
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
    # Apply context filter: keep non-attended always, keep attended only if matches filter
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
HAS_LATLON = "lat" in _schema_names and "lon" in _schema_names
HAS_EMPRESA_BASE = "empresa_base_atendida" in _schema_names

ss = st.session_state

# ---------------------------------------------------------------------------
# 1-a — Seleção de contexto
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Seleção de contexto")
st.caption(
    "Escolha os campos que definem o contexto da carteira a analisar. "
    "Os filtros identificam os clientes **atendidos** nesse contexto."
)

ss.setdefault("ad_filters", [{"field": CLASS_COLS[0] if CLASS_COLS else "", "values": []}])
ss.setdefault("ad_ctx_ready", False)

if not CLASS_COLS:
    st.warning(
        "O parquet selecionado não possui colunas de classificação de carteira. "
        "Regere o parquet no Módulo 5 selecionando colunas de filtro (canal, segmento, etc.)."
    )
    st.stop()

to_remove = None
for _idx, _flt in enumerate(ss["ad_filters"]):
    _c1, _c2, _c3 = st.columns([2, 4, 0.5])
    with _c1:
        _chosen_field = st.selectbox(
            "Campo",
            options=CLASS_COLS,
            index=CLASS_COLS.index(_flt["field"]) if _flt["field"] in CLASS_COLS else 0,
            key=f"ad_flt_field_{_idx}",
            label_visibility="collapsed",
        )
    with _c2:
        _vals_available = _col_values(FOCO_P, _chosen_field)
        _chosen_vals = st.multiselect(
            "Valores",
            options=_vals_available,
            default=[v for v in _flt["values"] if v in _vals_available],
            key=f"ad_flt_vals_{_idx}",
            label_visibility="collapsed",
            placeholder=f"Todos os valores de «{_chosen_field}»",
        )
    with _c3:
        if st.button("✕", key=f"ad_flt_rm_{_idx}", help="Remover filtro",
                     disabled=len(ss["ad_filters"]) == 1):
            to_remove = _idx
    ss["ad_filters"][_idx] = {"field": _chosen_field, "values": _chosen_vals}

if to_remove is not None:
    ss["ad_filters"].pop(to_remove)
    ss["ad_ctx_ready"] = False
    st.rerun()

_btn1, _btn2, _btn3 = st.columns([1, 1, 4])
with _btn1:
    if st.button("➕ Adicionar filtro", key="ad_add_filter"):
        ss["ad_filters"].append({"field": CLASS_COLS[0], "values": []})
        ss["ad_ctx_ready"] = False
        st.rerun()
with _btn2:
    if st.button("🔍 Analisar CNAEs de clientes", type="primary", key="ad_run_ctx"):
        ss["ad_ctx_ready"] = True
        ss.pop("ad_cnae_result", None)
        ss.pop("ad_subclasses_sel", None)
        ss.pop("ad_market_df", None)
        ss.pop("ad_1d_calc_ready", None)

# ---------------------------------------------------------------------------
# 1-b — CNAEs da carteira
# ---------------------------------------------------------------------------

if ss.get("ad_ctx_ready"):
    st.markdown("---")
    st.subheader("CNAEs da carteira atendida")

    _active_filters = [
        (flt["field"], tuple(flt["values"]))
        for flt in ss["ad_filters"]
        if flt["field"] and flt["values"]
    ]
    _filters_tuple = tuple(_active_filters)

    with st.spinner("Calculando CNAEs da carteira…"):
        _cnae_df = _cnae_pareto(FOCO_P, _filters_tuple)

    if _cnae_df.empty:
        st.warning(
            "Nenhum cliente atendido encontrado com esses filtros. "
            "Verifique os critérios em 1-a ou tente sem filtros de valor."
        )
    else:
        ss["ad_cnae_result"] = _cnae_df

        _top_n = 10
        _total_cnpjs = _cnae_df["n_cnpjs"].sum()
        _total_val   = _cnae_df["soma_valor"].sum()
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
            # Reverse so the largest bar sits at the top of the horizontal chart
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
            # ABC legend (dummy scatter markers)
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
                use_container_width=True, key="fig_pareto_cnt")
        with _cb:
            st.plotly_chart(
                _pareto_fig(_df10_val, "soma_valor",
                            "Top 10 CNAEs por valor agregado (relevancia_valor)",
                            "Valor"),
                use_container_width=True, key="fig_pareto_val")

# ---------------------------------------------------------------------------
# 1-c — Avaliação do mercado similar
# ---------------------------------------------------------------------------

if ss.get("ad_cnae_result") is not None:
    st.markdown("---")
    st.subheader("Avaliação do mercado similar")
    st.caption("Selecione as subclasses de interesse para analisar o mercado completo.")

    with st.spinner("Carregando subclasses…"):
        _all_sub = _all_subclasses(FOCO_P)

    _cnae_res = ss["ad_cnae_result"]
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
        key="ad_sub_multisel",
        placeholder="Digite ou selecione…",
    )

    if st.button("📊 Analisar mercado", type="primary", key="ad_run_market",
                 disabled=not _sel_labels):
        ss["ad_subclasses_sel"] = tuple(_opts_map[lbl] for lbl in _sel_labels)
        ss.pop("ad_market_df", None)
        ss.pop("ad_1d_calc_ready", None)

    if ss.get("ad_subclasses_sel"):
        _filters_1c = tuple(
            (flt["field"], tuple(flt["values"]))
            for flt in ss["ad_filters"]
            if flt["field"] and flt["values"]
        )
        with st.spinner("Carregando dados do mercado…"):
            _mkt_df = _market_data(
                FOCO_P, ss["ad_subclasses_sel"], _filters_1c,
                HAS_EMPRESA_BASE, expand_empresa=False,
            )
        ss["ad_market_df"] = _mkt_df

        if _mkt_df.empty:
            st.warning("Nenhum registro encontrado para as subclasses selecionadas.")
        else:
            _atend_m  = _mkt_df[_mkt_df["cliente_atendido"] == True]
            _natend_m = _mkt_df[_mkt_df["cliente_atendido"] != True]

            # Chart 1 — stacked bars top 10 subclasses
            _sagg = (_mkt_df.groupby(["cnae_principal", "cnae_principal_desc", "cliente_atendido"])
                     ["cnpj"].nunique().reset_index(name="n"))
            _top10_sub = _sagg.groupby("cnae_principal")["n"].sum().nlargest(10).index
            _sagg10 = _sagg[_sagg["cnae_principal"].isin(_top10_sub)].copy()
            _sagg10["lbl"] = (_sagg10["cnae_principal"].astype(str) + " – "
                              + _sagg10["cnae_principal_desc"].fillna("").str[:35])
            _fig_sub = go.Figure()
            for _flag, _col, _lbl in [(True, "#0079c1", "Atendido"), (False, "#ff6b35", "Não atendido")]:
                _g = _sagg10[_sagg10["cliente_atendido"] == _flag]
                _fig_sub.add_trace(go.Bar(y=_g["lbl"], x=_g["n"], orientation="h",
                                          name=_lbl, marker_color=_col))
            _fig_sub.update_layout(barmode="stack",
                                   title="CNPJs atendidos vs não atendidos por subclasse",
                                   height=400, margin=dict(l=10, r=10, t=40, b=10),
                                   legend=dict(orientation="h", y=1.06))
            st.plotly_chart(_fig_sub, use_container_width=True, key="fig_sub_stack")

            # Chart 2 — CNPJs por UF + top 10 cidades
            _uf_agg = (_mkt_df.groupby("uf")["cnpj"].nunique().reset_index(name="n_cnpjs")
                       .sort_values("n_cnpjs", ascending=False))
            _fig_uf = px.bar(_uf_agg, x="uf", y="n_cnpjs",
                             title="CNPJs por UF", labels={"uf": "UF", "n_cnpjs": "CNPJs"},
                             color_discrete_sequence=["#0079c1"])
            _fig_uf.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(_fig_uf, use_container_width=True, key="fig_uf_total")

            if "municipio_nome" in _mkt_df.columns:
                _ctop = (_mkt_df.groupby(["municipio_nome", "uf"])["cnpj"].nunique()
                         .reset_index(name="n").sort_values("n", ascending=False).head(10))
                st.markdown("**Top 10 cidades por concentração de CNPJs**")
                st.dataframe(_ctop.rename(columns={"municipio_nome": "Cidade", "uf": "UF", "n": "CNPJs"}),
                             use_container_width=True, hide_index=True)

            # Chart 4 — Top 20 palavras nome_fantasia (não atendidos)
            if "nome_fantasia" in _natend_m.columns:
                _words = {}
                for _nf in _natend_m["nome_fantasia"].dropna():
                    for w in _norm(_nf).split():
                        if len(w) > 2 and w not in _PT_STOPWORDS:
                            _words[w] = _words.get(w, 0) + 1
                if _words:
                    _wdf = (pd.Series(_words).sort_values(ascending=False)
                            .head(20).reset_index())
                    _wdf.columns = ["palavra", "freq"]
                    _wdf = _wdf.sort_values("freq", ascending=True)
                    _fig_w = px.bar(_wdf, x="freq", y="palavra", orientation="h",
                                    title="Top 20 palavras em Nome Fantasia (não atendidos)",
                                    labels={"freq": "Frequência", "palavra": ""},
                                    color_discrete_sequence=["#ff6b35"])
                    _fig_w.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(_fig_w, use_container_width=True, key="fig_words")

# ---------------------------------------------------------------------------
# 1-d — Avaliação de Atendimento Direto
# ---------------------------------------------------------------------------

if ss.get("ad_subclasses_sel") and ss.get("ad_market_df") is not None:
    st.markdown("---")
    st.subheader("Avaliação de Atendimento Direto")

    _cc1, _cc2, _cc3 = st.columns(3)
    with _cc1:
        _skip_blank = st.checkbox("Ignorar CNPJs com Nome Fantasia em branco",
                                  key="ad_skip_blank_nf", value=True)
    with _cc2:
        _expand_empr = st.checkbox(
            "Considerar empresa_base_atendida como atendida",
            key="ad_expand_empresa",
            value=True,
            disabled=not HAS_EMPRESA_BASE,
            help="Sinaliza como atendido qualquer CNPJ de empresa que já tenha outro estabelecimento atendido.",
        )
    with _cc3:
        _use_excl_kw = st.checkbox("Excluir não atendidos por palavras-chave",
                                   key="ad_excl_kw_on", value=True)

    ss.setdefault("ad_excl_kw", [])
    if _use_excl_kw:
        _kw_c1, _kw_c2 = st.columns([4, 1])
        with _kw_c1:
            _new_kw = st.text_input("Palavra-chave (Enter para adicionar)", key="ad_excl_kw_input",
                                    placeholder="Ex: matriz, holding…")
        with _kw_c2:
            st.write(""); st.write("")
            if st.button("➕ Incluir", key="ad_excl_kw_add") and _new_kw.strip():
                _nkw = _norm(_new_kw.strip())
                if _nkw and _nkw not in ss["ad_excl_kw"]:
                    ss["ad_excl_kw"].append(_nkw)
        if ss["ad_excl_kw"]:
            st.caption("Excluindo: " + " · ".join(f"`{k}`" for k in ss["ad_excl_kw"]))
            if st.button("🗑️ Limpar keywords", key="ad_excl_kw_clear"):
                ss["ad_excl_kw"] = []
                st.rerun()

    ss.setdefault("ad_1d_calc_ready", False)
    if st.button("🔄 Calcular", type="primary", key="ad_1d_calc_btn"):
        ss["ad_1d_calc_ready"] = True
        ss["_1d_skip_blank"]   = _skip_blank
        ss["_1d_expand_empr"]  = _expand_empr
        ss["_1d_use_excl_kw"]  = _use_excl_kw
        ss["_1d_excl_kw_snap"] = list(ss["ad_excl_kw"])

    if not ss.get("ad_1d_calc_ready"):
        st.info("Configure as opções acima e clique em **🔄 Calcular** para gerar o relatório.")
        st.stop()

    _filters_1d = tuple(
        (flt["field"], tuple(flt["values"]))
        for flt in ss["ad_filters"]
        if flt["field"] and flt["values"]
    )
    with st.spinner("Aplicando critérios…"):
        _df = _market_data(
            FOCO_P, ss["ad_subclasses_sel"], _filters_1d,
            HAS_EMPRESA_BASE, expand_empresa=ss.get("_1d_expand_empr", True),
        ).copy()

    if ss.get("_1d_skip_blank", True) and "nome_fantasia" in _df.columns:
        _df = _df[_df["nome_fantasia"].notna() & (_df["nome_fantasia"].str.strip() != "")]

    if ss.get("_1d_use_excl_kw", True) and ss.get("_1d_excl_kw_snap", []) and "nome_fantasia" in _df.columns:
        _kw_set = ss.get("_1d_excl_kw_snap", [])
        _norm_nf = _df["nome_fantasia"].fillna("").apply(_norm)
        _excl = _norm_nf.apply(lambda nf: any(kw in nf for kw in _kw_set))
        _is_atend = _df["cliente_atendido"] == True
        _df = _df[_is_atend | (~_is_atend & ~_excl)]

    # KPIs
    _tot_cnpj   = _df["cnpj"].nunique()
    _atend_cnpj = _df.loc[_df["cliente_atendido"] == True, "cnpj"].nunique()
    _pct_cnpj   = _atend_cnpj / _tot_cnpj * 100 if _tot_cnpj else 0
    _tot_emp    = _df["cnpj_basico"].nunique() if "cnpj_basico" in _df.columns else 0
    _atend_emp  = (_df.loc[_df["cliente_atendido"] == True, "cnpj_basico"].nunique()
                   if "cnpj_basico" in _df.columns else 0)
    _pct_emp    = _atend_emp / _tot_emp * 100 if _tot_emp else 0

    _k1, _k2, _k3, _k4, _k5, _k6 = st.columns(6)
    _k1.metric("Total CNPJs",          f"{_tot_cnpj:,}")
    _k2.metric("CNPJs Atendidos",      f"{_atend_cnpj:,}")
    _k3.metric("% CNPJs Atendidos",    f"{_pct_cnpj:.1f}%")
    _k4.metric("Total Empresas",       f"{_tot_emp:,}")
    _k5.metric("Empresas Atendidas",   f"{_atend_emp:,}")
    _k6.metric("% Empresas Atendidas", f"{_pct_emp:.1f}%")

    def _stacked_h(df, groupby, id_col, title, top_n=25):
        _agg = (df.groupby([groupby, "cliente_atendido"])[id_col].nunique()
                .reset_index(name="n"))
        _totals = _agg.groupby(groupby)["n"].sum().nlargest(top_n).index
        _agg_f = _agg[_agg[groupby].isin(_totals)]
        fig = go.Figure()
        for _flag, _color, _lbl in [(True, "#0079c1", "Atendido"), (False, "#ff6b35", "Não atendido")]:
            _g = _agg_f[_agg_f["cliente_atendido"] == _flag].set_index(groupby)["n"]
            _g = _g.reindex(_totals, fill_value=0)
            fig.add_trace(go.Bar(y=list(_totals), x=list(_g.values),
                                 orientation="h", name=_lbl, marker_color=_color))
        fig.update_layout(barmode="stack", title=title,
                          height=max(350, len(_totals) * 22),
                          margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=1.06))
        return fig

    _u1, _u2 = st.columns(2)
    with _u1:
        st.plotly_chart(_stacked_h(_df, "uf", "cnpj", "CNPJs por UF"),
                        use_container_width=True, key="fig_1d_cnpj_uf")
    with _u2:
        if "cnpj_basico" in _df.columns:
            st.plotly_chart(_stacked_h(_df, "uf", "cnpj_basico", "Empresas por UF"),
                            use_container_width=True, key="fig_1d_emp_uf")

    if "municipio_nome" in _df.columns:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.plotly_chart(_stacked_h(_df, "municipio_nome", "cnpj",
                                       "CNPJs por cidade (top 20)", top_n=20),
                            use_container_width=True, key="fig_1d_cnpj_cid")
        with _c2:
            if "cnpj_basico" in _df.columns:
                st.plotly_chart(_stacked_h(_df, "municipio_nome", "cnpj_basico",
                                           "Empresas por cidade (top 20)", top_n=20),
                                use_container_width=True, key="fig_1d_emp_cid")

    # Mapa
    st.markdown("**Mapa de calor de não atendimento**")
    _natend_1d = _df[_df["cliente_atendido"] != True]

    if HAS_LATLON and "lat" in _natend_1d.columns and _natend_1d["lat"].notna().any():
        # -------------------------------------------------------------------
        # Aggregate raw points into a spatial grid BEFORE building the figure.
        # Sending hundreds-of-thousands of individual lat/lon rows directly to
        # Plotly causes Chrome OOM because the entire dataset is serialised into
        # the figure JSON.  Aggregating to a 0.05° grid (≈5 km) collapses the
        # payload to at most a few thousand weighted centroids.
        # -------------------------------------------------------------------
        _MAP_GRID = 0.05          # degrees per cell side
        _raw = _natend_1d[["lat", "lon"]].dropna().copy()
        _raw["_lat_g"] = (_raw["lat"] / _MAP_GRID).round() * _MAP_GRID
        _raw["_lon_g"] = (_raw["lon"] / _MAP_GRID).round() * _MAP_GRID
        _mdf = (
            _raw.groupby(["_lat_g", "_lon_g"], sort=False)
            .size()
            .reset_index(name="n_cnpjs")
            .rename(columns={"_lat_g": "lat", "_lon_g": "lon"})
        )
        st.caption(
            f"ℹ️ {len(_raw):,} CNPJs não atendidos agregados em "
            f"{len(_mdf):,} células de ~5 km (grid 0.05°)"
        )
        # Cap colour scale at 90th percentile so SP doesn't dominate the palette
        _zmax = float(_mdf["n_cnpjs"].quantile(0.90))
        _fig_map = px.density_mapbox(
            _mdf, lat="lat", lon="lon", z="n_cnpjs", radius=18, zoom=3,
            center={"lat": -14.2, "lon": -51.9},
            mapbox_style="carto-positron",
            title="Concentração de não atendidos",
            color_continuous_scale="YlOrRd",
            range_color=[0, max(_zmax, 1)],
        )
        _fig_map.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(_fig_map, use_container_width=True, key="fig_1d_heatmap")
    else:
        st.caption(
            "ℹ️ Colunas lat/lon não disponíveis — exibindo choropleth por UF. "
            "Para o mapa por município regere o parquet com lat/lon no Módulo 5."
        )
        _uf_na = (_natend_1d.groupby("uf")["cnpj"].nunique()
                  .reset_index(name="n_nao_atendidos"))
        _fig_ch = px.choropleth(
            _uf_na, locations="uf", color="n_nao_atendidos",
            geojson=(
                "https://raw.githubusercontent.com/codeforamerica/"
                "click_that_hood/master/public/data/brazil-states.geojson"
            ),
            featureidkey="properties.sigla",
            scope="south america",
            color_continuous_scale="YlOrRd",
            title="CNPJs não atendidos por UF",
            labels={"n_nao_atendidos": "Não atendidos"},
        )
        _fig_ch.update_geos(fitbounds="locations", visible=False)
        _fig_ch.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(_fig_ch, use_container_width=True, key="fig_1d_choro")

    # Export
    st.markdown("---")
    st.subheader("📋 Exportação — não atendidos do contexto selecionado")
    _exp = _natend_1d.copy()
    if "capital_social" in _exp.columns:
        _exp["_cap_n"] = pd.to_numeric(
            _exp["capital_social"].astype(str).str.replace(",", "."), errors="coerce").fillna(0)
        _exp = _exp.sort_values("_cap_n", ascending=False).drop(columns=["_cap_n"])

    _prev_cols = [c for c in ["cnpj", "razao_social", "nome_fantasia",
                               "cnae_principal", "cnae_principal_desc",
                               "capital_social", "porte_desc",
                               "uf", "municipio_nome", "cep"] if c in _exp.columns]
    st.caption(f"**{len(_exp):,}** CNPJs não atendidos · preview: top 100 por capital social")
    st.dataframe(_exp[_prev_cols].head(100), use_container_width=True, hide_index=True)

    _csv = _exp.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        label="⬇️ Baixar CSV completo",
        data=_csv,
        file_name="nao_atendidos_direto.csv",
        mime="text/csv",
        key="ad_download_csv",
    )
