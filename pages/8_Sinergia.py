"""
Módulo 8 — Segmentação SG

Objetivo central: criar e manter a coluna "Segmentação SG" em uma base de
CNPJs, classificando cada registro a partir de regras e entradas manuais.

O trabalho é versionado: cada modificação gera uma nova versão do arquivo
(como commits git), permitindo retornar a qualquer estado anterior.

Estrutura de armazenamento:
    {data}/processed/sinergia/
        {projeto}/
            v000.parquet  +  v000.json   <- startpoint (upload original)
            v001.parquet  +  v001.json
            ...
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.config import get_subfolders
from utils.sg_versioning import (
    SEG_COL,
    build_meta,
    fmt_dt,
    list_projects,
    list_versions,
    load_version,
    next_version_number,
    save_version,
    slugify,
    unique_slug,
)
from utils.sidebar import render_sidebar
from utils.storage import parquet_exists

# ---------------------------------------------------------------------------
# Constants + pure helpers  (must be defined before st.set_page_config)
# ---------------------------------------------------------------------------

_subs     = get_subfolders()
_UNI_PATH = _subs["processed"] / "base_unificada.parquet"
_UNI_P    = str(_UNI_PATH).replace("\\", "/")

@st.cache_data(show_spinner=False, ttl=600)
def _cnae_stats_duckdb(cnpj_tuple: tuple, uni_path: str) -> pd.DataFrame:
    """Top 10 CNAEs para um conjunto de CNPJs, via base_unificada."""
    if not cnpj_tuple:
        return pd.DataFrame()
    try:
        con = duckdb.connect()
        frame = pd.DataFrame({"_c": list(cnpj_tuple)})
        con.register("sel", frame)
        return con.execute("""
            SELECT b.cnae_principal,
                   any_value(b.cnae_principal_desc) AS cnae_principal_desc,
                   COUNT(DISTINCT b.cnpj)           AS n_cnpjs
            FROM   '{uni}' b
            INNER  JOIN sel s ON s._c = b.cnpj
            WHERE  b.cnae_principal IS NOT NULL
            GROUP  BY b.cnae_principal
            ORDER  BY n_cnpjs DESC
            LIMIT  10
        """.replace("{uni}", uni_path)).df()
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def _compute_venn3(
    setA: set, setB: set, setC: set, val_map: dict
) -> dict:
    """7 regiões exclusivas de um Venn de 3 conjuntos. {frozenset[int]: (count, sum_value)}."""
    sets = [setA, setB, setC]
    universe = setA | setB | setC
    out: dict = {}
    for mask in range(1, 8):
        region = set(universe)
        for i in range(3):
            if mask & (1 << i):
                region &= sets[i]
            else:
                region -= sets[i]
        cnt = len(region)
        val = sum(val_map.get(c, 0.0) for c in region) if val_map else 0.0
        out[frozenset(i for i in range(3) if mask & (1 << i))] = (cnt, val)
    return out


def _draw_venn3(
    regions: dict,
    labels: list[str],
    set_sizes: list[int],
    has_values: bool = False,
) -> go.Figure:
    """
    Diagrama de Venn com 3 círculos em layout triangular clássico.
    A (topo-esquerda), B (topo-direita), C (base-centro).
    """
    colors = ["#0079c1", "#ff6b35", "#27ae60"]
    cx = [-0.35,  0.35,  0.00]
    cy = [ 0.22,  0.22, -0.28]
    r  = 0.50

    ann_pos = {
        frozenset([0]):       (-0.68,  0.54),
        frozenset([1]):       ( 0.68,  0.54),
        frozenset([2]):       ( 0.00, -0.72),
        frozenset([0, 1]):    ( 0.00,  0.50),
        frozenset([0, 2]):    (-0.42, -0.13),
        frozenset([1, 2]):    ( 0.42, -0.13),
        frozenset([0, 1, 2]): ( 0.00,  0.08),
    }

    theta = np.linspace(0, 2 * np.pi, 120)
    fig = go.Figure()

    for i, (x_c, y_c, col) in enumerate(zip(cx, cy, colors)):
        xs = x_c + r * np.cos(theta)
        ys = y_c + r * np.sin(theta)
        path = "M{:.4f},{:.4f}".format(xs[0], ys[0])
        for xi, yi in zip(xs[1:], ys[1:]):
            path += " L{:.4f},{:.4f}".format(xi, yi)
        path += " Z"
        fig.add_shape(
            type="path", path=path,
            fillcolor=col, opacity=0.22,
            line=dict(color=col, width=2.5),
        )

    lbl_pos = [(-0.80, 0.84), (0.80, 0.84), (0.00, -0.96)]
    for i, ((lx, ly), col) in enumerate(zip(lbl_pos, colors)):
        lbl = labels[i][:35] + ("…" if len(labels[i]) > 35 else "")
        n   = set_sizes[i]
        fig.add_annotation(
            x=lx, y=ly,
            text="<b>{}</b><br><sup>{:,} CNPJs</sup>".format(lbl, n),
            showarrow=False,
            font=dict(size=10, color=col),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor=col, borderwidth=1, borderpad=3,
        )

    for key, (ax, ay) in ann_pos.items():
        cnt, val = regions.get(key, (0, 0.0))
        if cnt <= 0:
            continue
        if has_values and val:
            txt = "{:,}<br><sub>R${:,.0f}</sub>".format(cnt, val)
        else:
            txt = "{:,}".format(cnt)
        fig.add_annotation(
            x=ax, y=ay, text=txt,
            showarrow=False,
            font=dict(size=10, color="#111"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#bbb", borderwidth=0.8, borderpad=3,
        )

    total = sum(v[0] for v in regions.values())
    fig.update_layout(
        title=dict(
            text="Diagrama de Venn — {:,} CNPJs no universo identificado".format(total),
            x=0.5, font=dict(size=13),
        ),
        xaxis=dict(range=[-1.12, 1.12], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            range=[-1.05, 1.05], showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="x",
        ),
        height=520, margin=dict(l=10, r=10, t=55, b=10),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Segmentação SG · SGGeoData",
    page_icon="🏷",
    layout="wide",
)
render_sidebar()
st.title("🏷 Módulo 8 — Segmentação SG")
st.caption(
    "Classifique cada CNPJ com a coluna **Segmentação SG** através de regras e "
    "entradas manuais. Cada modificação gera uma nova versão versionada do arquivo."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_SS: dict = {
    "sg_project":      None,
    "sg_version":      None,
    "sg_df":           None,
    "sg_meta":         None,
    "sg_col_map":      {},
    "sg_dirty":        False,
    # Regra 1 — Segmentação SG + CNAE
    "sg_r1_emp_a":     None,
    "sg_r1_segs_a":    [],
    "sg_r1_emp_b":     None,
    "sg_r1_segs_b":    [],
    "sg_r1_stats":     None,
    "sg_r1_top5":      None,
    "sg_r1_cnae":      None,
    "sg_r1_venn_data": None,
}
for _k, _v in _SS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Section 1 — Abrir projeto / criar novo
# ---------------------------------------------------------------------------

st.subheader("1 · Projeto")

_projects = list_projects()


def _proj_label(p: dict) -> str:
    last = "v{:03d}: {}".format(p["latest_version"], p["latest_label"][:45])
    return "{} [{} versão/versões · {} · {}]".format(
        p["source_file"], p["n_versions"], last, fmt_dt(p["latest_at"])
    )


_NOVO = "➕  Novo projeto (upload)"
_proj_display = [_NOVO] + [_proj_label(p) for p in _projects]
_proj_slugs   = [None]  + [p["slug"] for p in _projects]

_active_slug = st.session_state["sg_project"]
_default_proj_idx = (
    _proj_slugs.index(_active_slug)
    if _active_slug and _active_slug in _proj_slugs
    else 0
)

_sel_proj_idx = st.selectbox(
    "Selecione um projeto existente ou crie um novo",
    options=range(len(_proj_display)),
    format_func=lambda i: _proj_display[i],
    index=_default_proj_idx,
    key="sg_proj_sel",
)
_sel_slug = _proj_slugs[_sel_proj_idx]

# -- Branch A: novo upload --------------------------------------------------

if _sel_slug is None:
    st.markdown("Faça upload de um arquivo CSV ou Parquet para iniciar um novo projeto.")

    _uploaded = st.file_uploader(
        "Base de CNPJs",
        type=["csv", "txt", "parquet"],
        key="sg_uploader",
    )

    if _uploaded is None:
        if not _projects:
            st.info("Nenhum projeto encontrado. Faça upload de um arquivo para começar.")
        st.stop()

    with st.spinner("Lendo arquivo…"):
        try:
            if _uploaded.name.lower().endswith(".parquet"):
                _df_up = pd.read_parquet(_uploaded)
            else:
                _df_up = pd.read_csv(_uploaded, dtype=str, sep=None, engine="python")
        except Exception as _e:
            st.error("Erro ao ler arquivo: {}".format(_e))
            st.stop()

    st.success("**{:,} linhas · {} colunas**".format(len(_df_up), len(_df_up.columns)))

    _cols_up    = _df_up.columns.tolist()
    _cnpj_guess = next((c for c in _cols_up if re.search(r"cnpj", c, re.I)), _cols_up[0])

    _uc1, _uc2 = st.columns(2)
    with _uc1:
        _cnpj_col_up = st.selectbox(
            "Coluna de CNPJ",
            _cols_up,
            index=_cols_up.index(_cnpj_guess),
            key="sg_new_cnpj_col",
        )
    with _uc2:
        _proj_name_up = st.text_input(
            "Nome do projeto",
            value=slugify(_uploaded.name),
            key="sg_new_proj_name",
        )

    with st.expander("Prévia (5 primeiras linhas)", expanded=False):
        st.dataframe(_df_up.head(5), width="stretch", hide_index=True)

    _slug_preview = slugify(_proj_name_up.strip() or _uploaded.name)
    st.info(
        "Ao confirmar, o arquivo será salvo como **v000 (startpoint)** "
        "no projeto `{}`.".format(_slug_preview)
    )

    if st.button(
        "💾  Criar projeto e salvar startpoint",
        type="primary",
        width="content",
        key="sg_create_btn",
    ):
        _base_slug  = slugify(_proj_name_up.strip() or _uploaded.name)
        _final_slug = unique_slug(_base_slug, [p["slug"] for p in _projects])

        if SEG_COL not in _df_up.columns:
            _df_up[SEG_COL] = None

        _col_map_up = {"cnpj_col": _cnpj_col_up}
        _meta_v0 = build_meta(
            project=_final_slug,
            version=0,
            label="Startpoint — upload original",
            df=_df_up,
            source_file=_uploaded.name,
            col_map=_col_map_up,
        )
        with st.spinner("Salvando startpoint…"):
            save_version(_final_slug, _df_up, _meta_v0)

        st.session_state.update({
            "sg_project": _final_slug,
            "sg_version": 0,
            "sg_df":      _df_up.copy(),
            "sg_meta":    _meta_v0,
            "sg_col_map": _col_map_up,
            "sg_dirty":   False,
        })
        st.success(
            "✅ Projeto **{}** criado com **{:,}** registros. "
            "Versão `v000` salva.".format(_final_slug, len(_df_up))
        )
        st.rerun()

    st.stop()

# -- Branch B: projeto existente --------------------------------------------

if st.session_state["sg_project"] != _sel_slug:
    st.session_state.update({
        "sg_project": _sel_slug,
        "sg_version": None,
        "sg_df":      None,
        "sg_meta":    None,
        "sg_dirty":   False,
        "sg_col_map": {},
    })

_versions = list_versions(_sel_slug)
if not _versions:
    st.error("Projeto sem versões válidas no disco.")
    st.stop()

# ---------------------------------------------------------------------------
# Section 2 — Seletor de versão + histórico
# ---------------------------------------------------------------------------

st.divider()
st.subheader("2 · Versão")

_ver_nums   = [v["version"] for v in _versions]
_ver_latest = _ver_nums[-1]

_cur_ver = st.session_state.get("sg_version")
if _cur_ver not in _ver_nums:
    _cur_ver = _ver_latest
_cur_idx = _ver_nums.index(_cur_ver)


def _ver_label(v: dict) -> str:
    rc = v.get("rows_changed", 0)
    suffix = "  ·  {:,} linhas alteradas".format(rc) if rc else ""
    return "v{:03d}  —  {}  ({}{})".format(
        v["version"], v["label"][:55], fmt_dt(v["created_at"]), suffix
    )


_sel_ver_idx = st.selectbox(
    "Versão",
    options=range(len(_versions)),
    format_func=lambda i: _ver_label(_versions[i]),
    index=_cur_idx,
    key="sg_ver_sel",
)
_sel_ver_meta = _versions[_sel_ver_idx]
_sel_ver_num  = _sel_ver_meta["version"]

if st.session_state["sg_version"] != _sel_ver_num:
    st.session_state.update({
        "sg_version": _sel_ver_num,
        "sg_df":      None,
        "sg_col_map": _sel_ver_meta.get("col_map", {}),
    })

with st.expander("📜 Histórico de versões", expanded=False):
    _hist = [
        {
            "Versão":           "v{:03d}".format(v["version"]),
            "Label":            v["label"],
            "Data":             fmt_dt(v["created_at"]),
            "Linhas total":     "{:,}".format(v["rows"]) if isinstance(v.get("rows"), int) else "—",
            "Linhas alteradas": "{:,}".format(v["rows_changed"]) if v.get("rows_changed") else "—",
        }
        for v in reversed(_versions)
    ]
    st.dataframe(pd.DataFrame(_hist), width="stretch", hide_index=True)

if st.session_state["sg_df"] is None:
    with st.spinner("Carregando v{:03d}…".format(_sel_ver_num)):
        st.session_state["sg_df"]      = load_version(_sel_slug, _sel_ver_num)
        st.session_state["sg_meta"]    = _sel_ver_meta
        st.session_state["sg_col_map"] = _sel_ver_meta.get("col_map", {})

_df: pd.DataFrame = st.session_state["sg_df"]
_col_map: dict    = st.session_state["sg_col_map"]
_cnpj_col: str    = _col_map.get("cnpj_col", _df.columns[0])

# ---------------------------------------------------------------------------
# Section 3 — Visão geral dos dados
# ---------------------------------------------------------------------------

st.divider()
st.subheader("3 · Dados carregados")

_n_rows     = len(_df)
_has_seg    = SEG_COL in _df.columns
_n_filled   = int(_df[SEG_COL].notna().sum()) if _has_seg else 0
_pct_filled = (_n_filled / _n_rows * 100) if _n_rows else 0.0

_mc1, _mc2, _mc3, _mc4 = st.columns(4)
_mc1.metric("Registros",             "{:,}".format(_n_rows))
_mc2.metric("Colunas",               str(len(_df.columns)))
_mc3.metric(SEG_COL + " preenchida", "{:,}".format(_n_filled))
_mc4.metric("% classificado",        "{:.1f}%".format(_pct_filled))

if _has_seg and _n_filled > 0:
    with st.expander("📊 Distribuição de " + SEG_COL, expanded=True):
        _dist = (
            _df[SEG_COL]
            .value_counts(dropna=False)
            .rename_axis(SEG_COL)
            .reset_index(name="Qtd")
        )
        _dist["% Total"] = (_dist["Qtd"] / _n_rows * 100).round(1)
        st.dataframe(_dist, width="stretch", hide_index=True)
else:
    st.info(
        "A coluna **{}** ainda não possui valores preenchidos. "
        "Use as regras abaixo para iniciar a classificação.".format(SEG_COL)
    )

_preview_cols = (
    [_cnpj_col, SEG_COL] + [c for c in _df.columns if c not in (_cnpj_col, SEG_COL)]
    if _has_seg else list(_df.columns)
)
_preview_cols = [c for c in _preview_cols if c in _df.columns]

with st.expander("Prévia dos dados (10 primeiras linhas)", expanded=False):
    st.dataframe(_df[_preview_cols].head(10), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Section 4 — Regras de classificação
# ---------------------------------------------------------------------------

st.divider()
st.subheader("4 · Regras de " + SEG_COL)

# ── 4.0  Mapeamento de colunas ──────────────────────────────────────────────

_all_cols   = _df.columns.tolist()
_cm_ec  = _col_map.get("empresa_col", "")
_cm_sc  = _col_map.get("segmentacao_col", "")
_cm_vc  = _col_map.get("valor_col", "")
_mapped = bool(_cm_ec and _cm_ec in _all_cols and _cm_sc and _cm_sc in _all_cols)


def _ci(col: str) -> int:
    opts = ["—"] + _all_cols
    return opts.index(col) if col in opts else 0


with st.expander("⚙️ Mapeamento de colunas para regras", expanded=not _mapped):
    _r4c1, _r4c2, _r4c3 = st.columns(3)
    with _r4c1:
        _ec_s = st.selectbox("Empresa", ["—"] + _all_cols, index=_ci(_cm_ec), key="sg4_ec")
    with _r4c2:
        _sc_s = st.selectbox("Segmentação", ["—"] + _all_cols, index=_ci(_cm_sc), key="sg4_sc")
    with _r4c3:
        _vc_s = st.selectbox("Valor (R$)", ["—"] + _all_cols, index=_ci(_cm_vc), key="sg4_vc")

    if st.button("💾 Salvar mapeamento", key="sg4_save_cm", width="content"):
        _new_cm = dict(_col_map)
        _new_cm.update({
            "empresa_col":     _ec_s if _ec_s != "—" else "",
            "segmentacao_col": _sc_s if _sc_s != "—" else "",
            "valor_col":       _vc_s if _vc_s != "—" else "",
        })
        st.session_state["sg_col_map"] = _new_cm
        st.rerun()

_eff_ec = _col_map.get("empresa_col", "")
_eff_sc = _col_map.get("segmentacao_col", "")
_eff_vc = _col_map.get("valor_col", "")

if not (_eff_ec and _eff_ec in _df.columns and _eff_sc and _eff_sc in _df.columns):
    st.info(
        "Configure o **mapeamento de colunas** acima para ativar as regras de classificação."
    )
else:
    # ── Regra 1 — Segmentação SG + CNAE ─────────────────────────────────────
    st.markdown("##### Regra 1 — Segmentação Saint-Gobain + CNAE")

    _all_empresas = sorted(_df[_eff_ec].dropna().unique().tolist())

    # ----------- Seleção de empresas -----------------------------------------
    _r1_ca, _r1_cb = st.columns(2)

    with _r1_ca:
        st.markdown("**Empresa A** *(obrigatória)*")
        _ea_opts  = ["— selecione —"] + _all_empresas
        _ea_prev  = st.session_state["sg_r1_emp_a"]
        _ea_idx   = _ea_opts.index(_ea_prev) if _ea_prev in _ea_opts else 0
        _ea_sel   = st.selectbox(
            "Empresa A", _ea_opts, index=_ea_idx,
            label_visibility="collapsed", key="sg_r1_ea",
        )
        if _ea_sel != "— selecione —":
            _segs_a_opts = sorted(
                _df.loc[_df[_eff_ec] == _ea_sel, _eff_sc].dropna().unique().tolist()
            )
            _segs_a_prev = [s for s in st.session_state["sg_r1_segs_a"] if s in _segs_a_opts]
            _segs_a = st.multiselect(
                "Segmentações — A", _segs_a_opts, default=_segs_a_prev, key="sg_r1_sa"
            )
        else:
            _segs_a = []

    with _r1_cb:
        st.markdown("**Empresa B** *(opcional)*")
        _eb_opts_base = [e for e in _all_empresas if e != _ea_sel]
        _eb_opts  = ["— nenhuma —"] + _eb_opts_base
        _eb_prev  = st.session_state["sg_r1_emp_b"]
        _eb_idx   = _eb_opts.index(_eb_prev) if _eb_prev in _eb_opts else 0
        _eb_sel   = st.selectbox(
            "Empresa B", _eb_opts, index=_eb_idx,
            label_visibility="collapsed", key="sg_r1_eb",
        )
        if _eb_sel and _eb_sel != "— nenhuma —":
            _segs_b_opts = sorted(
                _df.loc[_df[_eff_ec] == _eb_sel, _eff_sc].dropna().unique().tolist()
            )
            _segs_b_prev = [s for s in st.session_state["sg_r1_segs_b"] if s in _segs_b_opts]
            _segs_b = st.multiselect(
                "Segmentações — B", _segs_b_opts, default=_segs_b_prev, key="sg_r1_sb"
            )
        else:
            _segs_b = []

    # Persist selections into session state
    st.session_state["sg_r1_emp_a"]  = _ea_sel if _ea_sel != "— selecione —" else None
    st.session_state["sg_r1_segs_a"] = _segs_a
    st.session_state["sg_r1_emp_b"]  = _eb_sel if _eb_sel and _eb_sel != "— nenhuma —" else None
    st.session_state["sg_r1_segs_b"] = _segs_b

    # ----------- Analisar CNAEs (botão) --------------------------------------
    _r1_ready = bool(_ea_sel != "— selecione —" and _segs_a)

    if st.button(
        "🔍 Analisar CNAEs e pré-estatísticas",
        type="primary", width="content",
        key="sg_r1_analyze",
        disabled=not _r1_ready,
    ):
        def _top_cnaes_for(mask: pd.Series, cnpj_col: str) -> tuple[int, pd.DataFrame]:
            """CNPJs únicos + top 10 CNAEs para a máscara dada."""
            cnpjs = set(_df.loc[mask, cnpj_col].dropna().astype(str))
            if "cnae_principal" in _df.columns:
                sub = _df.loc[mask & _df["cnae_principal"].notna()]
                desc_col = "cnae_principal_desc" if "cnae_principal_desc" in _df.columns else None
                grp = ["cnae_principal"] + ([desc_col] if desc_col else [])
                agg = (
                    sub.groupby(grp)[cnpj_col].nunique()
                    .reset_index(name="n_cnpjs")
                    .sort_values("n_cnpjs", ascending=False)
                    .head(5)
                )
            elif parquet_exists(_UNI_PATH):
                agg = _cnae_stats_duckdb(tuple(sorted(cnpjs)), _UNI_P).head(5)
            else:
                agg = pd.DataFrame()
            return len(cnpjs), agg

        _mask_a = (_df[_eff_ec] == _ea_sel) & (_df[_eff_sc].isin(_segs_a))
        _n_a, _top_a = _top_cnaes_for(_mask_a, _cnpj_col)
        _stats: dict = {"a": {"n_cnpjs": _n_a, "top_cnaes": _top_a}}

        if st.session_state["sg_r1_emp_b"] and _segs_b:
            _mask_b = (_df[_eff_ec] == _eb_sel) & (_df[_eff_sc].isin(_segs_b))
            _n_b, _top_b = _top_cnaes_for(_mask_b, _cnpj_col)
            _stats["b"] = {"n_cnpjs": _n_b, "top_cnaes": _top_b}

        # Consolidated top 5
        _all_tops = [_stats[k]["top_cnaes"] for k in ("a", "b") if k in _stats and not _stats[k]["top_cnaes"].empty]
        if _all_tops:
            _n_col = "n_cnpjs"
            _consol = (
                pd.concat(_all_tops, ignore_index=True)
                .groupby("cnae_principal")
                .agg(
                    cnae_principal_desc=("cnae_principal_desc", "first") if "cnae_principal_desc" in _all_tops[0].columns else (),
                    n_cnpjs=(_n_col, "sum"),
                )
                .reset_index()
                .sort_values("n_cnpjs", ascending=False)
                .head(5)
            )
        else:
            _consol = pd.DataFrame()

        st.session_state.update({
            "sg_r1_stats":     _stats,
            "sg_r1_top5":      _consol,
            "sg_r1_cnae":      None,
            "sg_r1_venn_data": None,
        })

    # ----------- Exibir estatísticas -----------------------------------------
    if st.session_state.get("sg_r1_stats"):
        _rs = st.session_state["sg_r1_stats"]

        _col_a, _col_b = st.columns(2)
        with _col_a:
            _sa = _rs.get("a", {})
            if _sa:
                st.metric(
                    "CNPJs atendidos — {}".format(_ea_sel),
                    "{:,}".format(_sa["n_cnpjs"]),
                )
                _tc_a = _sa.get("top_cnaes", pd.DataFrame())
                if not _tc_a.empty:
                    st.caption("**Top CNAEs — {}**".format(_ea_sel))
                    st.dataframe(_tc_a, width="stretch", hide_index=True)

        with _col_b:
            _sb = _rs.get("b", {})
            if _sb:
                st.metric(
                    "CNPJs atendidos — {}".format(st.session_state["sg_r1_emp_b"]),
                    "{:,}".format(_sb["n_cnpjs"]),
                )
                _tc_b = _sb.get("top_cnaes", pd.DataFrame())
                if not _tc_b.empty:
                    st.caption("**Top CNAEs — {}**".format(st.session_state["sg_r1_emp_b"]))
                    st.dataframe(_tc_b, width="stretch", hide_index=True)

        # ----------- Top 5 consolidados + seletor de CNAE --------------------
        _top5 = st.session_state.get("sg_r1_top5")
        if _top5 is not None and not _top5.empty:
            st.divider()
            st.markdown("**Top 5 CNAEs consolidados (A + B)**")
            st.dataframe(_top5, width="stretch", hide_index=True)

            _cnae_opts_raw = []
            _n_col2 = "n_cnpjs" if "n_cnpjs" in _top5.columns else _top5.columns[-1]
            for _, rw in _top5.iterrows():
                _desc = str(rw.get("cnae_principal_desc", ""))[:45]
                _lbl  = "{} — {}  ({:,} clientes)".format(
                    rw["cnae_principal"], _desc, rw[_n_col2]
                )
                _cnae_opts_raw.append((_lbl, str(rw["cnae_principal"])))

            _cnae_lbl_map = {lbl: code for lbl, code in _cnae_opts_raw}
            _prev_code    = st.session_state.get("sg_r1_cnae")
            _prev_lbl     = next(
                (lbl for lbl, code in _cnae_opts_raw if code == _prev_code),
                _cnae_opts_raw[0][0] if _cnae_opts_raw else "",
            )
            _cnae_lbl_sel = st.selectbox(
                "CNAE principal para análise",
                options=list(_cnae_lbl_map),
                index=list(_cnae_lbl_map).index(_prev_lbl) if _prev_lbl in _cnae_lbl_map else 0,
                key="sg_r1_cnae_pick",
            )
            _cnae_code = _cnae_lbl_map.get(_cnae_lbl_sel, "")

            if _cnae_code != st.session_state["sg_r1_cnae"]:
                st.session_state.update({
                    "sg_r1_cnae":      _cnae_code,
                    "sg_r1_venn_data": None,
                })

            if _cnae_code:
                # ----------- Botão "Visualizar Venn" -------------------------
                st.write("")
                _can_venn = bool(_cnae_code)

                if st.button(
                    "📊 Visualizar Diagrama de Venn",
                    type="primary", width="content",
                    key="sg_r1_venn_btn",
                    disabled=not _can_venn,
                ):
                    # -- Conjunto A --
                    _mask_va = (
                        (_df[_eff_ec] == st.session_state["sg_r1_emp_a"])
                        & (_df[_eff_sc].isin(st.session_state["sg_r1_segs_a"]))
                    )
                    _set_A = set(_df.loc[_mask_va, _cnpj_col].dropna().astype(str))

                    # -- Conjunto B --
                    _emp_b_v  = st.session_state["sg_r1_emp_b"]
                    _segs_b_v = st.session_state["sg_r1_segs_b"]
                    if _emp_b_v and _segs_b_v:
                        _mask_vb = (
                            (_df[_eff_ec] == _emp_b_v)
                            & (_df[_eff_sc].isin(_segs_b_v))
                        )
                        _set_B = set(_df.loc[_mask_vb, _cnpj_col].dropna().astype(str))
                    else:
                        _set_B = set()

                    # -- Conjunto C (CNAE) --
                    if "cnae_principal" in _df.columns:
                        _set_C = set(
                            _df.loc[
                                _df["cnae_principal"].astype(str) == str(_cnae_code),
                                _cnpj_col,
                            ].dropna().astype(str)
                        )
                    elif parquet_exists(_UNI_PATH):
                        try:
                            _all_cnpjs_v = set(_df[_cnpj_col].dropna().astype(str))
                            _con_c = duckdb.connect()
                            _cf    = pd.DataFrame({"_c": list(_all_cnpjs_v)})
                            _con_c.register("af", _cf)
                            _res_c = _con_c.execute(
                                "SELECT DISTINCT b.cnpj FROM '{}' b "
                                "INNER JOIN af a ON a._c = b.cnpj "
                                "WHERE b.cnae_principal = ?".format(_UNI_P),
                                [_cnae_code],
                            ).df()
                            _con_c.close()
                            _set_C = set(_res_c["cnpj"].astype(str))
                        except Exception:
                            _set_C = set()
                    else:
                        _set_C = set()

                    # -- Value map --
                    _val_map_v: dict = {}
                    if _eff_vc and _eff_vc in _df.columns:
                        _vs = pd.to_numeric(_df[_eff_vc].astype(str).str.replace(",", "."), errors="coerce").fillna(0)
                        _val_map_v = (
                            _df.assign(__v=_vs)
                            .groupby(_cnpj_col)["__v"]
                            .sum()
                            .to_dict()
                        )

                    _regions = _compute_venn3(_set_A, _set_B, _set_C, _val_map_v)

                    _emp_a_name = st.session_state["sg_r1_emp_a"] or ""
                    _segs_a_str = ", ".join(st.session_state["sg_r1_segs_a"])[:35]
                    _emp_b_name = _emp_b_v or "—"
                    _segs_b_str = (", ".join(_segs_b_v)[:35]) if _segs_b_v else "—"

                    st.session_state["sg_r1_venn_data"] = {
                        "regions":    _regions,
                        "has_values": bool(_val_map_v),
                        "labels": [
                            "{} · {}".format(_emp_a_name, _segs_a_str),
                            "{} · {}".format(_emp_b_name, _segs_b_str),
                            "CNAE {}".format(_cnae_code),
                        ],
                        "set_sizes": [
                            len(_set_A), len(_set_B), len(_set_C),
                        ],
                        "set_A": list(_set_A),
                        "set_B": list(_set_B),
                        "set_C": list(_set_C),
                    }

                # ----------- Venn chart + tabela detalhada -------------------
                _vd = st.session_state.get("sg_r1_venn_data")
                if _vd:
                    st.plotly_chart(
                        _draw_venn3(
                            _vd["regions"],
                            _vd["labels"],
                            _vd["set_sizes"],
                            _vd["has_values"],
                        ),
                        width="stretch",
                        key="sg_r1_venn_chart",
                    )

                    _mc_a, _mc_b, _mc_c = st.columns(3)
                    sz = _vd["set_sizes"]
                    _mc_a.metric("A — " + (_vd["labels"][0][:18]), "{:,}".format(sz[0]))
                    _mc_b.metric("B — " + (_vd["labels"][1][:18]), "{:,}".format(sz[1]))
                    _mc_c.metric("C — CNAE", "{:,}".format(sz[2]))

                    # ----------- Detalhamento por região ----------------------
                    st.divider()
                    st.markdown("#### 📋 Detalhamento por região")
                    st.caption(
                        "Expanda cada área para ver os **top 10 clientes por Valor Total**, "
                        "definir o rótulo de segmentação e aplicar ao grupo inteiro "
                        "ou a clientes específicos (modo granular)."
                    )

                    _set_names = [
                        _vd["labels"][0][:25],
                        _vd["labels"][1][:25],
                        "CNAE {}".format(st.session_state["sg_r1_cnae"]),
                    ]

                    # Columns to display inside each region expander
                    _disp_cols = [_cnpj_col]
                    for _dc in ("razao_social", "nome_fantasia", "uf", "municipio_nome"):
                        if _dc in _df.columns:
                            _disp_cols.append(_dc)
                    if _eff_vc and _eff_vc in _df.columns:
                        _disp_cols.append(_eff_vc)

                    # Numeric value series for sorting (NaN → 0)
                    if _eff_vc and _eff_vc in _df.columns:
                        _val_sort_s = pd.to_numeric(
                            _df[_eff_vc].astype(str).str.replace(",", "."),
                            errors="coerce",
                        ).fillna(0)
                        _df_r4 = _df.assign(__v=_val_sort_s)
                    else:
                        _df_r4 = _df.assign(__v=0)

                    # Sets used to recompute exclusive regions on-the-fly
                    _r4_sets = [
                        set(_vd["set_A"]),
                        set(_vd["set_B"]),
                        set(_vd["set_C"]),
                    ]
                    _r4_uni = _r4_sets[0] | _r4_sets[1] | _r4_sets[2]

                    for _rkey, (_rcnt, _rval) in sorted(
                        _vd["regions"].items(), key=lambda x: -x[1][0]
                    ):
                        if _rcnt == 0:
                            continue

                        # Stable string ID — e.g. "0", "1", "02", "012"
                        _rid = "".join(str(i) for i in sorted(_rkey))

                        # Human-readable region label
                        _rname = " ∩ ".join(_set_names[i] for i in sorted(_rkey))
                        _rexcl_str = ", ".join(
                            "¬" + _set_names[i] for i in range(3) if i not in _rkey
                        )

                        # Expander header
                        _ehdr = "**{}** — {:,} CNPJs".format(_rname, _rcnt)
                        if _rval > 0:
                            _ehdr += "  · R$ {:,.0f}".format(_rval)
                        if _rexcl_str:
                            _ehdr += "  _(excluindo {})_".format(_rexcl_str)

                        with st.expander(_ehdr, expanded=False):

                            # ---- Compute the exclusive CNPJs for this region ----
                            _reg_cnpjs: set = set(_r4_uni)
                            for _ri in range(3):
                                if _ri in _rkey:
                                    _reg_cnpjs &= _r4_sets[_ri]
                                else:
                                    _reg_cnpjs -= _r4_sets[_ri]

                            # Filtered + value-sorted slice of _df for this region
                            _df_reg = (
                                _df_r4[
                                    _df_r4[_cnpj_col].astype(str).isin(_reg_cnpjs)
                                ]
                                .sort_values("__v", ascending=False)
                            )
                            _show_c = [c for c in _disp_cols if c in _df_reg.columns]

                            # Top 10 table
                            st.caption("**Top 10 clientes por Valor Total**")
                            st.dataframe(
                                _df_reg[_show_c].head(10).reset_index(drop=True),
                                width="stretch",
                                hide_index=True,
                            )
                            st.caption(
                                "{:,} clientes nesta região{}".format(
                                    _rcnt,
                                    " · R$ {:,.0f} total".format(_rval) if _rval > 0 else "",
                                )
                            )
                            st.write("")

                            # Per-region session-state keys
                            _lbl_key = "sg_r1_lbl_{}".format(_rid)
                            _owt_key = "sg_r1_owt_{}".format(_rid)
                            _grn_key = "sg_r1_grn_{}".format(_rid)
                            for _k, _dv in (
                                (_lbl_key, ""), (_owt_key, False), (_grn_key, False)
                            ):
                                if _k not in st.session_state:
                                    st.session_state[_k] = _dv

                            # Label + overwrite + "Aplicar ao grupo" button
                            _rc1, _rc2 = st.columns([3, 1])
                            with _rc1:
                                _r_label = st.text_input(
                                    "Segmentação a aplicar ({})".format(SEG_COL),
                                    placeholder="ex: Mercado Alvo — CNAE {}".format(
                                        st.session_state["sg_r1_cnae"]
                                    ),
                                    key=_lbl_key,
                                )
                                _r_owt = st.checkbox(
                                    "Sobrescrever registros já classificados",
                                    key=_owt_key,
                                )
                            with _rc2:
                                st.write("")
                                st.write("")
                                if st.button(
                                    "✅ Aplicar ao grupo",
                                    key="sg_r1_grp_{}".format(_rid),
                                    type="primary",
                                    width="content",
                                    disabled=not bool(_r_label.strip()),
                                ):
                                    _ndf = st.session_state["sg_df"].copy()
                                    if SEG_COL not in _ndf.columns:
                                        _ndf[SEG_COL] = None
                                    _mk = _ndf[_cnpj_col].astype(str).isin(_reg_cnpjs)
                                    if not _r_owt:
                                        _mk = _mk & _ndf[SEG_COL].isna()
                                    _n_app = int(_mk.sum())
                                    _ndf.loc[_mk, SEG_COL] = _r_label.strip()
                                    st.session_state.update(
                                        {"sg_df": _ndf, "sg_dirty": True}
                                    )
                                    st.success(
                                        "✅ {:,} registros classificados como "
                                        "**{}**. Salve a versão na seção abaixo.".format(
                                            _n_app, _r_label.strip()
                                        )
                                    )
                                    st.rerun()

                            # ---- Modo granular (seleção individual) ----------
                            _gran_on = st.checkbox(
                                "🔧 Modo granular — selecionar clientes individuais",
                                key=_grn_key,
                            )
                            if _gran_on:
                                _MAX_GRAN = 500
                                if _rcnt > _MAX_GRAN:
                                    st.info(
                                        "Mostrando os {:,} clientes de maior valor "
                                        "(de {:,} no total).".format(_MAX_GRAN, _rcnt)
                                    )
                                _df_gran = (
                                    _df_reg[_show_c]
                                    .head(_MAX_GRAN)
                                    .reset_index(drop=True)
                                    .copy()
                                )
                                _df_gran.insert(0, "Incluir", True)
                                _col_cfg: dict = {
                                    "Incluir": st.column_config.CheckboxColumn(
                                        "Incluir", default=True
                                    )
                                }
                                for _cc in _show_c:
                                    _col_cfg[_cc] = st.column_config.Column(disabled=True)

                                _edited = st.data_editor(
                                    _df_gran,
                                    key="sg_r1_de_{}".format(_rid),
                                    width="stretch",
                                    hide_index=True,
                                    column_config=_col_cfg,
                                    num_rows="fixed",
                                )
                                _sel_cnpjs = set(
                                    _edited.loc[
                                        _edited["Incluir"] == True, _cnpj_col
                                    ].astype(str)
                                )
                                st.caption(
                                    "{:,} / {:,} clientes selecionados".format(
                                        len(_sel_cnpjs),
                                        min(_rcnt, _MAX_GRAN),
                                    )
                                )
                                if st.button(
                                    "✅ Aplicar selecionados",
                                    key="sg_r1_sel_{}".format(_rid),
                                    type="primary",
                                    width="content",
                                    disabled=not (
                                        bool(_r_label.strip()) and len(_sel_cnpjs) > 0
                                    ),
                                ):
                                    _ndf = st.session_state["sg_df"].copy()
                                    if SEG_COL not in _ndf.columns:
                                        _ndf[SEG_COL] = None
                                    _ms = _ndf[_cnpj_col].astype(str).isin(_sel_cnpjs)
                                    if not _r_owt:
                                        _ms = _ms & _ndf[SEG_COL].isna()
                                    _n_sel = int(_ms.sum())
                                    _ndf.loc[_ms, SEG_COL] = _r_label.strip()
                                    st.session_state.update(
                                        {"sg_df": _ndf, "sg_dirty": True}
                                    )
                                    st.success(
                                        "✅ {:,} registros selecionados classificados "
                                        "como **{}**. Salve a versão abaixo.".format(
                                            _n_sel, _r_label.strip()
                                        )
                                    )
                                    st.rerun()

# ---------------------------------------------------------------------------
# Section 5 — Salvar nova versão
# ---------------------------------------------------------------------------

st.divider()
st.subheader("5 · Salvar nova versão")

_next_ver = next_version_number(_sel_slug)

_sc1, _sc2 = st.columns([4, 1])
with _sc1:
    _save_label = st.text_input(
        "Descrição das alterações",
        placeholder="ex: Regra CNAE 4744 -> Construção Civil aplicada a 1.250 CNPJs",
        key="sg_save_label",
    )
    _dirty_hint = "  ⚠️ Há alterações não salvas." if st.session_state.get("sg_dirty") else ""
    st.caption(
        "Criará a versão **v{:03d}** ({:,} linhas · {:,} com {}).{}".format(
            _next_ver, _n_rows, _n_filled, SEG_COL, _dirty_hint
        )
    )
with _sc2:
    st.write("")
    _save_btn = st.button(
        "💾  Salvar v{:03d}".format(_next_ver),
        type="primary",
        width="content",
        key="sg_save_btn",
        disabled=not bool(_save_label.strip()),
    )

if _save_btn and _save_label.strip():
    _prev_df = None
    try:
        _prev_df = load_version(_sel_slug, _sel_ver_num)
    except Exception:
        pass

    _new_meta = build_meta(
        project=_sel_slug,
        version=_next_ver,
        label=_save_label.strip(),
        df=_df,
        source_file=_versions[0].get("source_file", ""),
        col_map=_col_map,
        prev_df=_prev_df,
    )
    with st.spinner("Salvando v{:03d}…".format(_next_ver)):
        save_version(_sel_slug, _df, _new_meta)

    st.session_state.update({
        "sg_version": _next_ver,
        "sg_meta":    _new_meta,
        "sg_dirty":   False,
    })
    st.success(
        "✅ Versão **v{:03d}** salva: *{}*".format(_next_ver, _save_label.strip())
    )
    st.rerun()
