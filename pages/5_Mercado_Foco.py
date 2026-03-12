"""
Módulo 5 — Análise de Mercado Foco

Upload de base de clientes (CSV) → mapeamento de colunas → enriquecimento
via CNPJ contra base_unificada → distribuição de CNAEs primários/secundários
→ identificação de oportunidades não atendidas por UF e cidade.
"""

import re
import sys
import unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px

try:
    import geopandas as gpd
    _HAS_GEOPANDAS = True
except ImportError:
    _HAS_GEOPANDAS = False

from utils.config import get_subfolders
from utils.storage import parquet_exists
from utils import ibge_geo
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Mercado Foco · SGGeoData", page_icon="🎯", layout="wide")
render_sidebar()
st.title("🎯 Módulo 5 — Análise de Mercado Foco")
st.caption("Enriqueça sua base de clientes com dados RFB · descubra a cobertura e as oportunidades por CNAE")

subs = get_subfolders()
UNIFIED_PATH = subs["processed"] / "base_unificada.parquet"
CNAE_PATH    = subs["cnae_ibge"] / "cnae_hierarquia.parquet"
_UNI_P    = str(UNIFIED_PATH).replace("\\", "/")
FOCO_PATH = subs["processed"] / "base_foco.parquet"
_FOCO_P   = str(FOCO_PATH).replace("\\", "/")

if not parquet_exists(UNIFIED_PATH):
    st.warning("⚠️ Base unificada não encontrada. Execute o Módulo 2 (ETL) primeiro.")
    st.stop()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_cnpj(series: pd.Series) -> pd.Series:
    """Remove formatação e zero-pad para 14 dígitos."""
    return (
        series.astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(14)
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _load_cnae_desc() -> dict:
    if parquet_exists(CNAE_PATH):
        df = pd.read_parquet(CNAE_PATH)
        return df.set_index("subclasse")["subclasse_desc"].to_dict()
    return {}


def _normalize_name(s: str) -> str:
    """Remove acentos, converte para maiúsculas e strip."""
    return (
        unicodedata.normalize("NFD", str(s).upper().strip())
        .encode("ascii", "ignore")
        .decode("ascii")
    )


@st.cache_data(show_spinner=False)
def _load_centroids(shp_path: str) -> pd.DataFrame:
    """Carrega shapefile IBGE, calcula centróides e retorna DataFrame com lat/lon."""
    gdf = gpd.read_file(shp_path)
    gdf_proj = gdf.to_crs(epsg=5880)
    gdf_proj["centroid"] = gdf_proj.geometry.centroid
    gdf_proj = gdf_proj.set_geometry("centroid").to_crs(epsg=4326)
    gdf_proj["lat"] = gdf_proj.geometry.y
    gdf_proj["lon"] = gdf_proj.geometry.x
    gdf_proj["_join_key"] = (
        gdf_proj["NM_MUN"].apply(_normalize_name)
        + "|"
        + gdf_proj["SIGLA_UF"].str.upper().str.strip()
    )
    return gdf_proj[["_join_key", "CD_MUN", "lat", "lon"]].drop_duplicates("_join_key")


# Colunas que buscamos na base (RFB "vence" conflito de nomes com o upload)
_BASE_COLS = [
    "uf", "municipio_nome", "cnae_principal", "cnae_principal_desc",
    "cnaes_secundarios", "porte_desc", "razao_social", "nome_fantasia",
    "capital_social", "cep",
    "tipo_logradouro", "logradouro", "numero", "complemento", "bairro",
]


def _enrich(df_upload: pd.DataFrame, cnpj_col: str) -> pd.DataFrame:
    """LEFT JOIN do upload com base_unificada via CNPJ. Usa DuckDB anti-scan."""
    df = df_upload.copy()
    df["__cnpj_limpo"] = _clean_cnpj(df[cnpj_col])

    # Remove colunas do usuário que conflitam com as colunas base (base vence)
    df = df.drop(columns=[c for c in _BASE_COLS if c in df.columns])

    cnpj_df = df[["__cnpj_limpo"]].drop_duplicates()

    con = duckdb.connect()
    con.register("cnpj_list", cnpj_df)
    df_base = con.execute(f"""
        SELECT
            b.cnpj            AS __cnpj_limpo,
            b.uf,
            b.municipio_nome,
            b.cnae_principal,
            b.cnae_principal_desc,
            b.cnaes_secundarios,
            b.porte_desc,
            b.razao_social,
            b.nome_fantasia,
            b.capital_social,
            b.cep,
            b.tipo_logradouro,
            b.logradouro,
            b.numero,
            b.complemento,
            b.bairro
        FROM '{_UNI_P}' b
        INNER JOIN cnpj_list c ON c.__cnpj_limpo = b.cnpj
    """).df()
    con.close()

    return df.merge(df_base, on="__cnpj_limpo", how="left")


def _enrich_full(df_upload: pd.DataFrame, cnpj_col: str) -> pd.DataFrame:
    """LEFT JOIN do upload com base_unificada via CNPJ, trazendo TODOS os campos da RFB."""
    df = df_upload.copy()
    df["__cnpj_limpo"] = _clean_cnpj(df[cnpj_col])

    cnpj_df = df[["__cnpj_limpo"]].drop_duplicates()

    con = duckdb.connect()
    con.register("cnpj_list", cnpj_df)
    df_rfb = con.execute(f"""
        SELECT b.*
        FROM '{_UNI_P}' b
        INNER JOIN cnpj_list c ON c.__cnpj_limpo = b.cnpj
    """).df()
    con.close()

    # Renomeia coluna 'cnpj' da RFB para fazer o merge
    if "cnpj" in df_rfb.columns:
        df_rfb = df_rfb.rename(columns={"cnpj": "__cnpj_limpo"})

    # Colunas do usuário têm prioridade — descarta RFB cols duplicadas
    user_cols = set(df.columns) - {"__cnpj_limpo"}
    df_rfb = df_rfb.drop(columns=[c for c in df_rfb.columns
                                   if c != "__cnpj_limpo" and c in user_cols])

    return df.merge(df_rfb, on="__cnpj_limpo", how="left").drop(
        columns=["__cnpj_limpo"], errors="ignore"
    )


def _enrich_full_pivot(
    df_upload: pd.DataFrame, cnpj_col: str, seg_col: str, valor_col: str
) -> pd.DataFrame:
    """
    Tabela de dimensões: 1 linha por CNPJ com colunas {col}_{valor_segmento}.
    valor_col é somado; demais colunas usam o primeiro valor encontrado.
    Ao final, faz LEFT JOIN com base_unificada (RFB).
    """
    df = df_upload.copy()
    df["__cnpj_limpo"] = _clean_cnpj(df[cnpj_col])

    # Cópia do seg_col para incluí-la como coluna de valor no pivot
    _seg_copy = f"__{seg_col}_copy__"
    df[_seg_copy] = df[seg_col]

    # Todas as colunas a pivotar (exceto CNPJ original, chave limpa e seg_col que vira eixo)
    value_cols = [c for c in df.columns if c not in (cnpj_col, "__cnpj_limpo", seg_col)]

    agg_dict = {
        c: ("sum" if c == valor_col else "first")
        for c in value_cols
    }

    df_grp = df.groupby(["__cnpj_limpo", seg_col], as_index=False, dropna=True).agg(agg_dict)

    df_wide = df_grp.pivot(
        index="__cnpj_limpo",
        columns=seg_col,
        values=[c for c in df_grp.columns if c not in ("__cnpj_limpo", seg_col)],
    )
    # Flatten MultiIndex: (col, seg_val) → col_seg_val; substituir cópia pelo nome original
    df_wide.columns = [
        f"{seg_col}_{seg}" if col == _seg_copy else f"{col}_{seg}"
        for col, seg in df_wide.columns
    ]
    df_wide = df_wide.reset_index()

    # RFB enrichment
    cnpj_df = df_wide[["__cnpj_limpo"]].drop_duplicates()
    con = duckdb.connect()
    con.register("cnpj_list", cnpj_df)
    df_rfb = con.execute(f"""
        SELECT b.*
        FROM '{_UNI_P}' b
        INNER JOIN cnpj_list c ON c.__cnpj_limpo = b.cnpj
    """).df()
    con.close()

    if "cnpj" in df_rfb.columns:
        df_rfb = df_rfb.rename(columns={"cnpj": "__cnpj_limpo"})

    pivot_cols_set = set(df_wide.columns) - {"__cnpj_limpo"}
    df_rfb = df_rfb.drop(columns=[c for c in df_rfb.columns
                                   if c != "__cnpj_limpo" and c in pivot_cols_set])

    result = df_wide.merge(df_rfb, on="__cnpj_limpo", how="left")
    result = result.rename(columns={"__cnpj_limpo": cnpj_col})
    return result


def _parse_cnaes_sec(series: pd.Series) -> pd.Series:
    """Transforma coluna de CNAEs secundários em lista de códigos numéricos."""
    def _parse(val):
        if pd.isna(val) or str(val).strip() in ("", "nan"):
            return []
        codes = re.split(r"[,;/\s]+", str(val).strip())
        return [re.sub(r"[^0-9]", "", c).zfill(7) for c in codes
                if re.sub(r"[^0-9]", "", c)]
    return series.apply(_parse)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("mf_raw",           None)
ss.setdefault("mf_col_map",       None)
ss.setdefault("mf_enriched",      None)
ss.setdefault("mf_enriched_full", None)

# ---------------------------------------------------------------------------
# PASSO 1 — Upload
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("1 · Upload da base de clientes")

uploaded = st.file_uploader(
    "Selecione o arquivo CSV da sua base de clientes",
    type=["csv", "txt"],
    key="mf_uploader",
)

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded, dtype=str, sep=None, engine="python")
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()
    # Reset downstream se novo upload
    if ss["mf_raw"] is None or list(df_raw.columns) != list(ss["mf_raw"].columns):
        ss["mf_col_map"]       = None
        ss["mf_enriched"]      = None
        ss["mf_enriched_full"] = None
        ss["mf_nao_atend"]     = None
    ss["mf_raw"] = df_raw

if ss["mf_raw"] is None:
    st.info("Faça o upload de um CSV para começar.")
    st.stop()

df_raw = ss["mf_raw"]
st.success(f"✅ **{len(df_raw):,} linhas · {len(df_raw.columns)} colunas**")
with st.expander("Prévia (5 primeiras linhas)", expanded=False):
    st.dataframe(df_raw.head(5), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# PASSO 2 — Mapeamento de colunas
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("2 · Mapeamento de colunas")

cols = df_raw.columns.tolist()

# Auto-suggest CNPJ
cnpj_guess = next((c for c in cols if re.search(r"cnpj", c, re.I)), cols[0])

# Auto-suggest numérico: primeira coluna (exceto CNPJ) com >50 % valores numéricos
numeric_guess = next(
    (c for c in cols
     if c != cnpj_guess
     and pd.to_numeric(
         df_raw[c].astype(str).str.replace(",", "."), errors="coerce"
     ).notna().mean() > 0.5),
    next((c for c in cols if c != cnpj_guess), cols[0]),
)

c1, c2 = st.columns(2)
cnpj_col  = c1.selectbox("Coluna de CNPJ:", cols, index=cols.index(cnpj_guess), key="mf_cnpj")
valor_col = c2.selectbox(
    "Coluna de Valor (vendas, volume, resultado…):", cols,
    index=cols.index(numeric_guess), key="mf_valor",
)

outras = [c for c in cols if c not in (cnpj_col, valor_col)]
filtro_cols = st.multiselect(
    "Colunas de classificação a incluir no parquet (Canal, Segmento, Tipo…):",
    outras,
    key="mf_filtros",
    help=(
        "Essas colunas serão adicionadas ao parquet foco para cada cliente atendido. "
        "Use-as para filtrar e segmentar nas análises seguintes sem precisar do CSV original."
    ),
)

_do_enrich_full = st.checkbox(
    "📋 Enriquecer minha base com todos os dados disponíveis da Receita Federal",
    key="mf_do_enrich_full",
    value=False,
    help=(
        "Une a sua base com TODOS os campos de base_unificada "
        "(estabelecimentos + empresas + auxiliares) e disponibiliza para download em CSV."
    ),
)

if st.button("✅ Confirmar e enriquecer base", type="primary", width="stretch"):
    ss["mf_col_map"] = {"cnpj": cnpj_col, "valor": valor_col, "filtros": filtro_cols}
    ss["mf_enriched"]      = None
    ss["mf_enriched_full"] = None
    ss["mf_nao_atend"]     = None
    with st.spinner("Cruzando CNPJs com a base unificada…"):
        try:
            df_enr = _enrich(df_raw, cnpj_col)
            df_enr[valor_col] = pd.to_numeric(
                df_enr[valor_col].astype(str).str.replace(",", "."), errors="coerce"
            )
            ss["mf_enriched"] = df_enr
            found = df_enr["cnae_principal"].notna().sum()
            st.success(
                f"✅ {found:,} de {len(df_enr):,} CNPJs encontrados na base RFB "
                f"({found/len(df_enr)*100:.1f} %)"
            )
        except Exception as e:
            st.error(f"Erro no enriquecimento: {e}")
            raise
if ss["mf_enriched"] is None:
    st.stop()

# ---------------------------------------------------------------------------
# Exportar base própria enriquecida com dados RFB
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Exportar base própria enriquecida com dados da Receita Federal
# ---------------------------------------------------------------------------

if ss.get("mf_do_enrich_full"):
    st.markdown("---")
    st.subheader("📋 Exportar base própria enriquecida com dados da Receita Federal")

    _cnpj_col_full  = ss["mf_col_map"]["cnpj"]
    _valor_col_full = ss["mf_col_map"]["valor"]
    _df_up          = ss["mf_raw"]

    # Verificar duplicação de CNPJs
    _cnpj_clean_full = _clean_cnpj(_df_up[_cnpj_col_full])
    _n_total_full    = len(_cnpj_clean_full)
    _n_unique_full   = _cnpj_clean_full.nunique()
    _has_dupes_full  = _n_unique_full < _n_total_full

    if _has_dupes_full:
        st.info(
            f"ℹ️ Sua base tem **{_n_total_full:,} linhas** e **{_n_unique_full:,} CNPJs únicos** "
            "— o mesmo CNPJ aparece em múltiplas linhas, provavelmente de empresas/grupos diferentes."
        )
        _mode_full = st.radio(
            "Modo de exportação:",
            options=["linha_a_linha", "dimensao"],
            format_func=lambda x: (
                "📄 Linha a linha — mantém todas as linhas originais, dados RFB repetidos para cada CNPJ"
                if x == "linha_a_linha" else
                "📐 Tabela de dimensões — uma linha por CNPJ, colunas sufixadas por valor de segmento"
            ),
            key="mf_enrich_mode",
            horizontal=False,
        )
    else:
        _mode_full = "linha_a_linha"
        st.success(f"✅ **{_n_unique_full:,} CNPJs únicos** — nenhuma duplicação detectada.")

    _seg_col_full  = None
    _can_generate  = True

    if _mode_full == "dimensao":
        _other_cols_full = [c for c in _df_up.columns if c != _cnpj_col_full]
        _seg_col_full = st.selectbox(
            "Coluna de segmentação (seus valores viram sufixos das colunas):",
            _other_cols_full,
            key="mf_enrich_seg_col",
        )
        _seg_vals_full = sorted(_df_up[_seg_col_full].dropna().unique().tolist())
        _n_seg_full    = len(_seg_vals_full)
        st.caption(
            f"**{_n_seg_full}** valores distintos em *{_seg_col_full}*: "
            + ", ".join(f"`{v}`" for v in _seg_vals_full[:15])
            + (" …" if _n_seg_full > 15 else "")
        )
        if _n_seg_full > 10:
            st.error(
                f"❌ **{_n_seg_full}** valores distintos em *{_seg_col_full}* — "
                "máximo permitido é **10** para evitar explosão de colunas. "
                "Escolha outra coluna ou use o modo linha a linha."
            )
            _can_generate = False
        else:
            _pivot_preview_cols = [
                c for c in _df_up.columns
                if c not in (_cnpj_col_full, _seg_col_full)
            ]
            _preview_ex = ", ".join(
                f"`{c}_{v}`"
                for c in _pivot_preview_cols[:2]
                for v in _seg_vals_full[:3]
            )
            st.caption(f"Exemplo de colunas geradas: {_preview_ex} … + campos RFB")

    if _can_generate:
        if st.button(
            "🔄 Gerar exportação enriquecida",
            key="mf_btn_gen_full",
            type="primary",
            width="stretch",
        ):
            ss["mf_enriched_full"] = None
            with st.spinner("Processando enriquecimento…"):
                try:
                    if _mode_full == "linha_a_linha":
                        ss["mf_enriched_full"] = _enrich_full(_df_up, _cnpj_col_full)
                    else:
                        ss["mf_enriched_full"] = _enrich_full_pivot(
                            _df_up, _cnpj_col_full, _seg_col_full, _valor_col_full
                        )
                    # ── Geocodificação lat/lon via shapefile IBGE ─────────────────
                    if _HAS_GEOPANDAS:
                        _shp_q = subs["shapefiles"]
                        _shp_auto = next(
                            iter(sorted(_shp_q.glob("BR_Municipios_*.shp"), reverse=True)),
                            None,
                        )
                        if _shp_auto is None:
                            _legacy_shp = (
                                Path(__file__).resolve().parent.parent
                                / "BR_Municipios_2024.shp"
                            )
                            if _legacy_shp.exists():
                                _shp_auto = _legacy_shp
                        _df_geo = ss["mf_enriched_full"]
                        if (
                            _shp_auto is not None
                            and "municipio_nome" in _df_geo.columns
                            and "uf" in _df_geo.columns
                        ):
                            try:
                                _centroids_exp = _load_centroids(str(_shp_auto))
                                _jk_exp = (
                                    _df_geo["municipio_nome"].apply(_normalize_name)
                                    + "|"
                                    + _df_geo["uf"].str.upper().str.strip()
                                )
                                _cidx_exp = _centroids_exp.set_index("_join_key")
                                _df_geo = _df_geo.copy()
                                _df_geo["lat"]    = _jk_exp.map(_cidx_exp["lat"])
                                _df_geo["lon"]    = _jk_exp.map(_cidx_exp["lon"])
                                _df_geo["CD_MUN"] = _jk_exp.map(_cidx_exp["CD_MUN"])
                                ss["mf_enriched_full"] = _df_geo
                            except Exception as _geo_err:
                                st.warning(f"⚠️ Geocodificação falhou: {_geo_err}")
                    st.success("✅ Enriquecimento concluído!")
                except Exception as _fe:
                    ss["mf_enriched_full"] = None
                    st.error(f"Erro no enriquecimento: {_fe}")

    if ss.get("mf_enriched_full") is not None:
        _df_full      = ss["mf_enriched_full"]
        _rfb_det_col  = next((c for c in _df_full.columns if c == "razao_social"), None)
        _n_found_full = int(_df_full[_rfb_det_col].notna().sum()) if _rfb_det_col else "?"
        st.caption(
            f"**{len(_df_full):,}** linhas · **{_n_found_full}** com dados RFB · "
            f"**{len(_df_full.columns)}** colunas"
        )
        with st.expander("Prévia (5 primeiras linhas)", expanded=False):
            st.dataframe(_df_full.head(5), width="stretch", hide_index=True)
        _csv_full = _df_full.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
        st.download_button(
            label="⬇️ Baixar CSV enriquecido completo",
            data=_csv_full,
            file_name="base_propria_enriquecida_rfb.csv",
            mime="text/csv",
            key="mf_dl_enriched_full",
        )

# ---------------------------------------------------------------------------
# PASSO 3 — Análise de CNAEs
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("3 · Distribuição de CNAEs na sua base")

df_enr    = ss["mf_enriched"]
col_map   = ss["mf_col_map"]
cnpj_col  = col_map["cnpj"]
valor_col = col_map["valor"]
filtro_cols = col_map["filtros"]
cnae_desc = _load_cnae_desc()

# ── Filtros do usuário ─────────────────────────────────────────────────────
if filtro_cols:
    with st.expander("🔎 Filtros da sua base", expanded=True):
        filter_vals: dict = {}
        fcols = st.columns(min(len(filtro_cols), 4))
        for i, fc in enumerate(filtro_cols):
            opts = sorted(df_enr[fc].dropna().unique().tolist())
            sel  = fcols[i % 4].multiselect(f"{fc}:", opts, key=f"mf_ff_{fc}")
            filter_vals[fc] = sel
    df_view = df_enr.copy()
    for fc, vals in filter_vals.items():
        if vals:
            df_view = df_view[df_view[fc].isin(vals)]
else:
    df_view = df_enr.copy()

df_found = df_view[df_view["cnae_principal"].notna()].copy()
df_found["cnae_principal"] = df_found["cnae_principal"].astype(str)
df_found["CNAE_Label"] = (
    df_found["cnae_principal"]
    + " · "
    + df_found["cnae_principal"].map(cnae_desc).fillna(
        df_found["cnae_principal_desc"] if "cnae_principal_desc" in df_found.columns
        else df_found["cnae_principal"]
    ).str[:55]
)

st.caption(
    f"**{len(df_view):,}** registros filtrados · "
    f"**{len(df_found):,}** com CNAE identificado · "
    f"**{len(df_view) - len(df_found):,}** sem correspondência"
)

if df_found.empty:
    st.warning("Nenhum CNPJ da sua base foi encontrado na base unificada.")
    st.stop()

# ── Top N por contagem e por valor ────────────────────────────────────────
top_n = st.slider("Top N CNAEs:", 5, 30, 10, key="mf_top_n")

col_left, col_right = st.columns(2)

with col_left:
    cnt = (
        df_found.groupby("CNAE_Label").size()
        .reset_index(name="Clientes")
        .sort_values("Clientes", ascending=False)
        .head(top_n)
    )
    fig_cnt = px.bar(
        cnt, x="Clientes", y="CNAE_Label", orientation="h",
        title=f"Top {top_n} CNAEs — por nº de clientes",
        labels={"CNAE_Label": ""},
    )
    fig_cnt.update_layout(
        height=max(400, top_n * 30),
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 10},
    )
    st.plotly_chart(fig_cnt, width="stretch")

with col_right:
    val_agg = (
        df_found.groupby("CNAE_Label")[valor_col]
        .sum().reset_index(name="Valor")
        .sort_values("Valor", ascending=False)
        .head(top_n)
    )
    fig_val = px.bar(
        val_agg, x="Valor", y="CNAE_Label", orientation="h",
        title=f"Top {top_n} CNAEs — por {valor_col}",
        labels={"CNAE_Label": ""},
    )
    fig_val.update_layout(
        height=max(400, top_n * 30),
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 10},
    )
    st.plotly_chart(fig_val, width="stretch")

# ── CNAEs secundários ─────────────────────────────────────────────────────
if "cnaes_secundarios" in df_found.columns:
    with st.expander("🔁 CNAEs secundários dos seus clientes", expanded=False):
        df_sec = df_found[["__cnpj_limpo", "cnaes_secundarios"]].copy()
        df_sec["cnaes_list"] = _parse_cnaes_sec(df_sec["cnaes_secundarios"])
        df_sec = df_sec.explode("cnaes_list").dropna(subset=["cnaes_list"])
        df_sec = df_sec[df_sec["cnaes_list"].str.len() == 7]
        df_sec["Label_Sec"] = (
            df_sec["cnaes_list"]
            + " · "
            + df_sec["cnaes_list"].map(cnae_desc).fillna("—").str[:55]
        )
        if not df_sec.empty:
            sec_counts = (
                df_sec.groupby("Label_Sec").size()
                .reset_index(name="Ocorrências")
                .sort_values("Ocorrências", ascending=False)
                .head(top_n)
            )
            fig_sec = px.bar(
                sec_counts, x="Ocorrências", y="Label_Sec", orientation="h",
                title=f"Top {top_n} CNAEs secundários",
                labels={"Label_Sec": ""},
            )
            fig_sec.update_layout(
                height=max(350, top_n * 28),
                yaxis={"categoryorder": "total ascending"},
                margin={"l": 10},
            )
            st.plotly_chart(fig_sec, width="stretch")
        else:
            st.info("Nenhum CNAE secundário identificado.")

# ── Ranking combinado (score = contagem norm. + valor norm.) ──────────────
cnt_rank = df_found.groupby("CNAE_Label").size().rename("cnt").reset_index()
val_rank = df_found.groupby("CNAE_Label")[valor_col].sum().rename("val").reset_index()
ranking  = cnt_rank.merge(val_rank, on="CNAE_Label", how="outer").fillna(0)
max_val  = ranking["val"].max()
ranking["score"] = (
    ranking["cnt"] / ranking["cnt"].max()
    + (ranking["val"] / max_val if max_val > 0 else 0)
)
ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# PASSO 4 — Base Foco (Parquet reduzido)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("4 · Geração de Base Foco")
st.caption(
    "Identifica as **Classes CNAE** (4 dígitos) que concentram 95 % dos seus clientes "
    "por volume e/ou por valor. Gera um Parquet compacto da base RFB filtrado "
    "para essas classes — ideal para migrar para outra máquina e eliminar "
    "a dependência dos arquivos pesados (~30 GB)."
)

# ── Código de classe (4 dígitos) + descrição via hierarquia ──────────────
if parquet_exists(CNAE_PATH):
    try:
        _hier_df = pd.read_parquet(CNAE_PATH, columns=["subclasse", "classe_desc"])
        _cls_desc_map = (
            _hier_df.assign(_k=_hier_df["subclasse"].str[:4])
            .drop_duplicates("_k")
            .set_index("_k")["classe_desc"]
            .to_dict()
        )
    except Exception:
        _cls_desc_map = {}
else:
    _cls_desc_map = {}

df_foco = df_found.copy()
df_foco["_cls_cod"]  = df_foco["cnae_principal"].str[:4]
df_foco["_cls_desc"] = df_foco["_cls_cod"].map(_cls_desc_map).fillna("")

# ── Agregações por classe ─────────────────────────────────────────────────
_total_cnt = len(df_foco)
_total_val = df_foco[valor_col].dropna().sum() if valor_col in df_foco.columns else 0.0

grp_cnt = (
    df_foco.groupby(["_cls_cod", "_cls_desc"])
    .size().reset_index(name="n_clientes")
    .sort_values("n_clientes", ascending=False)
    .reset_index(drop=True)
)
grp_cnt["_cumsum"] = grp_cnt["n_clientes"].cumsum() / _total_cnt

if _total_val > 0:
    grp_val = (
        df_foco.dropna(subset=[valor_col])
        .groupby("_cls_cod")[valor_col]
        .sum().reset_index(name="valor_total")
        .sort_values("valor_total", ascending=False)
        .reset_index(drop=True)
    )
    grp_val["_cumsum"] = grp_val["valor_total"].cumsum() / _total_val
else:
    grp_val = pd.DataFrame(columns=["_cls_cod", "valor_total", "_cumsum"])


def _top95(grp: pd.DataFrame) -> set:
    if grp.empty:
        return set()
    idx = grp[grp["_cumsum"] >= 0.95].index
    cutoff = int(idx[0]) if len(idx) > 0 else int(grp.index[-1])
    return set(grp.loc[:cutoff, "_cls_cod"])


classes_by_cnt = _top95(grp_cnt)
classes_by_val = _top95(grp_val) if _total_val > 0 else set()
classes_auto   = sorted(classes_by_cnt | classes_by_val)

# ── Tabela de resumo ──────────────────────────────────────────────────────
summary = grp_cnt[["_cls_cod", "_cls_desc", "n_clientes"]].copy()
if not grp_val.empty:
    summary = summary.merge(grp_val[["_cls_cod", "valor_total"]], on="_cls_cod", how="outer")
    summary["valor_total"] = summary["valor_total"].fillna(0.0)
else:
    summary["valor_total"] = 0.0
summary["n_clientes"] = summary["n_clientes"].fillna(0).astype(int)
summary["% Vol"]    = (summary["n_clientes"] / _total_cnt * 100).round(1)
summary["% Val"]    = (
    (summary["valor_total"] / _total_val * 100).round(1) if _total_val > 0 else 0.0
)
summary["Critério"] = summary["_cls_cod"].map(
    lambda c: "Vol + Val" if (c in classes_by_cnt and c in classes_by_val)
    else ("Volume" if c in classes_by_cnt else ("Valor" if c in classes_by_val else "—"))
)
summary_foco = (
    summary[summary["_cls_cod"].isin(classes_auto)]
    .sort_values("n_clientes", ascending=False)
    .reset_index(drop=True)
)

_mc1, _mc2, _mc3 = st.columns(3)
_mc1.metric("Classes — 95 % por volume",  len(classes_by_cnt))
_mc2.metric("Classes — 95 % por valor",   len(classes_by_val) if _total_val > 0 else "—")
_mc3.metric("Total classes selecionadas", len(classes_auto))

st.dataframe(
    summary_foco.rename(columns={
        "_cls_cod": "Classe", "_cls_desc": "Descrição",
        "n_clientes": "Clientes", "valor_total": f"Soma {valor_col}",
    })[["Classe", "Descrição", "Clientes", "% Vol",
        f"Soma {valor_col}", "% Val", "Critério"]],
    width="stretch",
    hide_index=True,
    column_config={
        "% Vol": st.column_config.NumberColumn("% Clientes", format="%.1f%%"),
        "% Val": st.column_config.NumberColumn(f"% {valor_col}", format="%.1f%%"),
        f"Soma {valor_col}": st.column_config.NumberColumn(
            f"Soma {valor_col}", format="%.0f"
        ),
    },
)

# ── Top subclasses por classe ─────────────────────────────────────────────
with st.expander("🔎 Principais subclasses por classe", expanded=False):
    for _, _row in summary_foco.iterrows():
        _cod  = _row["_cls_cod"]
        _desc = str(_row["_cls_desc"] or _cod)
        _sub_df = (
            df_foco[df_foco["_cls_cod"] == _cod]
            .groupby("CNAE_Label").size()
            .reset_index(name="Clientes")
            .sort_values("Clientes", ascending=False)
            .head(5)
        )
        st.markdown(
            f"**{_cod} · {_desc[:80]}** — "
            f"{_row['n_clientes']:,} clientes · {_row['% Vol']:.1f} %"
        )
        if not _sub_df.empty:
            st.dataframe(_sub_df, width="stretch", hide_index=True)

# ── Seleção e geração ─────────────────────────────────────────────────────
st.markdown("#### Selecione as classes para o Parquet Foco")
_all_cls = sorted(summary["_cls_cod"].dropna().unique().tolist())
_cls_fmt  = dict(zip(summary["_cls_cod"], summary["_cls_desc"].fillna("")))
classes_sel = st.multiselect(
    "Classes incluídas na base foco (pré-selecionadas = cobertura 95 %):",
    _all_cls,
    default=classes_auto,
    key="mf_classes_foco",
    format_func=lambda c: f"{c} · {_cls_fmt.get(c, '')}",
)

_nome_foco = st.text_input(
    "Nome do arquivo (sem extensão):",
    value="base_foco",
    key="mf_nome_foco",
).strip().removesuffix(".parquet") or "base_foco"
_foco_custom_path = subs["processed"] / f"{_nome_foco}.parquet"
_foco_custom_p    = str(_foco_custom_path).replace("\\", "/")

# ── Opcional: enriquecimento lat/lon via shapefile IBGE ───────────────────
_shp_folder = subs["shapefiles"]


def _find_existing_shp(folder: Path) -> Path | None:
    found = sorted(folder.glob("BR_Municipios_*.shp"), reverse=True)
    return found[0] if found else None


_existing_shp = _find_existing_shp(_shp_folder)
# Fallback: legacy location at sggeodata/ root (arquivo movido manualmente)
if _existing_shp is None:
    _legacy = Path(__file__).resolve().parent.parent / "BR_Municipios_2024.shp"
    if _legacy.exists():
        _existing_shp = _legacy

_enrich_latlon = False
_shp_path = str(_existing_shp) if _existing_shp else ""
if _HAS_GEOPANDAS:
    _enrich_latlon = st.checkbox(
        "📍 Enriquecer com lat/lon (centróide do município via shapefile IBGE)",
        value=True,
        key="mf_latlon",
    )
    if _enrich_latlon:
        if _existing_shp is None:
            # Shapefile ausente — oferecer download automático
            try:
                _ibge_year = ibge_geo.latest_municipios_year()
            except Exception:
                _ibge_year = 2024
            st.warning(
                f"📥 Shapefile não encontrado. "
                f"Clique em **Baixar** para obter automaticamente o "
                f"`BR_Municipios_{_ibge_year}.shp` do IBGE (≈ 199 MB)."
            )
            if st.button(
                f"📥 Baixar BR_Municipios_{_ibge_year}.shp do IBGE",
                key="mf_btn_dl_shp",
            ):
                _dl_bar = st.progress(0, text="Iniciando download…")

                def _dl_cb(dl: int, tot: int) -> None:
                    if tot:
                        _dl_bar.progress(
                            min(dl / tot, 1.0),
                            text=(
                                f"Baixando… {dl/1_048_576:.0f} /"
                                f" {tot/1_048_576:.0f} MB"
                            ),
                        )

                try:
                    _new_shp = ibge_geo.download_municipios_shp(
                        _shp_folder, _ibge_year, _dl_cb
                    )
                    _dl_bar.progress(1.0, text="✅ Download concluído!")
                    st.success(f"Shapefile salvo em `{_new_shp}`")
                    st.rerun()
                except Exception as _dl_err:
                    st.error(f"Falha no download: {_dl_err}")
            _shp_path = str(_shp_folder / f"BR_Municipios_{_ibge_year}.shp")
        else:
            _shp_path = st.text_input(
                "Caminho do shapefile `BR_Municipios_*.shp`:",
                value=str(_existing_shp),
                key="mf_shp_path",
            )
else:
    st.caption("ℹ️ `geopandas` não instalado — enriquecimento lat/lon indisponível.")

if parquet_exists(_foco_custom_path):
    _foco_mb = _foco_custom_path.stat().st_size / 1_048_576
    st.info(
        f"📦 `{_nome_foco}.parquet` já existe ({_foco_mb:.0f} MB). "
        "Regere para atualizar a seleção de classes."
    )

if classes_sel:
    if st.button(f"💾 Gerar {_nome_foco}.parquet", type="primary",
                 width="stretch", key="mf_btn_foco"):
        _foco_custom_path.parent.mkdir(parents=True, exist_ok=True)
        _cls_ph = ", ".join(f"'{c}'" for c in classes_sel)
        with st.spinner(
            f"Filtrando {len(classes_sel)} classes na base unificada… "
            "(pode demorar alguns minutos na primeira vez)"
        ):
            try:
                import pyarrow.parquet as _pq
                import pyarrow as _pa
                import pyarrow.compute as _pac
                _classes_arr = _pa.array(list(classes_sel))
                _pq_src = _pq.ParquetFile(_UNI_P)
                _pq_writer = None
                try:
                    for _batch in _pq_src.iter_batches(batch_size=200_000):
                        _cnae4 = _pac.utf8_slice_codeunits(
                            _batch.column("cnae_principal"), 0, 4
                        )
                        _mask = _pac.is_in(_cnae4, value_set=_classes_arr)
                        _filtered = _batch.filter(_mask)
                        if _filtered.num_rows > 0:
                            _tbl = _pa.Table.from_batches([_filtered])
                            if _pq_writer is None:
                                _pq_writer = _pq.ParquetWriter(
                                    _foco_custom_p, _tbl.schema, compression="snappy"
                                )
                            _pq_writer.write_table(_tbl)
                finally:
                    if _pq_writer is not None:
                        _pq_writer.close()
                    elif not _foco_custom_path.exists():
                        _pq.write_table(
                            _pa.Table.from_batches([], schema=_pq_src.schema_arrow),
                            _foco_custom_p, compression="snappy",
                        )
                # ── Enriquecimento com dados da carteira de clientes ──────────────
                with st.spinner("Marcando clientes atendidos e adicionando colunas de classificação…"):
                    try:
                        _raw_df = ss.get("mf_raw")
                        _cmap_   = ss.get("mf_col_map") or {}
                        _cnpj_c_ = _cmap_.get("cnpj", "")
                        _val_c_  = _cmap_.get("valor", "")
                        _cls_c_  = [
                            fc for fc in (_cmap_.get("filtros") or [])
                            if _raw_df is not None and fc in _raw_df.columns
                        ]
                        if _raw_df is not None and _cnpj_c_ and _val_c_:
                            _cli = _raw_df[
                                [c for c in [_cnpj_c_, _val_c_] + _cls_c_
                                 if c in _raw_df.columns]
                            ].copy()
                            _cli["__cnpj_limpo"] = _clean_cnpj(_cli[_cnpj_c_])
                            _cli[_val_c_] = pd.to_numeric(
                                _cli[_val_c_].astype(str).str.replace(",", "."),
                                errors="coerce",
                            )
                            _cli = _cli.drop(columns=[_cnpj_c_])
                            _agg_ = {_val_c_: "sum"} | {fc: "first" for fc in _cls_c_}
                            _cli_agg = (
                                _cli.groupby("__cnpj_limpo", as_index=False)
                                .agg(_agg_)
                                .rename(columns={_val_c_: "relevancia_valor"})
                            )
                            _cli_agg["cliente_atendido"] = True
                            _df_pq = pd.read_parquet(_foco_custom_path)
                            # Renomeia colunas de classificação que conflitem com parquet
                            _existing_ = set(_df_pq.columns)
                            _cls_rename_ = {
                                fc: f"cli_{fc}" for fc in _cls_c_ if fc in _existing_
                            }
                            if _cls_rename_:
                                _cli_agg = _cli_agg.rename(columns=_cls_rename_)
                            _final_cls_ = [_cls_rename_.get(fc, fc) for fc in _cls_c_]
                            _df_pq = _df_pq.merge(
                                _cli_agg,
                                left_on="cnpj",
                                right_on="__cnpj_limpo",
                                how="left",
                            ).drop(columns=["__cnpj_limpo"], errors="ignore")
                            _df_pq["cliente_atendido"] = (
                                _df_pq["cliente_atendido"].fillna(False)
                            )
                            # ── Empresa-base já atendida ──────────────────────────
                            _df_pq["_cnpj_base_"] = _df_pq["cnpj"].str[:8]
                            _atend_bases_ = set(
                                _df_pq.loc[_df_pq["cliente_atendido"], "_cnpj_base_"]
                            )
                            _df_pq["empresa_base_atendida"] = (
                                _df_pq["_cnpj_base_"].isin(_atend_bases_)
                            )
                            _df_pq = _df_pq.drop(columns=["_cnpj_base_"])
                            _df_pq.to_parquet(_foco_custom_path, index=False)
                            _n_atend_ = int(_df_pq["cliente_atendido"].sum())
                            _n_base_atend_ = int(_df_pq["empresa_base_atendida"].sum())
                            _added_ = ["cliente_atendido", "empresa_base_atendida", "relevancia_valor"] + _final_cls_
                            st.caption(
                                f"🤝 **{_n_atend_:,}** estabelecimentos atendidos · "
                                f"**{_n_base_atend_:,}** com empresa-base atendida — "
                                "colunas adicionadas: "
                                + ", ".join(f"`{c}`" for c in _added_)
                            )
                    except Exception as _cli_err:
                        st.warning(
                            f"⚠️ Marcação de carteira falhou (parquet salvo sem esses campos): {_cli_err}"
                        )
                # ── Enriquecimento lat/lon ────────────────────────────────────────
                if _enrich_latlon and Path(_shp_path).exists():
                    with st.spinner("Calculando centróides e adicionando lat/lon…"):
                        try:
                            centroids = _load_centroids(_shp_path)
                            df_pq = pd.read_parquet(_foco_custom_path)
                            _join_keys = (
                                df_pq["municipio_nome"].apply(_normalize_name)
                                + "|"
                                + df_pq["uf"].str.upper().str.strip()
                            )
                            _cent_idx = centroids.set_index("_join_key")
                            df_pq["lat"]    = _join_keys.map(_cent_idx["lat"])
                            df_pq["lon"]    = _join_keys.map(_cent_idx["lon"])
                            df_pq["CD_MUN"] = _join_keys.map(_cent_idx["CD_MUN"])
                            matched = int(df_pq["lat"].notna().sum())
                            df_pq.to_parquet(_foco_custom_path, index=False)
                            st.caption(
                                f"📍 lat/lon adicionados — "
                                f"{matched:,}/{len(df_pq):,} registros geocodificados "
                                f"({matched/len(df_pq)*100:.1f}%)"
                            )
                        except Exception as _geo_err:
                            st.warning(
                                f"⚠️ Geocodificação falhou (parquet salvo sem lat/lon): {_geo_err}"
                            )
                _foco_mb = _foco_custom_path.stat().st_size / 1_048_576
                st.success(
                    f"✅ **{_nome_foco}.parquet** gerado com sucesso!  \n"
                    f"📦 Tamanho: **{_foco_mb:.0f} MB** · Localização: `{_foco_custom_path}`"
                )
                st.info(
                    f"💡 **Migração para outra máquina:** copie `{_nome_foco}.parquet` e "
                    "`cnae_hierarquia.parquet` para a pasta `processed/` no computador de destino. "
                    "Os passos seguintes utilizarão automaticamente esse arquivo compacto "
                    "no lugar da base completa."
                )
            except Exception as e:
                st.error(f"Erro ao gerar {_nome_foco}.parquet: {e}")
                raise

# ---------------------------------------------------------------------------
# → Oportunidades não atendidas
# ---------------------------------------------------------------------------

st.markdown("---")
st.info(
    "🔭 **Análise de oportunidades não atendidas** foi movida para o "
    "**Módulo 6**, onde você pode selecionar qualquer base foco gerada aqui "
    "e cruzar com sua carteira de clientes.  \n"
    "Use o menu lateral para acessar **6 · Oportunidades**."
)

