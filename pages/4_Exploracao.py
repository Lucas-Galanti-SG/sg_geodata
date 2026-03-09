"""
Módulo 4 — Exploração de Dados

Visualizações interativas da base unificada RFB + busca textual nos CNAEs do IBGE.
UmFiltros hierárquicos (Seção → Divisão → Grupo → Classe) restringem a consulta ao parquet
via DuckDB antes de carregar qualquer dado para memória.
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.config import get_subfolders
from utils.storage import parquet_exists
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Exploração · SGGeoData", page_icon="📊", layout="wide")
render_sidebar()
st.title("📊 Módulo 4 — Exploração de Dados")
st.caption("Visualizações da base pública RFB e busca textual em CNAEs do IBGE")

subs = get_subfolders()
UNIFIED_PATH = subs["processed"] / "base_unificada.parquet"
CNAE_PATH    = subs["cnae_ibge"] / "cnae_hierarquia.parquet"
_UNI_P = str(UNIFIED_PATH).replace("\\", "/")

has_unified = parquet_exists(UNIFIED_PATH)
has_cnae    = parquet_exists(CNAE_PATH)

if not has_unified and not has_cnae:
    st.warning("⚠️ Nenhum dado processado encontrado. Execute os Módulos 2 e 3 primeiro.")
    st.stop()

# ---------------------------------------------------------------------------
# CNAE hierarchy — carrega sempre (1.332 linhas, instantâneo)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _load_cnae() -> pd.DataFrame:
    return pd.read_parquet(CNAE_PATH)

cnae = _load_cnae() if has_cnae else None

# ---------------------------------------------------------------------------
# Consulta filtrada via DuckDB (nunca carrega o parquet inteiro)
# ---------------------------------------------------------------------------

_COLS_EXPLO = (
    "uf, porte_desc, cnae_principal, cnae_principal_desc, "
    "data_inicio_atividade, municipio_nome, capital_social, cnpj_basico"
)

@st.cache_data(show_spinner="Consultando base de dados…", ttl=300)
def _query_unified(
    cnae_codes: tuple,
    ufs: tuple,
    portes: tuple,
    ano_de: int,
    ano_ate: int,
) -> pd.DataFrame:
    clauses: list[str] = []
    params: list = []

    if cnae_codes:
        ph = ", ".join(["?"] * len(cnae_codes))
        clauses.append(f"cnae_principal IN ({ph})")
        params.extend(cnae_codes)
    if ufs:
        ph = ", ".join(["?"] * len(ufs))
        clauses.append(f"uf IN ({ph})")
        params.extend(ufs)
    if portes:
        ph = ", ".join(["?"] * len(portes))
        clauses.append(f"porte_desc IN ({ph})")
        params.extend(portes)
    clauses.append(
        "(data_inicio_atividade IS NULL OR "
        "(YEAR(data_inicio_atividade) BETWEEN ? AND ?))"
    )
    params.extend([ano_de, ano_ate])

    where = " AND ".join(clauses) if clauses else "TRUE"
    sql = f"SELECT {_COLS_EXPLO} FROM '{_UNI_P}' WHERE {where}"
    return duckdb.connect().execute(sql, params).df()

@st.cache_data(show_spinner=False, ttl=300)
def _count_unified(cnae_code: str) -> int:
    sql = f"SELECT count(*) FROM '{_UNI_P}' WHERE cnae_principal = ?"
    return duckdb.connect().execute(sql, [cnae_code]).fetchone()[0]

# ---------------------------------------------------------------------------
# Tabs principais
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🏢 Base RFB",
    "🔍 Busca CNAE",
    "📋 Detalhes CNAE",
])

# ===========================================================================
# Tab 1 — Filtros hierárquicos + visualizações da base RFB
# ===========================================================================

_UFS_BR = [
    "AC","AL","AP","AM","BA","CE","DF","ES","GO","MA",
    "MT","MS","MG","PA","PB","PR","PE","PI","RJ","RN",
    "RS","RO","RR","SC","SP","SE","TO",
]

with tab1:
    if not has_unified:
        st.info("Execute o Módulo 2 (ETL) para gerar a base unificada.")
    else:
        # ── Painel de filtros ──────────────────────────────────────────────
        with st.expander("🔎 Filtros — defina antes de carregar os dados", expanded=True):

            # Linha 1 — Filtros CNAE hierárquicos (cascade)
            if cnae is not None:
                st.markdown("**Hierarquia CNAE** *(deixe em branco para todas)*")
                c1, c2, c3, c4 = st.columns(4)

                secao_opts = sorted(
                    cnae[["secao", "secao_desc"]].drop_duplicates()
                    .apply(lambda r: f"{r['secao']} · {r['secao_desc']}", axis=1)
                    .tolist()
                )
                secao_sel = c1.multiselect("Seção:", secao_opts, key="f_secao")
                secao_codes = [s.split(" · ")[0] for s in secao_sel]

                cnae_s = cnae[cnae["secao"].isin(secao_codes)] if secao_codes else cnae
                div_opts = sorted(
                    cnae_s[["divisao", "divisao_desc"]].drop_duplicates()
                    .apply(lambda r: f"{r['divisao']} · {r['divisao_desc']}", axis=1)
                    .tolist()
                )
                div_sel = c2.multiselect("Divisão:", div_opts, key="f_div")
                div_codes = [d.split(" · ")[0] for d in div_sel]

                cnae_d = cnae_s[cnae_s["divisao"].isin(div_codes)] if div_codes else cnae_s
                grp_opts = sorted(
                    cnae_d[["grupo", "grupo_desc"]].drop_duplicates()
                    .apply(lambda r: f"{r['grupo']} · {r['grupo_desc']}", axis=1)
                    .tolist()
                )
                grp_sel = c3.multiselect("Grupo:", grp_opts, key="f_grp")
                grp_codes = [g.split(" · ")[0] for g in grp_sel]

                cnae_g = cnae_d[cnae_d["grupo"].isin(grp_codes)] if grp_codes else cnae_d
                cls_opts = sorted(
                    cnae_g[["classe", "classe_desc"]].drop_duplicates()
                    .apply(lambda r: f"{r['classe']} · {r['classe_desc']}", axis=1)
                    .tolist()
                )
                cls_sel = c4.multiselect("Classe:", cls_opts, key="f_cls")
                cls_codes = [c_.split(" · ")[0] for c_ in cls_sel]

                cnae_cls = cnae_g[cnae_g["classe"].isin(cls_codes)] if cls_codes else cnae_g
                sub_opts = sorted(
                    cnae_cls[["subclasse", "subclasse_desc"]].drop_duplicates()
                    .apply(lambda r: f"{r['subclasse']} · {r['subclasse_desc']}", axis=1)
                    .tolist()
                )
                sub_sel = st.multiselect("Subclasse (opcional):", sub_opts, key="f_sub")
                sub_codes = [s.split(" · ")[0] for s in sub_sel]

                if sub_codes:
                    cnae_final = cnae_cls[cnae_cls["subclasse"].isin(sub_codes)]
                else:
                    cnae_final = cnae_cls

                cnae_codes_sel = tuple(cnae_final["subclasse"].tolist()) if (
                    secao_codes or div_codes or grp_codes or cls_codes or sub_codes
                ) else ()

                n_sub = len(cnae_final)
                st.caption(
                    f"Subclasses abrangidas pelo filtro: **{n_sub:,}** de {len(cnae):,}"
                    if cnae_codes_sel
                    else f"Sem filtro CNAE — todas as **{len(cnae):,}** subclasses incluídas"
                )
            else:
                cnae_codes_sel = ()
                st.info("Banco CNAE não disponível (Módulo 3). Filtro hierárquico desabilitado.")

            st.divider()

            # Linha 2 — UF, Porte, Período
            c_uf, c_porte, c_ano1, c_ano2 = st.columns(4)
            uf_sel   = c_uf.multiselect("UF:", _UFS_BR, key="f_uf")
            porte_sel = c_porte.multiselect(
                "Porte:", ["ME", "EPP", "Demais"], key="f_porte"
            )
            ano_de  = c_ano1.number_input("Ano início (de):",  1970, 2026, 1970, key="f_ano_de")
            ano_ate = c_ano2.number_input("Ano início (até):", 1970, 2026, 2026, key="f_ano_ate")

        # ── Botão de carga ─────────────────────────────────────────────────
        col_btn, col_reset = st.columns([3, 1])
        carregar = col_btn.button(
            "🔄 Carregar / Atualizar dados", type="primary", width="stretch"
        )
        if col_reset.button("🗑️ Limpar cache", width="stretch"):
            _query_unified.clear()
            st.rerun()

        if carregar:
            st.session_state["t1_params"] = {
                "cnae_codes": cnae_codes_sel,
                "ufs":        tuple(uf_sel),
                "portes":     tuple(porte_sel),
                "ano_de":     int(ano_de),
                "ano_ate":    int(ano_ate),
            }

        params = st.session_state.get("t1_params")
        if params is None:
            st.info("Defina os filtros acima e clique em **Carregar / Atualizar dados**.")
        else:
            df_f = _query_unified(**params)
            st.caption(f"**{len(df_f):,}** estabelecimentos carregados com os filtros aplicados")

            # ── KPIs ───────────────────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Estabelecimentos", f"{len(df_f):,}")
            col2.metric("UFs",      df_f["uf"].nunique() if "uf" in df_f.columns else "—")
            col3.metric("CNAEs",    df_f["cnae_principal"].nunique() if "cnae_principal" in df_f.columns else "—")
            col4.metric("Municípios", df_f["municipio_nome"].nunique() if "municipio_nome" in df_f.columns else "—")

            col_a, col_b = st.columns(2)

            # Distribuição por UF
            with col_a:
                if "uf" in df_f.columns:
                    uf_counts = df_f["uf"].value_counts().reset_index()
                    uf_counts.columns = ["UF", "Qtd"]
                    fig = px.bar(
                        uf_counts.head(27), x="UF", y="Qtd",
                        title="Estabelecimentos por UF",
                        labels={"Qtd": "Qtd. Estabelecimentos"},
                    )
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, width="stretch")

            # Distribuição por Porte
            with col_b:
                if "porte_desc" in df_f.columns:
                    porte_counts = (
                        df_f["porte_desc"].fillna("Não Informado")
                        .value_counts().reset_index()
                    )
                    porte_counts.columns = ["Porte", "Qtd"]
                    fig2 = px.pie(
                        porte_counts, names="Porte", values="Qtd",
                        title="Distribuição por Porte", hole=0.35,
                    )
                    fig2.update_layout(height=380)
                    st.plotly_chart(fig2, width="stretch")

            # Boxplot de Capital Social por CNAE
            if "capital_social" in df_f.columns and "cnae_principal" in df_f.columns:
                cap_df = (
                    df_f[["cnpj_basico", "cnae_principal", "capital_social"]]
                    .drop_duplicates(subset=["cnpj_basico"])
                    .dropna(subset=["capital_social"])
                    .copy()
                )
                cap_df["capital_social"] = cap_df["capital_social"].astype(float)
                cap_df = cap_df[cap_df["capital_social"] > 0]

                if not cap_df.empty:
                    # Remove outliers por CNAE: mantém [P5, P95]
                    p05 = cap_df["capital_social"].quantile(0.05)
                    p95 = cap_df["capital_social"].quantile(0.95)
                    cap_df = cap_df[
                        cap_df["capital_social"].between(p05, p95)
                    ]

                    # Label legível para cada CNAE
                    if cnae is not None:
                        desc_map = cnae.set_index("subclasse")["subclasse_desc"].to_dict()
                        cap_df["CNAE_Label"] = (
                            cap_df["cnae_principal"].astype(str)
                            .map(desc_map)
                            .fillna(cap_df["cnae_principal"].astype(str))
                            .str[:50]
                        )
                    else:
                        cap_df["CNAE_Label"] = cap_df["cnae_principal"].astype(str)

                    # Limita aos top 10 CNAEs por volume para não poluir
                    top_cnaes = (
                        cap_df["CNAE_Label"].value_counts().head(10).index.tolist()
                    )
                    cap_plot = cap_df[cap_df["CNAE_Label"].isin(top_cnaes)].copy()
                    # Adiciona linha "Total"
                    cap_total = cap_plot.copy()
                    cap_total["CNAE_Label"] = "TOTAL"
                    cap_plot = pd.concat([cap_plot, cap_total], ignore_index=True)

                    # Médias por grupo
                    medias = cap_plot.groupby("CNAE_Label")["capital_social"].mean()

                    col_box, col_avg = st.columns([4, 1])
                    with col_box:
                        fig_box = px.box(
                            cap_plot,
                            x="capital_social",
                            y="CNAE_Label",
                            orientation="h",
                            title="Capital Social por CNAE — sem outliers (P5–P95)",
                            labels={"capital_social": "Capital Social (R$)", "CNAE_Label": ""},
                            points=False,
                        )
                        fig_box.update_layout(
                            height=max(400, len(top_cnaes) * 38 + 80),
                            yaxis={"categoryorder": "mean ascending"},
                            margin={"l": 10},
                        )
                        st.plotly_chart(fig_box, width="stretch")

                    with col_avg:
                        st.markdown("**Média por CNAE**")
                        for label, val in medias.sort_values().items():
                            st.markdown(
                                f"<small>**{label[:30]}**  \nR$ {val:,.0f}</small>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("Capital social não disponível para o filtro atual.")

            # Top CNAEs
            if "cnae_principal" in df_f.columns:
                top_n = st.slider("Top N CNAEs:", 5, 30, 15, key="top_n")
                cnae_counts = df_f["cnae_principal"].value_counts().head(top_n).reset_index()
                cnae_counts.columns = ["CNAE", "Qtd"]

                # Garante que o código CNAE seja string (evita eixo numérico)
                cnae_counts["CNAE"] = cnae_counts["CNAE"].astype(str)

                if cnae is not None:
                    desc_map = cnae.set_index("subclasse")["subclasse_desc"].to_dict()
                    mapped = cnae_counts["CNAE"].map(desc_map)
                    if "cnae_principal_desc" in df_f.columns:
                        fallback = (
                            df_f.drop_duplicates("cnae_principal")
                            .set_index("cnae_principal")["cnae_principal_desc"]
                            .reindex(cnae_counts["CNAE"].values)
                        )
                        fallback.index = cnae_counts.index
                        mapped = mapped.fillna(fallback).fillna("—")
                    else:
                        mapped = mapped.fillna("—")
                    cnae_counts["Descrição"] = mapped
                elif "cnae_principal_desc" in df_f.columns:
                    desc_lkp = df_f.drop_duplicates("cnae_principal").set_index("cnae_principal")["cnae_principal_desc"].astype(str)
                    cnae_counts["Descrição"] = cnae_counts["CNAE"].map(desc_lkp).fillna("—")

                # Label do eixo Y: "CÓDIGO · Descrição" (truncado)
                if "Descrição" in cnae_counts.columns:
                    cnae_counts["Label"] = cnae_counts.apply(
                        lambda r: f"{r['CNAE']} · {r['Descrição'][:60]}", axis=1
                    )
                else:
                    cnae_counts["Label"] = cnae_counts["CNAE"]

                fig3 = px.bar(
                    cnae_counts, x="Qtd", y="Label", orientation="h",
                    title=f"Top {top_n} CNAEs por número de estabelecimentos",
                    hover_data={"CNAE": True, "Descrição": True} if "Descrição" in cnae_counts.columns else None,
                    labels={"Qtd": "Estabelecimentos", "Label": "CNAE"},
                )
                fig3.update_layout(
                    height=max(400, top_n * 28),
                    yaxis={"categoryorder": "total ascending"},
                    margin={"l": 380},
                )
                st.plotly_chart(fig3, width="stretch")

            # Evolução temporal
            if "data_inicio_atividade" in df_f.columns:
                df_tempo = df_f.dropna(subset=["data_inicio_atividade"]).copy()
                df_tempo["data_inicio_atividade"] = pd.to_datetime(
                    df_tempo["data_inicio_atividade"], errors="coerce"
                )
                df_tempo["Ano"] = df_tempo["data_inicio_atividade"].dt.year
                ano_counts = df_tempo["Ano"].value_counts().sort_index().reset_index()
                ano_counts.columns = ["Ano", "Qtd"]
                ano_counts = ano_counts[ano_counts["Ano"] >= 1970]
                fig4 = px.area(
                    ano_counts, x="Ano", y="Qtd",
                    title="Abertura de estabelecimentos por ano",
                    labels={"Qtd": "Estabelecimentos abertos"},
                )
                fig4.update_layout(height=320)
                st.plotly_chart(fig4, width="stretch")

# ===========================================================================
# Tab 2 — Busca textual em CNAEs
# ===========================================================================

with tab2:
    if cnae is None:
        st.info("Execute o Módulo 3 (CNAEs IBGE) para habilitar a busca.")
    else:
        st.markdown(f"Banco com **{len(cnae):,} subclasses** em **{cnae['secao'].nunique()} seções**")

        query = st.text_input(
            "🔍 Buscar por texto (código, nome ou descrição):",
            placeholder="Ex: tintas, materiais de construção, comércio varejista…",
        )

        search_fields = st.multiselect(
            "Campos de busca:",
            ["subclasse", "subclasse_cod", "subclasse_desc", "classe_desc",
             "grupo_desc", "divisao_desc", "secao_desc"],
            default=["subclasse", "subclasse_desc"],
        )

        if query and search_fields:
            mask = pd.Series(False, index=cnae.index)
            for field in search_fields:
                if field in cnae.columns:
                    mask |= cnae[field].astype(str).str.contains(
                        query, case=False, na=False, regex=False
                    )

            results = cnae[mask].copy()
            st.success(f"**{len(results)}** subclasse(s) encontrada(s) para: *{query}*")

            display_cols = [c for c in [
                "secao", "secao_desc", "divisao", "grupo",
                "classe", "subclasse", "subclasse_cod", "subclasse_desc",
            ] if c in results.columns]

            st.dataframe(results[display_cols], width="stretch", hide_index=True)

            if not results.empty:
                sel_sub = st.selectbox(
                    "Ver detalhes da subclasse:",
                    results["subclasse"].tolist(),
                    format_func=lambda c: (
                        f"{c} — {results.loc[results['subclasse']==c, 'subclasse_desc'].values[0]}"
                    ),
                )
                row = results[results["subclasse"] == sel_sub].iloc[0]
                cod_fmt = row.get("subclasse_cod", row["subclasse"])
                st.markdown(f"### {cod_fmt} — {row['subclasse_desc']}")
                st.markdown(f"**Seção:** {row.get('secao','')} · {row.get('secao_desc','')}")
                st.markdown(f"**Divisão:** {row.get('divisao','')} · {row.get('divisao_desc','')}")
                st.markdown(f"**Grupo:** {row.get('grupo','')} · {row.get('grupo_desc','')}")
                st.markdown(f"**Classe:** {row.get('classe','')} · {row.get('classe_desc','')}")

                if has_unified:
                    if st.button("🔢 Contar estabelecimentos neste CNAE na base RFB"):
                        n = _count_unified(sel_sub)
                        st.metric("Estabelecimentos (base RFB)", f"{n:,}")

        elif query:
            st.warning("Selecione ao menos um campo de busca.")

# ===========================================================================
# Tab 3 — Detalhes de um CNAE específico
# ===========================================================================

with tab3:
    if cnae is None:
        st.info("Execute o Módulo 3 (CNAEs IBGE) para habilitar os detalhes.")
    else:
        cod_input = st.text_input(
            "Código CNAE (7 dígitos ou formato 0000-0/00):",
            placeholder="Ex: 4744099",
        )

        if cod_input:
            cod_norm = re.sub(r"[^0-9]", "", cod_input)
            row_match = cnae[cnae["subclasse"] == cod_norm]

            if row_match.empty:
                st.warning(f"CNAE `{cod_input}` não encontrado no banco do IBGE.")
            else:
                row = row_match.iloc[0]
                cod_fmt = row.get("subclasse_cod", row["subclasse"])
                st.markdown(f"## {cod_fmt} — {row['subclasse_desc']}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Hierarquia:**")
                    st.markdown(f"- **Seção:** `{row.get('secao','')}` — {row.get('secao_desc','')}")
                    st.markdown(f"- **Divisão:** `{row.get('divisao','')}` — {row.get('divisao_desc','')}")
                    st.markdown(f"- **Grupo:** `{row.get('grupo','')}` — {row.get('grupo_desc','')}")
                    st.markdown(f"- **Classe:** `{row.get('classe','')}` — {row.get('classe_desc','')}")
                    st.markdown(f"- **Subclasse:** `{cod_fmt}` — {row.get('subclasse_desc','')}")
                with col2:
                    if has_unified:
                        if st.button("🔢 Consultar total na base RFB", key="btn_t3"):
                            n = _count_unified(cod_norm)
                            st.metric("Estabelecimentos na base RFB", f"{n:,}")

