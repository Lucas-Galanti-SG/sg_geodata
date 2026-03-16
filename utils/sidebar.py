"""
Sidebar compartilhada do SGGeoData.

Chame `render_sidebar()` em cada página para manter a navegação
rica e consistente em todos os módulos.
"""

import streamlit as st
from pathlib import Path

from utils.config import load_config, set_data_folder, get_data_folder
from utils.storage import folder_size_mb, free_disk_gb


def render_sidebar() -> None:
    """Renderiza o painel lateral completo (configuração + navegação agrupada)."""
    with st.sidebar:
        st.title("🗺️ SGGeoData")
        st.caption("Inteligência Geográfica e Comercial — RFB + IBGE")
        st.divider()

        # ── Base Foco (Analysis) ──────────────────────────────────────────
        _cfg2 = load_config()
        _data_root = _cfg2.get("data_folder", "")
        _proc_dir = Path(_data_root) / "processed" if _data_root else None
        _SKIP_PARQUETS = frozenset([
            "base_unificada.parquet", "empresas.parquet",
            "estabelecimentos.parquet", "socios.parquet",
        ])
        _foco_files: list[Path] = []
        if _proc_dir and _proc_dir.exists():
            _foco_files = sorted(
                [
                    p for p in _proc_dir.glob("*.parquet")
                    if p.name not in _SKIP_PARQUETS
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        with st.expander("📊 Base Foco", expanded=not bool(
            st.session_state.get("_sb_foco_path")
        )):
            if not _foco_files:
                st.caption("Nenhuma base foco encontrada.\nExecute o Módulo 5 primeiro.")
            else:
                _foco_names = [p.name for p in _foco_files]
                _cur_path = st.session_state.get("_sb_foco_path")
                _cur_idx = 0
                if _cur_path:
                    try:
                        _cur_idx = _foco_names.index(Path(_cur_path).name)
                    except ValueError:
                        _cur_idx = 0
                _sel_name = st.selectbox(
                    "Parquet de trabalho",
                    options=_foco_names,
                    index=_cur_idx,
                    key="_sb_foco_sel",
                    label_visibility="collapsed",
                )
                _sel_path = _proc_dir / _sel_name
                import pyarrow.parquet as _pqlib
                try:
                    _meta = _pqlib.read_metadata(str(_sel_path))
                    _nrows = _meta.num_rows
                    _mb = _sel_path.stat().st_size / 1_048_576
                    st.caption(
                        f"📦 `{_sel_name}`  \n"
                        f"{_nrows:,} registros · {_mb:.0f} MB"
                    )
                except Exception:
                    pass
                if st.button("✅ Usar esta base", width="stretch", key="_sb_foco_btn"):
                    st.session_state["_sb_foco_path"] = str(_sel_path)
                    st.rerun()
        if st.session_state.get("_sb_foco_path"):
            _active = Path(st.session_state["_sb_foco_path"]).name
            st.caption(f"🗂️ Base ativa: `{_active}`")

        # ── Pasta de dados ────────────────────────────────────────────────
        cfg = load_config()
        current_folder = cfg.get("data_folder", "")


        with st.expander("⚙️ Pasta de Dados", expanded=not bool(current_folder)):
            st.markdown(
                "Pasta externa onde os dados brutos e processados são armazenados. "
                "Pode estar fora do diretório do projeto."
            )
            new_folder = st.text_input(
                "Caminho da pasta de dados",
                value=current_folder,
                placeholder="Ex: D:\\Dados\\SGGeoData",
                key="_sb_data_folder",
            )
            if st.button("💾 Salvar pasta", width="stretch", key="_sb_save_folder"):
                if new_folder.strip():
                    try:
                        p = set_data_folder(new_folder.strip())
                        st.success(f"Pasta configurada: {p}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")
                else:
                    st.warning("Informe um caminho válido.")

        if current_folder and Path(current_folder).exists():
            used = folder_size_mb(current_folder)
            free = free_disk_gb(current_folder)
            col1, col2 = st.columns(2)
            col1.metric("Usado", f"{used:.0f} MB")
            col2.metric("Livre", f"{free:.1f} GB")

        # ── Navegação agrupada ────────────────────────────────────────────
        st.divider()
        st.caption("🏗️ GERAÇÃO DE DADOS DE TRABALHO")
        st.page_link("pages/1_Mineracao.py",    label="1 · Mineração de Dados",   icon="⬇️")
        st.page_link("pages/2_ETL.py",          label="2 · ETL e Base Unificada", icon="🔧")
        st.page_link("pages/3_CNAE_IBGE.py",    label="3 · CNAEs IBGE",           icon="📚")
        st.page_link("pages/4_Exploracao.py",   label="4 · Exploração de Dados",  icon="📊")
        st.page_link("pages/5_Mercado_Foco.py", label="5 · Mercado Foco",         icon="🎯")
        st.divider()
        st.caption("📈 ANÁLISE E PROSPECÇÃO")
        st.page_link("pages/6_Oportunidades.py",         label="6 · Oportunidades",         icon="🔭")
        st.page_link("pages/7_Atendimento_Indireto.py",  label="7 · Atendimento Indireto",  icon="🏭")
        st.page_link("pages/8_Sinergia.py",              label="8 · Sinergia e Multiclass.", icon="🔀")
        st.divider()
        st.caption("v0.1 · Março 2026")
