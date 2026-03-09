"""
Módulo 1 — Mineração de Dados

Download automatizado dos dados abertos da Receita Federal do Brasil
via repositório SERPRO Nextcloud.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from utils.config import get_data_folder, get_subfolders
from utils.storage import free_disk_gb, folder_size_mb
from utils.rfb_download import (
    list_versions, list_version_files,
    categorize_files, download_file, extract_zip,
)
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Mineração · SGGeoData", page_icon="⬇️", layout="wide")
render_sidebar()
st.title("⬇️ Módulo 1 — Mineração de Dados")
st.caption("Fonte: repositório SERPRO · https://arquivos.receitafederal.gov.br")

data_folder = get_data_folder()
subs = get_subfolders()


# ---------------------------------------------------------------------------
# Helpers de formatação
# ---------------------------------------------------------------------------

def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    else:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _fmt_speed(mbps: float) -> str:
    if mbps < 0.05:
        return "—"
    return f"{mbps:.1f} MB/s"


# ---------------------------------------------------------------------------
# 1. Seleção de versão
# ---------------------------------------------------------------------------

st.subheader("1 · Versão dos dados")

@st.cache_data(ttl=3600, show_spinner="Consultando repositório SERPRO...")
def _get_versions():
    return list_versions()

col_btn, col_info = st.columns([1, 3])
with col_btn:
    if st.button("🔄 Atualizar", width="stretch"):
        st.cache_data.clear()
        st.rerun()

try:
    versions = _get_versions()
    latest = versions[-1] if versions else None
except ConnectionError as e:
    st.error("⚠️ Não foi possível acessar o repositório SERPRO.")
    with st.expander("🔍 Detalhes do erro"):
        st.code(str(e))
    st.stop()

with col_info:
    st.caption(f"✅ {len(versions)} versões disponíveis · mais recente: **{latest}**")

selected_version = st.selectbox(
    "Selecione a versão para download:",
    options=list(reversed(versions)),
    index=0,
    help="Cada versão corresponde a um mês de atualização dos dados.",
)

# ---------------------------------------------------------------------------
# 2. Arquivos disponíveis
# ---------------------------------------------------------------------------

st.subheader("2 · Arquivos disponíveis")

@st.cache_data(ttl=1800, show_spinner="Listando arquivos...")
def _get_files(version: str):
    return list_version_files(version)

try:
    files = _get_files(selected_version)
    categorized = categorize_files(files)
except ConnectionError as e:
    st.error(f"⚠️ Erro ao listar arquivos da versão {selected_version}: {e}")
    st.stop()

total_rows = []
for group, flist in categorized.items():
    sizes = [f["size_mb"] for f in flist if f["size_mb"]]
    total_rows.append({
        "Grupo":    group.capitalize(),
        "Arquivos": len(flist),
        "~GB":      f"{sum(sizes)/1024:.2f}" if sizes else "<0.1",
    })
st.dataframe(pd.DataFrame(total_rows), width="stretch", hide_index=True)

with st.expander("Ver todos os arquivos"):
    detail_rows = [
        {"Grupo": g.capitalize(), "Arquivo": f["name"],
         "MB": f"{f['size_mb']:.0f}" if f["size_mb"] else "<1"}
        for g, flist in categorized.items() for f in flist
    ]
    st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 3. Espaço em disco
# ---------------------------------------------------------------------------

st.subheader("3 · Espaço em disco")

free_gb = free_disk_gb(data_folder)
used_mb = folder_size_mb(data_folder)
total_mb = sum(f["size_mb"] or 0 for f in files)

col1, col2, col3 = st.columns(3)
col1.metric("Livre no disco", f"{free_gb:.1f} GB")
col2.metric("Já baixado", f"{used_mb:.0f} MB")
col3.metric("Total estimado", f"{total_mb/1024:.1f} GB" if total_mb else "?")

if free_gb < total_mb / 1024:
    st.error(f"⚠️ Espaço insuficiente: {free_gb:.1f} GB livres, ~{total_mb/1024:.1f} GB necessários.")
elif free_gb < (total_mb / 1024) * 1.2:
    st.warning("⚠️ Pouco espaço livre. Recomendamos ao menos 20% de margem.")
else:
    st.success("✅ Espaço suficiente para o download.")

# ---------------------------------------------------------------------------
# 4. Seleção e download
# ---------------------------------------------------------------------------

st.subheader("4 · Selecionar grupos para download")

all_groups = list(categorized.keys())
group_labels = {g: f"{g.capitalize()} ({len(categorized[g])} arq.)" for g in all_groups}

selected_groups = st.multiselect(
    "Grupos a baixar:",
    options=all_groups,
    default=[g for g in ["empresas", "estabelecimentos"] if g in all_groups],
    format_func=lambda g: group_labels[g],
)

skip_existing = st.checkbox("Pular arquivos já baixados", value=True)

selected_files = [f for g in selected_groups for f in categorized[g]]

if selected_files:
    total_sel_mb = sum(f["size_mb"] or 0 for f in selected_files)
    st.info(
        f"**{len(selected_files)}** arquivo(s) selecionado(s) · "
        f"~{total_sel_mb/1024:.2f} GB estimados"
    )

if st.button(
    "⬇️ Iniciar Download + Extração",
    disabled=not selected_files,
    type="primary",
    width="stretch",
):
    total_files   = len(selected_files)
    total_all_b   = sum((f["size_mb"] or 0) * 1024 * 1024 for f in selected_files)
    bytes_done    = 0
    t0_total      = time.time()

    dest_map = {
        "empresas":         subs["raw_empresas"],
        "estabelecimentos": subs["raw_estab"],
        "socios":           subs["raw_socios"],
    }

    # UI containers — serão atualizados durante o download
    st.markdown("---")
    hdr_slot      = st.empty()
    bar_file_slot = st.empty()
    metrics_slot  = st.empty()
    bar_tot_slot  = st.empty()
    st.markdown("---")
    log_slot      = st.empty()
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(msg)
        log_slot.text_area("📋 Log", value="\n".join(log_lines[-40:]), height=220, disabled=True)

    def _refresh_total(done_b: float) -> None:
        pct = min(done_b / total_all_b, 1.0) if total_all_b > 0 else 0
        bar_tot_slot.progress(
            pct,
            text=f"📊 Total: {done_b/1024/1024:.0f} / {total_all_b/1024/1024:.0f} MB  ({pct*100:.0f}%)",
        )

    for idx, f in enumerate(selected_files):
        name        = f["name"]
        file_b      = (f["size_mb"] or 0) * 1024 * 1024
        zip_path    = subs["raw_zip"] / name
        grp         = next((g for g in dest_map if name.lower().startswith(g)), None)
        extract_dst = dest_map.get(grp, subs["raw_aux"])

        size_label = f"{f['size_mb']:.0f} MB" if f["size_mb"] else "tamanho desconhecido"
        hdr_slot.markdown(f"**📦 [{idx+1}/{total_files}] {name}** · {size_label}")

        zip_exists  = zip_path.exists()
        csvs_exist  = extract_dst.exists() and len(list(extract_dst.glob("*.csv"))) > 0

        # Só pula se ZIP já existe E CSVs já foram extraídos
        if skip_existing and zip_exists and csvs_exist:
            _log(f"⏭  [{idx+1}/{total_files}] Pulando (ZIP + CSVs já existem): {name}")
            bytes_done += file_b
            bar_file_slot.progress(1.0, text=f"⏭ {name} — já extraído")
            _refresh_total(bytes_done)
            continue

        # --- Download (apenas se o ZIP ainda não existe) ---
        if not zip_exists or not skip_existing:
            _log(f"⬇️  [{idx+1}/{total_files}] Baixando: {name}")

            file_t0      = time.time()
            last_ui_upd  = [0.0]

            def _on_progress(dl: int, total_b: int,
                             _idx=idx, _name=name, _file_t0=file_t0, _file_b=file_b):
                now = time.time()
                if now - last_ui_upd[0] < 0.4 and dl < total_b:
                    return
                last_ui_upd[0] = now

                elapsed   = max(now - _file_t0, 0.01)
                speed     = dl / elapsed / 1024 / 1024
                pct_f     = dl / total_b if total_b > 0 else 0
                eta_f     = (total_b - dl) / (dl / elapsed) if dl > 0 else 0

                so_far    = bytes_done + dl
                el_tot    = max(now - t0_total, 0.01)
                eta_tot   = (total_all_b - so_far) / (so_far / el_tot) if so_far > 0 else 0
                pct_tot   = min(so_far / total_all_b, 1.0) if total_all_b > 0 else 0

                try:
                    bar_file_slot.progress(
                        min(pct_f, 1.0),
                        text=(
                            f"{dl/1024/1024:.1f} / {total_b/1024/1024:.1f} MB"
                            f"  ({pct_f*100:.0f}%)  ·  {_fmt_speed(speed)}"
                        ),
                    )
                    with metrics_slot.container():
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("🚀 Velocidade",  _fmt_speed(speed))
                        c2.metric("⏱ ETA Arquivo",  _fmt_eta(eta_f))
                        c3.metric("📦 Arquivos",     f"{_idx + 1} / {total_files}")
                        c4.metric("⏳ ETA Total",    _fmt_eta(eta_tot))
                    bar_tot_slot.progress(
                        pct_tot,
                        text=f"📊 Total: {so_far/1024/1024:.0f} / {total_all_b/1024/1024:.0f} MB  ({pct_tot*100:.0f}%)",
                    )
                except Exception:
                    pass  # Ignora erros de UI durante progresso

            try:
                download_file(f["url"], zip_path, progress_callback=_on_progress)
                elapsed_dl = time.time() - file_t0
                speed_avg  = file_b / elapsed_dl / 1024 / 1024 if elapsed_dl > 0 else 0
                bytes_done += file_b
                bar_file_slot.progress(1.0, text=f"✅ {name} — {_fmt_eta(elapsed_dl)}  ·  média {_fmt_speed(speed_avg)}")
                _log(f"   ✅ Download: {name}  ({_fmt_eta(elapsed_dl)}, média {_fmt_speed(speed_avg)})")
            except Exception as exc:
                _log(f"   ❌ Erro no download de {name}: {exc}")
                st.error(f"❌ Erro baixando **{name}**: {exc}")
                bar_file_slot.progress(0.0, text=f"❌ Erro: {name}")
                continue
        else:
            _log(f"⏭  [{idx+1}/{total_files}] ZIP já existe, extraindo: {name}")
            bytes_done += file_b
            _refresh_total(bytes_done)

        # --- Extração ---
        _log(f"📦  Extraindo: {name} → {extract_dst}")
        bar_file_slot.progress(1.0, text=f"📦 Extraindo {name}…")
        try:
            extracted = extract_zip(zip_path, extract_dst, rename_to_csv=True)
            _log(f"   ✅ {len(extracted)} arquivo(s) extraído(s) em {extract_dst}")
        except Exception as exc:
            _log(f"   ❌ Erro na extração de {name}: {exc}")
            st.error(f"❌ Erro extraindo **{name}**: {exc}")

        _refresh_total(bytes_done)

    # Finalização
    elapsed_total = time.time() - t0_total
    bar_file_slot.progress(1.0, text="✅ Todos os arquivos processados!")
    bar_tot_slot.progress(1.0, text=f"✅ Total concluído — {bytes_done/1024/1024:.0f} MB")
    hdr_slot.markdown(f"**✅ Download e extração finalizados em {_fmt_eta(elapsed_total)}!**")
    st.success(f"✅ {total_files} arquivo(s) processado(s) em {_fmt_eta(elapsed_total)}.")

# ---------------------------------------------------------------------------
# 5. Conteúdo atual das pastas
# ---------------------------------------------------------------------------

st.subheader("5 · Conteúdo atual das pastas de dados")

folder_info = []
for key, path in subs.items():
    csv_count = len(list(path.glob("*.csv"))) if path.exists() else 0
    zip_count = len(list(path.glob("*.zip"))) if path.exists() else 0
    size = folder_size_mb(path)
    folder_info.append({
        "Pasta":   key,
        "Caminho": str(path),
        "CSVs":    csv_count,
        "ZIPs":    zip_count,
        "Tam.":    f"{size:.1f} MB",
    })

st.dataframe(pd.DataFrame(folder_info), width="stretch", hide_index=True)


st.set_page_config(page_title="Mineração · SGGeoData", page_icon="⬇️", layout="wide")
st.title("⬇️ Módulo 1 — Mineração de Dados")
st.caption("Download automatizado dos dados abertos da Receita Federal do Brasil")

data_folder = get_data_folder()
subs = get_subfolders()

