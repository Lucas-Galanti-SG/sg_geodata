"""
Download automático do shapefile de municípios do IBGE.

FTP público:
  https://geoftp.ibge.gov.br/organizacao_do_territorio/
          malhas_territoriais/malhas_municipais/

A estrutura é:
  municipio_{year}/Brasil/BR_Municipios_{year}.zip  (~199 MB)

Uso:
  from utils.ibge_geo import download_municipios_shp
  shp = download_municipios_shp(dest_folder)   # escolhe o ano mais recente
"""

import re
import urllib.request
import zipfile
from pathlib import Path

_BASE = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/"
    "malhas_territoriais/malhas_municipais/"
)

_SHP_EXTS = {".shp", ".dbf", ".shx", ".prj", ".cpg"}


def latest_municipios_year() -> int:
    """Consulta o diretório IBGE e retorna o ano mais recente disponível."""
    req = urllib.request.Request(_BASE, headers={"User-Agent": "SGGeoData/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    years = [int(m) for m in re.findall(r"municipio_(\d{4})/", html)]
    if not years:
        raise RuntimeError(
            "Não foi possível determinar o ano disponível no servidor IBGE."
        )
    return max(years)


def municipios_zip_url(year: int) -> str:
    """Retorna a URL do ZIP nacional para o ano informado."""
    return f"{_BASE}municipio_{year}/Brasil/BR_Municipios_{year}.zip"


def download_municipios_shp(
    dest_folder: Path,
    year: int | None = None,
    progress_cb=None,
) -> Path:
    """
    Baixa e extrai BR_Municipios_{year}.zip do servidor IBGE.

    Parâmetros
    ----------
    dest_folder : pasta de destino (criada se não existir)
    year        : ano do shapefile; None → detecta o mais recente
    progress_cb : callable(downloaded_bytes: int, total_bytes: int)
                  chamado a cada chunk de 256 KB durante o download

    Retorna
    -------
    Path para o arquivo .shp extraído
    """
    if year is None:
        year = latest_municipios_year()

    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    shp_path = dest_folder / f"BR_Municipios_{year}.shp"
    if shp_path.exists():
        return shp_path

    url = municipios_zip_url(year)
    zip_path = dest_folder / f"BR_Municipios_{year}.zip"

    # ── Download ──────────────────────────────────────────────────────────
    req = urllib.request.Request(url, headers={"User-Agent": "SGGeoData/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.getheader("Content-Length") or 0)
        downloaded = 0
        chunk_size = 256 * 1024  # 256 KB
        with open(zip_path, "wb") as fh:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(downloaded, total)

    # ── Extração (flatten — sem subdiretórios) ────────────────────────────
    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.namelist():
            if Path(entry).suffix.lower() in _SHP_EXTS:
                data = zf.read(entry)
                (dest_folder / Path(entry).name).write_bytes(data)

    zip_path.unlink(missing_ok=True)
    return shp_path
