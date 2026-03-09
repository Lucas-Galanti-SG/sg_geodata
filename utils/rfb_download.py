"""
Download dos dados abertos da Receita Federal do Brasil via repositório SERPRO.

Repositório: https://arquivos.receitafederal.gov.br/index.php/s/YggdBLfdninEJX9
Protocolo:   Nextcloud WebDAV (HTTPS, SSL não verificado — CA corporativa SERPRO)
Autenticação: HTTP Basic com share token (sem senha)
"""

from __future__ import annotations

import base64
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests
import urllib3

# Suprime InsecureRequestWarning do urllib3 (CA corporativa SERPRO não está no bundle)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SERPRO_BASE  = "https://arquivos.receitafederal.gov.br"
SHARE_TOKEN  = "YggdBLfdninEJX9"
WEBDAV_BASE  = f"{SERPRO_BASE}/public.php/webdav"

_AUTH_HEADER = {
    "Authorization": "Basic " + base64.b64encode(f"{SHARE_TOKEN}:".encode()).decode()
}

_NUMBERED_GROUPS = ["empresas", "estabelecimentos", "socios"]


# ---------------------------------------------------------------------------
# WebDAV helpers
# ---------------------------------------------------------------------------

def _propfind(path: str, depth: int = 1) -> ET.Element:
    """Executa PROPFIND WebDAV e retorna o XML parseado."""
    url = WEBDAV_BASE + path
    headers = {**_AUTH_HEADER, "Depth": str(depth)}
    resp = requests.request("PROPFIND", url, headers=headers, verify=False, timeout=30)
    resp.raise_for_status()
    return ET.fromstring(resp.content)


def _href_name(href: str) -> str:
    return urllib.parse.unquote(href.rstrip("/").split("/")[-1])


# ---------------------------------------------------------------------------
# Listagem de versões e arquivos
# ---------------------------------------------------------------------------

def list_versions() -> list[str]:
    """
    Lista as pastas de versão disponíveis no repositório SERPRO.

    Returns
    -------
    Lista de strings no formato 'YYYY-MM', ordenada.
    """
    try:
        xml_data = _propfind("/", depth=1)
    except requests.RequestException as e:
        raise ConnectionError(f"Não foi possível listar versões: {e}") from e

    versions = []
    for resp in xml_data.findall(".//{DAV:}response"):
        href = resp.find("{DAV:}href").text
        col  = resp.find(".//{DAV:}collection")
        if col is not None and href != "/public.php/webdav/":
            name = _href_name(href)
            if name:
                versions.append(name)
    return sorted(versions)


def get_latest_version() -> str:
    """Retorna a versão mais recente disponível (ex: '2026-02')."""
    versions = list_versions()
    if not versions:
        raise RuntimeError("Nenhuma versão encontrada no repositório.")
    return versions[-1]


def list_version_files(version: str) -> list[dict]:
    """
    Lista os arquivos de uma versão específica.

    Returns
    -------
    Lista de dicts: {name, url, size_mb}
    """
    try:
        xml_data = _propfind(f"/{version}/", depth=1)
    except requests.RequestException as e:
        raise ConnectionError(f"Não foi possível listar arquivos de '{version}': {e}") from e

    files = []
    for resp in xml_data.findall(".//{DAV:}response"):
        href     = resp.find("{DAV:}href").text
        col      = resp.find(".//{DAV:}collection")
        size_el  = resp.find(".//{DAV:}getcontentlength")
        name     = _href_name(href)
        if col is None and name:
            size = int(size_el.text) if size_el is not None else 0
            url  = f"{WEBDAV_BASE}/{version}/{urllib.parse.quote(name)}"
            files.append({
                "name":    name,
                "url":     url,
                "size_mb": size / 1024 / 1024 if size else None,
            })
    return sorted(files, key=lambda f: f["name"])


def list_rfb_files(version: str | None = None, **_) -> list[dict]:
    """Alias para list_version_files. Usa a versão mais recente se omitida."""
    if version is None:
        version = get_latest_version()
    return list_version_files(version)


# ---------------------------------------------------------------------------
# Categorização
# ---------------------------------------------------------------------------

def categorize_files(files: list[dict]) -> dict:
    """
    Separa os arquivos em grupos: empresas, estabelecimentos, socios, auxiliares.
    """
    result = {g: [] for g in _NUMBERED_GROUPS}
    result["auxiliares"] = []

    for f in files:
        name_lower = f["name"].lower()
        matched = False
        for group in _NUMBERED_GROUPS:
            if name_lower.startswith(group):
                result[group].append(f)
                matched = True
                break
        if not matched:
            result["auxiliares"].append(f)

    return result


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(
    url: str,
    dest_path: Path,
    progress_callback=None,
    **_,  # absorve parâmetros legados como proxy=...
) -> Path:
    """
    Baixa um arquivo via WebDAV SERPRO com suporte a progresso.

    Parameters
    ----------
    url : URL WebDAV do arquivo
    dest_path : caminho destino
    progress_callback : callable(bytes_downloaded: int, total_bytes: int)
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers=_AUTH_HEADER, verify=False, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(dest_path, "wb") as fh:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)

    return dest_path


# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------

def extract_zip(zip_path: Path, dest_folder: Path, rename_to_csv: bool = True) -> list[Path]:
    """
    Extrai um arquivo ZIP para a pasta destino.

    Parameters
    ----------
    rename_to_csv : se True, renomeia arquivos sem extensão para .csv (padrão RFB)
    """
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    extracted = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            zf.extract(member, dest_folder)
            extracted_path = dest_folder / member
            if rename_to_csv and extracted_path.suffix.lower() != ".csv":
                # Acrescenta .csv ao nome COMPLETO para preservar unicidade
                # Ex: F.K03200$Z.D60214.CNAECSV  →  F.K03200$Z.D60214.CNAECSV.csv
                new_path = extracted_path.parent / (extracted_path.name + ".csv")
                if new_path.exists():
                    new_path.unlink()
                extracted_path.rename(new_path)
                extracted.append(new_path)
            else:
                extracted.append(extracted_path)

    return extracted


# ---------------------------------------------------------------------------
# Fallback estático
# ---------------------------------------------------------------------------

def get_known_file_list(version: str = "2026-02") -> list[dict]:
    """Lista estática de arquivos RFB (fallback quando o repositório está inacessível)."""
    files = []
    approx = {"empresas": 150, "estabelecimentos": 500, "socios": 100}
    for group in _NUMBERED_GROUPS:
        for i in range(10):
            name = f"{group.capitalize()}{i}.zip"
            files.append({
                "name":    name,
                "url":     f"{WEBDAV_BASE}/{version}/{name}",
                "size_mb": approx.get(group),
            })
    for aux in ["Cnaes", "Motivos", "Municipios", "Naturezas", "Paises", "Qualificacoes", "Simples"]:
        name = f"{aux}.zip"
        files.append({
            "name":    name,
            "url":     f"{WEBDAV_BASE}/{version}/{name}",
            "size_mb": None,
        })
    return files
