"""
Gerenciamento de configuração do SGGeoData.

Salva e lê config.json na raiz do projeto (sggeodata/).
A configuração principal é `data_folder`: pasta externa onde os dados
brutos e processados são armazenados (pode ser fora do repo).
"""

import json
import os
from pathlib import Path

# Pasta raiz do projeto (sggeodata/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Caminho do arquivo de configuração
CONFIG_FILE = _PROJECT_ROOT / "config.json"

_DEFAULTS = {
    "data_folder": str(_PROJECT_ROOT / "data"),
    "proxy_http": "",
    "proxy_https": "",
}


def load_config() -> dict:
    """Lê config.json e retorna o dicionário de configurações.
    Se o arquivo não existir, retorna os valores padrão."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Garante que todas as chaves padrão existam
        for key, val in _DEFAULTS.items():
            cfg.setdefault(key, val)
        return cfg
    return dict(_DEFAULTS)


def save_config(cfg: dict) -> None:
    """Persiste o dicionário de configurações em config.json."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def get_data_folder() -> Path:
    """Retorna a pasta de dados configurada e garante que ela existe."""
    cfg = load_config()
    folder = Path(cfg["data_folder"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def set_data_folder(path: str) -> Path:
    """Define uma nova pasta de dados, persiste e retorna o Path."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["data_folder"] = str(folder)
    save_config(cfg)
    return folder


def get_proxy() -> dict | None:
    """Retorna o dict de proxy para requests, ou None se não configurado."""
    cfg = load_config()
    http = cfg.get("proxy_http", "").strip()
    https = cfg.get("proxy_https", "").strip()
    if http or https:
        return {"http": http or None, "https": https or None}
    return None


def set_proxy(http: str = "", https: str = "") -> None:
    """Salva as configurações de proxy em config.json."""
    cfg = load_config()
    cfg["proxy_http"] = http.strip()
    cfg["proxy_https"] = https.strip()
    save_config(cfg)


def get_subfolders() -> dict:
    """
    Retorna os subdiretórios padronizados dentro da pasta de dados.
    Cria todos se não existirem.
    """
    base = get_data_folder()
    subs = {
        "raw_zip":      base / "raw" / "zips",
        "raw_empresas": base / "raw" / "Empresas",
        "raw_estab":    base / "raw" / "Estabelecimentos",
        "raw_socios":   base / "raw" / "Socios",
        "raw_aux":      base / "raw" / "Auxiliares",
        "processed":    base / "processed",
        "cnae_ibge":    base / "processed" / "cnae_ibge",
        "shapefiles":   base / "processed" / "shapefiles",
    }
    for p in subs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subs
