"""
FindCEP — Geolocalização por CEP (API Saint-Gobain)

Grupo por prefixo de 5 dígitos (micro-região) para minimizar chamadas à API.
Cache persistente em `ceps_database.parquet`.

Credenciais necessárias em config.json:
    findcep_endpoint  ex: "saintgobain-57f2a16059b2e31e.api.findcep.com"
    findcep_fid       ex: "E4N6YR5GRZXMP"
    findcep_client_id ex: "saintgobain"
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_COLS = [
    "cep5",
    "cep_usado",
    "lat",
    "lon",
    "logradouro",
    "bairro",
    "localidade",
    "uf",
    "status",   # "ok" | "not_found" | "error"
    "timestamp",
]

_EMPTY_CACHE = pd.DataFrame(columns=_CACHE_COLS)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clean_cep(s) -> str | None:
    """Extrai 8 dígitos de um CEP; retorna None se inválido."""
    if s is None:
        return None
    digits = "".join(c for c in str(s) if c.isdigit())
    return digits if len(digits) == 8 else None


def _cep5(cep8: str) -> str:
    """Retorna os 5 primeiros dígitos (prefixo de micro-região)."""
    return cep8[:5]


def _call_api(
    cep8: str,
    endpoint: str,
    fid: str,
    client_id: str,
    timeout: int = 8,
) -> dict | None:
    """
    Chama a API FindCEP para um CEP de 8 dígitos.
    Retorna dict com lat, lon e campos de endereço, ou None em caso de falha.
    """
    url = f"https://{endpoint}/{cep8}"
    headers = {
        "Referer": f"FID={fid};CLIENT_ID={client_id}",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, verify=False, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            # Normaliza campo de longitude (API pode retornar lng, lon ou longitude)
            lon_val = (
                data.get("lon")
                or data.get("lng")
                or data.get("longitude")
            )
            lat_val = data.get("lat") or data.get("latitude")
            if lat_val and lon_val:
                return {
                    "lat": float(lat_val),
                    "lon": float(lon_val),
                    "logradouro": data.get("logradouro", ""),
                    "bairro": data.get("bairro", ""),
                    "localidade": data.get("localidade", "") or data.get("cidade", ""),
                    "uf": data.get("uf", "") or data.get("estado", ""),
                    "status": "ok",
                }
            # Resposta válida mas sem coordenadas — CEP existe mas sem geo
            return {"status": "not_found"}
        if resp.status_code == 404:
            return {"status": "not_found"}
    except requests.exceptions.Timeout:
        return {"status": "error"}
    except Exception:
        return {"status": "error"}
    return {"status": "not_found"}


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_cep_cache(path: Path) -> pd.DataFrame:
    """Carrega o cache de CEPs. Retorna DataFrame vazio se não existir."""
    if path.exists():
        try:
            df = pd.read_parquet(path)
            # Garante que todas as colunas existem
            for col in _CACHE_COLS:
                if col not in df.columns:
                    df[col] = None
            return df[_CACHE_COLS].copy()
        except Exception:
            pass
    return _EMPTY_CACHE.copy()


def save_cep_cache(df: pd.DataFrame, path: Path) -> None:
    """Salva o cache de CEPs em Parquet (snappy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[_CACHE_COLS].to_parquet(path, index=False, compression="snappy")


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich_microregioes(
    cep_series: pd.Series,
    cache_path: Path,
    endpoint: str,
    fid: str,
    client_id: str,
    progress_cb: Callable[[int, int, str], None] | None = None,
    rate_s: float = 0.2,
) -> pd.DataFrame:
    """
    Enriquece uma Series de CEPs com lat/lon usando prefixo de 5 dígitos (micro-região).

    Algoritmo:
    1. Extrai prefixos cep5 únicos.
    2. Verifica cache — já resolvidos (status='ok' ou 'not_found') são reutilizados.
    3. Para cep5 desconhecidos, busca candidatos ordenados por frequência no input.
    4. Chama a API para o representante mais frequente e armazena o resultado pelo cep5.
    5. A cada 50 chamadas faz uma pausa de 2 s (back-off leve).

    Parâmetros:
        cep_series   : Series com CEPs (8 dígitos ou formatados)
        cache_path   : Path para `ceps_database.parquet`
        endpoint     : Host da API FindCEP (sem https://)
        fid          : FindCEP FID credential
        client_id    : FindCEP CLIENT_ID credential
        progress_cb  : callback(current, total, message) para barra de progresso
        rate_s       : pausa entre chamadas API (segundos)

    Retorna DataFrame com colunas:
        cep5, lat, lon, localidade, uf, status,
        stats_api_calls, stats_cache_hits, stats_new_ok, stats_new_nf
    """
    # ── 1. Limpar e extrair cep5 únicos com frequência ──────────────────────
    cleaned = cep_series.apply(_clean_cep)
    valid_mask = cleaned.notna()
    cleaned_valid = cleaned[valid_mask]

    # Mapa: cep5 → list of cep8 sorted by frequency descending
    cep5_candidates: dict[str, list[str]] = {}
    for cep8 in cleaned_valid:
        prefix = _cep5(cep8)
        cep5_candidates.setdefault(prefix, [])
        cep5_candidates[prefix].append(cep8)

    # Representante mais frequente por cep5
    cep5_repr: dict[str, str] = {}
    for prefix, candidates in cep5_candidates.items():
        freq = {}
        for c in candidates:
            freq[c] = freq.get(c, 0) + 1
        cep5_repr[prefix] = max(freq, key=lambda x: freq[x])

    all_cep5 = list(cep5_repr.keys())

    # ── 2. Carregar cache ────────────────────────────────────────────────────
    cache_df = load_cep_cache(cache_path)
    cache_ok = cache_df[cache_df["status"].isin(["ok", "not_found"])].copy()
    known_cep5 = set(cache_ok["cep5"].astype(str))

    unknown = [p for p in all_cep5 if p not in known_cep5]

    stats_cache_hits = len(all_cep5) - len(unknown)
    stats_api_calls = 0
    stats_new_ok = 0
    stats_new_nf = 0

    # ── 3. Resolver desconhecidos via API ────────────────────────────────────
    new_rows: list[dict] = []
    total_unknown = len(unknown)

    for i, prefix in enumerate(unknown):
        if progress_cb:
            progress_cb(
                i + 1,
                total_unknown,
                f"CEP {prefix}… ({i+1}/{total_unknown})",
            )

        repr_cep = cep5_repr[prefix]
        result = _call_api(repr_cep, endpoint, fid, client_id)
        stats_api_calls += 1

        ts = datetime.now(timezone.utc).isoformat()
        if result and result.get("status") == "ok":
            stats_new_ok += 1
            row = {
                "cep5": prefix,
                "cep_usado": repr_cep,
                "lat": result["lat"],
                "lon": result["lon"],
                "logradouro": result.get("logradouro", ""),
                "bairro": result.get("bairro", ""),
                "localidade": result.get("localidade", ""),
                "uf": result.get("uf", ""),
                "status": "ok",
                "timestamp": ts,
            }
        else:
            stats_new_nf += 1
            row = {
                "cep5": prefix,
                "cep_usado": repr_cep,
                "lat": None,
                "lon": None,
                "logradouro": "",
                "bairro": "",
                "localidade": "",
                "uf": "",
                "status": result.get("status", "error") if result else "error",
                "timestamp": ts,
            }
        new_rows.append(row)

        # Pausa entre chamadas
        time.sleep(rate_s)

        # Back-off leve a cada 50 chamadas
        if stats_api_calls % 50 == 0:
            time.sleep(2.0)

    # ── 4. Atualizar cache ───────────────────────────────────────────────────
    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=_CACHE_COLS)
        updated_cache = pd.concat([cache_df, new_df], ignore_index=True)
        # Deduplica: mantém a entrada mais recente por cep5 com status='ok'
        updated_cache = updated_cache.sort_values("timestamp", ascending=False)
        updated_cache = updated_cache.drop_duplicates(subset=["cep5"], keep="first")
        save_cep_cache(updated_cache, cache_path)
        # Atualiza cache_ok para o merge final
        cache_ok = updated_cache[updated_cache["status"].isin(["ok", "not_found"])].copy()

    # ── 5. Montar resultado por cep5 ─────────────────────────────────────────
    result_df = (
        cache_ok[cache_ok["cep5"].isin(all_cep5)]
        [["cep5", "lat", "lon", "localidade", "uf", "status"]]
        .drop_duplicates(subset=["cep5"])
        .copy()
    )
    # Adiciona stats como colunas escalares (mesmo valor em todas as linhas)
    result_df["stats_api_calls"] = stats_api_calls
    result_df["stats_cache_hits"] = stats_cache_hits
    result_df["stats_new_ok"] = stats_new_ok
    result_df["stats_new_nf"] = stats_new_nf

    return result_df.reset_index(drop=True)
