"""
Baixa a hierarquia completa CNAE do IBGE via API REST e salva como parquet e CSV.
API: https://servicodados.ibge.gov.br/api/v2/cnae/subclasses

Estrutura resultante:
  secao | secao_desc | divisao | divisao_desc | grupo | grupo_desc |
  classe | classe_desc | subclasse | subclasse_desc
"""

import json
import re
import urllib.request
import pathlib

import pandas as pd

API_URL = "https://servicodados.ibge.gov.br/api/v2/cnae/subclasses"

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed" / "cnae_ibge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = OUTPUT_DIR / "cnae_hierarquia.parquet"
CSV_PATH = OUTPUT_DIR / "cnae_hierarquia.csv"


def fetch_subclasses() -> list[dict]:
    print(f"Baixando subclasses de {API_URL} ...")
    req = urllib.request.Request(API_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(f"  {len(data)} subclasses recebidas.")
    return data


def parse_row(item: dict) -> dict:
    classe = item.get("classe", {}) or {}
    grupo = classe.get("grupo", {}) or {}
    divisao = grupo.get("divisao", {}) or {}
    secao = divisao.get("secao", {}) or {}
    raw_id = item.get("id", "")

    return {
        "secao": secao.get("id", ""),
        "secao_desc": secao.get("descricao", ""),
        "divisao": divisao.get("id", ""),
        "divisao_desc": divisao.get("descricao", ""),
        "grupo": grupo.get("id", ""),
        "grupo_desc": grupo.get("descricao", ""),
        "classe": classe.get("id", ""),
        "classe_desc": classe.get("descricao", ""),
        "subclasse": re.sub(r"[^0-9]", "", raw_id),  # digits-only — join key com dados RFB
        "subclasse_cod": raw_id,                       # código formatado para exibição
        "subclasse_desc": item.get("descricao", ""),
    }


def main():
    data = fetch_subclasses()
    rows = [parse_row(item) for item in data]
    df = pd.DataFrame(rows)
    df = df.sort_values(["secao", "divisao", "grupo", "classe", "subclasse"]).reset_index(drop=True)

    print(f"\nHierarquia construída: {len(df)} subclasses")
    print(f"  Seções:   {df['secao'].nunique()}")
    print(f"  Divisões: {df['divisao'].nunique()}")
    print(f"  Grupos:   {df['grupo'].nunique()}")
    print(f"  Classes:  {df['classe'].nunique()}")
    print()
    print(df.head(5).to_string(index=False))

    df.to_parquet(PARQUET_PATH, index=False)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSalvo em:")
    print(f"  {PARQUET_PATH}")
    print(f"  {CSV_PATH}")


if __name__ == "__main__":
    main()
