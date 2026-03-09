"""
Recuperação rápida: reconstrói base_unificada.parquet a partir dos
parquets intermediários já salvos (empresas + estabelecimentos).

Use quando o run_etl falhar após salvar os parquets intermediários.
Execute: python _recover_unified.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from utils.rfb_etl import load_aux_tables, load_simples, clean_simples
from utils.storage import load_parquet, save_parquet

BASE       = Path(__file__).parent / "data"
OUTPUT     = BASE / "processed"
AUX_FOLDER = BASE / "raw" / "Auxiliares"


def _log(msg: str) -> None:
    print(msg, flush=True)


def recover_unified(include_simples: bool = False) -> None:
    _log("=== Recuperando base_unificada.parquet ===")

    # ── Carrega parquets intermediários ─────────────────────────────────
    _log("Carregando empresas.parquet…")
    emp = load_parquet(OUTPUT / "empresas.parquet")
    _log(f"  {len(emp):,} empresas, colunas: {list(emp.columns)}")

    _log("Carregando estabelecimentos.parquet…")
    est = load_parquet(OUTPUT / "estabelecimentos.parquet")
    _log(f"  {len(est):,} estabelecimentos")

    # ── Tabelas auxiliares ───────────────────────────────────────────────
    _log("Carregando tabelas auxiliares…")
    aux = load_aux_tables(AUX_FOLDER)
    _log(f"  {list(aux.keys())}")

    # ── Merge Estabelecimentos × Empresas ────────────────────────────────
    _log("Unificando…")
    emp_cols = [
        "cnpj_basico", "razao_social", "capital_social",
        "porte", "porte_desc", "natureza_juridica", "qualificacao_responsavel",
    ]
    for opt in ["natureza_juridica_desc"]:
        if opt in emp.columns:
            emp_cols.append(opt)

    unified = est.merge(emp[emp_cols], on="cnpj_basico", how="left")
    _log(f"  {len(unified):,} registros após merge")

    # ── Lookups auxiliares ───────────────────────────────────────────────
    if "cnaes" in aux:
        unified = unified.merge(
            aux["cnaes"].rename(columns={"codigo": "cnae_principal", "descricao": "cnae_principal_desc"}),
            on="cnae_principal", how="left",
        )
        _log("  + CNAE descrições")

    if "municipios" in aux:
        unified = unified.merge(
            aux["municipios"].rename(columns={"codigo": "municipio", "descricao": "municipio_nome"}),
            on="municipio", how="left",
        )
        _log("  + Nomes de municípios")

    if "motivos" in aux:
        unified = unified.merge(
            aux["motivos"].rename(columns={"codigo": "motivo_situacao", "descricao": "motivo_situacao_desc"}),
            on="motivo_situacao", how="left",
        )
        _log("  + Motivos de situação")

    if "paises" in aux:
        unified = unified.merge(
            aux["paises"].rename(columns={"codigo": "pais", "descricao": "pais_nome"}),
            on="pais", how="left",
        )
        _log("  + Nomes de países")

    if "qualificacoes" in aux:
        unified = unified.merge(
            aux["qualificacoes"].rename(columns={"codigo": "qualificacao_responsavel", "descricao": "qualificacao_responsavel_desc"}),
            on="qualificacao_responsavel", how="left",
        )
        _log("  + Qualificações do responsável")

    # ── Simples Nacional (opcional) ──────────────────────────────────────
    if include_simples:
        try:
            _log("Carregando Simples Nacional…")
            cnpj_set = set(est["cnpj_basico"].dropna().unique())
            sim = load_simples(AUX_FOLDER, cnpj_filter=cnpj_set, log=_log)
            sim = clean_simples(sim)
            unified = unified.merge(
                sim[["cnpj_basico", "opcao_simples", "data_opcao_simples",
                     "data_exclusao_simples", "opcao_mei",
                     "data_opcao_mei", "data_exclusao_mei"]],
                on="cnpj_basico", how="left",
            )
            _log(f"  + Simples/MEI ({len(sim):,} registros)")
        except Exception as exc:
            _log(f"  ⚠ Simples ignorado: {exc}")

    # ── Salva ────────────────────────────────────────────────────────────
    _log(f"Total: {len(unified):,} registros, {len(unified.columns)} colunas")
    out_path = OUTPUT / "base_unificada.parquet"
    save_parquet(unified, out_path)
    _log(f"✅ Salvo: {out_path}")


if __name__ == "__main__":
    recover_unified(include_simples=False)
