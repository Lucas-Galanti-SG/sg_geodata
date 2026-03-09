"""
Download e parsing dos PDFs de notas explicativas de CNAE do IBGE.

Fontes:
 - CNAE 2.0: https://cnae.ibge.gov.br/images/concla/downloads/revisao2007/PropCNAE20/CNAE20_NotasExplicativas.pdf
 - CNAE-Subclasses 2.2: https://cnae.ibge.gov.br/images/concla/downloads/cnae-subclasses-2-2-notas-explicativas.pdf

Estrutura hierárquica extraída:
  Seção (letra) → Divisão (2 dig) → Grupo (3 dig) → Classe (4 dig) → Subclasse (7 dig)
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Callable

import requests
import pandas as pd
from pdfminer.high_level import extract_text

from utils.storage import save_parquet, parquet_exists, load_parquet

# ---------------------------------------------------------------------------
# URLs dos PDFs
# ---------------------------------------------------------------------------

PDF_URLS = {
    "cnae20": (
        "https://cnae.ibge.gov.br/images/concla/downloads/"
        "revisao2007/PropCNAE20/CNAE20_NotasExplicativas.pdf"
    ),
    "cnae22": (
        "https://cnae.ibge.gov.br/images/concla/downloads/"
        "cnae-subclasses-2-2-notas-explicativas.pdf"
    ),
}

# Regex patterns para identificar cada nível hierárquico
# Seção: linha como "SEÇÃO A - AGRICULTURA, PECUÁRIA E ..."
RE_SECAO = re.compile(
    r"^SEÇ[ÃA]O\s+([A-Z])\s*[-–]\s*(.+)$", re.IGNORECASE
)
# Divisão: linha como "DIVISÃO 01 - AGRICULTURA, ..."
RE_DIVISAO = re.compile(
    r"^DIVIS[ÃA]O\s+(\d{2})\s*[-–]\s*(.+)$", re.IGNORECASE
)
# Grupo: linha como "GRUPO 01.1 - Produção ..."
RE_GRUPO = re.compile(
    r"^GRUPO\s+(\d{2}[.,]\d)\s*[-–]\s*(.+)$", re.IGNORECASE
)
# Classe: "CLASSE 0111-3 - Cultivo de cereais..."
RE_CLASSE = re.compile(
    r"^CLASSE\s+(\d{4}[-/]\d)\s*[-–]\s*(.+)$", re.IGNORECASE
)
# Subclasse: "0111-3/01 - Cultivo de trigo"  ou  "01113 - Cultivo de trigo"
RE_SUBCLASSE = re.compile(
    r"^(\d{4}[-/]\d[/]\d{2}|\d{7})\s*[-–]\s*(.+)$"
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: Path, log: Callable[[str], None] | None = None) -> Path:
    """Baixa um PDF e salva localmente. Pula se já existir."""
    dest = Path(dest)
    if dest.exists():
        if log:
            log(f"PDF já existe: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"Baixando {dest.name}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    if log:
        log(f"  {dest.stat().st_size / 1024:.0f} KB salvos.")
    return dest


# ---------------------------------------------------------------------------
# Extração de texto
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando pdfminer.six."""
    return extract_text(str(pdf_path))


# ---------------------------------------------------------------------------
# Parsing do texto
# ---------------------------------------------------------------------------

def parse_cnae_text(text: str) -> list[dict]:
    """
    Percorre o texto do PDF linha a linha e extrai a hierarquia de CNAEs.

    Returns
    -------
    Lista de dicts com campos:
        secao_cod, secao_nome,
        divisao_cod, divisao_nome,
        grupo_cod, grupo_nome,
        classe_cod, classe_nome,
        subclasse_cod, subclasse_nome,
        notas_explicativas
    """
    records = []
    lines = [l.strip() for l in text.splitlines()]

    # Estado atual
    secao_cod = secao_nome = ""
    divisao_cod = divisao_nome = ""
    grupo_cod = grupo_nome = ""
    classe_cod = classe_nome = ""
    subclasse_cod = subclasse_nome = ""
    notas_buffer: list[str] = []

    def flush_notas():
        """Salva as notas acumuladas para a subclasse atual."""
        if subclasse_cod:
            notes = " ".join(notas_buffer).strip()
            records.append({
                "secao_cod":      secao_cod,
                "secao_nome":     secao_nome,
                "divisao_cod":    divisao_cod,
                "divisao_nome":   divisao_nome,
                "grupo_cod":      grupo_cod,
                "grupo_nome":     grupo_nome,
                "classe_cod":     classe_cod,
                "classe_nome":    classe_nome,
                "subclasse_cod":  subclasse_cod,
                "subclasse_nome": subclasse_nome,
                "notas_explicativas": notes,
            })
        notas_buffer.clear()

    for line in lines:
        if not line:
            continue

        m = RE_SECAO.match(line)
        if m:
            flush_notas()
            secao_cod, secao_nome = m.group(1).strip(), m.group(2).strip()
            divisao_cod = divisao_nome = grupo_cod = grupo_nome = ""
            classe_cod = classe_nome = subclasse_cod = subclasse_nome = ""
            continue

        m = RE_DIVISAO.match(line)
        if m:
            flush_notas()
            divisao_cod, divisao_nome = m.group(1).strip(), m.group(2).strip()
            grupo_cod = grupo_nome = classe_cod = classe_nome = ""
            subclasse_cod = subclasse_nome = ""
            continue

        m = RE_GRUPO.match(line)
        if m:
            flush_notas()
            grupo_cod, grupo_nome = m.group(1).strip(), m.group(2).strip()
            classe_cod = classe_nome = subclasse_cod = subclasse_nome = ""
            continue

        m = RE_CLASSE.match(line)
        if m:
            flush_notas()
            classe_cod, classe_nome = m.group(1).strip(), m.group(2).strip()
            subclasse_cod = subclasse_nome = ""
            continue

        m = RE_SUBCLASSE.match(line)
        if m:
            flush_notas()
            subclasse_cod = m.group(1).strip()
            subclasse_nome = m.group(2).strip()
            continue

        # Linha de texto livre → acumula como nota explicativa da subclasse atual
        if subclasse_cod:
            notas_buffer.append(line)

    flush_notas()
    return records


def _normalize_subclasse_cod(cod: str) -> str:
    """Normaliza código de subclasse para 7 dígitos numéricos (sem '-' ou '/')."""
    return re.sub(r"[^0-9]", "", cod)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_cnae_database(
    pdf_folder: Path,
    output_path: Path,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Baixa os PDFs do IBGE, parseia e retorna/salva o DataFrame de CNAEs.

    Parameters
    ----------
    pdf_folder : pasta para armazenar os PDFs baixados
    output_path : caminho do Parquet a salvar (.parquet)
    log : callable opcional para logging

    Returns
    -------
    DataFrame com hierarquia completa de CNAEs + notas explicativas
    """
    def _log(msg):
        if log:
            log(msg)

    all_records = []

    for key, url in PDF_URLS.items():
        pdf_name = url.split("/")[-1]
        pdf_path = Path(pdf_folder) / pdf_name
        try:
            download_pdf(url, pdf_path, log=_log)
            _log(f"Extraindo texto de {pdf_name}...")
            text = extract_pdf_text(pdf_path)
            _log(f"  {len(text):,} caracteres extraídos")
            records = parse_cnae_text(text)
            _log(f"  {len(records):,} subclasses identificadas")
            all_records.extend(records)
        except Exception as e:
            _log(f"  AVISO: erro ao processar {pdf_name}: {e}")

    if not all_records:
        _log("Nenhum registro extraído dos PDFs.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Normaliza código da subclasse para 7 dígitos
    df["subclasse_cod_num"] = df["subclasse_cod"].apply(_normalize_subclasse_cod)

    # Remove duplicatas (PDF 2.2 sobrepõe 2.0)
    df = df.drop_duplicates(subset=["subclasse_cod_num"], keep="last")
    df = df.sort_values(["secao_cod", "divisao_cod", "grupo_cod", "classe_cod", "subclasse_cod"])
    df = df.reset_index(drop=True)

    _log(f"Total final: {len(df):,} subclasses únicas")
    save_parquet(df, output_path)
    _log(f"Banco de CNAEs salvo em {output_path}")
    return df


def load_cnae_database(output_path: Path) -> pd.DataFrame | None:
    """Carrega o banco de CNAEs se existir, retorna None caso contrário."""
    if parquet_exists(output_path):
        return load_parquet(output_path)
    return None
