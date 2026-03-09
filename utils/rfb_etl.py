"""
ETL dos dados brutos da Receita Federal do Brasil.

Usa DuckDB para ler, filtrar e unificar os CSVs diretamente em SQL, sem carregar
tudo em memória. É 3–5× mais rápido que pandas chunked e usa muito menos RAM.

Schema dos arquivos (header=None, sep=';', encoding='ISO_8859_1'):
──────────────────────────────────────────────────────────────
EMPRESAS (7 colunas):
  0  cnpj_basico          – 8 dígitos, chave principal
  1  razao_social
  2  natureza_juridica    – código 4 dig → NATUREZAS
  3  qualificacao_responsavel – código 2 dig → QUALIFICACOES
  4  capital_social       – "1000,00" (vírgula decimal)
  5  porte                – 00=não informado, 01=ME, 03=EPP, 05=Demais
  6  ente_federativo_responsavel

ESTABELECIMENTOS (30 colunas):
  0  cnpj_basico          – FK → EMPRESAS
  1  cnpj_ordem           – 4 dígitos
  2  cnpj_dv              – 2 dígitos; CNPJ completo = col0+col1+col2
  3  matriz_filial        – 1=Matriz, 2=Filial
  4  nome_fantasia
  5  situacao_cadastral   – 01=Nula, 02=Ativa, 03=Suspensa, 04=Inapta, 08=Baixada
  6  data_situacao_cadastral – YYYYMMDD
  7  motivo_situacao      – código 2 dig → MOTIVOS
  8  nome_cidade_exterior
  9  pais                 – código 3 dig → PAISES
  10 data_inicio_atividade – YYYYMMDD
  11 cnae_principal       – código 7 dig → CNAES
  12 cnaes_secundarios    – múltiplos códigos separados por vírgula
  13 tipo_logradouro … 27 email … 29 data_situacao_especial

SOCIOS (11 colunas):
  0  cnpj_basico          – FK → EMPRESAS
  1  identificador_socio … 10 faixa_etaria

SIMPLES / MEI (7 colunas):
  0  cnpj_basico … 6 data_exclusao_mei

TABELAS AUXILIARES (2 colunas: código; descrição):
  CNAES, MOTIVOS, MUNICIPIOS, NATUREZAS, PAISES, QUALIFICACOES
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

from utils.storage import save_parquet

# ---------------------------------------------------------------------------
# Nomes de colunas (posicionais – os CSVs não têm cabeçalho)
# ---------------------------------------------------------------------------

EMPRESAS_COLS = [
    "cnpj_basico", "razao_social", "natureza_juridica",
    "qualificacao_responsavel", "capital_social", "porte",
    "ente_federativo_responsavel",
]

ESTAB_COLS = [
    "cnpj_basico", "cnpj_ordem", "cnpj_dv",
    "matriz_filial", "nome_fantasia", "situacao_cadastral",
    "data_situacao_cadastral", "motivo_situacao",
    "nome_cidade_exterior", "pais",
    "data_inicio_atividade", "cnae_principal", "cnaes_secundarios",
    "tipo_logradouro", "logradouro", "numero", "complemento",
    "bairro", "cep", "uf", "municipio",
    "ddd1", "telefone1", "ddd2", "telefone2",
    "ddd_fax", "fax", "email",
    "situacao_especial", "data_situacao_especial",
]

# Colunas que guardamos no Parquet de estabelecimentos
ESTAB_KEEP = [
    "cnpj_basico", "cnpj_ordem", "cnpj_dv",
    "matriz_filial", "nome_fantasia", "situacao_cadastral",
    "data_situacao_cadastral", "motivo_situacao",
    "nome_cidade_exterior", "pais",
    "data_inicio_atividade", "cnae_principal", "cnaes_secundarios",
    "tipo_logradouro", "logradouro", "numero", "complemento",
    "bairro", "cep", "uf", "municipio",
    "ddd1", "telefone1", "email",
    "situacao_especial", "data_situacao_especial",
]

SOCIOS_COLS = [
    "cnpj_basico", "identificador_socio", "nome_socio",
    "cpf_cnpj_socio", "qualificacao_socio",
    "data_entrada_sociedade", "pais_socio",
    "cpf_representante", "nome_representante",
    "qualificacao_representante", "faixa_etaria",
]

SIMPLES_COLS = [
    "cnpj_basico", "opcao_simples",
    "data_opcao_simples", "data_exclusao_simples",
    "opcao_mei", "data_opcao_mei", "data_exclusao_mei",
]

# Padrões glob para localizar os arquivos auxiliares extraídos
_AUX_GLOBS: dict[str, str] = {
    "cnaes":         "*CNAECSV*",
    "motivos":       "*MOTICSV*",
    "municipios":    "*MUNICCSV*",
    "naturezas":     "*NATJUCSV*",
    "paises":        "*PAISCSV*",
    "qualificacoes": "*QUALSCSV*",
    "simples":       "*SIMPLES*",
}

# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _glob_one(folder: Path, pattern: str) -> Path | None:
    matches = list(folder.glob(pattern))
    return matches[0] if matches else None


def _cols_expr(cols: list[str]) -> str:
    """Gera expressão SQL 'column_0 AS nome, column_1 AS nome, ...'"""
    return ", ".join(f"column{i} AS {c}" for i, c in enumerate(cols))


def _read_aux_csv(path: Path) -> pd.DataFrame:
    """Lê uma tabela auxiliar de 2 colunas (código;descrição)."""
    df = pd.read_csv(
        path, encoding="latin1", sep=";", header=None,
        names=["codigo", "descricao"], dtype=str, on_bad_lines="skip",
    )
    df["codigo"] = df["codigo"].str.strip()
    df["descricao"] = df["descricao"].str.strip()
    return df


# ---------------------------------------------------------------------------
# Leitura das tabelas auxiliares (pandas, são pequenas)
# ---------------------------------------------------------------------------

def load_aux_tables(aux_folder: Path) -> dict[str, pd.DataFrame]:
    """Descobre e lê todas as tabelas auxiliares. Retorna dict nome→DataFrame."""
    aux_folder = Path(aux_folder)
    tables: dict[str, pd.DataFrame] = {}
    for key, pattern in _AUX_GLOBS.items():
        if key == "simples":
            continue
        path = _glob_one(aux_folder, pattern)
        if path:
            try:
                tables[key] = _read_aux_csv(path)
            except Exception:
                pass
    return tables


# ---------------------------------------------------------------------------
# ETL principal via DuckDB
# ---------------------------------------------------------------------------

def _duckdb_read_folder(
    con: duckdb.DuckDBPyConnection,
    folder: Path,
    col_names: list[str],
    view_name: str,
    where: str = "",
) -> None:
    """
    Registra todos os CSVs da pasta como uma view DuckDB.
    Os arquivos não têm cabeçalho, separador ';', encoding latin1.
    """
    glob_path = str(folder / "*.csv").replace("\\", "/")
    cols_expr = _cols_expr(col_names)
    where_clause = f"WHERE {where}" if where else ""
    con.execute(f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT {cols_expr}
        FROM read_csv(
            '{glob_path}',
            header=false,
            sep=';',
            quote='"',
            encoding='ISO_8859_1',
            ignore_errors=true,
            columns={{
                {", ".join(f"'column{i}': 'VARCHAR'" for i in range(len(col_names)))}
            }}
        )
        {where_clause}
    """)


def run_etl(
    empresas_folder: Path,
    estab_folder: Path,
    socios_folder: Path,
    aux_folder: Path,
    output_folder: Path,
    only_active: bool = True,
    include_socios: bool = True,
    include_simples: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """
    Executa o ETL completo usando DuckDB:
      Estabelecimentos + Empresas + Auxiliares → base_unificada.parquet
      Sócios (opcional)                        → socios.parquet
      Simples Nacional (opcional)              → aderido ao base_unificada

    DuckDB lê, filtra e faz os JOINs em SQL sem carregar tudo em memória,
    sendo 3–5× mais rápido que a abordagem pandas chunked.
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    con = duckdb.connect()

    # ── Tabelas auxiliares (pequenas, carregam em pandas e registram no DuckDB) ──
    _log("Carregando tabelas auxiliares…")
    aux = load_aux_tables(Path(aux_folder))
    for name, df in aux.items():
        con.register(f"aux_{name}", df)
    _log(f"  Tabelas carregadas: {', '.join(aux.keys()) or '(nenhuma)'}")

    # ── Empresas ──────────────────────────────────────────────────────────
    emp_path = output_folder / "empresas.parquet"
    if emp_path.exists():
        _log(f"Empresas já existe ({emp_path.stat().st_size / 1e9:.1f} GB) — pulando regeneração.")
    else:
        _log("Processando Empresas…")
        _duckdb_read_folder(con, Path(empresas_folder), EMPRESAS_COLS, "v_empresas")
        nat_join = (
            "LEFT JOIN aux_naturezas ON trim(e.natureza_juridica) = aux_naturezas.codigo"
            if "naturezas" in aux else ""
        )
        nat_col = "aux_naturezas.descricao AS natureza_juridica_desc," if "naturezas" in aux else ""
        con.execute(f"""
            COPY (
                SELECT
                    lpad(trim(e.cnpj_basico), 8, '0')  AS cnpj_basico,
                    trim(e.razao_social)                AS razao_social,
                    trim(e.natureza_juridica)           AS natureza_juridica,
                    trim(e.qualificacao_responsavel)    AS qualificacao_responsavel,
                    TRY_CAST(replace(trim(e.capital_social), ',', '.') AS DOUBLE) AS capital_social,
                    TRY_CAST(trim(e.porte) AS INTEGER)  AS porte,
                    CASE TRY_CAST(trim(e.porte) AS INTEGER)
                        WHEN 1 THEN 'Micro Empresa'
                        WHEN 3 THEN 'Empresa de Pequeno Porte'
                        WHEN 5 THEN 'Demais'
                        ELSE 'Não Informado'
                    END                                 AS porte_desc,
                    {nat_col}
                    trim(e.ente_federativo_responsavel) AS ente_federativo_responsavel
                FROM v_empresas e
                {nat_join}
            ) TO '{str(emp_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        _log(f"  Salvo: {emp_path.name}")
    result["empresas"] = emp_path

    # ── Estabelecimentos ──────────────────────────────────────────────────
    _log(f"Processando Estabelecimentos (only_active={only_active})…")
    active_filter = "trim(situacao_cadastral) = '02'" if only_active else ""
    _duckdb_read_folder(con, Path(estab_folder), ESTAB_COLS, "v_estab", where=active_filter)

    # SELECT com zero-padding para os campos de código
    est_select = """
        lpad(trim(cnpj_basico), 8, '0')  AS cnpj_basico,
        lpad(trim(cnpj_ordem),  4, '0')  AS cnpj_ordem,
        lpad(trim(cnpj_dv),     2, '0')  AS cnpj_dv,
        lpad(trim(cnpj_basico), 8, '0') ||
        lpad(trim(cnpj_ordem),  4, '0') ||
        lpad(trim(cnpj_dv),     2, '0')  AS cnpj,
        trim(matriz_filial)              AS matriz_filial,
        trim(nome_fantasia)              AS nome_fantasia,
        trim(situacao_cadastral)         AS situacao_cadastral,
        TRY_STRPTIME(trim(data_situacao_cadastral), '%Y%m%d') AS data_situacao_cadastral,
        trim(motivo_situacao)            AS motivo_situacao,
        trim(nome_cidade_exterior)       AS nome_cidade_exterior,
        trim(pais)                       AS pais,
        TRY_STRPTIME(trim(data_inicio_atividade), '%Y%m%d')   AS data_inicio_atividade,
        trim(cnae_principal)             AS cnae_principal,
        trim(cnaes_secundarios)          AS cnaes_secundarios,
        trim(tipo_logradouro)            AS tipo_logradouro,
        trim(logradouro)                 AS logradouro,
        trim(numero)                     AS numero,
        trim(complemento)                AS complemento,
        trim(bairro)                     AS bairro,
        lpad(trim(cep), 8, '0')          AS cep,
        trim(uf)                         AS uf,
        trim(municipio)                  AS municipio,
        trim(ddd1)                       AS ddd1,
        trim(telefone1)                  AS telefone1,
        trim(email)                      AS email,
        trim(situacao_especial)          AS situacao_especial,
        TRY_STRPTIME(trim(data_situacao_especial), '%Y%m%d')  AS data_situacao_especial
    """

    est_path = output_folder / "estabelecimentos.parquet"
    con.execute(f"""
        COPY (SELECT {est_select} FROM v_estab)
        TO '{str(est_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    n_est = con.execute(f"SELECT count(*) FROM '{str(est_path).replace(chr(92), '/')}'").fetchone()[0]
    _log(f"  {n_est:,} estabelecimentos ativos salvos: {est_path.name}")
    result["estabelecimentos"] = est_path

    # Registra o parquet de estabelecimentos como view para os JOINs seguintes
    con.execute(f"CREATE OR REPLACE VIEW est AS SELECT * FROM '{str(est_path).replace(chr(92), '/')}'")

    # ── Sócios (opcional) ─────────────────────────────────────────────────
    if include_socios:
        try:
            _log("Processando Sócios…")
            _duckdb_read_folder(con, Path(socios_folder), SOCIOS_COLS, "v_socios")
            soc_path = output_folder / "socios.parquet"
            qual_join = (
                "LEFT JOIN aux_qualificacoes aq ON trim(s.qualificacao_socio) = aq.codigo"
                if "qualificacoes" in aux else ""
            )
            qual_col = "aq.descricao AS qualificacao_socio_desc," if "qualificacoes" in aux else ""
            con.execute(f"""
                COPY (
                    SELECT
                        lpad(trim(s.cnpj_basico), 8, '0') AS cnpj_basico,
                        trim(s.identificador_socio)       AS identificador_socio,
                        trim(s.nome_socio)                AS nome_socio,
                        trim(s.cpf_cnpj_socio)            AS cpf_cnpj_socio,
                        trim(s.qualificacao_socio)        AS qualificacao_socio,
                        {qual_col}
                        TRY_STRPTIME(trim(s.data_entrada_sociedade), '%Y%m%d') AS data_entrada_sociedade,
                        trim(s.pais_socio)                AS pais_socio,
                        trim(s.cpf_representante)         AS cpf_representante,
                        trim(s.nome_representante)        AS nome_representante,
                        trim(s.qualificacao_representante) AS qualificacao_representante,
                        trim(s.faixa_etaria)              AS faixa_etaria
                    FROM v_socios s
                    {qual_join}
                ) TO '{str(soc_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION SNAPPY)
            """)
            n_soc = con.execute(f"SELECT count(*) FROM '{str(soc_path).replace(chr(92), '/')}'").fetchone()[0]
            _log(f"  {n_soc:,} sócios salvos: {soc_path.name}")
            result["socios"] = soc_path
        except Exception as exc:
            _log(f"  ⚠ Sócios ignorados: {exc}")

    # ── Base unificada: Estabelecimentos × Empresas × Auxiliares ──────────
    _log("Construindo base_unificada…")

    # Registra o parquet de empresas como view
    con.execute(f"CREATE OR REPLACE VIEW emp AS SELECT * FROM '{str(emp_path).replace(chr(92), '/')}'")

    # Monta os LEFT JOINs dos auxiliares dinamicamente
    aux_joins = ""
    aux_cols  = ""
    if "cnaes" in aux:
        aux_joins += " LEFT JOIN aux_cnaes     ac  ON est.cnae_principal    = ac.codigo"
        aux_cols  += " ac.descricao  AS cnae_principal_desc,"
    if "municipios" in aux:
        aux_joins += " LEFT JOIN aux_municipios am  ON est.municipio         = am.codigo"
        aux_cols  += " am.descricao  AS municipio_nome,"
    if "motivos" in aux:
        aux_joins += " LEFT JOIN aux_motivos    amo ON est.motivo_situacao   = amo.codigo"
        aux_cols  += " amo.descricao AS motivo_situacao_desc,"
    if "paises" in aux:
        aux_joins += " LEFT JOIN aux_paises     ap  ON est.pais              = ap.codigo"
        aux_cols  += " ap.descricao  AS pais_nome,"
    if "qualificacoes" in aux:
        aux_joins += " LEFT JOIN aux_qualificacoes aq ON emp.qualificacao_responsavel = aq.codigo"
        aux_cols  += " aq.descricao  AS qualificacao_responsavel_desc,"

    uni_path = output_folder / "base_unificada.parquet"
    nat_desc_col = "emp.natureza_juridica_desc," if "naturezas" in aux else ""
    con.execute(f"""
        COPY (
            SELECT
                est.*,
                emp.razao_social,
                emp.capital_social,
                emp.porte,
                emp.porte_desc,
                emp.natureza_juridica,
                {nat_desc_col}
                emp.qualificacao_responsavel,
                {aux_cols}
                NULL::VARCHAR AS opcao_simples,
                NULL::TIMESTAMP AS data_opcao_simples,
                NULL::TIMESTAMP AS data_exclusao_simples,
                NULL::VARCHAR AS opcao_mei,
                NULL::TIMESTAMP AS data_opcao_mei,
                NULL::TIMESTAMP AS data_exclusao_mei
            FROM est
            LEFT JOIN emp ON est.cnpj_basico = emp.cnpj_basico
            {aux_joins}
        ) TO '{str(uni_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    _log(f"  Base unificada (sem Simples) salva: {uni_path.name}")
    result["unified"] = uni_path

    # ── Simples Nacional (opcional) ───────────────────────────────────────
    if include_simples:
        try:
            simples_path = _glob_one(Path(aux_folder), _AUX_GLOBS["simples"])
            if not simples_path:
                raise FileNotFoundError("Arquivo Simples não encontrado")
            _log(f"  Incorporando Simples Nacional ({simples_path.stat().st_size / 1e9:.1f} GB)…")
            sim_glob = str(simples_path).replace("\\", "/")
            sim_cols_expr = _cols_expr(SIMPLES_COLS)
            con.execute(f"""
                CREATE OR REPLACE VIEW v_simples AS
                SELECT {sim_cols_expr}
                FROM read_csv(
                    '{sim_glob}',
                    header=false, sep=';', encoding='ISO_8859_1', ignore_errors=true,
                    columns={{
                        {", ".join(f"'column{i}': 'VARCHAR'" for i in range(len(SIMPLES_COLS)))}
                    }}
                )
            """)
            # Recria base_unificada com Simples
            con.execute(f"""
                COPY (
                    SELECT
                        u.* EXCLUDE (opcao_simples, data_opcao_simples, data_exclusao_simples,
                                     opcao_mei, data_opcao_mei, data_exclusao_mei),
                        trim(s.opcao_simples)  AS opcao_simples,
                        TRY_STRPTIME(trim(s.data_opcao_simples),    '%Y%m%d') AS data_opcao_simples,
                        TRY_STRPTIME(trim(s.data_exclusao_simples), '%Y%m%d') AS data_exclusao_simples,
                        trim(s.opcao_mei)      AS opcao_mei,
                        TRY_STRPTIME(trim(s.data_opcao_mei),        '%Y%m%d') AS data_opcao_mei,
                        TRY_STRPTIME(trim(s.data_exclusao_mei),     '%Y%m%d') AS data_exclusao_mei
                    FROM '{str(uni_path).replace(chr(92), "/")}' u
                    LEFT JOIN v_simples s
                        ON lpad(trim(s.cnpj_basico), 8, '0') = u.cnpj_basico
                ) TO '{str(uni_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION SNAPPY)
            """)
            _log(f"  Simples/MEI incorporado: {uni_path.name}")
        except Exception as exc:
            _log(f"  ⚠ Simples ignorado: {exc}")
    # ── Hierarquia CNAE (opcional) ─────────────────────────────────────────
    cnae_hier_path = output_folder / "cnae_ibge" / "cnae_hierarquia.parquet"
    if cnae_hier_path.exists():
        try:
            _log("Enriquecendo com hierarquia CNAE (seção/divisão/grupo/classe)…")
            hier_p = str(cnae_hier_path).replace("\\", "/")
            con.execute(f"""
                COPY (
                    SELECT
                        u.*,
                        h.secao,
                        h.secao_desc,
                        h.divisao,
                        h.divisao_desc,
                        h.grupo,
                        h.grupo_desc,
                        h.classe,
                        h.classe_desc
                    FROM '{str(uni_path).replace(chr(92), "/")}' u
                    LEFT JOIN '{hier_p}' h
                        ON u.cnae_principal = h.subclasse
                ) TO '{str(uni_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION SNAPPY)
            """)
            _log("  Hierarquia CNAE incorporada: secao, divisao, grupo, classe")
        except Exception as exc:
            _log(f"  ⚠ Hierarquia CNAE ignorada: {exc}")
    else:
        _log("  ℹ Hierarquia CNAE não encontrada — execute utils/build_cnae_hierarquia.py")

    n_uni = con.execute(f"SELECT count(*) FROM '{str(uni_path).replace(chr(92), '/')}'").fetchone()[0]
    cols_uni = len(con.execute(f"DESCRIBE SELECT * FROM '{str(uni_path).replace(chr(92), '/')}' LIMIT 1").fetchdf())
    _log(f"  Total: {n_uni:,} registros, {cols_uni} colunas")

    con.close()
    return result
