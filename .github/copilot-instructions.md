# Contexto do projeto SGGeoData para GitHub Copilot

## Visão geral

Aplicativo **Streamlit** de inteligência geográfica e comercial baseado nos dados públicos da
Receita Federal do Brasil (RFB). Baixa, extrai, limpa e enriquece os dados de CNPJs para análise
de cobertura comercial e prospecção.

- **Python 3.13** — instalação global (sem `.venv`)
- **Streamlit 1.55** — app multi-página em `sggeodata/`
- **Pandas 2 + PyArrow** — processamento e Parquet
- **DuckDB** — queries opcionais sobre Parquet
- **requests + urllib3** — HTTP com `verify=False` (proxy Zscaler)

---

## Fonte dos dados

- **SERPRO Nextcloud WebDAV**: `https://arquivos.receitafederal.gov.br/public.php/webdav/`
- **Share token**: `YggdBLfdninEJX9` — autenticação Basic `token:` (sem senha)
- **`verify=False`** obrigatório: CA do SERPRO não está no bundle do Python
- **34 versões** disponíveis (`2023-05` a `2026-02`); versão mais recente: `2026-02`
- **37 arquivos por versão**: 10 Empresas + 10 Estabelecimentos + 10 Sócios + 7 Auxiliares
- Nomes no ZIP: extensões proprietárias (`.EMPRECSV`, `.ESTABELE`, `.SOCIOCSV`, `.SIMPLES.CSV`, etc.)
  → após extração appenda `.csv` ao nome completo para evitar colisões

---

## Schema dos arquivos RFB (header=None, sep=';', encoding='latin1')

### EMPRESAS — 7 colunas posicionais
```
0  cnpj_basico (8 dig, PK)
1  razao_social
2  natureza_juridica  → NATUREZAS (código 4 dig)
3  qualificacao_responsavel  → QUALIFICACOES (código 2 dig)
4  capital_social  ("1000,00" — vírgula decimal)
5  porte  (01=ME, 03=EPP, 05=Demais)
6  ente_federativo_responsavel
```

### ESTABELECIMENTOS — 30 colunas posicionais
```
0  cnpj_basico (FK → EMPRESAS)    1  cnpj_ordem (4 dig)    2  cnpj_dv (2 dig)
3  matriz_filial (1=Matriz/2=Filial)
4  nome_fantasia
5  situacao_cadastral (02=Ativa, 08=Baixada)
6  data_situacao_cadastral (YYYYMMDD)
7  motivo_situacao → MOTIVOS (código 2 dig)
8  nome_cidade_exterior    9  pais → PAISES (código 3 dig)
10 data_inicio_atividade (YYYYMMDD)
11 cnae_principal → CNAES (código 7 dig)
12 cnaes_secundarios (múltiplos, separados por vírgula)
13 tipo_logradouro  14 logradouro  15 numero  16 complemento  17 bairro
18 cep (8 dig)  19 uf  20 municipio → MUNICIPIOS (código 4 dig TOM/RFB ≠ IBGE)
21 ddd1  22 telefone1  23 ddd2  24 telefone2  25 ddd_fax  26 fax  27 email
28 situacao_especial  29 data_situacao_especial (YYYYMMDD)
```

### SOCIOS — 11 colunas posicionais
```
0  cnpj_basico (FK → EMPRESAS)
1  identificador_socio (1=PJ, 2=PF, 3=Estrangeiro)
2  nome_socio      3  cpf_cnpj_socio (mascarado)
4  qualificacao_socio → QUALIFICACOES
5  data_entrada_sociedade (YYYYMMDD)
6  pais_socio → PAISES
7  cpf_representante  8  nome_representante
9  qualificacao_representante → QUALIFICACOES
10 faixa_etaria (1=0-12 … 9=81+, 0=não informado)
```

### SIMPLES / MEI — 7 colunas posicionais (~2,8 GB descomprimido)
```
0  cnpj_basico (8 dig)
1  opcao_simples (S/N)  2  data_opcao_simples  3  data_exclusao_simples
4  opcao_mei (S/N)      5  data_opcao_mei      6  data_exclusao_mei
```

### Tabelas de domínio — 2 colunas (código; descrição)
```
Glob *CNAECSV*    → CNAES        (1.359 registros)
Glob *MOTICSV*    → MOTIVOS      (63 registros)
Glob *MUNICCSV*   → MUNICIPIOS   (5.572 registros)
Glob *NATJUCSV*   → NATUREZAS    (91 registros)
Glob *PAISCSV*    → PAISES       (255 registros)
Glob *QUALSCSV*   → QUALIFICACOES (68 registros)
Glob *SIMPLES*    → SIMPLES/MEI  (~60 M registros)
```

---

## Estrutura de pastas

```
sggeodata/
├── app.py                 # Home Streamlit
├── iniciar.bat            # Launcher Windows
├── config.json            # Pasta de dados + proxy (NÃO versionar)
├── pages/
│   ├── 1_Mineracao.py     # Download WebDAV + extração ZIPs
│   ├── 2_ETL.py           # ETL → Parquets enriquecidos
│   ├── 3_CNAE_IBGE.py     # API REST IBGE → cnae_hierarquia.parquet
│   ├── 4_Exploracao.py    # KPIs, gráficos DuckDB lazy, cascading CNAE filters
│   └── 5_Mercado_Foco.py  # Upload CSV clientes → enriquecimento → oportunidades
└── utils/
    ├── config.py          # get_subfolders(), load_config(), get_proxy()
    ├── storage.py         # save_parquet(), load_parquet(), parquet_exists()
    ├── rfb_download.py    # list_versions(), download_file(), extract_zip()
    ├── rfb_etl.py         # run_etl(), load_empresas(), load_estabelecimentos()
    ├── build_cnae_hierarquia.py  # fetch_subclasses(), parse_row() → parquet
    └── cnae_ibge.py       # (legado) build_cnae_database()
```

### Subpastas de dados (`get_subfolders()`)
```python
{
  "raw_zip":      data / "raw" / "zips",
  "raw_empresas": data / "raw" / "Empresas",
  "raw_estab":    data / "raw" / "Estabelecimentos",
  "raw_socios":   data / "raw" / "Socios",
  "raw_aux":      data / "raw" / "Auxiliares",
  "processed":    data / "processed",
  "cnae_ibge":    data / "processed" / "cnae_ibge",
}
```

---

## Parquets gerados pelo ETL

| Arquivo | Granularidade | Joins aplicados |
|---|---|---|
| `empresas.parquet` | 1 linha por cnpj_basico | + NATUREZAS + QUALIFICACOES |
| `estabelecimentos.parquet` | 1 linha por CNPJ 14 dig | Sem joins (raw limpo) |
| `socios.parquet` | N linhas por cnpj_basico | + QUALIFICACOES |
| `base_unificada.parquet` | 1 linha por CNPJ 14 dig | + Empresas + todos auxiliares + Simples/MEI |

---

## Estado atual dos módulos (atualizado 2026-03-08)

| Módulo | Arquivo | Status |
|--------|---------|--------|
| 1 · Mineração | `pages/1_Mineracao.py` | ✅ Completo |
| 2 · ETL | `pages/2_ETL.py` | ✅ Completo |
| 3 · CNAE IBGE | `pages/3_CNAE_IBGE.py` | ✅ Completo — usa API REST IBGE |
| 4 · Exploração | `pages/4_Exploracao.py` | ✅ Completo — DuckDB lazy, cascading filters, boxplot |
| 5 · Mercado Foco | `pages/5_Mercado_Foco.py` | ✅ Completo — 4 passos, enriquecimento + oportunidades |
| 6 · Mapas | — | 🔲 Planejado |
| 7 · Geolocalização | — | 🔲 Planejado |

---

## Padrões críticos estabelecidos neste projeto

### Streamlit — parâmetros depreciados
- **`use_container_width`** foi depreciado no Streamlit 1.55 para todos os componentes (`st.dataframe`, `st.plotly_chart`, `st.button`, etc.)  
  - `use_container_width=True` → **`width="stretch"`**  
  - `use_container_width=False` → **`width="content"`**  
- Nunca usar `use_container_width`; sempre usar `width=`.

### DuckDB — queries sobre Parquet
```python
# Sempre usar string path com forward-slash
_PATH = str(Path(...)).replace("\\\\", "/")

# Query parametrizada simples
con = duckdb.connect()
df = con.execute("SELECT ... FROM '" + _PATH + "' WHERE col = ?", [val]).df()
con.close()

# Registro de DataFrame como tabela virtual (para JOINs)
con.register("nome", df_pandas)
df_result = con.execute("SELECT ... FROM parquet JOIN nome ON ...").df()
```

### CNAE — chave de join
- `subclasse` = dígitos apenas, ex `"4744099"` — é a chave de join com `cnae_principal` em `base_unificada`
- `subclasse_cod` = formatado para exibição, ex `"47-44-0/99"`
- Hierarquia: `secao` (letra) → `divisao` (2 dig) → `grupo` (3 dig) → `classe` (4 dig) → `subclasse` (7 dig)

### Módulo 5 — session state keys
Todos com prefixo `mf_` para não colidir com Módulo 4:
```
mf_raw, mf_col_map, mf_enriched, mf_nao_atend, mf_at_sel
```

### Módulo 5 — colunas trazidas da base para enriquecimento
```python
_BASE_COLS = [
    "uf", "municipio_nome", "cnae_principal", "cnae_principal_desc",
    "cnaes_secundarios", "porte_desc", "razao_social", "nome_fantasia",
    "capital_social", "cep",
    "tipo_logradouro", "logradouro", "numero", "complemento", "bairro",
]
```

---

## Próximos passos (roadmap)

### Módulo 6 — Mapas de Atendimento e Cobertura
- Baseado em `mapa_atendimento.ipynb` e `mapa_cobertura_dashboard.html` (já existentes)
- Entrada: resultado do Módulo 5 (base enriquecida) ou diretamente CSV de clientes
- Geolocalização por município (centróide shapefile `BR_Municipios_2024.shp`) ou por CEP (API FindCEP)
- `plot_cobertura()` — BallTree/Haversine para classificar oportunidades dentro/fora do raio
- Saídas: mapa Folium interativo (`heatmap`, clusters, círculos de cobertura) + dashboard HTML
- Parâmetros: raio fixo ou proporcional ao faturamento (`3Y_Fat`), filtro por canal/CNAE/UF

### Módulo 7 — Geolocalização por CEP
- API FindCEP da Saint-Gobain: `CLIENT_ENDPOINT`, `CLIENT_FID`, `CLIENT_ID`, `CLIENT_URL_HASH` (não versionar)
- `_clean_cep()` → `_find_cep_lat_lon()` → `_find_cep_lat_lon_batch()` (já implementados no notebook)
- Integra ao Módulo 6 para precisão de endereço quando centróide de município for insuficiente

---

## Regras de desenvolvimento

- **`use_container_width=`** está deprecated — usar `width='stretch'` (True) ou `width='content'` (False)
- `st.plotly_chart(fig, width='stretch')` — idem
- `requests` sempre com `verify=False` e `timeout=600` para downloads grandes
- CSVs da RFB: **nunca** assumir cabeçalho — usar `header=None` + `names=[...]`
- **`situacao_cadastral` ativo = `"02"`** (NÃO `"2"`) — valores são zero-padded como strings
- Leitura de CSVs grandes: **sempre** usar `chunksize=_CHUNK_SIZE` (200_000) para evitar OOM
- `load_simples` recebe `cnpj_filter=set(est["cnpj_basico"])` para filtrar antes de acumular (arquivo 2,8 GB)
- Dados de Sócios e Simples são opcionais no ETL (checkboxes na UI) — nunca forçar
- Parquets salvos com `pyarrow` (snappy) via `utils.storage.save_parquet()`
- `config.json` nunca versionado — armazena caminho da pasta de dados e proxy corporativo

---

## Convenções de nomes

- Colunas em `snake_case` minúsculo (ex: `cnpj_basico`, `razao_social`)
- Colunas de descrição: `{campo}_desc` (ex: `cnae_principal_desc`, `municipio_nome`)
- Funções de leitura: `load_{entidade}(folder, log=None)`
- Funções de limpeza: `clean_{entidade}(df)`
- Globs de auxiliares definidos em `_AUX_GLOBS` em `rfb_etl.py`
