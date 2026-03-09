# SGGeoData — Plataforma de Inteligência Geográfica e Comercial

> Aplicativo Streamlit que unifica mineração de dados públicos da Receita Federal do Brasil (RFB), classificações do IBGE (CNAE), geolocalização de municípios e análise de cobertura comercial — incluindo cobertura via rede de distribuidores.

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Fluxo Recomendado](#fluxo-recomendado)
3. [Schema dos Dados RFB](#schema-dos-dados-rfb)
4. [Módulos do Aplicativo](#módulos-do-aplicativo)
5. [Parquet de Base Foco](#parquet-de-base-foco)
6. [Instalação e Execução](#instalação-e-execução)
7. [Estrutura de Pastas](#estrutura-de-pastas)
8. [Privacidade e Segurança](#privacidade-e-segurança)
9. [Changelog](#changelog)
10. [Histórico — Pipeline de Notebooks](#histórico--pipeline-de-notebooks)

---

## Visão Geral

O SGGeoData consolida em um único aplicativo Streamlit todo o processo de:

- **Mineração** de dados públicos de empresas brasileiras (RFB/Fazenda) via SERPRO Nextcloud
- **Limpeza e cruzamento** dos dados de estabelecimentos, empresas, sócios, Simples Nacional e CNAEs IBGE
- **Enriquecimento** com a base interna de clientes — mapeando quais CNPJs já são atendidos, relevância de valor e classificações personalizadas (canal, segmento, etc.)
- **Geolocalização** automática por centróide de município via shapefile oficial do IBGE (download automático via FTP)
- **Análise de atendimento direto**: identificação de oportunidades não atendidas com perfil CNAE similar à carteira atual
- **Análise de atendimento indireto**: mapeamento de quais clientes finais estão cobertos pela rede de distribuidores, usando distância geográfica real (BallTree haversine)

---

## Fluxo Recomendado

O aplicativo é dividido em duas fases claramente separadas no menu lateral:

### 🏗️ Geração de Dados de Trabalho
> Execute uma vez (ou quando houver nova versão dos dados RFB). Etapas sequenciais.

| Módulo | Nome | Saída principal |
|--------|------|-----------------|
| 1 | ⬇️ Mineração de Dados | CSVs brutos extraídos em `raw/` |
| 2 | 🔧 ETL e Base Unificada | `processed/base_unificada.parquet` |
| 3 | 📚 CNAEs IBGE | `processed/cnae_ibge/cnae_hierarquia.parquet` |
| 4 | 📊 Exploração de Dados | (interativo — sem saída em disco) |
| 5 | 🎯 Mercado Foco | `processed/foco_<nome>.parquet` |

### 📈 Análise e Prospecção
> Use quantas vezes quiser, apenas com o parquet foco gerado no Módulo 5.

| Módulo | Nome | Descrição |
|--------|------|-----------|
| 6 | 🔭 Atendimento Direto | Análise de cobertura direta da carteira por CNAE, mercado similar e oportunidades não atendidas |
| 7 | 🏭 Atendimento Indireto | Cobertura por rede de distribuidores com raio geográfico (BallTree haversine) |

---

## Schema dos Dados RFB

Os arquivos baixados do SERPRO Nextcloud **não têm cabeçalho** (`header=None`), encoding `latin1`, separador `;`.
Todos os arquivos principais usam `cnpj_basico` (8 dígitos, zero-padded) como chave de relacionamento.

### Diagrama de relacionamento

```
SIMPLES ──────────────────────────────┐
(1:1 por cnpj_basico)                 │
                                       ▼
EMPRESAS ─────── cnpj_basico ──── ESTABELECIMENTOS ──── MUNICIPIOS  (col 20)
   │                   │               │                 MOTIVOS     (col 7)
   │            SOCIOS (N:1)           │                 CNAES       (col 11, 12)
   │                                   │                 PAISES      (col 9)
   ├── natureza_juridica ────────── NATUREZAS
   └── qualificacao_responsavel ─── QUALIFICACOES
              ▲ (também em SOCIOS cols 4, 9)
```

### EMPRESAS — 7 colunas

| # | Campo | Notas |
|---|---|---|
| 0 | `cnpj_basico` | 8 dígitos, PK |
| 1 | `razao_social` | |
| 2 | `natureza_juridica` | código 4 dig → NATUREZAS |
| 3 | `qualificacao_responsavel` | código 2 dig → QUALIFICACOES |
| 4 | `capital_social` | `"1000,00"` (vírgula decimal) |
| 5 | `porte` | `01`=ME · `03`=EPP · `05`=Demais |
| 6 | `ente_federativo_responsavel` | |

### ESTABELECIMENTOS — 30 colunas

| # | Campo | Notas |
|---|---|---|
| 0 | `cnpj_basico` | FK → EMPRESAS |
| 1 | `cnpj_ordem` | 4 dígitos |
| 2 | `cnpj_dv` | 2 dígitos; CNPJ = col0+col1+col2 |
| 3 | `matriz_filial` | `1`=Matriz · `2`=Filial |
| 4 | `nome_fantasia` | |
| 5 | `situacao_cadastral` | `02`=Ativa · `08`=Baixada (filtro padrão: `02`) |
| 6 | `data_situacao_cadastral` | YYYYMMDD |
| 7 | `motivo_situacao` | código 2 dig → MOTIVOS |
| 8 | `nome_cidade_exterior` | para estrangeiras |
| 9 | `pais` | código 3 dig → PAISES |
| 10 | `data_inicio_atividade` | YYYYMMDD |
| 11 | `cnae_principal` | código 7 dig → CNAES |
| 12 | `cnaes_secundarios` | múltiplos códigos separados por vírgula |
| 13–17 | `tipo_logradouro`, `logradouro`, `numero`, `complemento`, `bairro` | |
| 18 | `cep` | 8 dígitos |
| 19 | `uf` | |
| 20 | `municipio` | código 4 dig TOM/RFB → MUNICIPIOS |
| 21–27 | `ddd1`, `telefone1`, `ddd2`, `telefone2`, `ddd_fax`, `fax`, `email` | |
| 28–29 | `situacao_especial`, `data_situacao_especial` | |

### SOCIOS — 11 colunas

| # | Campo | Notas |
|---|---|---|
| 0 | `cnpj_basico` | FK → EMPRESAS |
| 1 | `identificador_socio` | `1`=PJ · `2`=PF · `3`=Estrangeiro |
| 2 | `nome_socio` | |
| 3 | `cpf_cnpj_socio` | mascarado (`***XXXXXX**`) |
| 4 | `qualificacao_socio` | código 2 dig → QUALIFICACOES |
| 5 | `data_entrada_sociedade` | YYYYMMDD |
| 6 | `pais_socio` | código 3 dig → PAISES |
| 7–8 | `cpf_representante`, `nome_representante` | |
| 9 | `qualificacao_representante` | código 2 dig → QUALIFICACOES |
| 10 | `faixa_etaria` | `1`=0-12 … `9`=81+ · `0`=não informado |

### SIMPLES / MEI — 7 colunas (~2,8 GB descomprimido)

| # | Campo | Notas |
|---|---|---|
| 0 | `cnpj_basico` | 8 dígitos |
| 1 | `opcao_simples` | S/N |
| 2 | `data_opcao_simples` | YYYYMMDD |
| 3 | `data_exclusao_simples` | YYYYMMDD |
| 4 | `opcao_mei` | S/N |
| 5 | `data_opcao_mei` | YYYYMMDD |
| 6 | `data_exclusao_mei` | YYYYMMDD |

### Tabelas de domínio (lookup)

| Arquivo glob | Tabela | Registros | Chave |
|---|---|---|---|
| `*CNAECSV*` | CNAES | 1.359 | código 7 dig |
| `*MOTICSV*` | MOTIVOS | 63 | código 2 dig |
| `*MUNICCSV*` | MUNICIPIOS | 5.572 | código 4 dig (TOM/RFB ≠ IBGE) |
| `*NATJUCSV*` | NATUREZAS | 91 | código 4 dig |
| `*PAISCSV*` | PAISES | 255 | código 3 dig |
| `*QUALSCSV*` | QUALIFICACOES | 68 | código 2 dig |
| `*SIMPLES*` | SIMPLES/MEI | ~60 M | cnpj_basico 8 dig |

### Parquet `base_unificada` — colunas resultantes

A granularidade é **CNPJ completo de 14 dígitos** (ESTABELECIMENTOS como âncora).

```
Identidade:  cnpj · cnpj_basico · cnpj_ordem · cnpj_dv · matriz_filial
Empresa:     razao_social · nome_fantasia · capital_social · porte · porte_desc
             natureza_juridica · natureza_juridica_desc
             qualificacao_responsavel · qualificacao_responsavel_desc
Situação:    situacao_cadastral · data_situacao_cadastral · motivo_situacao · motivo_situacao_desc
             situacao_especial · data_situacao_especial
Atividade:   cnae_principal · cnae_principal_desc · cnaes_secundarios · data_inicio_atividade
             secao · secao_desc · divisao · divisao_desc · grupo · grupo_desc · classe · classe_desc
Endereço:    tipo_logradouro · logradouro · numero · complemento · bairro · cep · uf
             municipio · municipio_nome · nome_cidade_exterior · pais · pais_nome
Contato:     ddd1 · telefone1 · email
Simples/MEI: opcao_simples · data_opcao_simples · data_exclusao_simples
             opcao_mei · data_opcao_mei · data_exclusao_mei
```

Sócios ficam em **`socios.parquet`** separado (relação N:1 com EMPRESAS).

---

## Módulos do Aplicativo

### Módulo 1 — Mineração de Dados ✅

Download automatizado dos dados abertos da RFB via **SERPRO Nextcloud WebDAV**:

- URL: `https://arquivos.receitafederal.gov.br/index.php/s/YggdBLfdninEJX9`
- Autenticação Basic com share token; `verify=False` (CA SERPRO)
- Listagem de todas as versões disponíveis (`YYYY-MM`); seleção pelo usuário
- 37 arquivos por versão: 10 Empresas + 10 Estabelecimentos + 10 Sócios + 7 Auxiliares
- Indicação de espaço em disco necessário antes de iniciar o download
- Barra de progresso por arquivo (MB, %, velocidade, ETA) + barra de progresso global
- Extração automática dos ZIPs com renomeação segura para `.csv`
- `skip_existing`: pula download se ZIP já existe e CSVs já foram extraídos

---

### Módulo 2 — ETL e Base Unificada ✅

- Leitura dos CSVs brutos de Empresas, Estabelecimentos, Sócios e Simples Nacional
- Filtro de situação cadastral (padrão: apenas ativas = `02`)
- Schema posicional correto: `header=None`, `sep=';'`, `encoding='latin1'`
- Descoberta automática das tabelas auxiliares por glob (`*CNAECSV*`, `*MUNICCSV*`, etc.)
- Joins left com todas as 6 tabelas de domínio
- Decodificação de porte (`01`→ME, `03`→EPP, `05`→Demais)
- **Parquets gerados:**
  - `processed/empresas.parquet` — empresas deduplicadas
  - `processed/estabelecimentos.parquet` — estabelecimentos limpos
  - `processed/socios.parquet` — sócios com qualificação decodificada (opcional)
  - `processed/base_unificada.parquet` — consolidado com todos os joins + hierarquia CNAE

---

### Módulo 3 — CNAEs IBGE ✅

- Consulta à **API REST do IBGE** para hierarquia CNAE completa:
  - Endpoint: `https://servicodados.ibge.gov.br/api/v2/cnae/subclasses`
  - 1.332 subclasses com cadeia hierárquica completa (subclasse → seção)
- Estrutura: **Seção** (21) → **Divisão** (87) → **Grupo** (283) → **Classe** (671) → **Subclasse** (1.332)
- Salvo em `processed/cnae_ibge/cnae_hierarquia.parquet`
- **Integrado automaticamente** na `base_unificada` durante o ETL do Módulo 2

---

### Módulo 4 — Exploração de Dados ✅

- Visualizações interativas (Plotly): distribuição por UF, porte, natureza jurídica, top CNAEs, evolução temporal
- **Busca textual em CNAEs** em qualquer nível da hierarquia (Seção → Subclasse)
- Carregamento lazy via **DuckDB** — filtra diretamente no Parquet sem carregar 28 M linhas em memória
- Filtros em cascata: Seção → Divisão → Grupo → Classe → Subclasse com multiselect
- Boxplot de capital social (P5-P95, top 10 CNAEs + TOTAL)

---

### Módulo 5 — Mercado Foco ✅

O módulo central de inteligência comercial. Fluxo em etapas:

1. **Upload CSV** da base de clientes (auto-detecta separador)
2. **Mapeamento de colunas** — auto-sugere coluna de CNPJ (regex) e coluna de valor (>50% numérico); permite selecionar colunas de classificação adicionais (canal, segmento, etc.)
3. **Enriquecimento via CNPJ** — DuckDB INNER JOIN com `base_unificada.parquet`; nunca carrega o parquet inteiro em memória
4. **Seleção de CNAEs de interesse** e **construção do parquet foco** (ver schema abaixo)
5. **Opcão de enriquecimento lat/lon** via centróide de município (shapefile IBGE, download automático)

Após a geração do parquet foco, o módulo também exibe:
- Top CNAEs por nº de clientes (contagem) + por valor (soma) + ranking combinado
- CNAEs secundários explodidos a partir de `cnaes_secundarios`
- Anti-join DuckDB para identificar não atendidos com CNAE de interesse
- Gráfico Atendidos × Não Atendidos por UF
- Top 10 cidades por potencial não atendido
- Tabela detalhada exportável para CSV

---

### Módulo 6 — Atendimento Direto ✅

Analisa a cobertura da carteira atual e identifica oportunidades de atendimento direto. Estruturado em quatro seções progressivas:

**1 — Seleção de contexto**
- Filtros dinâmicos sobre colunas de classificação do parquet foco (canal, segmento, etc.) que definem os clientes atendidos do contexto analisado
- Suporte a múltiplos filtros simultâneos com botão de adição/remoção
- Botão **🔍 Analisar CNAEs de clientes** dispara o cálculo das seções seguintes

**2 — CNAEs da carteira atendida**
- Pareto de CNAEs por nº de CNPJs e por valor agregado (`relevancia_valor`)
- Cores ABC: dourado (A ≤ 80%), safira (B 80–95%), cinza (C > 95%)
- Percentual individual + percentual acumulado em cada barra (sem eixo Y secundário)

**3 — Avaliação do mercado similar**
- Multiselect de subclasses CNAE com pré-seleção dos top-10 já presentes na carteira
- Stacked bar de CNPJs atendidos × não atendidos por subclasse
- Distribuição por UF, top 10 cidades e nuvem de palavras em Nome Fantasia dos não atendidos

**4 — Avaliação de Atendimento Direto**
- Opções: ignora nomes fantasia em branco, considera `empresa_base_atendida`, exclui por palavras-chave configuráveis
- Botão **🔄 Calcular** com padrão *lazy* (parâmetros são capturados no clique; widgtes não re-executam o cálculo)
- **Mapa de calor** (Plotly `Densitymapbox`, fundo `carto-positron`): vermelho = não atendidos, azul = atendidos; grade 0,05° para evitar sobrecarga de memória; escala limitada ao P90 de densidade para evitar que SP domine a paleta
- KPIs: total de CNPJs, atendidos e não atendidos
- Exportação CSV dos não atendidos com todos os metadados RFB

---

### Módulo 7 — Atendimento Indireto ✅

Analisa a cobertura de clientes finais (varejistas, revendas, etc.) pela rede de distribuidores (atacadistas, homecenters, etc.) já presentes na base atendida. As três primeiras seções replicam a estrutura do Módulo 6; a quarta é exclusiva deste módulo.

**1–3 — Seleção de contexto / CNAEs / Mercado similar**
- Idênticas ao Módulo 6, porém o contexto aqui define os **clientes finais** (ex.: Canal = Varejo)
- Estados de sessão com prefixo `ai_` (isolados do Módulo 6, que usa `ad_`)

**4 — Avaliação via Distribuição**

*Configuração:*
- Campo + valores que identificam os distribuidores na base atendida (ex.: Canal = Atacado Regional, Atacado Nacional)
- Raio de cobertura em km (padrão: 150 km)
- Mesmas opções de limpeza do Módulo 6 (nome em branco, empresa-base, palavras-chave)
- Botão **🔄 Calcular distribuição** com padrão *lazy*

*Separação de universos:*
- `_df_distrib` — carregado **diretamente do parquet sem filtro de CNAE**, pois atacadistas têm CNAEs completamente diferentes dos clientes finais; query: `cliente_atendido = true AND campo IN (valores)`
- `_df_finais` — clientes não atendidos dentro das CNAEs selecionadas nas seções anteriores
- `_df_diretos` — clientes atendidos que não são distribuidores

*Cálculo BallTree haversine:*
```
sklearn.neighbors.BallTree(coords_distribuidores, metric="haversine")
→ para cada cliente final: distribuidor mais próximo + distância em km
→ atendimento_via = "Dentro do raio" se dist_km ≤ raio, senão "Fora do raio"
```

*Mapa:*
- Multiselect de distribuidores visíveis, ordenado por `relevancia_valor` decrescente; opção **◉ Ver todos**
- Heatmap duplo: vermelho = fora do raio, azul = dentro do raio (grade 0,05°, P90 zmax)
- Marcadores dos distribuidores selecionados (`Scattermapbox`)

*KPIs:* total de clientes finais · atendidos direto · dentro do raio · fora do raio

*Gráfico de distribuidores:*
- Todos os distribuidores identificados (inclusive com 0 clientes próximos)
- Ordenado por "Dentro do raio" decrescente — maior cobertura no topo
- Altura fixa (≈ 10 barras visíveis); scroll + zoom para ver todos
- Filtro por UF embutido (não re-executa o BallTree)

*Exportação:* fora do raio / distribuidor específico / todos — CSV utf-8 com `atendimento_via`, `cnpj_distrib_proximo`, `nome_distrib`, `dist_km_min`

---

## Parquet de Base Foco

O arquivo `processed/foco_<nome>.parquet` é gerado pelo Módulo 5 e contém **apenas os CNPJs** que se enquadram nos CNAEs de interesse selecionados pelo usuário. Além das colunas da `base_unificada`, ele adiciona:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `cliente_atendido` | bool | `True` se o CNPJ está na base interna de clientes |
| `empresa_base_atendida` | bool | `True` se **qualquer estabelecimento** do mesmo `cnpj_basico` já é cliente atendido — sinaliza oportunidade de expansão dentro de uma empresa conhecida |
| `relevancia_valor` | float | Soma do valor da coluna de referência para esse CNPJ na base interna |
| `<colunas de classificação>` | str | Quaisquer colunas adicionais mapeadas no upload (canal, segmento, etc.) — usadas como filtro de contexto nos Módulos 6 e 7 |
| `lat` | float | Latitude do centróide do município (opcional — requer enriquecimento lat/lon no M5) |
| `lon` | float | Longitude do centróide do município (opcional) |
| `CD_MUN` | str | Código IBGE do município — chave para o join com o shapefile |

### Conceito: `empresa_base_atendida`

O `cnpj_basico` é formado pelos primeiros 8 dígitos do CNPJ completo (14 dígitos) e identifica a **empresa-mãe** de todos os seus estabelecimentos. `empresa_base_atendida = True` significa que existe pelo menos um outro estabelecimento da mesma empresa-base já atendido — oportunidade de expansão com menor fricção comercial.

### Conceito: colunas de classificação como duplo filtro (M6 e M7)

As colunas de classificação (ex.: `Canal`) servem para **dois propósitos distintos** nos módulos de análise:

- **Módulo 6 (Atendimento Direto):** o filtro de contexto seleciona os clientes *atendidos* cujo canal corresponde ao segmento analisado (ex.: Varejo). Todos os outros registros com os mesmos CNAEs são tratados como mercado potencial não atendido.
- **Módulo 7 (Atendimento Indireto):** o filtro de contexto identifica os *clientes finais* (ex.: Canal = Varejo); um segundo filtro independente identifica os *distribuidores* (ex.: Canal = Atacado Regional). Os distribuidores são carregados sem restrição de CNAE — eles tipicamente têm CNAEs completamente diferentes dos clientes finais.

---

## Instalação e Execução

### Pré-requisitos

- **Python 3.10+** (testado em Python 3.13)
- Acesso à rede SERPRO (`arquivos.receitafederal.gov.br`) — requer `verify=False` em redes com proxy Zscaler
- **geopandas** instalado no Python do sistema (necessário para enriquecimento lat/lon do Módulo 5)
- **scikit-learn** — incluso em `requirements.txt`; usado pelo Módulo 7 (BallTree haversine)

### Instalação das dependências

```bash
pip install -r requirements.txt
pip install geopandas  # instalação separada — depende de GDAL/Fiona
```

> Não utilizar `.venv` nem executar como módulo — o projeto é executado no ambiente Python corporativo.

### Execução

```bash
cd sggeodata
streamlit run app.py
```

Ou clique duplo em **`iniciar.bat`**. O bat verifica automaticamente as dependências antes de iniciar.

### Verificação de dependências

O script `_check_deps.py` verifica todas as bibliotecas antes de iniciar o Streamlit:

```
python _check_deps.py
```

### Configuração da pasta de dados

Na primeira execução, configure a **pasta de dados** no painel lateral (⚙️ Pasta de Dados).
Pode ser **fora** do diretório do projeto (ex: `D:\Dados\RFB\`).
A configuração é salva em `sggeodata/config.json`.

### Estrutura esperada da pasta de dados

```
data/
├── raw/
│   ├── zips/                        ← ZIPs baixados pelo Módulo 1
│   ├── Empresas/                    ← CSVs extraídos
│   ├── Estabelecimentos/            ← CSVs extraídos
│   ├── Socios/                      ← CSVs extraídos
│   └── Auxiliares/                  ← tabelas de domínio (CNAE, MUNIC, etc.)
│
└── processed/
    ├── empresas.parquet
    ├── estabelecimentos.parquet
    ├── socios.parquet
    ├── base_unificada.parquet
    ├── cnae_ibge/
    │   └── cnae_hierarquia.parquet
    ├── foco_<nome>.parquet          ← gerado pelo Módulo 5 (um por configuração)
    └── shapefiles/
        └── BR_Municipios_<ANO>/     ← shapefile IBGE (download automático)
            ├── BR_Municipios_<ANO>.shp
            └── ...
```

---

## Estrutura de Pastas

```
sggeodata/
├── app.py                    # Entrada principal do Streamlit
├── iniciar.bat               # Atalho de execução para Windows
├── _check_deps.py            # Verificação de dependências antes de iniciar
├── requirements.txt          # Dependências Python (inclui scikit-learn)
├── config.json               # Configuração local (pasta de dados) — não versionar
├── README.md                 # Esta documentação
│
├── pages/
│   ├── 1_Mineracao.py        # Módulo 1 — Download SERPRO Nextcloud + extração ZIPs
│   ├── 2_ETL.py              # Módulo 2 — ETL completo → parquets enriquecidos
│   ├── 3_CNAE_IBGE.py        # Módulo 3 — Hierarquia CNAE via API REST IBGE
│   ├── 4_Exploracao.py       # Módulo 4 — KPIs, gráficos, busca textual DuckDB
│   ├── 5_Mercado_Foco.py     # Módulo 5 — Upload base clientes → parquet foco enriquecido
│   ├── 6_Oportunidades.py    # Módulo 6 — Atendimento Direto: contexto, CNAEs, mercado, prospecção
│   └── 7_Atendimento_Indireto.py  # Módulo 7 — Atendimento Indireto: distribuidores + BallTree haversine
│
└── utils/
    ├── config.py             # Gerenciamento de config.json e subpastas de dados
    ├── storage.py            # Leitura/escrita Parquet, DuckDB e métricas de disco
    ├── sidebar.py            # Sidebar compartilhada (render_sidebar) + seletor de parquet foco
    ├── rfb_download.py       # WebDAV SERPRO Nextcloud: listagem, download, extração
    ├── rfb_etl.py            # ETL: Empresas/Estab/Socios/Simples → parquet unificado
    ├── build_cnae_hierarquia.py  # API REST IBGE → cnae_hierarquia.parquet
    └── ibge_geo.py           # Download automático do shapefile IBGE via FTP
```

---

## Privacidade e Segurança

> Os arquivos abaixo contêm dados comerciais sensíveis e **não devem ser versionados**. Já constam no `.gitignore`:

- `config.json` — contém o caminho da pasta de dados
- `CNPJs_SGA.csv` — base de clientes internos
- `df_interesse_norton_enriquecido.csv` — dataset final enriquecido
- `oportunidades_fora_cobertura*.xlsx` — listas de prospecção
- `data/` — todos os dados brutos e processados da RFB (pasta externa ao repo)
- Parquets foco gerados pelo Módulo 5

---

## Changelog

### v0.2 — 2026-03-09

#### Módulo 7 — Atendimento Indireto (`pages/7_Atendimento_Indireto.py`)

**Seção "CNAEs da carteira atendida"**
- **Fix: título sobrepondo legenda ABC** — `margin top` aumentado de `t=40` → `t=70` e posição da legenda ajustada (`y=1.14`), eliminando sobreposição.
- **Fix: % e ACC calculados sobre a base completa** — os indicadores de porcentagem e acumulado agora são calculados sobre o total de todos os CNAEs da carteira, não apenas sobre a soma dos top-10 exibidos. O parâmetro `base_total` é passado explicitamente para `_pareto_fig()`.
- **Fix: limite de caracteres na descrição do CNAE** — tamanho máximo ampliado de 40 → 55 caracteres no label de cada barra.
- **Fix: stacked bar "CNPJs atendidos vs não atendidos" ordenado por não atendido decrescente** — o gráfico agora ordena as subclasses pelo total de CNPJs *não atendidos* (maior no topo).

**Seção "Avaliação via Distribuição" — Mapa de cobertura**
- **Mapa com 3 camadas de cor:**
  - 🔴 **Vermelho** — clientes fora do alcance de qualquer distribuidor.
  - 🔵 **Azul** — clientes dentro do raio dos distribuidores selecionados no filtro.
  - 🟢 **Verde** — clientes dentro do raio de outros distribuidores (não selecionados). Quando "todos" estão selecionados, apenas azul e vermelho são exibidos.
- **Checkbox "Considerar apenas esses distribuidores na alocação"** — aparece somente quando distribuidores específicos estão selecionados. Se marcado, a alocação prioriza sempre o distribuidor selecionado mais próximo (dentro do raio); se desmarcado, o cliente é atribuído ao distribuidor mais próximo globalmente.
- **Pontos dos distribuidores amarelos** — marcadores na cor amarelo (`#FFD700`) para contrastar com o mapa de calor azul. Hover exibe: empresa mãe (cnpj_basico), CNPJ filho, razão social e nome fantasia.
- **KPIs expandidos para 5 métricas** — Clientes finais, Atendidos direto, 🔵 Dentro raio (selecionados), 🟢 Dentro raio (outros), 🔴 Fora do raio.
- **Gráfico "Clientes finais mais próximos de cada distribuidor"** — exibe apenas distribuidores com clientes alocados (removidos distribuidores sem alocação); eixo X forçado a iniciar em zero (`rangemode="tozero"`); suporte às 3 categorias de cor.
- **Tabela de exportação enriquecida** — adicionadas colunas `distrib_empresa_mae`, `distrib_razao_social` e `distrib_nome_fantasia` identificando o distribuidor mais próximo com dados completos. Seletor "Distribuidor específico" também exibe razão social + nome fantasia.
- **Bugfix: `KeyError: 'cnpj_basico_distrib'`** — corrigida duplicação de chave ao fazer merge de `_dist_info` no DataFrame de exportação (a coluna já existia em `_df_result`; `cnpj_basico` foi removido do rename).
- **Reorganização de código** — o multiselect de distribuidores visíveis e o checkbox de priorização foram movidos para *antes* do cálculo BallTree, garantindo que `_todos_selecionados`, `_dist_geo_vis`, `_sel_cb_vis` e `_priorizar_sel` estejam disponíveis na lógica de classificação.

---

## Histórico — Pipeline de Notebooks

> Esta seção documenta o pipeline legado de notebooks utilizado antes da criação do SGGeoData. Mantido apenas para referência histórica.

### `auto_download.py`
Script legado de scraping do portal `dadosabertos.rfb.gov.br` — domínio bloqueado por proxies corporativos. **Substituído pelo Módulo 1** que usa a nova fonte oficial SERPRO Nextcloud.

### `unzip_and_unite.py`
Descompacta os ZIPs nas subpastas (`Empresas/`, `Estabelecimentos/`, `Socios/`) e renomeia os arquivos extraídos com extensão `.csv`. **Substituído pelo Módulo 1**.

### `estabelecimentos_data_cleaning_full.ipynb` / `empresas_EDA.ipynb`
ETL legado que gerava `estabelecimentos_ETLd.csv` e `empresas_ETL.csv`. **Substituídos pelo Módulo 2**.

### `Fazenda_Vs_SGA.ipynb`
Notebook de análise e inteligência comercial (cruzamento RFB × SGA, CNAEs 95%, `df_interesse_norton_enriquecido.csv`). **Substituído pelos Módulos 5, 6 e 7**.

### `mapa_atendimento.ipynb`
Notebook de visualização geográfica com Folium. A geolocalização por centróide de município foi absorvida pelo Módulo 5 via `utils/ibge_geo.py`. A análise de cobertura por raio com BallTree/Haversine foi implementada no **Módulo 7**.

### `findcep_geoloc_notebook.ipynb`
Funções utilitárias para converter CEPs em coordenadas via API FindCEP. **Não utilizado no SGGeoData** — a geolocalização é feita via centróide de município (IBGE), sem dependência de API externa.

