"""
SGGeoData — Aplicativo Streamlit principal.

Navegação via sidebar + painel de configuração da pasta de dados.
"""

import streamlit as st
from pathlib import Path

# Importa helpers de configuração
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_data_folder
from utils.sidebar import render_sidebar

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SGGeoData",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — Configuração global
# ---------------------------------------------------------------------------

render_sidebar()

# ---------------------------------------------------------------------------
# Página inicial
# ---------------------------------------------------------------------------

st.title("🗺️ SGGeoData")
st.subheader("Plataforma de Inteligência Geográfica e Comercial")

st.markdown("""
Bem-vindo ao **SGGeoData**. Use o menu lateral para navegar entre os módulos.

---

### Módulos desta versão

| # | Módulo | Descrição |
|---|--------|-----------|
| 1 | ⬇️ **Mineração de Dados** | Download automático dos dados abertos da Receita Federal |
| 2 | 🔧 **ETL e Base Unificada** | Limpeza, filtros e consolidação de Empresas + Estabelecimentos |
| 3 | 📚 **CNAEs IBGE** | Download e parsing dos PDFs de notas explicativas CNAE |
| 4 | 📊 **Exploração de Dados** | Visualizações e busca por texto nos CNAEs |
| 5 | 🎯 **Mercado Foco** | Upload base de clientes → enriquecimento CNPJ → análise de cobertura → parquet foco com carteira, relevância, classificações e lat/lon |
| 6 | 🔭 **Atendimento Direto** | Selecione um parquet foco, defina o contexto da carteira e analise cobertura de atendimento, mercado similar e oportunidades não atendidas |
| 7 | 🏭 **Atendimento Indireto** | Selecione distribuidores na carteira, defina o raio de cobertura e veja quais clientes finais estão cobertos pela rede de distribuição |

---

### Antes de começar

Configure a **pasta de dados** no painel lateral (⚙️). Ela pode ser externa ao projeto —
os arquivos brutos (~30 GB) e processados serão salvos nela, sem interferir no versionamento Git.

**Fluxo recomendado:**

🏗️ **Geração de Dados de Trabalho** (baixa frequência — execute uma vez)  
`⬇️ Mineração` → `🔧 ETL` → `📚 CNAEs IBGE` → `📊 Exploração` → `🎯 Mercado Foco`

📈 **Análise e Prospecção** (use apenas o parquet foco gerado acima)  
`🔭 Oportunidades`
""")

# Alerta se pasta não configurada
data_folder = get_data_folder()
if not Path(data_folder).exists():
    st.warning("⚠️ Pasta de dados não configurada. Expanda o painel **⚙️ Pasta de Dados** na barra lateral.")
else:
    st.info(f"📁 Pasta de dados: `{data_folder}`")
