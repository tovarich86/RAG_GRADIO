# app.py (versão com Melhoria 1 e 2)

import streamlit as st
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import requests
import re
import unicodedata
import logging
from pathlib import Path
import zipfile
import io
import shutil
import random
from models import get_embedding_model, get_cross_encoder_model
from concurrent.futures import ThreadPoolExecutor # <<< MELHORIA 4 ADICIONADA
from tools import (
    find_companies_by_topic,
    get_final_unified_answer,
    suggest_alternative_query,
    analyze_topic_thematically, 
    get_summary_for_topic_at_company,
    rerank_with_cross_encoder,
    create_hierarchical_alias_map,
    rerank_by_recency
    )
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
HF_REPO_ID = "tovarich86/analise-ilp-dados"

# --- Módulos do Projeto (devem estar na mesma pasta) ---
from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
from analytical_engine import AnalyticalEngine

# --- Configurações Gerais ---
st.set_page_config(page_title="Agente de Análise LTIP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
TOP_K_SEARCH = 7
TOP_K_INITIAL_RETRIEVAL = 30
TOP_K_FINAL = 15             # Número final de chunks a usar no contexto
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
CVM_SEARCH_URL = "https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx"

CACHE_DIR = Path("data_cache_gradio") # Usamos um novo diretório de cache para evitar conflitos

# Mapeia o nome do arquivo no repositório para o nome que ele terá localmente
FILES_TO_DOWNLOAD = {
    "item_8_4_chunks_map.json": "item_8_4_chunks_map_final.json",
    "item_8_4_faiss_index.bin": "item_8_4_faiss_index_final.bin",
    "outros_documentos_chunks_map.json": "outros_documentos_chunks_map_final.json",
    "outros_documentos_faiss_index.bin": "outros_documentos_faiss_index_final.bin",
    "resumo_fatos_e_topicos_v4_por_data.json": "resumo_fatos_e_topicos_final_enriquecido.json"
}
SUMMARY_FILENAME = "resumo_fatos_e_topicos_final_enriquecido.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CARREGADOR DE DADOS ---
# --- CARREGADOR DE DADOS ---
@st.cache_resource(show_spinner="Configurando o ambiente e baixando dados...")
def setup_and_load_data():
    """
    Baixa dados do Hugging Face Hub, carrega modelos e prepara todos os artefatos
    necessários para a aplicação. É executada uma única vez no início.
    """
    logger.info("Iniciando configuração e carregamento de dados...")
    CACHE_DIR.mkdir(exist_ok=True)

    # 1. Baixar todos os arquivos de dados do Hugging Face Hub
    for repo_filename, local_filename in FILES_TO_DOWNLOAD.items():
        local_path = CACHE_DIR / local_filename
        if not local_path.exists():
            logger.info(f"Baixando '{repo_filename}' do repo '{HF_REPO_ID}'...")
            try:
                # A função hf_hub_download baixa o arquivo e o coloca em um cache gerenciado.
                downloaded_path_str = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=repo_filename,
                    cache_dir=CACHE_DIR, # Define um diretório de cache local
                    force_filename=local_filename # Força o nome do arquivo final
                )
                logger.info(f"Arquivo salvo em: {downloaded_path_str}")
            except Exception as e:
                logger.error(f"Falha ao baixar '{repo_filename}': {e}")
                # Em um app real, você poderia querer sair do programa aqui
                raise e

    # 2. Carregar os modelos de ML
    logger.info("Carregando modelos de embedding e cross-encoder...")
    embedding_model = get_embedding_model()
    cross_encoder_model = get_cross_encoder_model()

    # 3. Carregar os artefatos (índices FAISS e mapas de chunks)
    artifacts = {}
    for index_file_path in CACHE_DIR.glob('*_faiss_index_final.bin'):
        category = index_file_path.stem.replace('_faiss_index_final', '')
        chunks_file_path = CACHE_DIR / f"{category}_chunks_map_final.json"
        try:
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                # Acessa diretamente a lista de chunks, como no seu código corrigido
                list_of_chunks = json.load(f)

            artifacts[category] = {
                'index': faiss.read_index(str(index_file_path)),
                'chunks': list_of_chunks
            }
            logger.info(f"Artefatos para a categoria '{category}' carregados com sucesso.")
        except Exception as e:
            logger.error(f"Falha ao carregar artefatos para a categoria '{category}': {e}")
            raise

    # 4. Carregar os dados de resumo
    summary_file_path = CACHE_DIR / "resumo_fatos_e_topicos_final_enriquecido.json"
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        logger.info("Dados de resumo carregados com sucesso.")
    except FileNotFoundError as e:
        logger.error(f"Erro crítico ao carregar dados de resumo: {e}")
        raise

    # 5. Gerar listas de filtros dinâmicos
    setores = set()
    controles = set()
    for artifact_data in artifacts.values():
        for metadata in artifact_data.get('chunks', []):
            setor = metadata.get('setor')
            if isinstance(setor, str) and setor.strip():
                setores.add(setor.strip().capitalize())

            controle = metadata.get('controle_acionario')
            if isinstance(controle, str) and controle.strip():
                controles.add(controle.strip().capitalize())

    all_setores = ["Todos"] + sorted(list(setores))
    all_controles = ["Todos"] + sorted(list(controles))
    logger.info("Listas de filtros dinâmicos geradas.")

    logger.info("✅ Ambiente pronto!")
    # Retorna todos os objetos que a aplicação precisará
    return (
        artifacts, summary_data, all_setores, all_controles,
        embedding_model, cross_encoder_model
    )

# --- Carregamento Global dos Dados e Modelos ---
# Esta parte já deve existir no seu script
(
    artifacts, summary_data, setores_disponiveis, controles_disponiveis,
    embedding_model, cross_encoder_model
) = setup_and_load_data()

def run_full_analysis(query, setor, controle, priorizar_recente, progress=gr.Progress(track_tqdm=True)):
    """
    Função completa que orquestra a análise, chamada pela interface do Gradio.
    Usa um gerador (`yield`) para fornecer atualizações de status para a UI.
    """
    # Validação inicial
    if not query.strip():
        # A função deve retornar um valor para cada output definido no .click()
        # Retornamos (status_text, markdown_text, dataframe_data)
        yield "Pronto.", "⚠️ Por favor, digite uma pergunta.", None
        return

    # 1. Preparar filtros e status inicial
    active_filters = {}
    if setor != "Todos": active_filters['setor'] = setor.lower()
    if controle != "Todos": active_filters['controle_acionario'] = controle.lower()
    
    progress(0.1, desc="Analisando intenção da pergunta...")

    # 2. Roteamento de Intenção (Híbrido: Regras + LLM)
    intent = None
    query_lower = query.lower()
    quantitative_keywords = [
        'liste', 'quais empresas', 'quais companhias', 'quantas', 'média',
        'mediana', 'estatísticas', 'mais comuns', 'prevalência', 'contagem'
    ]
    if any(keyword in query_lower for keyword in quantitative_keywords):
        intent = "quantitativa"
    else:
        # A função get_query_intent_with_llm foi importada de tools.py
        # e já foi adaptada para receber a api_key
        intent = get_query_intent_with_llm(query, GEMINI_API_KEY, GEMINI_MODEL)

    # --- ROTA QUANTITATIVA ---
    if intent == "quantitativa":
        progress(0.5, desc="Executando análise quantitativa rápida...")
        report_text, data_result = analytical_engine.answer_query(query, filters=active_filters)
        
        df_to_show = None
        if isinstance(data_result, pd.DataFrame):
            df_to_show = data_result
        elif isinstance(data_result, dict) and data_result:
            # Se a análise retornar múltiplos dataframes, exibimos o primeiro e avisamos no texto.
            first_key = next(iter(data_result))
            df_to_show = data_result[first_key]
            if len(data_result) > 1:
                report_text += f"\n\n*Nota: Múltiplas tabelas foram geradas. Exibindo a primeira: '{first_key}'.*"

        progress(1.0, desc="Análise quantitativa concluída!")
        yield "Análise Concluída!", report_text, df_to_show
        return

    # --- ROTA QUALITATIVA (RAG) ---
    progress(0.2, desc="Gerando plano de análise RAG...")
    # Todas as funções necessárias (create_dynamic_analysis_plan, etc.) foram importadas de tools.py
    plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, DICIONARIO_UNIFICADO_HIERARQUICO, summary_data, active_filters)

    if plan_response['status'] != "success":
        progress(1.0, desc="Falha na identificação.")
        suggestion = suggest_alternative_query(query, DICIONARIO_UNIFICADO_HIERARQUICO)
        error_message = f"Não consegui identificar uma intenção clara na sua pergunta.\n\n**Sugestão:**\n`{suggestion}`"
        yield "Falha no Plano", error_message, None
        return

    plan = plan_response['plan']

    # --- Lógica de Comparação (Múltiplas Empresas) ---
    if len(plan.get('empresas', [])) > 1:
        progress(0.4, desc=f"Analisando {len(plan['empresas'])} empresas em paralelo...")
        with ThreadPoolExecutor(max_workers=len(plan['empresas'])) as executor:
            futures = [
                executor.submit(
                    analyze_single_company, empresa, plan, query, artifacts, embedding_model,
                    cross_encoder_model, DICIONARIO_UNIFICADO_HIERARQUICO, company_catalog_rich,
                    company_lookup_map, execute_dynamic_plan, get_final_unified_answer, GEMINI_API_KEY, GEMINI_MODEL
                )
                for empresa in plan['empresas']
            ]
            results = [future.result() for future in futures]
        
        progress(0.8, desc="Gerando relatório comparativo final...")
        structured_context = json.dumps(results, indent=2, ensure_ascii=False)
        comparison_prompt = f"""Sua tarefa é criar um relatório comparativo detalhado sobre "{query}", usando os dados estruturados no CONTEXTO JSON abaixo. Comece com uma análise textual e, em seguida, apresente uma TABELA MARKDOWN clara que compare os tópicos lado a lado para cada empresa. CONTEXTO: {structured_context}"""
        final_answer = get_final_unified_answer(comparison_prompt, "", GEMINI_API_KEY, GEMINI_MODEL)
        
        progress(1.0, desc="Relatório comparativo gerado!")
        yield "Análise Concluída!", final_answer, None
        return

    # --- Lógica de Análise Única ou Geral ---
    else:
        progress(0.5, desc="Recuperando e re-ranqueando contexto...")
        context, sources = execute_dynamic_plan(
            query, plan, artifacts, embedding_model, cross_encoder_model,
            DICIONARIO_UNIFICADO_HIERARQUICO, company_catalog_rich, company_lookup_map,
            search_by_tags, expand_search_terms, priorizar_recente
        )

        if not context:
            progress(1.0, desc="Nenhuma informação encontrada.")
            yield "Análise Concluída!", "❌ Não encontrei informações relevantes nos documentos para a sua consulta.", None
            return

        progress(0.8, desc="Gerando resposta final com LLM...")
        final_answer = get_final_unified_answer(query, context, GEMINI_API_KEY, GEMINI_MODEL)

        # Anexar fontes ao final da resposta
        if sources:
            sources_md = "\n\n---\n\n### 📚 Documentos Consultados\n"
            for src in sorted(sources, key=lambda x: x.get('company_name', '')):
                company_name = src.get('company_name', 'N/A')
                doc_date = src.get('document_date', 'N/A')
                doc_type_raw = src.get('doc_type', '')
                url = src.get('source_url', '#')
                display_doc_type = 'Plano de Remuneração' if doc_type_raw == 'outros_documentos' else doc_type_raw.replace('_', ' ')
                display_text = f"**{company_name}** - {display_doc_type} (Data: {doc_date})"
                sources_md += f"- {display_text} [Link]({url})\n"
            final_answer += sources_md

        progress(1.0, desc="Análise concluída!")
        yield "Análise Concluída!", final_answer, None
        return
# --- Construção da Interface Visual com Gradio Blocks ---
logger.info("Construindo a interface do Gradio...")

with gr.Blocks(title="Agente de Análise ILP", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky")) as demo:
    # --- Título Principal ---
    gr.Markdown(
        """
        # 🤖 Agente de Análise de Planos de Incentivo (ILP)
        **Faça perguntas quantitativas (listas, médias) ou qualitativas (comparações, detalhes) sobre planos de ILP.**
        """
    )

    # --- Layout Principal em Linhas e Colunas ---
    with gr.Row():
        # --- Coluna da Esquerda (Sidebar do Streamlit) ---
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### ⚙️ Filtros e Controles")
            
            # Dropdowns para filtros, preenchidos com os dados carregados
            filtro_setor = gr.Dropdown(label="Filtrar por Setor", choices=setores_disponiveis, value="Todos")
            filtro_controle = gr.Dropdown(label="Filtrar por Controle Acionário", choices=controles_disponiveis, value="Todos")
            
            # Checkbox para priorizar recência
            check_priorizar_recente = gr.Checkbox(label="Priorizar documentos mais recentes", value=True, info="Dá um bônus de relevância para os documentos mais novos.")

            # Um componente de texto para mostrar o status da análise
            status_component = gr.Textbox(label="Status da Análise", value="Pronto.", interactive=False)

            # Acordeão para a lista de empresas (equivalente ao st.expander)
            with gr.Accordion("Empresas com Dados de Resumo", open=False):
                lista_empresas_df = pd.DataFrame(sorted(list(summary_data.keys())), columns=["Empresa"])
                gr.DataFrame(lista_empresas_df, interactive=False, height=400)

        # --- Coluna da Direita (Área Principal) ---
        with gr.Column(scale=3):
            # Acordeão para o Guia do Usuário
            with gr.Accordion("ℹ️ Guia Rápido de Uso", open=False):
                gr.Markdown(
                    """
                    #### 1. Perguntas de Listagem e Estatística (Análise Rápida)
                    Use palavras como `liste`, `quais empresas`, `média de`, `mais comuns`.
                    *Ex: "Qual o período médio de vesting (em anos)?"*

                    #### 2. Análise Profunda (RAG)
                    Faça perguntas abertas sobre uma empresa específica.
                    *Ex: "Como funciona o plano de vesting da Vale?"*

                    #### 3. Comparação (RAG)
                    Mencione duas ou mais empresas na sua pergunta.
                    *Ex: "Compare o tratamento de dividendos da Localiza e da Movida."*
                    """
                )
            
            # Componente de Input para a pergunta do usuário
            query_input = gr.Textbox(
                label="Faça sua pergunta:",
                lines=5,
                placeholder="Ex: Compare as cláusulas de Malus/Clawback da Vale com as do Itaú."
            )

            # Botão de Ação Principal
            analisar_btn = gr.Button("🔍 Analisar", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### 📋 Resultado da Análise")
            
            # Componentes de Saída que serão atualizados pela lógica do backend
            output_markdown = gr.Markdown(label="Relatório Analítico")
            output_dataframe = gr.DataFrame(label="Dados Tabulares", interactive=False)

# --- Lançamento da Aplicação ---
# O if __name__ == "__main__": garante que o servidor só iniciará quando o script for executado diretamente
if __name__ == "__main__":
    # O método launch() inicia o servidor web do Gradio
    demo.launch(debug=True) # debug=True ajuda a ver erros no console


# --- FUNÇÕES GLOBAIS E DE RAG ---

def _create_flat_alias_map(kb: dict) -> dict:
    alias_to_canonical = {}
    for section, topics in kb.items():
        for topic_name_raw, aliases in topics.items():
            canonical_name = topic_name_raw.replace('_', ' ')
            alias_to_canonical[canonical_name.lower()] = canonical_name
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical

AVAILABLE_TOPICS = list(set(_create_flat_alias_map(DICIONARIO_UNIFICADO_HIERARQUICO).values()))

def expand_search_terms(base_term: str, kb: dict) -> list[str]:
    base_term_lower = base_term.lower()
    expanded_terms = {base_term_lower}
    for section, topics in kb.items():
        for topic, aliases in topics.items():
            all_terms_in_group = {alias.lower() for alias in aliases} | {topic.lower().replace('_', ' ')}
            if base_term_lower in all_terms_in_group:
                expanded_terms.update(all_terms_in_group)
    return list(expanded_terms)

# Em app.py, substitua esta função
def search_by_tags(query: str, kb: dict) -> list[str]:
    """
    Versão melhorada que busca por palavras-chave na query e retorna as tags correspondentes.
    Evita o uso de expressões regulares complexas para cada chunk.
    """
    found_tags = set()
    # Converte a query para minúsculas e remove pontuação para uma busca mais limpa
    clean_query = query.lower().strip()
    
    # Itera sobre todas as tags e seus sinônimos no dicionário de conhecimento
    for tag, details in kb.items():
        search_terms = [tag.lower()] + [s.lower() for s in details.get("sinonimos", [])]
        
        # Se qualquer um dos termos de busca estiver na query, adiciona a tag
        if any(term in clean_query for term in search_terms):
            found_tags.add(tag)
            
    return list(found_tags)

def get_final_unified_answer(query: str, context: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    has_complete_8_4 = "formulário de referência" in query.lower() and "8.4" in query.lower()
    has_tagged_chunks = "--- CONTEÚDO RELEVANTE" in context
    structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
    if has_complete_8_4:
        structure_instruction = "ESTRUTURA OBRIGATÓRIA PARA ITEM 8.4: Use a estrutura oficial do item 8.4 do Formulário de Referência (a, b, c...)."
    elif has_tagged_chunks:
        structure_instruction = "PRIORIZE as informações dos chunks recuperados e organize a resposta de forma lógica."
    prompt = f"""Você é um consultor especialista em planos de incentivo de longo prazo (ILP).
    PERGUNTA ORIGINAL DO USUÁRIO: "{query}"
    CONTEXTO COLETADO DOS DOCUMENTOS:
    {context}
    {structure_instruction}
    INSTRUÇÕES PARA O RELATÓRIO FINAL:
    1. Responda diretamente à pergunta do usuário com base no contexto fornecido.
    2. Seja detalhado, preciso e profissional na sua linguagem. Use formatação Markdown.
    3. Se uma informação específica pedida não estiver no contexto, declare explicitamente: "Informação não encontrada nas fontes analisadas.". Não invente dados.
    RELATÓRIO ANALÍTICO FINAL:"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        return f"Ocorreu um erro ao contatar o modelo de linguagem. Detalhes: {str(e)}"

# <<< MELHORIA 1 ADICIONADA >>>
def get_query_intent_with_llm(query: str) -> str:
    """
    Usa um LLM para classificar a intenção do usuário em 'quantitativa' ou 'qualitativa'.
    Retorna 'qualitativa' como padrão em caso de erro.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
    Analise a pergunta do usuário e classifique a sua intenção principal. Responda APENAS com uma única palavra em JSON.
    
    As opções de classificação são:
    1. "quantitativa": Se a pergunta busca por números, listas diretas, contagens, médias, estatísticas ou agregações. 
       Exemplos: "Quantas empresas têm TSR Relativo?", "Qual a média de vesting?", "Liste as empresas com desconto no strike.".
    2. "qualitativa": Se a pergunta busca por explicações, detalhes, comparações, descrições ou análises aprofundadas.
       Exemplos: "Como funciona o plano da Vale?", "Compare os planos da Hypera e Movida.", "Detalhe o tratamento de dividendos.".

    Pergunta do Usuário: "{query}"

    Responda apenas com o JSON da classificação. Exemplo de resposta: {{"intent": "qualitativa"}}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 50
        }
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        
        response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        intent_json = json.loads(re.search(r'\{.*\}', response_text, re.DOTALL).group())
        intent = intent_json.get("intent", "qualitativa").lower()
        
        logger.info(f"Intenção detectada pelo LLM: '{intent}' para a pergunta: '{query}'")
        
        if intent in ["quantitativa", "qualitativa"]:
            return intent
        else:
            logger.warning(f"Intenção não reconhecida '{intent}'. Usando 'qualitativa' como padrão.")
            return "qualitativa"

    except Exception as e:
        logger.error(f"ERRO ao determinar intenção com LLM: {e}. Usando 'qualitativa' como padrão.")
        return "qualitativa"


# Em app.py, substitua sua função pela versão ABAIXO, que é a sua versão original e robusta, apenas com os erros corrigidos.

from datetime import datetime # Certifique-se que 'datetime' está importado no topo do seu script

def execute_dynamic_plan(
    query: str,
    plan: dict,
    artifacts: dict,
    model,  # SentenceTransformer
    cross_encoder_model,  # CrossEncoder
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    search_by_tags: callable,
    expand_search_terms: callable,
    prioritize_recency: bool = True,
) -> tuple[str, list[dict]]:
    """
    Versão Completa de execute_dynamic_plan
    """

    import re
    import random
    from collections import defaultdict
    from datetime import datetime
    import faiss


    # -------------- HELPERS --------------
    def _is_company_match(plan_canonical_name: str, metadata_name: str) -> bool:
        if not plan_canonical_name or not metadata_name:
            return False
        return plan_canonical_name.lower() in metadata_name.lower()

    candidate_chunks_dict = {}

    def add_candidate(chunk):
        """Add chunk de forma única por sua origem e id/texto."""
        key = chunk.get('source_url', '') + str(chunk.get('chunk_id', hash(chunk.get('text', ''))))
        if key not in candidate_chunks_dict:
            candidate_chunks_dict[key] = chunk

    # -------------- LOG INICIAL --------------
    logger.info(f"Executando plano dinâmico para query: '{query}'")
    plan_type = plan.get("plan_type", "default")
    empresas = plan.get("empresas", [])
    topicos = plan.get("topicos", [])

    # -------------- CARREGAMENTO E NORMALIZAÇÃO DOS CHUNKS --------------
    all_chunks = [
        chunk_meta
        for artifact_data in artifacts.values()
        for chunk_meta in artifact_data.get('chunks', [])
    ]
    for chunk in all_chunks:
        if 'chunk_text' in chunk and 'text' not in chunk:
            chunk['text'] = chunk.pop('chunk_text')
        if 'doc_type' not in chunk:
            if 'frmExibirArquivoFRE' in chunk.get('source_url', ''):
                chunk['doc_type'] = 'item_8_4'
            else:
                chunk['doc_type'] = 'outros_documentos'
        # Prévias para busca rápida nos tópicos (pode expandir conforme necessidade)
        if "topics_in_chunk" not in chunk:
            chunk["topics_in_chunk"] = []

    # -------------- FILTROS ----------
    filtros = plan.get("filtros", {})

    pre_filtered_chunks = all_chunks
    if filtros.get('setor'):
        pre_filtered_chunks = [
            c for c in pre_filtered_chunks
            if c.get('setor', '').lower() == filtros['setor'].lower()
        ]
    if filtros.get('controle_acionario'):
        pre_filtered_chunks = [
            c for c in pre_filtered_chunks
            if c.get('controle_acionario', '').lower() == filtros['controle_acionario'].lower()
        ]
    logger.info(f"Após pré-filtragem, {len(pre_filtered_chunks)} chunks são candidatos.")

    # -------------- BUSCA POR TAGS E EXPANSÃO DE TERMOS ----------------
    logger.info("Executando busca por tags...")
    tags = search_by_tags(query, kb)
    logger.info(f"Tags encontradas: {tags}")

    # Expansão dos termos de busca para potencializar recuperação semântica
    if tags:
        expanded_terms = {query.lower()}
        for tag in tags:
            expanded_terms.update(expand_search_terms(tag, kb))
        query_to_search = " ".join(list(expanded_terms))
        logger.info(f"Query expandida: {query_to_search}")
    else:
        logger.info("Nenhuma tag relevante encontrada. Usando query original.")
        query_to_search = query

    # -------------- ROTEAMENTO PRINCIPAL --------------
    if plan_type == "section_8_4" and empresas:
        canonical_name_from_plan = empresas[0]
        search_name = next(
            (
                e.get("search_alias", canonical_name_from_plan)
                for e in company_catalog_rich
                if e.get("canonical_name") == canonical_name_from_plan
            ),
            canonical_name_from_plan,
        )
        logger.info(f"ROTA ESPECIAL section_8_4: Usando nome de busca '{search_name}'.")
        chunks_to_search = [
            c for c in pre_filtered_chunks
            if c.get('doc_type') == 'item_8_4' and _is_company_match(canonical_name_from_plan, c.get('company_name', ''))
        ]
        if chunks_to_search:
            temp_embeddings = model.encode(
                [c.get('text', '') for c in chunks_to_search],
                normalize_embeddings=True
            ).astype('float32')
            temp_index = faiss.IndexFlatIP(temp_embeddings.shape[1])
            temp_index.add(temp_embeddings)

            all_search_queries = []
            for topico in topicos:
                for term in expand_search_terms(topico, kb)[:3]:
                    all_search_queries.append(f"explicação detalhada sobre o conceito e funcionamento de {term}")

            if not all_search_queries:
                return "Não encontrei informações relevantes para esta combinação.", []

            logger.info(f"Codificando {len(all_search_queries)} variações de busca...")
            k_per_query = max(1, TOP_K_FINAL // len(all_search_queries))
            query_embeddings = model.encode(
                all_search_queries,
                normalize_embeddings=True
            ).astype('float32')
            _, all_indices = temp_index.search(query_embeddings, k_per_query)

            for indices_row in all_indices:
                for idx in indices_row:
                    if idx != -1:
                        add_candidate(chunks_to_search[idx])

    else:
        if not empresas and topicos:
            # Busca conceitual em todos os docs, amostra randomizada p/ eficiência
            logger.info(f"ROTA Default (Geral): busca conceitual para tópicos: {topicos}")
            sample_size = 100
            chunks_to_search = random.sample(
                pre_filtered_chunks,
                min(sample_size, len(pre_filtered_chunks))
            )
            if chunks_to_search:
                temp_embeddings = model.encode(
                    [c['text'] for c in chunks_to_search],
                    normalize_embeddings=True
                ).astype('float32')
                temp_index = faiss.IndexFlatIP(temp_embeddings.shape[1])
                temp_index.add(temp_embeddings)
                for topico in topicos:
                    for term in expand_search_terms(topico, kb)[:3]:
                        search_query = f"explicação detalhada sobre o conceito e funcionamento de {term}"
                        query_embedding = model.encode(
                            [search_query],
                            normalize_embeddings=True
                        ).astype('float32')
                        _, indices = temp_index.search(query_embedding, TOP_K_FINAL)
                        for idx in indices[0]:
                            if idx != -1:
                                add_candidate(chunks_to_search[idx])

        elif empresas and topicos:
            # Busca híbrida empresa+tópico: selecione docs por empresa e combine busca por tag/semântica
            logger.info(f"ROTA HÍBRIDA: Empresas: {empresas}, Tópicos: {topicos}")
            target_topic_paths = plan.get("topicos", [])

            for empresa_canonica in empresas:
                chunks_for_company = [
                    c for c in pre_filtered_chunks
                    if _is_company_match(empresa_canonica, c.get('company_name', ''))
                ]
                if not chunks_for_company:
                    continue

                # Deduplicação e recorte por data (recency)
                docs_by_url = defaultdict(list)
                for chunk in chunks_for_company:
                    docs_by_url[chunk.get('source_url')].append(chunk)
                MAX_DOCS_PER_COMPANY = 3
                if len(docs_by_url) > MAX_DOCS_PER_COMPANY:
                    sorted_urls = sorted(
                        docs_by_url.keys(),
                        key=lambda url: docs_by_url[url][0].get('document_date', '0000-00-00'),
                        reverse=True
                    )
    
                    latest_urls = sorted_urls[:MAX_DOCS_PER_COMPANY]
                    chunks_for_company = [chunk for url in latest_urls for chunk in docs_by_url[url]]
                    logger.info(f"Para '{empresa_canonica}', selecionando os {MAX_DOCS_PER_COMPANY} documentos mais recentes pela DATA REAL.")

                # Etapa 1: Busca por tags (precisão)
                logger.info(f"[{empresa_canonica}] Etapa 1: Busca por tags nos metadados...")
                for chunk in chunks_for_company:
                    if any(
                        target_path in path
                        for path in chunk.get("topics_in_chunk", [])
                        for target_path in target_topic_paths
                    ):
                        add_candidate(chunk)

                # Etapa 2: Busca vetorial semântica
                logger.info(f"[{empresa_canonica}] Etapa 2: Busca por similaridade semântica...")
                if chunks_for_company:
                    temp_embeddings = model.encode(
                        [c.get('text', '') for c in chunks_for_company],
                        normalize_embeddings=True
                    ).astype('float32')
                    temp_index = faiss.IndexFlatIP(temp_embeddings.shape[1])
                    temp_index.add(temp_embeddings)
                    search_name = next(
                        (
                            e.get("search_alias", empresa_canonica)
                            for e in company_catalog_rich
                            if e.get("canonical_name") == empresa_canonica
                        ),
                        empresa_canonica
                    )
                    search_query = (f"informações detalhadas sobre "
                                    f"{' e '.join(topicos)} no plano da empresa {search_name}")
                    query_embedding = model.encode(
                        [search_query], normalize_embeddings=True
                    ).astype('float32')
                    _, indices = temp_index.search(
                        query_embedding,
                        min(TOP_K_INITIAL_RETRIEVAL, len(chunks_for_company))
                    )
                    for idx in indices[0]:
                        if idx != -1:
                            add_candidate(chunks_for_company[idx])

    # -------------------- RE-RANKING FINAL ----------------------------
    if not candidate_chunks_dict:
        logger.warning(
            f"Nenhum chunk candidato encontrado para a query: '{query}' com os filtros aplicados."
        )
        return "Não encontrei informações relevantes para esta combinação específica de consulta e filtros.", []

    candidate_list = list(candidate_chunks_dict.values())
    if prioritize_recency:
        logger.info("Re-ranking adicional por recência ativado.")
        candidate_list = rerank_by_recency(candidate_list, datetime.now())

    reranked_chunks = rerank_with_cross_encoder(
        query, candidate_list, cross_encoder_model, top_n=TOP_K_FINAL
    )

    # -------------- CONSTRUÇÃO DO CONTEXTO FINAL PARA RETORNO ---------------
    full_context = ""
    retrieved_sources = []
    seen_sources = set()
    for chunk in reranked_chunks:
        company_name = chunk.get('company_name', 'N/A')
        source_url = chunk.get('source_url', 'N/A')
        source_header = (
            f"(Empresa: {company_name}, Setor: {chunk.get('setor', 'N/A')}, "
            f"Documento: {chunk.get('doc_type', 'N/A')})"
        )
        clean_text = re.sub(r'\[.*?\]', '', chunk.get('text', '')).strip()
        full_context += (
            f"--- CONTEÚDO RELEVANTE {source_header} ---\n{clean_text}\n\n"
        )
        source_tuple = (company_name, source_url)
        if source_tuple not in seen_sources:
            seen_sources.add(source_tuple)
            retrieved_sources.append(chunk)

    logger.info(
        f"Contexto final construído a partir de {len(reranked_chunks)} chunks re-ranqueados."
    )
    return full_context, retrieved_sources

    
def create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data, filters: dict):
    """
    Versão 3.0 (Unificada) do planejador dinâmico.

    Esta versão combina o melhor de ambas as propostas:
    1.  EXTRAI filtros de metadados (setor, controle acionário).
    2.  EXTRAI tópicos hierárquicos completos.
    3.  RESTAURA a detecção de intenção de "Resumo Geral" para perguntas abertas.
    4.  MANTÉM a detecção da intenção especial "Item 8.4".
    """
    logger.info(f"Gerando plano dinâmico v3.0 para a pergunta: '{query}'")
    query_lower = query.lower().strip()
    
    plan = {
        "empresas": [],
        "topicos": [],
        "filtros": filters.copy(),
        "plan_type": "default" # O tipo de plano default aciona a busca RAG padrão.
    }



    # --- PASSO 2: Identificação Robusta de Empresas (Lógica Original Mantida) ---
    mentioned_companies = []
    if company_catalog_rich:
        companies_found_by_alias = {}
        for company_data in company_catalog_rich:
            canonical_name = company_data.get("canonical_name")
            if not canonical_name: continue
            
            all_aliases = company_data.get("aliases", []) + [canonical_name]
            for alias in all_aliases:
                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                    score = len(alias.split())
                    if canonical_name not in companies_found_by_alias or score > companies_found_by_alias[canonical_name]:
                        companies_found_by_alias[canonical_name] = score
        if companies_found_by_alias:
            mentioned_companies = [c for c, s in sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)]

    if not mentioned_companies:
        for empresa_nome in summary_data.keys():
            if re.search(r'\b' + re.escape(empresa_nome.lower()) + r'\b', query_lower):
                mentioned_companies.append(empresa_nome)
    
    plan["empresas"] = mentioned_companies
    logger.info(f"Empresas identificadas: {plan['empresas']}")

    # --- PASSO 3: Detecção de Intenções Especiais (LÓGICA UNIFICADA) ---
    # Palavras-chave para as intenções especiais
    summary_keywords = ['resumo geral', 'plano completo', 'como funciona o plano', 'descreva o plano', 'resumo do plano', 'detalhes do plano']
    section_8_4_keywords = ['item 8.4', 'seção 8.4', '8.4 do fre']
    
    is_summary_request = any(keyword in query_lower for keyword in summary_keywords)
    is_section_8_4_request = any(keyword in query_lower for keyword in section_8_4_keywords)

    if plan["empresas"] and is_section_8_4_request:
        plan["plan_type"] = "section_8_4"
        # O tópico é o caminho hierárquico para a seção inteira
        plan["topicos"] = ["FormularioReferencia,Item_8_4"]
        logger.info("Plano especial 'section_8_4' detectado.")
        return {"status": "success", "plan": plan}
    
    # [LÓGICA RESTAURADA E ADAPTADA]
    # Se for uma pergunta de resumo para uma empresa, define um conjunto de tópicos essenciais.
    elif plan["empresas"] and is_summary_request:
        plan["plan_type"] = "summary" # Um tipo especial para indicar um resumo completo
        logger.info("Plano especial 'summary' detectado. Montando plano com tópicos essenciais.")
        # Define os CAMINHOS HIERÁRQUICOS essenciais para um bom resumo.
        plan["topicos"] = [
            "TiposDePlano",
            "ParticipantesCondicoes,Elegibilidade",
            "Mecanicas,Vesting",
            "Mecanicas,Lockup",
            "IndicadoresPerformance",
            "GovernancaRisco,MalusClawback",
            "EventosFinanceiros,DividendosProventos"
        ]
        return {"status": "success", "plan": plan}

    # --- PASSO 4: Extração de Tópicos Hierárquicos (Se Nenhuma Intenção Especial Foi Ativada) ---
    alias_map = create_hierarchical_alias_map(kb)
    found_topics = set()
    
    # Ordena os aliases por comprimento para encontrar o mais específico primeiro
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        # Usamos uma regex mais estrita para evitar matches parciais (ex: 'TSR' em 'TSR Relativo')
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            found_topics.add(alias_map[alias])
    
    plan["topicos"] = sorted(list(found_topics))
    if plan["topicos"]:
        logger.info(f"Caminhos de tópicos identificados: {plan['topicos']}")
    if plan["empresas"] and not plan["topicos"]:
        logger.info("Nenhum tópico específico encontrado. Ativando modo de resumo/comparação geral.")
        plan["plan_type"] = "summary"
        # Define os CAMINHOS HIERÁRQUICOS essenciais para um bom resumo/comparação.
        plan["topicos"] = [
            "TiposDePlano",
            "ParticipantesCondicoes,Elegibilidade",
            "MecanicasCicloDeVida,Vesting",
            "MecanicasCicloDeVida,Lockup",
            "IndicadoresPerformance",
            "GovernancaRisco,MalusClawback",
            "EventosFinanceiros,DividendosProventos"
        ]
        logger.info(f"Tópicos de resumo geral adicionados ao plano: {plan['topicos']}")    

    # --- PASSO 5: Validação Final ---
    if not plan["empresas"] and not plan["topicos"] and not plan["filtros"]:
        logger.warning("Planejador não conseguiu identificar empresa, tópico ou filtro na pergunta.")
        return {"status": "error", "message": "Não foi possível identificar uma intenção clara na sua pergunta. Tente ser mais específico."}
        
    return {"status": "success", "plan": plan}

    
def analyze_single_company(
    empresa: str,
    plan: dict,
    query: str,
    artifacts: dict,
    model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    execute_dynamic_plan_func: callable,
    get_final_unified_answer_func: callable
) -> dict:
    """
    Executa o plano de análise para uma única empresa e retorna um dicionário estruturado.
    Esta função é projetada para ser executada em um processo paralelo.
    """
    single_plan = {'empresas': [empresa], 'topicos': plan['topicos']}
    
    # --- CORREÇÃO APLICADA AQUI ---
    # Adicionado o argumento 'is_summary_plan=False' na chamada.
    context, sources_list = execute_dynamic_plan_func(query, single_plan, artifacts, model, cross_encoder_model, kb, company_catalog_rich,
        company_lookup_map, search_by_tags, expand_search_terms)
    
    result_data = {
        "empresa": empresa,
        "resumos_por_topico": {topico: "Informação não encontrada" for topico in plan['topicos']},
        "sources": sources_list
    }

    if context:
        summary_prompt = f"""
        Com base no CONTEXTO abaixo sobre a empresa {empresa}, crie um resumo para cada um dos TÓPICOS solicitados.
        Sua resposta deve ser APENAS um objeto JSON válido, sem nenhum texto adicional antes ou depois.
        
        TÓPICOS PARA RESUMIR: {json.dumps(plan['topicos'])}
        
        CONTEXTO:
        {context}
        
        FORMATO OBRIGATÓRIO DA RESPOSTA (APENAS JSON):
        {{
          "resumos_por_topico": {{
            "Tópico 1": "Resumo conciso sobre o Tópico 1...",
            "Tópico 2": "Resumo conciso sobre o Tópico 2...",
            "...": "..."
          }}
        }}
        """
        
        try:
            json_response_str = get_final_unified_answer_func(summary_prompt, context)
            json_match = re.search(r'\{.*\}', json_response_str, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group())
                result_data["resumos_por_topico"] = parsed_json.get("resumos_por_topico", result_data["resumos_por_topico"])
            else:
                logger.warning(f"Não foi possível extrair JSON da resposta para a empresa {empresa}.")

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Erro ao processar o resumo JSON para {empresa}: {e}")
            
    return result_data


def handle_rag_query(
    query: str,
    artifacts: dict,
    embedding_model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    summary_data: dict,
    filters: dict,
    prioritize_recency: bool = False
) -> tuple[str, list[dict]]:
    """
    Orquestra o pipeline de RAG para perguntas qualitativas, incluindo a geração do plano,
    a execução da busca (com re-ranking) e a síntese da resposta final.
    """
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data, filters)
        
        if plan_response['status'] != "success":
            status.update(label="⚠️ Falha na identificação", state="error", expanded=True)
            
            st.warning("Não consegui identificar uma empresa conhecida na sua pergunta para realizar uma análise profunda.")
            st.info("Para análises detalhadas, por favor, use o nome de uma das empresas listadas na barra lateral.")
            
            with st.spinner("Estou pensando em uma pergunta alternativa que eu possa responder..."):
                alternative_query = suggest_alternative_query(query, kb) # Passe o kb
            
            st.markdown("#### Que tal tentar uma pergunta mais geral?")
            st.markdown("Você pode copiar a sugestão abaixo ou reformular sua pergunta original.")
            st.code(alternative_query, language=None)
            
            # Retornamos uma string vazia para o texto e para as fontes, encerrando a análise de forma limpa.
            return "", []
        # --- FIM DO NOVO BLOCO ---
            
        plan = plan_response['plan']
        
        summary_keywords = ['resumo', 'geral', 'completo', 'visão geral', 'como funciona o plano', 'detalhes do plano']
        is_summary_request = any(keyword in query.lower() for keyword in summary_keywords)
        
        specific_topics_in_query = list({canonical for alias, canonical in _create_flat_alias_map(kb).items() if re.search(r'\b' + re.escape(alias) + r'\b', query.lower())})
        is_summary_plan = is_summary_request and not specific_topics_in_query
        
        if plan['empresas']:
            st.write(f"**🏢 Empresas identificadas:** {', '.join(plan['empresas'])}")
        else:
            st.write("**🏢 Nenhuma empresa específica identificada. Realizando busca geral.**")
            
        st.write(f"**📝 Tópicos a analisar:** {', '.join(plan['topicos'])}")
        if is_summary_plan:
            st.info("💡 Modo de resumo geral ativado. A busca será otimizada para os tópicos encontrados.")
            
        status.update(label="✅ Plano gerado com sucesso!", state="complete")

    final_answer, all_sources_structured = "", []
    seen_sources_tuples = set()

    # --- Lógica para Múltiplas Empresas (Comparação) ---
    if len(plan.get('empresas', [])) > 1:
        st.info(f"Modo de comparação ativado para {len(plan['empresas'])} empresas. Executando análises em paralelo...")
        
        with st.spinner(f"Analisando {len(plan['empresas'])} empresas..."):
            with ThreadPoolExecutor(max_workers=len(plan['empresas'])) as executor:
                futures = [
                    executor.submit(
                        analyze_single_company, empresa, plan, query, artifacts, embedding_model, cross_encoder_model, 
                        kb, company_catalog_rich, company_lookup_map, execute_dynamic_plan, get_final_unified_answer) 
                    for empresa in plan['empresas']
                ]
                results = [future.result() for future in futures]

        for result in results:
            for src_dict in result.get('sources', []):
                company_name = src_dict.get('company_name')
                source_url = src_dict.get('source_url')
                
                if company_name and source_url:
                    src_tuple = (company_name, source_url)
                    if src_tuple not in seen_sources_tuples:
                        seen_sources_tuples.add(src_tuple)
                        all_sources_structured.append(src_dict)

        with st.status("Gerando relatório comparativo final...", expanded=True) as status:
            clean_results = []
            for company_result in results:
                # Remove a chave 'sources' temporariamente para limpeza
                sources = company_result.pop("sources", [])
                clean_sources = []
                for source_chunk in sources:
                    # Remove a chave 'relevance_score' de cada chunk
                    source_chunk.pop('relevance_score', None)
                    clean_sources.append(source_chunk)
                
                # Adiciona as fontes limpas de volta
                company_result["sources"] = clean_sources
                clean_results.append(company_result)
            structured_context = json.dumps(results, indent=2, ensure_ascii=False)
            comparison_prompt = f"""
            Sua tarefa é criar um relatório comparativo detalhado sobre "{query}".
            Use os dados estruturados fornecidos no CONTEXTO JSON abaixo.
            O relatório deve começar com uma breve análise textual e, em seguida, apresentar uma TABELA MARKDOWN clara e bem formatada.

            CONTEXTO (em formato JSON):
            {structured_context}
            """
            final_answer = get_final_unified_answer(comparison_prompt, structured_context)
            status.update(label="✅ Relatório comparativo gerado!", state="complete")
            
    # --- Lógica para Empresa Única ou Busca Geral ---
    else:
        with st.status("2️⃣ Recuperando e re-ranqueando contexto...", expanded=True) as status:
            context, all_sources_structured = execute_dynamic_plan(
                query, plan, artifacts, embedding_model, cross_encoder_model, kb,company_catalog_rich, company_lookup_map, search_by_tags, expand_search_terms)
            
            if not context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", []
                
            st.write(f"**📄 Contexto recuperado de:** {len(all_sources_structured)} documento(s)")
            status.update(label="✅ Contexto relevante selecionado!", state="complete")
        
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            final_answer = get_final_unified_answer(query, context)
            status.update(label="✅ Análise concluída!", state="complete")

    return final_answer, all_sources_structured

def main():
    st.title("🤖 Agente de Análise de Planos de Incentivo (ILP)")
    st.markdown("---")

    # 2. Carregue os dados (a função agora só retorna 4 valores)
    artifacts, summary_data, setores_disponiveis, controles_disponiveis, embedding_model, cross_encoder_model = setup_and_load_data()
        
    if not summary_data or not artifacts:
        st.error("❌ Falha crítica no carregamento dos dados. O app não pode continuar.")
        st.stop()
    
    engine = AnalyticalEngine(summary_data, DICIONARIO_UNIFICADO_HIERARQUICO) 
    
    try:
        from catalog_data import company_catalog_rich 
    except ImportError:
        company_catalog_rich = [] 
    
    st.session_state.company_catalog_rich = company_catalog_rich

   
    from tools import _create_company_lookup_map
    st.session_state.company_lookup_map = _create_company_lookup_map(company_catalog_rich)


    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Categorias de Documentos (RAG)", len(artifacts))
        st.markdown("---")

        # Adicione o checkbox para re-ranking por recência
        prioritize_recency = st.checkbox(
            "Priorizar documentos mais recentes",
            value=True, # Deixe ativado por padrão
            help="Dá um bônus de relevância para os documentos mais novos.")
        st.metric("Empresas no Resumo", len(summary_data))
                # --- MODIFICAÇÃO 2: Usar as listas dinâmicas ---
        st.header("⚙️ Filtros da Análise")
        st.caption("Filtre a base de dados antes de fazer sua pergunta.")
        
        selected_setor = st.selectbox(
            label="Filtrar por Setor",
            options=setores_disponiveis, # Usa a lista dinâmica
            index=0
        )
        
        selected_controle = st.selectbox(
            label="Filtrar por Controle Acionário",
            options=controles_disponiveis, # Usa a lista dinâmica
            index=0
        )
        st.markdown("---") 
        with st.expander("Empresas com dados no resumo"):
            st.dataframe(pd.DataFrame(sorted(list(summary_data.keys())), columns=["Empresa"]), use_container_width=True, hide_index=True)
        st.success("✅ Sistema pronto para análise")
        st.info(f"Embedding Model: `{MODEL_NAME}`")
        st.info(f"Generative Model: `{GEMINI_MODEL}`")
    
    st.header("💬 Faça sua pergunta")
    
    # Em app.py, localize o bloco `with st.expander(...)` e substitua seu conteúdo por este:

    with st.expander("ℹ️ **Guia do Usuário: Como Extrair o Máximo do Agente**", expanded=False): # `expanded=False` é uma boa prática para não poluir a tela inicial
        st.markdown("""
        Este agente foi projetado para atuar como um consultor especialista em Planos de Incentivo de Longo Prazo (ILP), analisando uma base de dados de documentos públicos da CVM. Para obter os melhores resultados, formule perguntas que explorem suas principais capacidades.
        """)

        st.subheader("1. Perguntas de Listagem (Quem tem?) 🎯")
        st.info("""
        Use estas perguntas para identificar e listar empresas que adotam uma prática específica. Ideal para mapeamento de mercado.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Liste as empresas que pagam dividendos ou JCP durante o período de carência (vesting).
        - Quais companhias possuem cláusulas de Malus ou Clawback?
        - Gere uma lista de empresas que oferecem planos com contrapartida do empregador (Matching/Coinvestimento).
        - Quais organizações mencionam explicitamente o Comitê de Remuneração como órgão aprovador dos planos?""")

        st.subheader("2. Análise Estatística (Qual a média?) 📈")
        st.info("""
        Pergunte por médias, medianas e outros dados estatísticos para entender os números por trás das práticas de mercado e fazer benchmarks.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Qual o período médio de vesting (em anos) entre as empresas analisadas?
        - Qual a diluição máxima média (% do capital social) que os planos costumam aprovar?
        - Apresente as estatísticas do desconto no preço de exercício (mínimo, média, máximo).
        - Qual o prazo de lock-up (restrição de venda) mais comum após o vesting das ações?""")

        st.subheader("3. Padrões de Mercado (Como é o normal?) 🗺️")
        st.info("""
        Faça perguntas abertas para que o agente analise diversos planos e descreva os padrões e as abordagens mais comuns para um determinado tópico.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Analise os modelos típicos de planos de Ações Restritas (RSU), o tipo mais comum no mercado.
        - Além do TSR, quais são as metas de performance (ESG, Financeiras) mais utilizadas pelas empresas?
        - Descreva os padrões de tratamento para condições de saída (Good Leaver vs. Bad Leaver) nos planos.
        - Quais as abordagens mais comuns para o tratamento de dividendos em ações ainda não investidas?""")

        st.subheader("4. Análise Profunda e Comparativa (Me explique em detalhes) 🧠")
        st.info("""
        Use o poder do RAG para pedir análises detalhadas sobre uma ou mais empresas, comparando regras e estruturas específicas.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Como o plano da Vale trata a aceleração de vesting em caso de mudança de controle?
        - Compare as cláusulas de Malus/Clawback da Vale com as do Itaú.
        - Descreva em detalhes o plano de Opções de Compra da Localiza, incluindo prazos, condições e forma de liquidação.
        - Descreva o Item 8.4 da M.dias Braco.
        - Quais as diferenças na elegibilidade de participantes entre os planos da Magazine Luiza e da Lojas Renner?""")


        st.subheader("❗ Conhecendo as Limitações")
        st.warning("""
        - **Fonte dos Dados:** Minha análise se baseia em documentos públicos da CVM com data de corte 31/07/2025. Não tenho acesso a informações em tempo real ou privadas.
        - **Identificação de Nomes:** Para análises profundas, preciso que o nome da empresa seja claro e reconhecível. Se o nome for ambíguo ou não estiver na minha base, posso não encontrar os detalhes.
        - **Escopo:** Sou altamente especializado em Incentivos de Longo Prazo. Perguntas fora deste domínio podem não ter respostas adequadas.
        """)

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Quais são os modelos típicos de vesting? ou Como funciona o plano da Vale?")
    
    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            st.stop()
        active_filters = {}
        if selected_setor != "Todos":
            active_filters['setor'] = selected_setor.lower()
        if selected_controle != "Todos":
            # A chave 'controle_acionario' deve ser exatamente como nos metadados dos chunks.
            active_filters['controle_acionario'] = selected_controle.lower()
        if active_filters:
            # Formata o dicionário para uma exibição amigável.
            filter_text_parts = []
            if 'setor' in active_filters:
                filter_text_parts.append(f"**Setor**: {active_filters['setor'].capitalize()}")
            if 'controle_acionario' in active_filters:
                filter_text_parts.append(f"**Controle**: {active_filters['controle_acionario'].capitalize()}")

            filter_text = " e ".join(filter_text_parts)
            st.info(f"🔎 Análise sendo executada com os seguintes filtros: {filter_text}")

        st.markdown("---")
        st.subheader("📋 Resultado da Análise")
                # --- INÍCIO DA NOVA LÓGICA DE ROTEAMENTO HÍBRIDO ---
        
        intent = None
        query_lower = user_query.lower()
        
        # 1. Camada de Regras: Verifica palavras-chave quantitativas óbvias primeiro.
        quantitative_keywords = [
            'liste', 'quais empresas', 'quais companhias', 'quantas', 'média', 
            'mediana', 'estatísticas', 'mais comuns', 'prevalência', 'contagem'
        ]
        
        if any(keyword in query_lower for keyword in quantitative_keywords):
            intent = "quantitativa"
            logger.info("Intenção 'quantitativa' detectada por regras de palavras-chave.")
        
        # 2. Camada de LLM: Se nenhuma regra correspondeu, consulta o LLM.
        if intent is None:
            with st.spinner("Analisando a intenção da sua pergunta..."):
                intent = get_query_intent_with_llm(user_query)

        # --- FIM DA NOVA LÓGICA DE ROTEAMENTO HÍBRIDO ---

        if intent == "quantitativa":
            query_lower = user_query.lower()
            listing_keywords = ["quais empresas", "liste as empresas", "quais companhias"]
            thematic_keywords = ["modelos típicos", "padrões comuns", "analise os planos", "formas mais comuns"]
                # --- INÍCIO DA LÓGICA CORRIGIDA E FINAL ---
            
            # 1. Usa a nova função para criar o mapa hierárquico
            alias_map = create_hierarchical_alias_map(DICIONARIO_UNIFICADO_HIERARQUICO)
            found_topics = set()
            
            # 2. Itera nos aliases para encontrar os tópicos mencionados na query
            for alias in sorted(alias_map.keys(), key=len, reverse=True):
                if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
                    full_path = alias_map[alias]
                    topic_leaf = full_path.split(',')[-1].replace('_', ' ')
                    found_topics.add(topic_leaf)
            
            topics_to_search = list(found_topics)
            # Remove palavras-chave genéricas da lista de tópicos
            topics_to_search = [t for t in topics_to_search if t.lower() not in listing_keywords and t.lower() not in thematic_keywords]

            # --- FIM DA LÓGICA CORRIGIDA E FINAL ---

            # Rota 1: Análise Temática
            if any(keyword in query_lower for keyword in thematic_keywords) and topics_to_search:
                primary_topic = topics_to_search[0]
                with st.spinner(f"Iniciando análise temática... Este processo é detalhado e pode levar alguns minutos."):
                    st.write(f"**Tópico identificado para análise temática:** `{topics_to_search}`")
                    final_report = analyze_topic_thematically(
                        topic=topics_to_search, query=user_query, artifacts=artifacts, model=embedding_model, kb=DICIONARIO_UNIFICADO_HIERARQUICO,
                        execute_dynamic_plan_func=execute_dynamic_plan, get_final_unified_answer_func=get_final_unified_answer,filters=active_filters,
                        company_catalog_rich=st.session_state.company_catalog_rich, company_lookup_map=st.session_state.company_lookup_map,
                    )
                    st.markdown(final_report)

            # Rota 2: Listagem de Empresas
            elif any(keyword in query_lower for keyword in listing_keywords) and topics_to_search:
                with st.spinner(f"Usando ferramentas para encontrar empresas..."):
                    st.write(f"**Tópicos identificados para busca:** `{', '.join(topics_to_search)}`")
        
                    all_found_companies = set()
        
                    # CORREÇÃO: Itera sobre a lista e chama a ferramenta para CADA tópico.
                    for topic_item in topics_to_search:
                        companies = find_companies_by_topic(
                            topic=topic_item,  # Passa um único tópico (string)
                            artifacts=artifacts, 
                            model=embedding_model, 
                            kb=DICIONARIO_UNIFICADO_HIERARQUICO,
                            filters=active_filters
                        )
                        all_found_companies.update(companies)

                    if all_found_companies:
                        sorted_companies = sorted(list(all_found_companies))
                        final_answer = f"#### Foram encontradas {len(sorted_companies)} empresas para os tópicos relacionados:\n"
                        final_answer += "\n".join([f"- {company}" for company in sorted_companies])
                    else:
                        final_answer = "Nenhuma empresa encontrada nos documentos para os tópicos identificados."
                
                st.markdown(final_answer)
                     # --- INÍCIO DA NOVA ROTA 2.5 ---
            # Rota 2.5: Listagem de Empresas APENAS POR FILTRO
            elif any(keyword in query_lower for keyword in listing_keywords) and active_filters and not topics_to_search:
                with st.spinner("Listando empresas com base nos filtros selecionados..."):
                    st.write("Nenhum tópico técnico identificado. Listando todas as empresas que correspondem aos filtros.")
                    
                    companies_from_filter = set()
                    # Itera em todos os documentos para encontrar empresas que correspondem ao filtro
                    for artifact_data in artifacts.values():
                        
                        list_of_chunks = artifact_data.get('chunks', [])
                        for metadata in list_of_chunks:
                            # --- INÍCIO DA CORREÇÃO ---
                            setor_metadata = metadata.get('setor', '')
                            controle_metadata = metadata.get('controle_acionario', '')

                            setor_match = (not active_filters.get('setor') or 
                                           (isinstance(setor_metadata, str) and setor_metadata.lower() == active_filters['setor']))
                        
                            controle_match = (not active_filters.get('controle_acionario') or 
                                              (isinstance(controle_metadata, str) and controle_metadata.lower() == active_filters['controle_acionario']))
                            # --- FIM DA CORREÇÃO ---
                            if setor_match and controle_match:
                                company_name = metadata.get('company_name')
                                if company_name:
                                    companies_from_filter.add(company_name)
                    
                    if companies_from_filter:
                        sorted_companies = sorted(list(companies_from_filter))
                        final_answer = f"#### Foram encontradas {len(sorted_companies)} empresas para os filtros selecionados:\n"
                        final_answer += "\n".join([f"- {company}" for company in sorted_companies])
                    else:
                        final_answer = "Nenhuma empresa foi encontrada para a combinação de filtros selecionada."
                    
                    st.markdown(final_answer)

            # Rota 3: Fallback para o AnalyticalEngine
            else:
                st.info("Intenção quantitativa detectada. Usando o motor de análise rápida...")
                with st.spinner("Executando análise quantitativa rápida..."):
                    report_text, data_result = engine.answer_query(user_query, filters=active_filters)
                    if report_text: st.markdown(report_text)
                    if data_result is not None:
                        if isinstance(data_result, pd.DataFrame):
                            if not data_result.empty: st.dataframe(data_result, use_container_width=True, hide_index=True)
                        elif isinstance(data_result, dict):
                            for df_name, df_content in data_result.items():
                                if df_content is not None and not df_content.empty:
                                    st.markdown(f"#### {df_name}")
                                    st.dataframe(df_content, use_container_width=True, hide_index=True)
                    else: 
                        st.info("Nenhuma análise tabular foi gerada para a sua pergunta ou dados insuficientes.")
        
        else: # intent == 'qualitativa'
            final_answer, sources = handle_rag_query(
                user_query, 
                artifacts, 
                embedding_model, 
                cross_encoder_model, 
                kb=DICIONARIO_UNIFICADO_HIERARQUICO,
                company_catalog_rich=st.session_state.company_catalog_rich, 
                company_lookup_map=st.session_state.company_lookup_map, 
                summary_data=summary_data,
                filters=active_filters,
                prioritize_recency=prioritize_recency
            )
            st.markdown(final_answer)
            
            if sources:
                with st.expander(f"📚 Documentos consultados ({len(sources)})", expanded=True):
                    st.caption("Nota: Links diretos para a CVM podem falhar. Use a busca no portal com o protocolo como plano B.")
        
        # --- BLOCO CORRIGIDO ---
                    for src in sorted(sources, key=lambda x: x.get('company_name', '')):
                        company_name = src.get('company_name', 'N/A')
                        # Recupere a data do documento dos metadados
                        doc_date = src.get('document_date', 'N/A')
                        doc_type_raw = src.get('doc_type', '')
                        url = src.get('source_url', '')

                        if doc_type_raw == 'outros_documentos':
                            display_doc_type = 'Plano de Remuneração'
                        else:
                            display_doc_type = doc_type_raw.replace('_', ' ')
    
                        # Adicione a data do documento ao texto de exibição
                        display_text = f"{company_name} - {display_doc_type} - (Data: **{doc_date}**)"
            
                       
            
                        # A lógica de exibição agora está corretamente separada por tipo de documento
                        if "frmExibirArquivoIPEExterno" in url:
                            # O protocolo SÓ é definido e usado dentro deste bloco
                            protocolo_match = re.search(r'NumeroProtocoloEntrega=(\d+)', url)
                            protocolo = protocolo_match.group(1) if protocolo_match else "N/A"
                            st.markdown(f"**{display_text}** (Protocolo: **{protocolo}**)")
                            st.markdown(f"↳ [Link Direto para Plano de ILP]({url}) ", unsafe_allow_html=True)
            
                        elif "frmExibirArquivoFRE" in url:
                            # Este bloco não usa a variável 'protocolo'
                            st.markdown(f"**{display_text}**")
                            st.markdown(f"↳ [Link Direto para Formulário de Referência]({url})", unsafe_allow_html=True)
            
                        else:
                            # Este bloco também não usa a variável 'protocolo'
                            st.markdown(f"**{display_text}**: [Link]({url})")


if __name__ == "__main__":
    main()
