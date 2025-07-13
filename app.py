# -*- coding: utf-8 -*-
"""
AGENTE DE ANÁLISE LTIP - VERSÃO STREAMLIT (HÍBRIDO)
Aplicação web para análise de planos de incentivo de longo prazo, com
capacidades de busca profunda (RAG) e análise agregada (resumo).
"""

import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import glob
import os
import re
import unicodedata
import logging
import pandas as pd
from pathlib import Path

try:
    from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
except ImportError:
    st.error("ERRO CRÍTICO: Crie o arquivo 'knowledge_base.py' e cole o 'DICIONARIO_UNIFICADO_HIERARQUICO' nele.")
    st.stop()


# --- FUNÇÕES AUXILIARES GLOBAIS ---
# Estas funções são usadas pelo fluxo de análise profunda (RAG)



def normalize_name(name):
    """Normaliza nomes de empresas removendo acentos, pontuação e sufixos comuns."""
    try:
        # Converte para minúsculas e remove acentos
        nfkd_form = unicodedata.normalize('NFKD', name.lower())
        name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        
        # Remove pontuação e caracteres especiais
        name = re.sub(r'[.,-]', '', name)
        
        # Remove sufixos comuns de empresas
        suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b']
        for suffix in suffixes:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)
            
        # Remove espaços extras
        return re.sub(r'\s+', ' ', name).strip()
    except Exception as e:
        # Fallback em caso de erro
        return name.lower()

# --- CONFIGURAÇÕES GERAIS ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-lite"  # Modelo Gemini unificado
DADOS_PATH = "dados" # Centraliza o caminho para a pasta de dados

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




# --- CARREGAMENTO DE DADOS E CACHING ---


@st.cache_resource
def load_all_artifacts():
    """
    (MODIFICADO) Garante que os artefatos de dados existam localmente,
    baixando-os do GitHub Releases se necessário, antes de carregar na memória.
    """
    DADOS_PATH.mkdir(exist_ok=True)

    ARQUIVOS_REMOTOS = {
        "item_8_4_chunks_map_final.json": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/item_8_4_chunks_map_final.json",
        "item_8_4_faiss_index_final.bin": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/item_8_4_faiss_index_final.bin",
        "outros_documentos_chunks_map_final.json": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/outros_documentos_chunks_map_final.json",
        "outros_documentos_faiss_index_final.bin": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/outros_documentos_faiss_index_final.bin",
        "resumo_fatos_e_topicos_final_enriquecido.json": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/resumo_fatos_e_topicos_final_enriquecido.json"
    }

    for nome_arquivo, url in ARQUIVOS_REMOTOS.items():
        caminho_arquivo = DADOS_PATH / nome_arquivo
        if not caminho_arquivo.exists():
            with st.spinner(f"Baixando artefato: {nome_arquivo}..."):
                try:
                    r = requests.get(url, stream=True)
                    r.raise_for_status()
                    with open(caminho_arquivo, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Arquivo {nome_arquivo} baixado.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Falha ao baixar {nome_arquivo}: {e}")
                    st.stop()

    st.success("Artefatos de dados prontos para uso.")

    model = SentenceTransformer(MODEL_NAME)
    artifacts = {}
    index_files = glob.glob(os.path.join(DADOS_PATH, '*_faiss_index_final.bin'))
    for index_file_path in index_files:
        category = os.path.basename(index_file_path).replace('_faiss_index_final.bin', '')
        chunks_file_path = DADOS_PATH / f"{category}_chunks_map_final.json"
        try:
            index = faiss.read_index(index_file_path)
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            artifacts[category] = {'index': index, 'chunks': chunk_data}
        except Exception as e:
            logger.error(f"Erro ao carregar '{category}': {e}")
            continue

    summary_data = None
    summary_file_path = DADOS_PATH / 'resumo_fatos_e_topicos_final_enriquecido.json'
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Arquivo de resumo '{summary_file_path}' não encontrado.")

    return model, artifacts, summary_data

# --- FUNÇÕES DE LÓGICA DE NEGÓCIO (ROTEADOR E MANIPULADORES) ---

def handle_aggregate_query(query: str, summary_data: dict, alias_map: dict):
    """
    (VERSÃO APRIMORADA) Processa uma pergunta agregada (ex: "Quais empresas têm RSU?").
    
    Esta função foi reescrita para utilizar o formato do 'resumo enriquecido', que contém
    os aliases específicos encontrados para cada tópico em cada empresa. Isso permite
    uma filtragem muito mais precisa e poderosa.

    Args:
        query (str): A pergunta feita pelo usuário.
        summary_data (dict): O dicionário carregado do JSON de resumo enriquecido.
        alias_map (dict): O mapa que converte qualquer alias para seu tópico canônico.
                          (gerado pela função criar_mapa_de_alias).
    """
    
    # --- ETAPA 1: Identificar os termos-chave na pergunta do usuário ---
    query_lower = query.lower()
    query_keywords = set()
    
    # Ordena os aliases conhecidos pelo comprimento, do maior para o menor.
    # Isso garante que "ações restritas" seja encontrado antes de "ações", evitando ambiguidades.
    sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
    
    temp_query = query_lower
    for alias in sorted_aliases:
        # Usa regex com `\b` para garantir que estamos combinando palavras inteiras.
        if re.search(r'\b' + re.escape(alias) + r'\b', temp_query):
            query_keywords.add(alias)
            # Remove o alias encontrado da string de busca para não contá-lo novamente.
            temp_query = temp_query.replace(alias, "")

    # Se nenhum termo conhecido for encontrado, informa o usuário e encerra.
    if not query_keywords:
        st.warning("Não consegui identificar um termo técnico conhecido (como 'TSR', 'vesting', 'RSU') na sua pergunta.")
        return

    st.info(f"Termos específicos identificados para a busca: **{', '.join(sorted(list(query_keywords)))}**")

    # --- ETAPA 2: Filtrar as empresas usando a nova estrutura de dados enriquecida ---
    empresas_encontradas = []
    
    # Itera sobre cada empresa no nosso arquivo de resumo.
    for empresa, data in summary_data.items():
        
        # Para cada empresa, cria um conjunto com todos os seus termos conhecidos.
        company_terms = set()
        
        # Acessa a nova estrutura aninhada: {Seção: {Tópico: [alias1, alias2]}}
        topicos_encontrados = data.get("topicos_encontrados", {})
        for section, topics in topicos_encontrados.items():
            for topic, aliases in topics.items():
                # Adiciona o nome do tópico canônico (ex: 'AcoesRestritas')
                company_terms.add(topic.lower())
                # Adiciona todos os aliases específicos encontrados para aquele tópico (ex: 'rsu')
                company_terms.update([a.lower() for a in aliases])
        
        # A condição de filtro: a empresa é uma candidata se TODOS os termos da pergunta
        # do usuário estiverem presentes no conjunto de termos da empresa.
        if query_keywords.issubset(company_terms):
            empresas_encontradas.append(empresa)

    # --- ETAPA 3: Apresentar os resultados ao usuário ---
    if not empresas_encontradas:
        st.warning(f"Nenhuma empresa foi encontrada com **TODOS** os termos específicos mencionados: `{', '.join(query_keywords)}`.")
        return

    st.success(f"✅ **{len(empresas_encontradas)} empresa(s)** encontrada(s) com os critérios definidos.")
    
    # Usa o Pandas para criar uma tabela bonita e organizada com os resultados.
    df = pd.DataFrame(sorted(empresas_encontradas), columns=["Empresa"])
    st.dataframe(df, use_container_width=True, hide_index=True)

def handle_rag_query(query, artifacts, model, company_catalog_rich):
    """
    Lida com perguntas detalhadas e comparativas usando o fluxo RAG completo.
    """
    # ETAPA 1: GERAÇÃO DO PLANO
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        # Nota: Idealmente, company_catalog_rich seria carregado uma vez fora.
        # Por simplicidade, mantemos aqui.
        plan_response = create_dynamic_analysis_plan_v2(query, company_catalog_rich, list(artifacts.keys()))
        if plan_response['status'] != "success" or not plan_response['plan']['empresas']:
            st.error("❌ Não consegui identificar empresas na sua pergunta. Tente usar nomes conhecidos (ex: Magalu, Vivo, Itaú).")
            return "Análise abortada.", set()
        
        plan = plan_response['plan']
        empresas = plan.get('empresas', [])
        st.write(f"**🏢 Empresas identificadas:** {', '.join(empresas)}")
        st.write(f"**📝 Tópicos a analisar:** {len(plan.get('topicos', []))}")
        status.update(label="✅ Plano gerado com sucesso!", state="complete")

    # ETAPA 2: LÓGICA DE EXECUÇÃO (com tratamento para comparações)
    final_answer = ""
    sources = set()

    # MODO COMPARATIVO
    if len(empresas) > 1:
        st.info(f"Modo de comparação ativado para {len(empresas)} empresas. Analisando sequencialmente...")
        summaries = []
        for i, empresa in enumerate(empresas):
            with st.status(f"Analisando {i+1}/{len(empresas)}: {empresa}...", expanded=True):
                single_company_plan = {'empresas': [empresa], 'topicos': plan['topicos']}
                query_intent = 'item_8_4_query' if any(term in query.lower() for term in ['8.4', 'formulário']) else 'general_query'
                retrieved_context, retrieved_sources = execute_dynamic_plan(single_company_plan, query_intent, artifacts, model)
                sources.update(retrieved_sources)

                if "Nenhuma informação" in retrieved_context or not retrieved_context.strip():
                    summary = f"## Análise para {empresa.upper()}\n\nNenhuma informação encontrada nos documentos para os tópicos solicitados."
                else:
                    summary_prompt = f"Com base no contexto a seguir sobre a empresa {empresa}, resuma os pontos principais sobre os seguintes tópicos: {', '.join(plan['topicos'])}. Contexto: {retrieved_context}"
                    summary = get_final_unified_answer(summary_prompt, retrieved_context)
                
                summaries.append(f"--- RESUMO PARA {empresa.upper()} ---\n\n{summary}")

        with st.status("Gerando relatório comparativo final...", expanded=True):
            comparison_prompt = f"Com base nos resumos individuais a seguir, crie um relatório comparativo detalhado e bem estruturado entre as empresas, focando nos pontos levantados na pergunta original do usuário.\n\nPergunta original do usuário: '{query}'\n\n" + "\n\n".join(summaries)
            final_answer = get_final_unified_answer(comparison_prompt, "\n\n".join(summaries))
            status.update(label="✅ Relatório comparativo gerado!", state="complete")

    # MODO DE ANÁLISE ÚNICA
    else:
        with st.status("2️⃣ Recuperando contexto relevante...", expanded=True) as status:
            query_intent = 'item_8_4_query' if any(term in query.lower() for term in ['8.4', 'formulário']) else 'general_query'
            st.write(f"**🎯 Estratégia detectada:** {'Item 8.4 completo' if query_intent == 'item_8_4_query' else 'Busca geral'}")
            
            retrieved_context, retrieved_sources = execute_dynamic_plan(plan, query_intent, artifacts, model)
            sources.update(retrieved_sources)
            
            if not retrieved_context.strip() or "Nenhuma informação encontrada" in retrieved_context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", set()
            
            st.write(f"**📄 Contexto recuperado de:** {len(sources)} documento(s)")
            status.update(label="✅ Contexto recuperado com sucesso!", state="complete")
        
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            final_answer = get_final_unified_answer(query, retrieved_context)
            status.update(label="✅ Análise concluída!", state="complete")

    return final_answer, sources

def handle_direct_fact_query(query: str, summary_data: dict, alias_map: dict, company_catalog: list):
    """(NOVO) Responde a perguntas diretas de fatos usando o resumo."""
    query_lower = query.lower()
    empresa_encontrada, fato_encontrado_alias = None, None

    # Tenta extrair a empresa e o fato da pergunta
    for company_data in company_catalog:
        for alias in company_data.get("aliases", []):
            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                empresa_encontrada = company_data["canonical_name"].upper()
                break
        if empresa_encontrada: break

    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            fato_encontrado_alias = alias
            break

    # Se não encontrou empresa e fato, não é uma query de fato direto
    if not empresa_encontrada or not fato_encontrado_alias:
        return False

    # Busca o fato nos dados da empresa
    empresa_data = summary_data.get(empresa_encontrada, {})
    st.subheader(f"Fato Direto para: {empresa_encontrada}")
    fato_encontrado = False
    for fact_key, fact_value in empresa_data.get("fatos_extraidos", {}).items():
        if fato_encontrado_alias in fact_key.lower():
            valor = fact_value.get('valor', '')
            unidade = fact_value.get('unidade', '')
            st.metric(label=f"Fato: {fact_key.replace('_', ' ').title()}", value=f"{valor} {unidade}".strip())
            fato_encontrado = True
            break

    if not fato_encontrado:
        st.info(f"O tópico '{fato_encontrado_alias}' foi mencionado, mas um fato estruturado (com valor) não pôde ser extraído.")

    return True # Query tratada com sucesso

# --- FUNÇÕES DE BACKEND (RAG) - com modelo atualizado ---
@st.cache_data
def criar_mapa_de_alias():
    """(MODIFICADO) Cria o mapa de alias a partir do dicionário hierárquico unificado."""
    alias_to_canonical = {}
    if 'DICIONARIO_UNIFICADO_HIERARQUICO' not in globals():
         return {} # Evita erro se o dicionário não for importado
    for section, topics in DICIONARIO_UNIFICADO_HIERARQUICO.items():
        for canonical_name, aliases in topics.items():
            alias_to_canonical[canonical_name.lower()] = canonical_name
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical

def create_dynamic_analysis_plan_v2(query, company_catalog_rich, available_indices):
    # Esta função agora é chamada apenas pelo `handle_rag_query`
    api_key = GEMINI_API_KEY
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    query_lower = query.lower().strip()
    
    # Identificação de Empresas
    mentioned_companies = []
    companies_found_by_alias = {}
    if company_catalog_rich:
        for company_data in company_catalog_rich:
            for alias in company_data.get("aliases", []):
                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                    score = len(alias.split())
                    canonical_name = company_data["canonical_name"]
                    if canonical_name not in companies_found_by_alias or score > companies_found_by_alias[canonical_name]:
                        companies_found_by_alias[canonical_name] = score
        if companies_found_by_alias:
            sorted_companies = sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)
            mentioned_companies = [company for company, score in sorted_companies]
    
    if not mentioned_companies:
        return {"status": "error", "plan": {}}
    
    # Identificação de Tópicos
    topics = []
    found_topics = set()
    alias_map = criar_mapa_de_alias() # Reutiliza o mapa de alias
    for alias, canonical_name in alias_map.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            found_topics.add(canonical_name)

    if found_topics:
        topics = list(found_topics)
    else:
        # Fallback para LLM se nenhum tópico for encontrado
        prompt = f"""Você é um consultor de ILP. Identifique os TÓPICOS CENTRAIS da pergunta: "{query}".
        Retorne APENAS uma lista JSON com os tópicos mais relevantes de: {json.dumps(AVAILABLE_TOPICS)}.
        Se for genérica, selecione tópicos para uma análise geral. Formato: ["Tópico 1", "Tópico 2"]"""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            response.raise_for_status()
            text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
            json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
            if json_match:
                topics = json.loads(json_match.group(0))
            else:
                topics = ["Estrutura do Plano/Programa", "Vesting", "Opções de Compra de Ações"]
        except Exception as e:
            logger.error(f"Falha ao chamar LLM para tópicos: {e}")
            topics = ["Estrutura do Plano/Programa", "Vesting", "Opções de Compra de Ações"]
            
    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}


def execute_dynamic_plan(plan, query_intent, artifacts, model):
    """
    Executa o plano de busca com controle robusto de tokens e deduplicação.
    """
    full_context = ""
    all_retrieved_docs = set()
    unique_chunks_content = set()
    current_token_count = 0
    chunks_processed = 0

    class Config: # Usando uma classe interna para manter as constantes da função
        MAX_CONTEXT_TOKENS = 256000
        MAX_CHUNKS_PER_TOPIC = 10
        SCORE_THRESHOLD_GENERAL = 0.4
        SCORE_THRESHOLD_ITEM_84 = 0.5
        DEDUPLICATION_HASH_LENGTH = 100

    def estimate_tokens(text):
        return len(text) // 4

    def generate_chunk_hash(chunk_text):
        normalized = re.sub(r'\s+', '', chunk_text.lower())
        return hash(normalized[:Config.DEDUPLICATION_HASH_LENGTH])

    def add_unique_chunk_to_context(chunk_text, source_info):
        nonlocal full_context, current_token_count, chunks_processed, unique_chunks_content, all_retrieved_docs
        
        chunk_hash = generate_chunk_hash(chunk_text)
        if chunk_hash in unique_chunks_content:
            logger.debug(f"Chunk duplicado ignorado: {source_info[:50]}...")
            return "DUPLICATE"
        
        estimated_chunk_tokens = estimate_tokens(chunk_text) + estimate_tokens(source_info) + 10
        
        if current_token_count + estimated_chunk_tokens > Config.MAX_CONTEXT_TOKENS:
            logger.warning(f"Limite de tokens atingido. Atual: {current_token_count}")
            return "LIMIT_REACHED"
        
        unique_chunks_content.add(chunk_hash)
        full_context += f"--- {source_info} ---\n{chunk_text}\n\n"
        current_token_count += estimated_chunk_tokens
        chunks_processed += 1
        
        try:
            doc_name = source_info.split("(Doc: ")[1].split(")")[0]
            all_retrieved_docs.add(doc_name)
        except IndexError:
            pass # Ignora se não conseguir extrair o nome
        
        logger.debug(f"Chunk adicionado. Tokens atuais: {current_token_count}")
        return "SUCCESS"

    for empresa in plan.get("empresas", []):
        searchable_company_name = unicodedata.normalize('NFKD', empresa.lower()).encode('ascii', 'ignore').decode('utf-8').split(' ')[0]
        logger.info(f"Processando empresa: {empresa}")

        # Lógica para busca geral (pode ser adaptada para item_8_4 também)
        full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
        
        # Busca por tags
        target_tags = []
        for topico in plan.get("topicos", []):
            target_tags.extend(expand_search_terms(topico))
        target_tags = list(set([tag.title() for tag in target_tags if len(tag) > 3]))
        
        # A função search_by_tags precisa estar definida no seu script
        tagged_chunks = search_by_tags(artifacts, empresa, target_tags)
        
        if tagged_chunks:
            full_context += f"=== CHUNKS COM TAGS ESPECÍFICAS - {empresa.upper()} ===\n\n"
            for chunk_info in tagged_chunks:
                add_unique_chunk_to_context(
                    chunk_info['text'], 
                    f"Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']})"
                )

        # Busca semântica complementar
        for topico in plan.get("topicos", []):
            expanded_terms = expand_search_terms(topico)
            for term in expanded_terms[:3]:
                search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                
                for index_name, artifact_data in artifacts.items():
                    index = artifact_data['index']
                    chunk_data = artifact_data['chunks']
                    scores, indices = index.search(query_embedding, TOP_K_SEARCH)
                    
                    for i, idx in enumerate(indices[0]):
                        if idx != -1 and scores[0][i] > Config.SCORE_THRESHOLD_GENERAL:
                            document_path = chunk_data["map"][idx]['document_path']
                            if searchable_company_name in document_path.lower():
                                chunk_text = chunk_data["chunks"][idx]
                                add_unique_chunk_to_context(
                                    chunk_text,
                                    f"Contexto para '{topico}' via '{term}' (Fonte: {index_name}, Score: {scores[0][i]:.3f}, Doc: {document_path})"
                                )

    if not unique_chunks_content:
        logger.warning("Nenhum chunk único encontrado para o plano de execução.")
        return "Nenhuma informação única encontrada para os critérios especificados.", set()

    logger.info(f"Processamento concluído - Tokens: {current_token_count}, Chunks únicos: {len(unique_chunks_content)}")
    return full_context, all_retrieved_docs

def get_final_unified_answer(query, context):
    """Gera a resposta final usando o contexto recuperado e a API do Gemini."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    has_complete_8_4 = "=== SEÇÃO COMPLETA DO ITEM 8.4" in context
    has_tagged_chunks = "=== CHUNKS COM TAGS ESPECÍFICAS" in context
    
    structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
    if has_complete_8_4:
        structure_instruction = """
**ESTRUTURA OBRIGATÓRIA PARA ITEM 8.4:**
Use a estrutura oficial do item 8.4 do Formulário de Referência:
a) Termos e condições gerais; b) Data de aprovação e órgão; c) Máximo de ações; d) Máximo de opções; 
e) Condições de aquisição; f) Critérios de preço; g) Critérios de prazo; h) Forma de liquidação; 
i) Restrições à transferência; j) Suspensão/extinção; k) Efeitos da saída.
Para cada subitem, extraia e organize as informações encontradas na SEÇÃO COMPLETA DO ITEM 8.4.
"""
    elif has_tagged_chunks:
        structure_instruction = "**PRIORIZE** as informações dos CHUNKS COM TAGS ESPECÍFICAS e organize a resposta de forma lógica usando Markdown."
        
    prompt = f"""Você é um consultor especialista em planos de incentivo de longo prazo (ILP) e no item 8 do formulário de referência da CVM.
    
    PERGUNTA ORIGINAL DO USUÁRIO: "{query}"
    
    CONTEXTO COLETADO DOS DOCUMENTOS:
    {context}
    
    {structure_instruction}
    
    INSTRUÇÕES PARA O RELATÓRIO FINAL:
    1. Responda diretamente à pergunta do usuário com base no contexto fornecido.
    2. PRIORIZE informações da "SEÇÃO COMPLETA DO ITEM 8.4" ou de "CHUNKS COM TAGS ESPECÍFICAS" quando disponíveis. Use o resto do contexto para complementar.
    3. Seja detalhado, preciso e profissional na sua linguagem. Use formatação Markdown (negrito, listas) para clareza.
    4. Se uma informação específica pedida não estiver no contexto, declare explicitamente: "Informação não encontrada nas fontes analisadas.". Não invente dados.
    
    RELATÓRIO ANALÍTICO FINAL:
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        return f"ERRO ao gerar resposta final: Ocorreu um problema ao contatar o modelo de linguagem. Detalhes: {e}"


# --- INTERFACE STREAMLIT (Aplicação Principal) ---
def main():
    st.set_page_config(page_title="Agente de Análise LTIP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
    st.title("🤖 Agente de Análise de Planos de Incentivo (ILP)")
    st.markdown("---")

    # Carregamento centralizado dos artefatos
    model, artifacts, summary_data = load_all_artifacts()
    ALIAS_MAP = criar_mapa_de_alias()

    company_catalog = [{"canonical_name": name, "aliases": [name.split(' ')[0], name]} for name in summary_data.keys()]



    except ImportError:
        company_catalog_rich = []
        logger.warning("`catalog_data.py` não encontrado. A identificação de empresas por apelidos será limitada.")

    # Validação dos dados carregados
    if not artifacts:
        st.error("❌ Erro crítico: Nenhum artefato de busca (índices FAISS) foi carregado. A análise profunda está desativada.")
    if not summary_data:
        st.warning("⚠️ Aviso: O arquivo `resumo_caracteristicas.json` não foi encontrado. Análises de 'quais/quantas empresas' estão desativadas.")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Fontes de Documentos (RAG)", len(artifacts) if artifacts else 0)
        st.metric("Empresas no Resumo", len(summary_data) if summary_data else 0)
        
        if summary_data:
            with st.expander("empresas com características identificadas"):
                st.dataframe(sorted(list(summary_data.keys())), use_container_width=True)
        
        st.success("✅ Sistema pronto para análise")
        st.info(f"Embedding Model: `{MODEL_NAME}`")
        st.info(f"Generative Model: `{GEMINI_MODEL}`") # Mostra o modelo Gemini em uso

    # --- Corpo Principal ---
    st.header("💬 Faça sua pergunta")
    
        # Colunas para exemplos de perguntas
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Experimente uma análise agregada:**")
        st.code("Quais empresas possuem planos com matching?")
        st.code("Quantas empresas têm performance?")
        st.code("Quantas empresas têm Stock Options?")
    with col2:
        st.info("**Ou uma análise profunda :**")
        st.code("Compare dividendos da Vale com a Gerdau")
        st.code("Como é o plano Magazine Luiza?")
        st.code("Resumo item 8.4 Movida")
    st.caption("**Principais Termos-Chave:** `Item 8.4`, `Vesting`, `Stock Options`, `Ações Restritas`, `Performance`, `Matching`, `Lockup`, `SAR`, `ESPP`, `Malus e Clawback`, `Dividendos`")

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Quantas empresas oferecem ações restritas? ")

    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            return

        st.markdown("---")
        st.subheader("📋 Resultado da Análise")
        
        # --- O ROTEADOR DE INTENÇÃO EM AÇÃO ---
        final_answer = ""
        sources = set()
        
        query_lower = user_query.lower()
        aggregate_keywords = ["quais", "quantas", "liste", "qual a lista de"]

        # Rota 1: Pergunta agregada
        query_lower = user_query.lower()
        aggregate_keywords = ["quais", "quantas", "liste"]
        # Padrão simples para detectar perguntas como "Qual o/a ... da/de ..."
        direct_fact_pattern = r'qual\s*(?:é|o|a)\s*.*\s*d[aeo]\s*'

        # Nível 1: Pergunta Agregada
        if any(keyword in query_lower for keyword in aggregate_keywords):
            handle_aggregate_query(user_query, summary_data, ALIAS_MAP)

        # Nível 2: Pergunta de Fato Direto
        elif re.search(direct_fact_pattern, query_lower):
            st.info("Buscando resposta direta no resumo...")
            # A função retorna True se tratou a query, False se não
            if not handle_direct_fact_query(user_query, summary_data, ALIAS_MAP, company_catalog):
                 # Se não conseguiu tratar, cai para o RAG
                 st.info("Não foi possível responder diretamente. Acionando análise profunda...")
                 final_answer, sources = handle_rag_query(user_query, artifacts, model, company_catalog) # Passe o company_catalog aqui também
                 st.markdown(final_answer)

        # Nível 3: Pergunta Complexa (RAG)
        else:
            final_answer, sources = handle_rag_query(user_query, artifacts, model, company_catalog) # Passe o company_catalog aqui também
            st.markdown(final_answer)
            if 'sources' in locals() and sources:
                 with st.expander(f"📚 Documentos consultados ({len(sources)})"):
                      st.write(sorted(list(sources)))
        # Fontes consultadas (apenas para o RAG)
        if sources:
            st.markdown("---")
            with st.expander(f"📚 Documentos consultados na análise profunda ({len(sources)})", expanded=False):
                for i, source in enumerate(sorted(list(sources)), 1):
                    st.write(f"{i}. {source}")

if __name__ == "__main__":
    main()
