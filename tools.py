# tools_v3.0.py (Versão Unificada e Completa)
#
# DESCRIÇÃO:
# Este módulo combina as melhores características de ambas as versões anteriores.
#
# FUNCIONALIDADES PRESERVADAS DO ORIGINAL:
# 1. Busca Híbrida: Combina busca vetorial e busca por metadados para máxima relevância.
# 2. Motor de Sugestão Inteligente: Usa CAPABILITY_MAP e QUESTION_TEMPLATES para
#    gerar sugestões de perguntas diversificadas e garantidas.
# 3. Normalização de Nomes de Empresas: Reintroduzida a função _create_company_lookup_map.
#
# MELHORIAS INTEGRADAS DA VERSÃO ATUALIZADA:
# 1. Dicionário Hierárquico: Totalmente compatível com a estrutura de KB aninhada (v6.0).
# 2. Filtros de Metadados: Suporte para filtrar buscas por setor, controle acionário, etc.
# 3. Código Robusto: Inclui logging e tratamento de exceções aprimorados.

import faiss
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import requests
import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

# --- ESTRUTURAS DE DADOS DA VERSÃO ORIGINAL (PRESERVADAS PELA SUA ROBUSTEZ) ---
# Habilidades: 'thematic', 'listing', 'statistic', 'comparison'
CAPABILITY_MAP = {
    # Mapeia o TÓPICO FINAL (folha da hierarquia) para suas capacidades
    "AcoesRestritas": ["listing", "thematic", "comparison", "example_deep_dive"],
    "OpcoesDeCompra": ["listing", "thematic", "comparison", "example_deep_dive"],
    "AcoesFantasmas": ["listing", "thematic", "comparison"],
    "Matching Coinvestimento": ["listing", "thematic", "comparison"],
    "Vesting": ["statistic", "thematic", "listing", "comparison", "example_deep_dive"],
    "Lockup": ["statistic", "thematic", "listing", "comparison"],
    "PrecoDesconto": ["statistic", "listing", "thematic"],
    "VestingAcelerado": ["listing", "thematic", "comparison"],
    "Outorga": ["thematic", "listing"],
    "MalusClawback": ["listing", "thematic", "comparison", "example_deep_dive"],
    "Diluicao": ["statistic", "listing", "thematic"],
    "OrgaoDeliberativo": ["listing", "thematic"],
    "Elegibilidade": ["listing", "thematic", "comparison"],
    "CondicaoSaida": ["thematic", "listing", "comparison", "example_deep_dive"],
    "TSR Relativo": ["listing", "thematic", "comparison", "example_deep_dive"],
    "TSR Absoluto": ["listing", "thematic", "comparison"],
    "ESG": ["listing", "thematic"],
    "GrupoDeComparacao": ["thematic", "listing"],
    "DividendosProventos": ["listing", "thematic", "comparison", "example_deep_dive"],
    "MudancaDeControle": ["listing", "thematic", "comparison"],
}

QUESTION_TEMPLATES = {
    "thematic": [
        "Analise os modelos típicos de **{topic}** encontrados nos planos das empresas.",
        "Quais são as abordagens mais comuns para **{topic}** no mercado brasileiro?",
        "Descreva os padrões de mercado relacionados a **{topic}**."
    ],
    "listing": [
        "Quais empresas na base de dados possuem planos com **{topic}**?",
        "Gere uma lista de companhias que mencionam **{topic}** em seus documentos.",
        "Liste as empresas que adotam práticas de **{topic}**."
    ],
    "statistic": [
        "Qual o valor médio ou mais comum para o tópico de **{topic}** entre as empresas?",
        "Apresente as estatísticas (média, mediana, máximo) para **{topic}**.",
        "Qual a prevalência de **{topic}** nos planos analisados?"
    ],
    "comparison": [
        "Compare como a Vale e a Petrobras abordam o tópico de **{topic}** em seus planos.",
        "Quais as principais diferenças entre os planos da Magazine Luiza e da Localiza sobre **{topic}**?",
    ],
    "example_deep_dive": [
        "Como o plano da Vale define e aplica o conceito de **{topic}**?",
        "Descreva em detalhes como a Hypera trata a questão de **{topic}** em seu plano de remuneração.",
    ]
}

# --- NOVAS FUNÇÕES DE MAPEAMENTO HIERÁRQUICO (DA v2.1) ---
def rerank_by_recency(chunks_to_rerank: list[dict], current_query_date: datetime) -> list[dict]:
    """Dá um bônus de relevância a chunks mais recentes."""
    # Define uma data base para cálculo do score
    if not isinstance(current_query_date, datetime):
        current_query_date = datetime.now()

    def get_recency_score(chunk):
        date_str = chunk.get("document_date", "N/A")
        if date_str and date_str != "N/A":
            try:
                # Converte a string 'AAAA-MM-DD' para um objeto de data real
                doc_date = datetime.fromisoformat(date_str)
                # Calcula a diferença de dias. Um score maior para datas mais próximas.
                days_diff = (current_query_date - doc_date).days
                # Adiciona um score de relevância. Ex: 10 pontos para 0 dias de diferença.
                return max(0, 10 - (days_diff / 365))
            except (ValueError, TypeError):
                return 0
        return 0

    for chunk in chunks_to_rerank:
        recency_score = get_recency_score(chunk)
        # Adiciona o score de recência ao score de relevância existente
        chunk['relevance_score'] = chunk.get('relevance_score', 0) + recency_score

    # Re-ordena a lista com base no novo score
    return sorted(chunks_to_rerank, key=lambda x: x.get('relevance_score', 0), reverse=True)

def _recursive_alias_mapper(sub_dict, path_so_far, flat_map):
    """Função auxiliar recursiva para criar o mapa de aliases."""
    for topic_key, topic_data in sub_dict.items():
        current_path = path_so_far + [topic_key]
        path_str = ",".join(current_path)
        
        # Mapeia todos os aliases para o caminho completo
        for alias in topic_data.get("aliases", []):
            flat_map[alias.lower()] = path_str
        
        # Mapeia o nome canônico do tópico (ex: 'Acoes_Restritas' -> 'acoes restritas')
        canonical_alias = topic_key.replace('_', ' ').lower()
        if canonical_alias not in flat_map:
            flat_map[canonical_alias] = path_str

        # Continua a recursão para sub-tópicos
        if "subtopicos" in topic_data and topic_data["subtopicos"]:
            _recursive_alias_mapper(topic_data["subtopicos"], current_path, flat_map)

def create_hierarchical_alias_map(kb: dict) -> dict:
    """
    Cria um mapeamento plano de qualquer alias (em minúsculas) para seu
    caminho hierárquico completo.
    AGORA SUPORTA ALIASES NA CATEGORIA PAI.
    """
    alias_map = {}
    for section, data in kb.items():  # Ex: section="MecanicasCicloDeVida"
        path_str = section
        
        # --- LÓGICA ADICIONADA ---
        # 1. Mapeia os aliases da categoria principal (ex: "mecanicas")
        for alias in data.get("aliases", []):
            alias_map[alias.lower()] = path_str
        
        # 2. Mapeia o nome canônico da própria categoria
        canonical_alias = section.replace('_', ' ').lower()
        if canonical_alias not in alias_map:
            alias_map[canonical_alias] = path_str
        # --- FIM DA LÓGICA ADICIONADA ---

        # 3. Continua a recursão para os sub-tópicos, se existirem
        if "subtopicos" in data:
            _recursive_alias_mapper(data["subtopicos"], [section], alias_map)
            
    return alias_map

def _create_company_lookup_map(company_catalog_rich: list) -> dict:
    """
    (REINTRODUZIDA) Cria um dicionário de mapeamento reverso para normalizar nomes de empresas.
    """
    lookup_map = {}
    if not company_catalog_rich:
        return lookup_map
        
    for company_data in company_catalog_rich:
        canonical_name = company_data.get("canonical_name")
        if not canonical_name:
            continue
        
        all_names_to_map = [canonical_name] + company_data.get("aliases", [])
        
        for name in all_names_to_map:
            lookup_map[name.lower()] = canonical_name
            
    return lookup_map

def get_final_unified_answer(query: str, context: str, api_key: str) -> str:
    """Chama a API do LLM para gerar uma resposta final sintetizada."""
    GEMINI_MODEL = "gemini-2.0-flash-lite" # Mantém o modelo mais recente
    if not GEMINI_API_KEY: return "Erro: A chave da API do Gemini não foi configurada."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
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
        candidates = response.json().get('candidates', [])
        if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
            return candidates[0]['content']['parts'][0]['text'].strip()
        else:
            logger.error(f"Resposta inesperada da API Gemini: {response.json()}")
            return "Ocorreu um erro ao processar a resposta do modelo de linguagem."
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        return f"Ocorreu um erro ao contatar o modelo de linguagem. Detalhes: {str(e)}"

def rerank_with_cross_encoder(query: str, chunks: list[dict], cross_encoder_model: CrossEncoder, top_n: int = 10) -> list[dict]:
    """Usa um Cross-Encoder para reordenar chunks por relevância."""
    if not chunks or not query: return []
    pairs = [[query, chunk.get('text', '')] for chunk in chunks]
    try:
        scores = cross_encoder_model.predict(pairs, show_progress_bar=False)
        for i, chunk in enumerate(chunks):
            chunk['relevance_score'] = scores[i]
        reranked_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        return reranked_chunks[:top_n]
    except Exception as e:
        logger.error(f"Erro durante o re-ranking com CrossEncoder: {e}")
        return chunks[:top_n]

# --- MOTOR DE SUGESTÃO ROBUSTO (LÓGICA ORIGINAL RESTAURADA) ---
def suggest_alternative_query(failed_query: str, kb: dict) -> str:
    """
    Motor de sugestão robusto que usa os mapas enriquecidos (CAPABILITY_MAP) para gerar
    sugestões variadas, relevantes e garantidas de funcionar.
    """
    logger.info("---EXECUTANDO MOTOR DE SUGESTÃO ROBUSTO (v3 Unificado)---")

    alias_map = create_hierarchical_alias_map(kb)
    
    # Identifica tópicos na consulta falha
    topics_found_paths = set()
    for alias, path in alias_map.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', failed_query.lower()):
            topics_found_paths.add(path)

    safe_suggestions = []
    context_for_llm = ""

    if topics_found_paths:
        primary_path = list(topics_found_paths)[0]
        # Pega o tópico mais específico (última parte do caminho) para usar com os mapas
        primary_topic_key = primary_path.split(',')[-1]
        primary_topic_display = primary_topic_key.replace('_', ' ')
        
        context_for_llm = f"O usuário demonstrou interesse no tópico de '{primary_topic_display}'."
        logger.info(f"---TÓPICO IDENTIFICADO: '{primary_topic_key}'---")

        capabilities = CAPABILITY_MAP.get(primary_topic_key, ["listing", "thematic"])
        logger.info(f"---HABILIDADES ENCONTRADAS PARA O TÓPICO: {capabilities}---")
        
        for cap in capabilities:
            template_variations = QUESTION_TEMPLATES.get(cap)
            if template_variations:
                chosen_template = random.choice(template_variations)
                safe_suggestions.append(chosen_template.format(topic=primary_topic_display))
    else:
        context_for_llm = "O usuário fez uma pergunta genérica ou sobre um tópico não reconhecido."
        logger.warning("---NENHUM TÓPICO IDENTIFICADO. USANDO SUGESTÕES GERAIS.---")
        safe_suggestions = [
            "Liste as empresas que utilizam TSR Relativo como métrica de performance.",
            "Analise os modelos típicos de planos de Ações Restritas (RSU).",
            "Como funciona o plano de vesting da Vale?"
        ]

    safe_suggestions = safe_suggestions[:3]

    prompt = f"""
    Você é um assistente de IA prestativo. A pergunta de um usuário falhou.
    PERGUNTA ORIGINAL: "{failed_query}"
    CONTEXTO: {context_for_llm}
    Eu gerei uma lista de perguntas "seguras". Sua tarefa é apresentá-las de forma clara e convidativa. Você pode fazer pequenas melhorias para soar mais natural, mas mantenha a intenção original.

    PERGUNTAS SEGURAS PARA APRESENTAR:
    {json.dumps(safe_suggestions, indent=2, ensure_ascii=False)}

    Apresente o resultado como uma lista de marcadores em Markdown.
    """
    return get_final_unified_answer(prompt, "")


# --- FERRAMENTA DE BUSCA HÍBRIDA (LÓGICA ORIGINAL RESTAURADA E ADAPTADA) ---

def find_companies_by_topic(
    topic: str,
    artifacts: dict,
    model: SentenceTransformer,
    kb: dict,
    filters: dict = None,
    top_k: int = 20
) -> list[str]:
    """
    Ferramenta de Listagem HÍBRIDA. Busca por um tópico aplicando filtros e combinando
    busca por metadados (exata) e busca vetorial (semântica).
    """
    alias_map = create_hierarchical_alias_map(kb)
    topic_path = alias_map.get(topic.lower(), topic)
    logger.info(f"Buscando empresas para o tópico '{topic}' (caminho: {topic_path}) com filtros: {filters}")

    # --- Parte 1: Busca por Metadados (como na v2.1, mas com filtros) ---
    companies_from_metadata = set()
    for artifact_name, artifact_data in artifacts.items():
        chunk_map = artifact_data.get('chunks', [])
        if not chunk_map: continue

        # Aplica filtros, se existirem
        filtered_map = chunk_map
        if filters:
            if filters.get('setor'):
                filtered_map = [c for c in filtered_map if c.get('setor', '').lower() == filters['setor'].lower()]
            if filters.get('controle_acionario'):
                filtered_map = [c for c in filtered_map if c.get('controle_acionario', '').lower() == filters['controle_acionario'].lower()]
        
        if not filtered_map: continue

        for chunk_data in filtered_map:
            for path in chunk_data.get('topics_in_chunk', []):
                if path.lower().startswith(topic_path.lower()):
                    companies_from_metadata.add(chunk_data["company_name"])
                    break

    # --- Parte 2: Busca Vetorial Semântica (como na original, mas com filtros) ---
    companies_from_vector = set()
    search_query = f"regras, detalhes e funcionamento sobre {topic.replace('_', ' ')}"
    query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')

    for artifact_name, artifact_data in artifacts.items():
        index = artifact_data.get('index')
        chunk_map = artifact_data.get('chunks', [])
        if not index or not chunk_map: continue

        _, indices = index.search(query_embedding, top_k)
        for idx in indices[0]:
            if idx == -1: continue
            
            chunk_data = chunk_map[idx]
            company_name = chunk_data.get("company_name")
            
            # Valida o chunk encontrado contra os filtros
            passes_filter = True
            if filters:
                if filters.get('setor') and chunk_data.get('setor', '').lower() != filters['setor'].lower():
                    passes_filter = False
                if filters.get('controle_acionario') and chunk_data.get('controle_acionario', '').lower() != filters['controle_acionario'].lower():
                    passes_filter = False

            if passes_filter and company_name:
                companies_from_vector.add(company_name)

    # --- Parte 3: União dos Resultados ---
    final_companies = sorted(list(companies_from_metadata.union(companies_from_vector)))
    logger.info(f"Encontradas {len(final_companies)} empresas no total para o tópico '{topic}'. (Metadados: {len(companies_from_metadata)}, Vetor: {len(companies_from_vector)})")
    return final_companies


# --- FERRAMENTAS DE ANÁLISE DE ALTO NÍVEL (ADAPTADAS PARA A NOVA ESTRUTURA) ---

def get_summary_for_topic_at_company(
    company: str,
    topic: str,
    query: str,
    artifacts: dict,
    model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    execute_dynamic_plan_func: callable,
    get_final_unified_answer_func: callable,
    filters: dict = None  # <-- Movido para o final
) -> str:
    """Ferramenta de Extração: Busca, re-ranqueia e resume um tópico para uma empresa específica."""
    plan = {"empresas": [company], "topicos": [topic], "filtros": filters or {}}
    context, _ = execute_dynamic_plan_func(query, plan, artifacts, model, cross_encoder_model, kb, company_catalog_rich, company_lookup_map, search_by_tags, expand_search_terms)
    if not context:
        return "Não foi possível encontrar detalhes específicos sobre este tópico para esta empresa e filtros."
    
    # Prompt mais detalhado para melhor qualidade do resumo
    summary_prompt = f"""
    Com base no contexto fornecido sobre a empresa {company}, resuma em detalhes as regras e o funcionamento do plano relacionadas ao tópico: '{topic}'.
    Seja direto e foque apenas nas informações relevantes para o tópico.

    CONTEXTO:
    {context}
    """
    summary = get_final_unified_answer_func(summary_prompt, context)
    return summary


def analyze_topic_thematically(
    topic: str,
    query: str,
    artifacts: dict,
    model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    execute_dynamic_plan_func: callable,
    get_final_unified_answer_func: callable,
    filters: dict = None  # <-- Movido para o final
) -> str:
    """Ferramenta de Orquestração: Realiza uma análise temática completa de um tópico usando a busca híbrida."""
    logger.info(f"Iniciando análise temática para '{topic}' com filtros: {filters}")
    
    # Utiliza a nova função de busca híbrida
    companies_to_analyze = find_companies_by_topic(topic, artifacts, model, kb, filters)
    
    if not companies_to_analyze:
        return f"Não foram encontradas empresas com informações suficientes sobre '{topic}' para os filtros selecionados."
    
    # Limita o número de empresas para análise para evitar sobrecarga e custos
    limit = 15
    if len(companies_to_analyze) > limit:
        logger.warning(f"Muitas empresas ({len(companies_to_analyze)}) encontradas. Analisando uma amostra de {limit}.")
        companies_to_analyze = random.sample(companies_to_analyze, limit)
        
    logger.info(f"Analisando '{topic}' para {len(companies_to_analyze)} empresas...")
    company_summaries = []
    
    # Usa ThreadPool para paralelizar a coleta de resumos
    with ThreadPoolExecutor(max_workers=min(len(companies_to_analyze), 10)) as executor:
        futures = {
            executor.submit(
                get_summary_for_topic_at_company,
                company, topic, query, artifacts, model, cross_encoder_model,
                kb, company_catalog_rich, company_lookup_map, execute_dynamic_plan_func, get_final_unified_answer_func, filters
            ): company for company in companies_to_analyze
        }
        for future in futures:
            company = futures[future]
            try:
                summary_text = future.result()
                company_summaries.append({"empresa": company, "resumo_do_plano": summary_text})
            except Exception as e:
                logger.error(f"Erro ao analisar a empresa {company}: {e}")
                company_summaries.append({"empresa": company, "resumo_do_plano": f"Erro ao processar a análise."})

    synthesis_context = json.dumps(company_summaries, indent=2, ensure_ascii=False)
    
    # Usa o prompt de síntese mais detalhado da versão original
    synthesis_prompt = f"""
    Você é um consultor especialista em remuneração e planos de incentivo.
    Sua tarefa é responder à pergunta original do usuário: "{query}"
    Para isso, analise o CONTEXTO JSON abaixo, que contém resumos dos planos de várias empresas sobre o tópico '{topic}'.

    CONTEXTO:
    {synthesis_context}

    INSTRUÇÕES PARA O RELATÓRIO TEMÁTICO:
    1.  **Introdução:** Comece com um parágrafo que resume suas principais descobertas.
    2.  **Identificação de Padrões:** Analise todos os resumos e identifique de 2 a 4 "modelos" ou "padrões" comuns de mercado.
    3.  **Descrição dos Padrões:** Para cada padrão, descreva-o detalhadamente e liste as empresas que o seguem.
    4.  **Exceções e Casos Únicos:** Destaque abordagens que fogem ao padrão ou são inovadoras.
    5.  **Conclusão:** Finalize com uma breve conclusão sobre as práticas de mercado para '{topic}'.

    Seja analítico, estruturado e use Markdown para formatar sua resposta de forma clara e profissional.
    """
    final_report = get_final_unified_answer_func(synthesis_prompt, synthesis_context)
    return final_report
