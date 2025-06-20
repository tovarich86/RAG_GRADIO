# app.py
# -*- coding: utf-8 -*-
"""
AGENTE COM PLANEJAMENTO DINÂMICO - VERSÃO FINAL STREAMLIT (MAP-REDUCE E SEGURA)

Este script implementa o agente de análise de LTIP como uma aplicação web interativa
usando Streamlit. A arquitetura usa "Map-Reduce" para escalabilidade e inclui um
tratamento de erros centralizado e seguro para proteger as chaves de API.
"""

# --- IMPORTAÇÕES ---
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
import time

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(page_title="Agente de Análise LTIP", layout="wide")
st.title("🤖 Agente de Análise de Planos de Incentivo")
st.caption("Uma aplicação que usa IA para analisar Formulários de Referência da CVM de forma escalável e segura.")

# --- CONFIGURAÇÕES E ESTRUTURAS DE CONHECIMENTO ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
GOOGLE_DRIVE_PATH = './dados'

# (O resto das constantes como TERMOS_TECNICOS_LTIP e AVAILABLE_TOPICS permanecem as mesmas)
TERMOS_TECNICOS_LTIP = {
    "tratamento de dividendos": ["tratamento de dividendos", "equivalente em dividendos", "dividendos", "juros sobre capital próprio", "proventos", "dividend equivalent", "dividendos pagos em ações", "ajustes por dividendos"],
    "preço de exercício": ["preço de exercício", "strike price", "preço de compra", "preço fixo", "valor de exercício", "preço pré-estabelecido", "preço de aquisição"],
    "forma de liquidação": ["forma de liquidação", "liquidação", "pagamento", "entrega física", "pagamento em dinheiro", "transferência de ações", "settlement"],
    "vesting": ["vesting", "período de carência", "carência", "aquisição de direitos", "cronograma de vesting", "vesting schedule", "período de cliff"],
    "eventos corporativos": ["eventos corporativos", "desdobramento", "grupamento", "dividendos pagos em ações", "bonificação", "split", "ajustes", "reorganização societária"],
    "stock options": ["stock options", "opções de ações", "opções de compra", "SOP", "plano de opções", "ESOP", "opção de compra de ações"],
    "ações restritas": ["ações restritas", "restricted shares", "RSU", "restricted stock units", "ações com restrição", "plano de ações restritas"]
}
AVAILABLE_TOPICS = [
    "termos e condições gerais", "data de aprovação e órgão responsável", "número máximo de ações abrangidas", "número máximo de opções a serem outorgadas", "condições de aquisição de ações", "critérios para fixação do preço de aquisição ou exercício", "preço de exercício", "strike price", "critérios para fixação do prazo de aquisição ou exercício", "forma de liquidação", "liquidação", "pagamento", "restrições à transferência das ações", "critérios e eventos de suspensão/extinção", "efeitos da saída do administrador", "Tipos de Planos", "Condições de Carência", "Vesting", "período de carência", "cronograma de vesting", "Matching", "contrapartida", "co-investimento", "Lockup", "período de lockup", "restrição de venda", "Tratamento de Dividendos", "equivalente em dividendos", "proventos", "Stock Options", "opções de ações", "SOP", "Ações Restritas", "RSU", "restricted shares", "Eventos Corporativos", "IPO", "grupamento", "desdobramento"
]


# --- NOVA FUNÇÃO SEGURA PARA CHAMADAS DE API ---
def safe_api_call(url, payload, headers, timeout=90):
    """
    Realiza uma chamada POST para a API de forma segura, tratando erros
    sem expor a URL completa ou a chave de API.
    Retorna (dados_json, None) em caso de sucesso, ou (None, mensagem_de_erro_segura) em caso de falha.
    """
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        response.raise_for_status()  # Gera um erro para códigos de status 4xx ou 5xx
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        # Captura erros HTTP e extrai apenas informações seguras
        status_code = e.response.status_code
        reason = e.response.reason
        return None, f"Erro de API com código {status_code}: {reason}. Por favor, tente novamente mais tarde."
    except requests.exceptions.RequestException as e:
        # Captura outros erros de requisição (conexão, timeout, etc.)
        return None, f"Erro de conexão ao tentar contatar a API. Verifique sua conexão com a internet."


# --- FUNÇÕES DE LÓGICA (Refatoradas para usar safe_api_call) ---

# (As funções expand_search_terms, search_by_tags e load_all_artifacts permanecem iguais)
def expand_search_terms(base_term):
    expanded_terms = [base_term.lower()]
    for category, terms in TERMOS_TECNICOS_LTIP.items():
        if any(term.lower() in base_term.lower() for term in terms):
            expanded_terms.extend([term.lower() for term in terms])
    return list(set(expanded_terms))

def search_by_tags(artifacts, company_name, target_tags):
    results = []
    for index_name, artifact_data in artifacts.items():
        chunk_data = artifact_data['chunks']
        for i, mapping in enumerate(chunk_data.get('map', [])):
            document_path = mapping['document_path']
            if re.search(re.escape(company_name.split(' ')[0]), document_path, re.IGNORECASE):
                chunk_text = chunk_data["chunks"][i]
                for tag in target_tags:
                    if f"Tópicos:" in chunk_text and tag in chunk_text:
                        results.append({'text': chunk_text, 'path': document_path, 'source': index_name, 'tag_found': tag})
                        break
    return results

@st.cache_resource
def load_all_artifacts():
    artifacts = {}
    canonical_company_names = set()
    with st.spinner("Carregando modelo de embedding (isso só acontece na primeira vez)..."):
        model = SentenceTransformer(MODEL_NAME)
    
    index_files = glob.glob(os.path.join(GOOGLE_DRIVE_PATH, '*_faiss_index.bin'))
    if not index_files:
        st.error(f"ERRO CRÍTICO: Nenhum arquivo de índice (*_faiss_index.bin) encontrado na pasta '{GOOGLE_DRIVE_PATH}'. Verifique se os arquivos de dados estão na pasta 'dados' no GitHub e se o nome da pasta está em minúsculas.")
        return None, None, None

    for index_file in index_files:
        category = os.path.basename(index_file).replace('_faiss_index.bin', '')
        chunks_file = os.path.join(GOOGLE_DRIVE_PATH, f"{category}_chunks_map.json")
        try:
            index = faiss.read_index(index_file)
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            artifacts[category] = {'index': index, 'chunks': chunk_data}
            for mapping in chunk_data.get('map', []):
                parts = mapping['document_path'].split('/')
                if parts:
                    canonical_company_names.add(parts[0])
        except FileNotFoundError:
            st.warning(f"AVISO: Arquivo de chunks '{chunks_file}' não encontrado.")
            continue
    
    if not artifacts:
        st.error("ERRO CRÍTICO: Nenhum artefato foi carregado com sucesso.")
        return None, None, None

    return artifacts, model, list(canonical_company_names)


@st.cache_data
def create_dynamic_analysis_plan(_query, company_catalog, available_indices):
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    # (A lógica de identificação de empresa permanece a mesma)
    def normalize_name(name):
        try:
            nfkd_form = unicodedata.normalize('NFKD', name.lower()); name = "".join([c for c in nfkd_form if not unicodedata.combining(c)]); name = re.sub(r'[.,-]', '', name); suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b'];
            for suffix in suffixes: name = re.sub(suffix, '', name, flags=re.IGNORECASE)
            return re.sub(r'\s+', '', name).strip()
        except Exception: return name.lower()
    mentioned_companies = []; query_clean = _query.lower().strip()
    for canonical_name in company_catalog:
        if (canonical_name.lower() in query_clean or any(len(part) > 2 and re.search(r'\b' + re.escape(part.lower()) + r'\b', query_clean) for part in canonical_name.split(' ')) or (len(query_clean) > 2 and normalize_name(query_clean) in normalize_name(canonical_name))):
            if canonical_name not in mentioned_companies: mentioned_companies.append(canonical_name)
    if not mentioned_companies and len(query_clean) <= 6:
        for canonical_name in company_catalog:
            if query_clean.upper() in canonical_name.upper():
                if canonical_name not in mentioned_companies: mentioned_companies.append(canonical_name)
    
    prompt = f'Você é um planejador de análise... (seu prompt completo aqui)'
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}

    # USO DA FUNÇÃO SEGURA
    response_data, error_message = safe_api_call(url, payload, headers)

    if error_message:
        st.warning(f"Erro no planejamento: {error_message}")
        # Fallback em caso de erro
        plan = {"empresas": list(set(mentioned_companies)), "topicos": AVAILABLE_TOPICS}
        return {"status": "success", "plan": plan}

    text_response = response_data['candidates'][0]['content']['parts'][0]['text']
    json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
    if json_match:
        topics = json.loads(json_match.group(0))
        plan = {"empresas": list(set(mentioned_companies)), "topicos": topics}
        return {"status": "success", "plan": plan}
    
    # Fallback se o JSON não for encontrado
    plan = {"empresas": list(set(mentioned_companies)), "topicos": AVAILABLE_TOPICS}
    return {"status": "success", "plan": plan}

def execute_dynamic_plan(plan, query_intent, artifacts, model):
    # (Esta função não faz chamadas de API, então permanece inalterada)
    full_context = ""
    all_retrieved_docs = set()
    for empresa in plan.get("empresas", []):
        full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
        if query_intent == 'item_8_4_query' and 'item_8_4' in artifacts:
            artifact_data = artifacts['item_8_4']
            chunk_data = artifact_data['chunks']
            for i, mapping in enumerate(chunk_data.get('map', [])):
                if re.search(re.escape(empresa.split(' ')[0]), mapping['document_path'], re.IGNORECASE):
                    all_retrieved_docs.add(mapping['document_path'])
                    full_context += f"--- Chunk Item 8.4 (Doc: {mapping['document_path']}) ---\n{chunk_data['chunks'][i]}\n\n"
        target_tags = []
        for topico in plan.get("topicos", []): target_tags.extend(expand_search_terms(topico))
        target_tags = list(set([tag.title() for tag in target_tags if len(tag) > 3]))
        tagged_chunks = search_by_tags(artifacts, empresa, target_tags)
        for chunk_info in tagged_chunks:
            full_context += f"--- Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']}) ---\n{chunk_info['text']}\n\n"
            all_retrieved_docs.add(chunk_info['path'])
    return full_context, [str(doc) for doc in all_retrieved_docs]


# --- FUNÇÕES DE SÍNTESE (ESTRATÉGIA MAP-REDUCE SEGURA) ---

def summarize_chunk(chunk_text, query):
    """(MAP) Pede à IA para resumir um único chunk de texto de forma segura."""
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    prompt = f'Com base na pergunta do usuário e no chunk de texto abaixo, extraia e resuma APENAS as informações relevantes. Se o chunk não contiver informações relevantes, responda com "N/A".\n\nPergunta do Usuário: "{query}"\n\nChunk de Texto:\n---\n{chunk_text}\n---\n\nResumo Conciso das Informações Relevantes:'
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024}}
    headers = {'Content-Type': 'application/json'}
    
    # USO DA FUNÇÃO SEGURA
    response_data, error_message = safe_api_call(url, payload, headers)

    if error_message:
        # Se houver erro, apenas o ignoramos e continuamos para o próximo chunk
        print(f"Erro ao resumir chunk: {error_message}") # Log no console do servidor, não para o usuário
        return None

    summary = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
    if summary.upper() != "N/A" and len(summary) > 20:
        return summary
    return None


def get_final_unified_answer(query, context, plan):
    """(REDUCE) Sintetiza a resposta final de forma segura."""
    chunks = re.split(r'--- Chunk|--- INÍCIO DA ANÁLISE PARA:', context)
    relevant_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]
    st.info(f"Analisando {len(relevant_chunks)} chunks de informação relevantes...")
    summaries = []
    progress_bar = st.progress(0, text="Mapeando e resumindo chunks...")

    for i, chunk in enumerate(relevant_chunks):
        summary = summarize_chunk(chunk, query)
        if summary: summaries.append(summary)
        time.sleep(1.1)
        progress_bar.progress((i + 1) / len(relevant_chunks), text=f"Mapeando e resumindo chunks... ({i+1}/{len(relevant_chunks)})")
    
    progress_bar.empty()

    if not summaries:
        return "Não foi possível extrair informações relevantes dos documentos para montar um relatório."

    st.info("Todos os chunks foram analisados. Sintetizando o relatório final...")
    final_context = "\n\n---\n\n".join(summaries)
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    final_prompt = f'Você é um analista financeiro sênior... (seu prompt completo aqui)'
    payload = {"contents": [{"parts": [{"text": final_prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}}
    headers = {'Content-Type': 'application/json'}
    
    # USO DA FUNÇÃO SEGURA
    response_data, error_message = safe_api_call(url, payload, headers, timeout=180)

    if error_message:
        return f"ERRO ao gerar a síntese final do relatório: {error_message}"
    
    return response_data['candidates'][0]['content']['parts'][0]['text'].strip()


# --- LÓGICA PRINCIPAL DA APLICAÇÃO STREAMLIT ---
try:
    loaded_artifacts, embedding_model, company_catalog = load_all_artifacts()
except Exception as e:
    st.error(f"Ocorreu um erro fatal durante o carregamento inicial dos recursos: {e}")
    st.stop()

if loaded_artifacts:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Qual empresa ou plano de incentivo você gostaria de analisar?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Digite sua pergunta sobre CCR, Vibra, etc."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Iniciando análise..."):
                plan_response = create_dynamic_analysis_plan(prompt, company_catalog, list(loaded_artifacts.keys()))
                if plan_response['status'] != 'success' or not plan_response['plan'].get("empresas"):
                    response_text = "Não consegui identificar uma empresa em sua pergunta ou houve um erro no planejamento. Por favor, seja mais específico."
                else:
                    plan = plan_response['plan']
                    st.info(f"Plano de análise criado. Foco em: {plan['empresas'][0] if plan['empresas'] else 'N/A'}.")
                    query_intent = 'item_8_4_query' if any(term in prompt.lower() for term in ['8.4', '8-4', 'item 8.4', 'formulário']) else 'general_query'
                    retrieved_context, sources = execute_dynamic_plan(plan, query_intent, loaded_artifacts, embedding_model)
                    if not retrieved_context.strip():
                        response_text = "Não encontrei informações relevantes nos documentos para a sua solicitação."
                    else:
                        final_answer = get_final_unified_answer(prompt, retrieved_context, plan)
                        response_text = final_answer
                        if sources:
                            unique_sources = sorted(list(set(sources)))
                            with st.expander(f"Fontes Consultadas ({len(unique_sources)})"):
                                for source in unique_sources:
                                    st.write(f"- {source}")
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
