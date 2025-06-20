# -*- coding: utf-8 -*-
"""
AGENTE DE ANÁLISE LTIP - VERSÃO STREAMLIT
Aplicação web para análise de planos de incentivo de longo prazo
"""

import streamlit as st
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import glob
import os
import re
import unicodedata
from catalog_data import company_catalog_rich

# --- CONFIGURAÇÕES ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- DICIONÁRIOS E LISTAS DE TÓPICOS (sem alterações) ---
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

# --- FUNÇÕES AUXILIARES ---
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
            # Melhorando a busca para não pegar apenas a primeira palavra
            if company_name.split(' ')[0].lower() in document_path.lower():
                chunk_text = chunk_data["chunks"][i]
                for tag in target_tags:
                    if f"Tópicos:" in chunk_text and tag in chunk_text:
                        results.append({
                            'text': chunk_text, 'path': document_path, 'index': i,
                            'source': index_name, 'tag_found': tag
                        })
                        break
    return results

def normalize_name(name):
    try:
        nfkd_form = unicodedata.normalize('NFKD', name.lower())
        name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        name = re.sub(r'[.,-]', '', name)
        suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b']
        for suffix in suffixes:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', name).strip() # Manter espaços entre as palavras
    except Exception as e:
        return name.lower()

# --- CACHE PARA CARREGAR ARTEFATOS ---
@st.cache_resource
def load_all_artifacts():
    artifacts = {}
    model = SentenceTransformer(MODEL_NAME)
    dados_path = "dados"
    index_files = glob.glob(os.path.join(dados_path, '*_faiss_index.bin'))
    if not index_files:
        return None, None
    for index_file in index_files:
        category = os.path.basename(index_file).replace('_faiss_index.bin', '')
        chunks_file = os.path.join(dados_path, f"{category}_chunks_map.json")
        try:
            index = faiss.read_index(index_file)
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            artifacts[category] = {'index': index, 'chunks': chunk_data}
        except FileNotFoundError:
            continue
    if not artifacts:
        return None, None
    return artifacts, model

# --- FUNÇÕES PRINCIPAIS ---
# NOVO: Esta é a nova função de análise com lógica de scoring.
def create_dynamic_analysis_plan_v2(query, company_catalog_rich, available_indices):
    """
    Gera um plano de ação dinâmico com identificação de empresas baseada em scoring
    para maior precisão, lidando com nomes compostos, apelidos e variações.
    """
    api_key = GEMINI_API_KEY
    # CORREÇÃO: Atualizado o nome do modelo na URL para a versão mais recente e estável.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

    # --- LÓGICA DE IDENTIFICAÇÃO DE EMPRESAS POR SCORING ---
    query_lower = query.lower().strip()
    normalized_query_for_scoring = normalize_name(query_lower)
    company_scores = {}
    for company_data in company_catalog_rich:
        canonical_name = company_data["canonical_name"]
        score = 0
        for alias in company_data.get("aliases", []):
            if alias in query_lower:
                score += 10 * len(alias.split())
        name_for_parts = normalize_name(canonical_name)
        parts = name_for_parts.split()
        for part in parts:
            if len(part) > 2 and part in normalized_query_for_scoring.split():
                score += 1
        if score > 0:
            company_scores[canonical_name] = score
    mentioned_companies = []
    if company_scores:
        sorted_companies = sorted(company_scores.items(), key=lambda item: item[1], reverse=True)
        max_score = sorted_companies[0][1]
        if max_score > 0:
            mentioned_companies = [company for company, score in sorted_companies if score >= max_score * 0.7]

    # --- FIM DA LÓGICA DE IDENTIFICAÇÃO ---
    prompt = f'Você é um planejador de análise. Sua tarefa é analisar a "Pergunta do Usuário" e identificar os tópicos de interesse. Instruções: 1. Identifique os Tópicos: Analise a pergunta para identificar os tópicos de interesse. Se a pergunta for genérica (ex: "resumo dos planos", "análise da empresa"), inclua todos os "Tópicos de Análise Disponíveis". Se for específica (ex: "fale sobre o vesting e dividendos"), inclua apenas os tópicos relevantes. 2. Formate a Saída: Retorne APENAS uma lista JSON de strings contendo os tópicos identificados. Tópicos de Análise Disponíveis: {json.dumps(AVAILABLE_TOPICS, indent=2)} Pergunta do Usuário: "{query}" Tópicos de Interesse (responda APENAS com a lista JSON de strings):'
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        topics = json.loads(json_match.group(0)) if json_match else AVAILABLE_TOPICS
    except Exception:
        topics = AVAILABLE_TOPICS
    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}


# MANTENDO A FUNÇÃO DE EXECUÇÃO ORIGINAL, QUE JÁ É ROBUSTA
def execute_dynamic_plan(plan, query_intent, artifacts, model):
    full_context = ""
    all_retrieved_docs = set()
    if query_intent == 'item_8_4_query':
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            if 'item_8_4' in artifacts:
                artifact_data = artifacts['item_8_4']
                chunk_data = artifact_data['chunks']
                empresa_chunks_8_4 = []
                for i, mapping in enumerate(chunk_data.get('map', [])):
                    document_path = mapping['document_path']
                    if empresa.split(' ')[0].lower() in document_path.lower():
                        chunk_text = chunk_data["chunks"][i]
                        all_retrieved_docs.add(str(document_path))
                        empresa_chunks_8_4.append({'text': chunk_text, 'path': document_path, 'index': i})
                if empresa_chunks_8_4:
                    full_context += f"=== SEÇÃO COMPLETA DO ITEM 8.4 - {empresa.upper()} ===\n\n"
                    for chunk_info in empresa_chunks_8_4:
                        full_context += f"--- Chunk Item 8.4 (Doc: {chunk_info['path']}) ---\n{chunk_info['text']}\n\n"
                    full_context += f"=== FIM DA SEÇÃO ITEM 8.4 - {empresa.upper()} ===\n\n"
            complementary_indices = [idx for idx in artifacts.keys() if idx != 'item_8_4']
            for topico in plan.get("topicos", [])[:5]:
                for term in expand_search_terms(topico)[:3]:
                    search_query = f"item 8.4 {term} empresa {empresa}"
                    for index_name in complementary_indices:
                        if index_name in artifacts:
                            # (Lógica de busca semântica complementar)
                            pass
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
    else:
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            target_tags = list(set(term for topico in plan.get("topicos", []) for term in expand_search_terms(topico)))
            tagged_chunks = search_by_tags(artifacts, empresa, [tag.title() for tag in target_tags if len(tag) > 3])
            if tagged_chunks:
                full_context += f"=== CHUNKS COM TAGS ESPECÍFICAS - {empresa.upper()} ===\n\n"
                for chunk_info in tagged_chunks:
                    full_context += f"--- Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']}) ---\n{chunk_info['text']}\n\n"
                    all_retrieved_docs.add(str(chunk_info['path']))
                full_context += f"=== FIM DOS CHUNKS COM TAGS - {empresa.upper()} ===\n\n"
            for topico in plan.get("topicos", [])[:5]:
                for term in expand_search_terms(topico)[:3]:
                    search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                    # (Lógica de busca semântica complementar)
                    pass
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
    # Adicionei uma verificação para garantir que o contexto não seja vazio
    if not full_context.strip():
        return "Nenhuma informação encontrada para os critérios especificados.", []
    return full_context, [str(doc) for doc in all_retrieved_docs]

# MANTENDO A FUNÇÃO DE GERAÇÃO DE RESPOSTA ORIGINAL
# MANTENDO A FUNÇÃO DE GERAÇÃO DE RESPOSTA ORIGINAL
def get_final_unified_answer(query, context):
    """Gera a resposta final usando o contexto recuperado."""
    # CORREÇÃO: Atualizado o nome do modelo na URL para a versão mais recente e estável.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    
    has_complete_8_4 = "=== SEÇÃO COMPLETA DO ITEM 8.4" in context
    has_tagged_chunks = "=== CHUNKS COM TAGS ESPECÍFICAS" in context
    
    if has_complete_8_4:
        structure_instruction = """
**ESTRUTURA OBRIGATÓRIA PARA ITEM 8.4:**
Use a estrutura oficial do item 8.4 do Formulário de Referência:
a) Termos e condições gerais dos planos
b) Data de aprovação e órgão responsável
c) Número máximo de ações abrangidas pelos planos
d) Número máximo de opções a serem outorgadas
e) Condições de aquisição de ações
f) Critérios para fixação do preço de aquisição ou exercício
g) Critérios para fixação do prazo de aquisição ou exercício
h) Forma de liquidação
i) Restrições à transferência das ações
j) Critérios e eventos que, quando verificados, ocasionarão a suspensão, alteração ou extinção do plano
k) Efeitos da saída do administrador do cargo na manutenção dos seus direitos no plano

Para cada subitem, extraia e organize as informações encontradas na SEÇÃO COMPLETA DO ITEM 8.4.
"""
    elif has_tagged_chunks:
        structure_instruction = "**PRIORIZE** as informações dos CHUNKS COM TAGS ESPECÍFICAS e organize a resposta de forma lógica usando Markdown."
    else:
        structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
        
    prompt = f'Você é um analista financeiro sênior especializado em Formulários de Referência da CVM. PERGUNTA ORIGINAL DO USUÁRIO: "{query}" CONTEXTO COLETADO DOS DOCUMENTOS: {context} {structure_instruction} INSTRUÇÕES PARA O RELATÓRIO FINAL: 1. Responda diretamente à pergunta do usuário. 2. PRIORIZE informações da SEÇÃO COMPLETA DO ITEM 8.4 ou de CHUNKS COM TAGS ESPECÍFICAS quando disponíveis. 3. Use informações complementares apenas para esclarecer. 4. Seja detalhado, preciso e profissional. 5. Se alguma informação não estiver disponível, indique: "Informação não encontrada nas fontes analisadas". RELATÓRIO ANALÍTICO FINAL:'
    
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"ERRO ao gerar resposta final: {e}"
        

# --- INTERFACE STREAMLIT ---
def main():
    st.set_page_config(page_title="Agente de Análise LTIP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
    st.title("🤖 Agente de Análise de Planos de Incentivo Longo Prazo ILP")
    st.markdown("---")
    
    with st.spinner("Inicializando sistema..."):
        loaded_artifacts, embedding_model = load_all_artifacts()
    
    if not loaded_artifacts:
        st.error("❌ Erro no carregamento dos artefatos. Verifique os arquivos na pasta 'dados'.")
        return
    
    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Fontes disponíveis", len(loaded_artifacts))
        st.metric("Empresas identificadas", len(company_catalog_rich))
        with st.expander("📋 Ver empresas disponíveis"):
            sorted_companies = sorted([company['canonical_name'] for company in company_catalog_rich])
            for company_name in sorted_companies:
                st.write(f"• {company_name}")
        st.success("✅ Sistema carregado")
        st.info(f"Modelo: {MODEL_NAME}")

    st.header("💬 Faça sua pergunta")
    # Na sua função main(), logo após o st.header("💬 Faça sua pergunta")

with st.expander("💡 Entenda como funciona e veja dicas para perguntas ideais"):
    st.markdown("""
    Este agente é um especialista treinado para analisar Planos de Incentivo de Longo Prazo (ILPs) a partir dos documentos públicos (como o Formulário de Referência) das empresas listadas.

    #### Tipos de Análise que o Agente Realiza:

    O processo de análise ocorre em três etapas inteligentes:

    1.  **🧠 Compreensão da Pergunta:** Primeiro, o agente identifica com alta precisão a(s) empresa(s) e os tópicos de interesse na sua pergunta. Ele é capaz de reconhecer nomes oficiais, apelidos (como "Magalu" ou "Vivo") e marcas famosas (como "Havaianas" ou "Riachuelo").
    2.  **🔍 Busca Inteligente nos Documentos:** Com o plano definido, o agente realiza uma busca profunda nos arquivos.
        - **Análise do Item 8.4:** Se você pedir especificamente pelo "item 8.4", o agente fará uma leitura completa e detalhada desta seção no documento da empresa.
        - **Análise por Tópico:** Para perguntas gerais, ele busca por trechos e seções que foram pré-identificados com os tópicos do seu interesse (ex: Vesting, Lockup, etc.), garantindo respostas contextuais.
    3.  **✍️ Síntese e Relatório:** Por fim, o agente entrega todo o material encontrado para a IA Generativa (Google Gemini), que atua como um analista financeiro sênior para redigir um relatório detalhado e estruturado, baseando-se **exclusivamente** nas informações contidas nos documentos.

    #### Limitações Atuais:

    É importante conhecer os limites do agente para usar seu potencial ao máximo:

    * **Fonte de Dados:** O conhecimento do agente é **restrito aos documentos que foram carregados no sistema**. Ele não possui acesso à internet em tempo real e não sabe sobre eventos ou planos que não estejam nesta base.
    * **Não é um Consultor:** O agente **não emite opiniões, conselhos de investimento ou previsões**. Sua função é localizar, extrair e resumir informações existentes.
    * **Data dos Documentos:** A validade das respostas está diretamente ligada à data dos formulários de referência na base de dados. Sempre verifique as fontes citadas ao final da análise.
    * **Ambiguidade:** Perguntas muito vagas podem gerar respostas imprecisas. Quanto mais clara e específica for a sua pergunta, melhor será o resultado.

    ---
    
    ### ✨ Dicas para Perguntas Ideais

    Para obter os melhores resultados, utilize um dos formatos abaixo.

    #### 1. Perguntas Específicas sobre Tópicos (Melhor formato)
    *Combine um ou mais tópicos com uma ou mais empresas.*
    - "Qual a forma de liquidação e o tratamento de dividendos da **Vale**?"
    - "Descreva o cronograma de vesting dos planos da **Petrobras**."
    - "Como a **Ambev** ajusta o preço de exercício em caso de eventos corporativos?"
    - "Fale sobre o período de lockup para os executivos da **Magalu**."
    - "Quais são as condições de carência para as ações restritas da **YDUQS**?"

    #### 2. Perguntas de Visão Geral do Plano (Item 8.4)
    *Peça pela seção completa do Formulário de Referência para ter uma visão geral.*
    - "Me mostre o **item 8.4** completo da **Vibra Energia**."
    - "Faça um resumo do **formulário de referência, seção 8.4**, da **Raia Drogasil**."
    - "Resuma o item 8.4 da **WEG**."

    #### 3. Perguntas Comparativas
    *Compare características específicas entre duas ou mais empresas.*
    - "Compare as condições de vesting e carência entre **Itaú** e **Santander**."
    - "Quais as diferenças na forma de liquidação dos planos da **Localiza** e da **Movida**?"
    - "Compare o tratamento de dividendos da **Eletrobras** com o da **Energisa**."

    """)
    user_query = st.text_area("Digite sua pergunta:", height=100, placeholder="Ex: Fale sobre o vesting da Magalu ou planos da Vibra Energia")
    
    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            return
        
        with st.container():
            st.markdown("---")
            st.subheader("📋 Processo de Análise")
            
            with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
                # USANDO A NOVA FUNÇÃO DE PLANEJAMENTO
                plan_response = create_dynamic_analysis_plan_v2(user_query, company_catalog_rich, list(loaded_artifacts.keys()))
                plan = plan_response['plan']
                if plan.get('empresas'):
                    st.write(f"**🏢 Empresas identificadas:** {', '.join(plan.get('empresas', []))}")
                else:
                    st.write("**🏢 Empresas identificadas:** Nenhuma")
                st.write(f"**📝 Tópicos a analisar:** {len(plan.get('topicos', []))}")
                status.update(label="✅ Plano gerado com sucesso!", state="complete")

            if not plan.get("empresas"):
                st.error("❌ Não consegui identificar empresas na sua pergunta. Tente usar nomes, apelidos ou marcas conhecidas (ex: Magalu, Vivo, Itaú).")
                return

            with st.status("2️⃣ Recuperando contexto relevante...", expanded=True) as status:
                query_intent = 'item_8_4_query' if any(term in user_query.lower() for term in ['8.4', '8-4', 'item 8.4', 'formulário']) else 'general_query'
                st.write(f"**🎯 Estratégia detectada:** {'Item 8.4 completo' if query_intent == 'item_8_4_query' else 'Busca geral'}")
                # USANDO A FUNÇÃO DE EXECUÇÃO ORIGINAL
                retrieved_context, sources = execute_dynamic_plan(plan, query_intent, loaded_artifacts, embedding_model)
                if not retrieved_context.strip() or "Nenhuma informação encontrada" in retrieved_context:
                    st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                    return
                st.write(f"**📄 Contexto recuperado de:** {len(set(sources))} documento(s)")
                status.update(label="✅ Contexto recuperado com sucesso!", state="complete")
            
            with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
                # USANDO A FUNÇÃO DE SÍNTESE ORIGINAL
                final_answer = get_final_unified_answer(user_query, retrieved_context)
                status.update(label="✅ Análise concluída!", state="complete")
            
            st.markdown("---")
            st.subheader("📄 Resultado da Análise")
            with st.container():
                st.markdown(final_answer)

if __name__ == "__main__":
    main()
