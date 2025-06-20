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

# --- CONFIGURAÇÕES ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
AMBIGUITY_THRESHOLD = 3

# Configuração da API Key do Gemini
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Dicionário especializado para termos técnicos de LTIP
TERMOS_TECNICOS_LTIP = {
    "tratamento de dividendos": [
        "tratamento de dividendos", "equivalente em dividendos", "dividendos", 
        "juros sobre capital próprio", "proventos", "dividend equivalent",
        "dividendos pagos em ações", "ajustes por dividendos"
    ],
    "preço de exercício": [
        "preço de exercício", "strike price", "preço de compra", "preço fixo", 
        "valor de exercício", "preço pré-estabelecido", "preço de aquisição"
    ],
    "forma de liquidação": [
        "forma de liquidação", "liquidação", "pagamento", "entrega física", 
        "pagamento em dinheiro", "transferência de ações", "settlement"
    ],
    "vesting": [
        "vesting", "período de carência", "carência", "aquisição de direitos", 
        "cronograma de vesting", "vesting schedule", "período de cliff"
    ],
    "eventos corporativos": [
        "eventos corporativos", "desdobramento", "grupamento", "dividendos pagos em ações",
        "bonificação", "split", "ajustes", "reorganização societária"
    ],
    "stock options": [
        "stock options", "opções de ações", "opções de compra", "SOP", 
        "plano de opções", "ESOP", "opção de compra de ações"
    ],
    "ações restritas": [
        "ações restritas", "restricted shares", "RSU", "restricted stock units", 
        "ações com restrição", "plano de ações restritas"
    ]
}

# Tópicos expandidos de análise
AVAILABLE_TOPICS = [
    "termos e condições gerais", "data de aprovação e órgão responsável",
    "número máximo de ações abrangidas", "número máximo de opções a serem outorgadas",
    "condições de aquisição de ações", "critérios para fixação do preço de aquisição ou exercício",
    "preço de exercício", "strike price",
    "critérios para fixação do prazo de aquisição ou exercício", 
    "forma de liquidação", "liquidação", "pagamento",
    "restrições à transferência das ações", "critérios e eventos de suspensão/extinção",
    "efeitos da saída do administrador", "Tipos de Planos", "Condições de Carência", 
    "Vesting", "período de carência", "cronograma de vesting",
    "Matching", "contrapartida", "co-investimento",
    "Lockup", "período de lockup", "restrição de venda",
    "Tratamento de Dividendos", "equivalente em dividendos", "proventos",
    "Stock Options", "opções de ações", "SOP",
    "Ações Restritas", "RSU", "restricted shares",
    "Eventos Corporativos", "IPO", "grupamento", "desdobramento"
]

# --- FUNÇÕES AUXILIARES ---

def expand_search_terms(base_term):
    """Expande um termo base com sinônimos e variações técnicas."""
    expanded_terms = [base_term.lower()]
    
    for category, terms in TERMOS_TECNICOS_LTIP.items():
        if any(term.lower() in base_term.lower() for term in terms):
            expanded_terms.extend([term.lower() for term in terms])
    
    return list(set(expanded_terms))

def search_by_tags(artifacts, company_name, target_tags):
    """Busca chunks que contenham tags específicas para uma empresa."""
    results = []
    
    for index_name, artifact_data in artifacts.items():
        chunk_data = artifact_data['chunks']
        
        for i, mapping in enumerate(chunk_data.get('map', [])):
            document_path = mapping['document_path']
            if re.search(re.escape(company_name.split(' ')[0]), document_path, re.IGNORECASE):
                chunk_text = chunk_data["chunks"][i]
                
                # Verifica se o chunk contém as tags procuradas
                for tag in target_tags:
                    if f"Tópicos:" in chunk_text and tag in chunk_text:
                        results.append({
                            'text': chunk_text,
                            'path': document_path,
                            'index': i,
                            'source': index_name,
                            'tag_found': tag
                        })
                        break
    
    return results

def normalize_name(name):
    """Normaliza nomes de empresas para comparação."""
    try:
        nfkd_form = unicodedata.normalize('NFKD', name.lower())
        name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        name = re.sub(r'[.,-]', '', name)
        suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b']
        for suffix in suffixes:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)
        return re.sub(r'\s+', '', name).strip()
    except Exception as e:
        return name.lower()

# --- CACHE PARA CARREGAR ARTEFATOS ---

@st.cache_resource
def load_all_artifacts():
    """Carrega todos os artefatos e constrói um catálogo de nomes de empresas canônicos."""
    artifacts = {}
    canonical_company_names = set()
    
    # Carrega modelo de embedding
    model = SentenceTransformer(MODEL_NAME)
    
    # Busca arquivos na pasta dados
    dados_path = "dados"
    index_files = glob.glob(os.path.join(dados_path, '*_faiss_index.bin'))
    
    if not index_files:
        return None, None, None

    for index_file in index_files:
        category = os.path.basename(index_file).replace('_faiss_index.bin', '')
        chunks_file = os.path.join(dados_path, f"{category}_chunks_map.json")
        
        try:
            index = faiss.read_index(index_file)
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            artifacts[category] = {'index': index, 'chunks': chunk_data}
            
            for mapping in chunk_data.get('map', []):
                company_name = mapping['document_path'].split('/')[0]
                canonical_company_names.add(company_name)
                
        except FileNotFoundError:
            continue
    
    if not artifacts:
        return None, None, None

    return artifacts, model, list(canonical_company_names)

# --- FUNÇÕES PRINCIPAIS ---

def create_dynamic_analysis_plan(query, company_catalog, available_indices):
    """Gera um plano de ação dinâmico em JSON."""
    api_key = GEMINI_API_KEY
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # Identificação robusta de empresas
    mentioned_companies = []
    query_clean = query.lower().strip()
    
    for canonical_name in company_catalog:
        found = False
        
        # Busca exata
        if canonical_name.lower() == query_clean:
            mentioned_companies.append(canonical_name)
            found = True
            continue
        
        # Busca por substring
        if canonical_name.lower() in query_clean:
            mentioned_companies.append(canonical_name)
            found = True
            continue
        
        # Busca por partes do nome
        company_parts = canonical_name.split(' ')
        for part in company_parts:
            if len(part) > 2 and re.search(r'\b' + re.escape(part.lower()) + r'\b', query_clean):
                if canonical_name not in mentioned_companies:
                    mentioned_companies.append(canonical_name)
                    found = True
                break
        
        if found:
            continue
        
        # Busca simplificada
        normalized_canonical = normalize_name(canonical_name)
        normalized_query = normalize_name(query_clean)
        
        if normalized_query and len(normalized_query) > 2:
            if normalized_query in normalized_canonical:
                mentioned_companies.append(canonical_name)
    
    # Busca mais agressiva se não encontrou empresas
    if not mentioned_companies:
        for canonical_name in company_catalog:
            if len(query_clean) <= 6:  # Provável sigla
                if query_clean.upper() in canonical_name.upper():
                    mentioned_companies.append(canonical_name)
    
    prompt = f"""
Você é um planejador de análise. Sua tarefa é analisar a "Pergunta do Usuário" e identificar os tópicos de interesse.

**Instruções:**
1. **Identifique os Tópicos:** Analise a pergunta para identificar os tópicos de interesse. Se a pergunta for genérica (ex: "resumo dos planos", "análise da empresa"), inclua todos os "Tópicos de Análise Disponíveis". Se for específica (ex: "fale sobre o vesting e dividendos"), inclua apenas os tópicos relevantes.
2. **Formate a Saída:** Retorne APENAS uma lista JSON de strings contendo os tópicos identificados.

**Tópicos de Análise Disponíveis:** {json.dumps(AVAILABLE_TOPICS, indent=2)}

**Pergunta do Usuário:** "{query}"

**Tópicos de Interesse (responda APENAS com a lista JSON de strings):**
"""
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        if json_match:
            topics = json.loads(json_match.group(0))
            plan = {"empresas": mentioned_companies, "topicos": topics}
            return {"status": "success", "plan": plan}
        else:
            plan = {"empresas": mentioned_companies, "topicos": AVAILABLE_TOPICS}
            return {"status": "success", "plan": plan}
    except Exception as e:
        plan = {"empresas": mentioned_companies, "topicos": AVAILABLE_TOPICS}
        return {"status": "success", "plan": plan}

def execute_dynamic_plan(plan, query_intent, artifacts, model):
    """Executa o plano de busca e recupera contexto relevante."""
    full_context = ""
    all_retrieved_docs = set()
    
    if query_intent == 'item_8_4_query':
        # Busca exaustiva no item 8.4
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            
            if 'item_8_4' in artifacts:
                artifact_data = artifacts['item_8_4']
                chunk_data = artifact_data['chunks']
                
                empresa_chunks_8_4 = []
                for i, mapping in enumerate(chunk_data.get('map', [])):
                    document_path = mapping['document_path']
                    if re.search(re.escape(empresa.split(' ')[0]), document_path, re.IGNORECASE):
                        chunk_text = chunk_data["chunks"][i]
                        all_retrieved_docs.add(str(document_path))
                        empresa_chunks_8_4.append({
                            'text': chunk_text,
                            'path': document_path,
                            'index': i
                        })
                
                full_context += f"=== SEÇÃO COMPLETA DO ITEM 8.4 - {empresa.upper()} ===\n\n"
                for chunk_info in empresa_chunks_8_4:
                    full_context += f"--- Chunk Item 8.4 (Doc: {chunk_info['path']}) ---\n"
                    full_context += f"{chunk_info['text']}\n\n"
                
                full_context += f"=== FIM DA SEÇÃO ITEM 8.4 - {empresa.upper()} ===\n\n"
            
            # Busca complementar
            complementary_indices = [idx for idx in artifacts.keys() if idx != 'item_8_4']
            
            for topico in plan.get("topicos", [])[:10]:
                expanded_terms = expand_search_terms(topico)
                
                for term in expanded_terms[:5]:
                    search_query = f"item 8.4 {term} empresa {empresa}"
                    
                    for index_name in complementary_indices:
                        if index_name in artifacts:
                            artifact_data = artifacts[index_name]
                            index = artifact_data['index']
                            chunk_data = artifact_data['chunks']
                            
                            query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                            scores, indices = index.search(query_embedding, 3)
                            
                            chunks_found = 0
                            for i, idx in enumerate(indices[0]):
                                if idx != -1 and idx < len(chunk_data.get("chunks", [])) and scores[0][i] > 0.5:
                                    document_path = chunk_data["map"][idx]['document_path']
                                    if re.search(re.escape(empresa.split(' ')[0]), document_path, re.IGNORECASE):
                                        chunk_text = chunk_data["chunks"][idx]
                                        
                                        chunk_hash = hash(chunk_text[:100])
                                        if chunk_hash not in all_retrieved_docs:
                                            all_retrieved_docs.add(chunk_hash)
                                            score = scores[0][i]
                                            full_context += f"--- Contexto COMPLEMENTAR para '{topico}' via '{term}' (Fonte: {index_name}, Score: {score:.3f}) ---\n{chunk_text}\n\n"
                                            chunks_found += 1
                            
                            if chunks_found > 0:
                                break
                    
                    if chunks_found > 0:
                        break
            
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
    
    else:
        # Busca geral com tags e expansão de termos
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            
            # Busca por tags específicas
            target_tags = []
            for topico in plan.get("topicos", []):
                expanded_terms = expand_search_terms(topico)
                target_tags.extend(expanded_terms)
            
            target_tags = list(set([tag.title() for tag in target_tags if len(tag) > 3]))
            
            tagged_chunks = search_by_tags(artifacts, empresa, target_tags)
            
            if tagged_chunks:
                full_context += f"=== CHUNKS COM TAGS ESPECÍFICAS - {empresa.upper()} ===\n\n"
                for chunk_info in tagged_chunks:
                    full_context += f"--- Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']}) ---\n"
                    full_context += f"{chunk_info['text']}\n\n"
                    all_retrieved_docs.add(str(chunk_info['path']))
                full_context += f"=== FIM DOS CHUNKS COM TAGS - {empresa.upper()} ===\n\n"
            
            # Busca semântica complementar
            indices_to_search = list(artifacts.keys())
            
            for topico in plan.get("topicos", []):
                expanded_terms = expand_search_terms(topico)
                
                for term in expanded_terms[:3]:
                    search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                    
                    chunks_found = 0
                    for index_name in indices_to_search:
                        if index_name in artifacts:
                            artifact_data = artifacts[index_name]
                            index = artifact_data['index']
                            chunk_data = artifact_data['chunks']
                            
                            query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                            scores, indices = index.search(query_embedding, TOP_K_SEARCH)
                            
                            for i, idx in enumerate(indices[0]):
                                if idx != -1 and scores[0][i] > 0.4:
                                    document_path = chunk_data["map"][idx]['document_path']
                                    if re.search(re.escape(empresa.split(' ')[0]), document_path, re.IGNORECASE):
                                        chunk_text = chunk_data["chunks"][idx]
                                        
                                        chunk_hash = hash(chunk_text[:100])
                                        if chunk_hash not in all_retrieved_docs:
                                            all_retrieved_docs.add(chunk_hash)
                                            score = scores[0][i]
                                            full_context += f"--- Contexto para '{topico}' via '{term}' (Fonte: {index_name}, Score: {score:.3f}) ---\n{chunk_text}\n\n"
                                            chunks_found += 1
                    
                    if chunks_found > 0:
                        break
            
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
    
    return full_context, [str(doc) for doc in all_retrieved_docs]

def get_final_unified_answer(query, context):
    """Gera a resposta final usando o contexto recuperado."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
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
    
    prompt = f"""
Você é um analista financeiro sênior especializado em Formulários de Referência da CVM. 

**PERGUNTA ORIGINAL DO USUÁRIO:**
"{query}"

**CONTEXTO COLETADO DOS DOCUMENTOS:**
{context}

{structure_instruction}

**INSTRUÇÕES PARA O RELATÓRIO FINAL:**
1. Responda diretamente à pergunta do usuário
2. **PRIORIZE** as informações da SEÇÃO COMPLETA DO ITEM 8.4 quando disponível
3. **PRIORIZE** as informações dos CHUNKS COM TAGS ESPECÍFICAS quando disponível
4. Use informações complementares apenas para esclarecer ou expandir pontos específicos
5. Seja detalhado, preciso e profissional
6. Transcreva dados importantes como valores, datas e percentuais
7. Se alguma informação não estiver disponível, indique: "Informação não encontrada nas fontes analisadas"
8. Mantenha a estrutura técnica apropriada para administradores de LTIP

**RELATÓRIO ANALÍTICO FINAL:**
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
        return f"ERRO ao gerar resposta final: {e}"

# --- INTERFACE STREAMLIT ---

def main():
    st.set_page_config(
        page_title="Agente de Análise LTIP",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Agente de Análise de Planos de Incentivo Longo Prazo ILP")
    st.markdown("---")
    
    # Carrega os artefatos
    with st.spinner("Inicializando sistema..."):
        loaded_artifacts, embedding_model, company_catalog = load_all_artifacts()
    
    if not loaded_artifacts:
        st.error("❌ Erro no carregamento dos artefatos. Verifique os arquivos na pasta 'dados'.")
        st.info("Certifique-se de que os arquivos FAISS (.bin) e chunks (.json) estão na pasta 'dados'.")
        return
    
    # Sidebar com informações
    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Fontes disponíveis", len(loaded_artifacts))
        st.metric("Empresas identificadas", len(company_catalog))
        
        with st.expander("📋 Ver empresas disponíveis"):
            for company in sorted(company_catalog):
                st.write(f"• {company}")
        
        with st.expander("📁 Ver fontes de dados"):
            for source in loaded_artifacts.keys():
                st.write(f"• {source}")
        
        st.markdown("---")
        st.markdown("### 🔧 Status do Sistema")
        st.success("✅ Sistema carregado")
        st.info(f"🤖 Modelo: {MODEL_NAME}")
    
    # Interface principal
    st.header("💬 Faça sua pergunta")
    
    # Exemplos de perguntas
    with st.expander("💡 Exemplos de perguntas"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Análises específicas:**
            - "Fale sobre o vesting e dividendos da CCR"
            - "Como funciona a liquidação na Vibra?"
            - "Quais são os critérios de exercício da Vale?"
            """)
        
        with col2:
            st.markdown("""
            **📋 Análises completas:**
            - "Mostre o item 8.4 completo da Vibra"
            - "Compare os planos entre CCR e Vibra"
            - "Resumo dos planos de stock options"
            """)
    
    # Input da pergunta
    user_query = st.text_area(
        "Digite sua pergunta sobre planos de incentivo:",
        height=100,
        placeholder="Ex: Fale sobre o vesting e dividendos da CCR",
        help="Seja específico sobre a empresa e o tópico de interesse"
    )
    
    # Botões de ação
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button("🔍 Analisar", type="primary", use_container_width=True)
    
    if analyze_button:
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            return
        
        # Processo de análise
        with st.container():
            st.markdown("---")
            st.subheader("📋 Processo de Análise")
            
            # Etapa 1: Planejamento
            with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
                plan_response = create_dynamic_analysis_plan(
                    user_query, company_catalog, list(loaded_artifacts.keys())
                )
                
                if plan_response['status'] != 'success':
                    st.error("❌ Erro ao gerar plano de análise")
                    return
                
                plan = plan_response['plan']
                
                if plan.get('empresas'):
                    st.write(f"**🏢 Empresas identificadas:** {', '.join(plan.get('empresas', []))}")
                else:
                    st.write("**🏢 Empresas identificadas:** Nenhuma")
                
                st.write(f"**📝 Tópicos a analisar:** {len(plan.get('topicos', []))}")
                
                if plan.get('topicos'):
                    with st.expander("Ver tópicos identificados"):
                        for i, topico in enumerate(plan.get('topicos', [])[:10], 1):
                            st.write(f"{i}. {topico}")
                
                status.update(label="✅ Plano gerado com sucesso!", state="complete")
            
            if not plan.get("empresas"):
                st.error("❌ Não consegui identificar empresas na sua pergunta. Seja mais específico.")
                st.info("💡 Dica: Mencione o nome da empresa claramente (ex: CCR, Vibra, Petrobras)")
                return
            
            # Etapa 2: Recuperação de contexto
            with st.status("2️⃣ Recuperando contexto relevante...", expanded=True) as status:
                query_intent = 'item_8_4_query' if any(term in user_query.lower() for term in ['8.4', '8-4', 'item 8.4', 'formulário']) else 'general_query'
                
                st.write(f"**🎯 Estratégia detectada:** {'Item 8.4 completo' if query_intent == 'item_8_4_query' else 'Busca geral'}")
                
                retrieved_context, sources = execute_dynamic_plan(
                    plan, query_intent, loaded_artifacts, embedding_model
                )
                
                if not retrieved_context.strip():
                    st.error("❌ Não encontrei informações relevantes nos documentos.")
                    return
                
                st.write(f"**📄 Contexto recuperado de:** {len(set(sources))} documento(s)")
                status.update(label="✅ Contexto recuperado com sucesso!", state="complete")
            
            # Etapa 3: Geração da resposta
            with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
                final_answer = get_final_unified_answer(user_query, retrieved_context)
                status.update(label="✅ Análise concluída!", state="complete")
        
        # Resultado final
        st.markdown("---")
        st.subheader("📄 Resultado da Análise")
        
        # Exibe a resposta em um container
        with st.container():
            st.markdown(final_answer)
        
        # Fontes consultadas
        if sources:
            st.markdown("---")
            with st.expander(f"📚 Documentos consultados ({len(set(sources))})", expanded=False):
                unique_sources = sorted(set(sources))
                for i, source in enumerate(unique_sources, 1):
                    st.write(f"{i}. {source}")
        
        # Botão para nova consulta
        st.markdown("---")
        if st.button("🔄 Nova Consulta", use_container_width=True):
            st.rerun()

if __name__ == "__main__":
    main()
