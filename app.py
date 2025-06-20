# app.py
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

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="🚀 Agente de Análise LTIP",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURAÇÕES ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# CORREÇÃO: Caminho correto para a pasta data do repositório
GOOGLE_DRIVE_PATH = os.path.join(os.getcwd(), 'data')
TOP_K_SEARCH = 7

# Dicionário de termos técnicos (mantido do código original)
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
    "termos e condições gerais", "data de aprovação e órgão responsável",
    "número máximo de ações abrangidas", "número máximo de opções a serem outorgadas",
    "condições de aquisição de ações", "critérios para fixação do preço de aquisição ou exercício",
    "preço de exercício", "strike price", "critérios para fixação do prazo de aquisição ou exercício", 
    "forma de liquidação", "liquidação", "pagamento", "restrições à transferência das ações", 
    "critérios e eventos de suspensão/extinção", "efeitos da saída do administrador", 
    "Tipos de Planos", "Condições de Carência", "Vesting", "período de carência", 
    "cronograma de vesting", "Matching", "contrapartida", "co-investimento",
    "Lockup", "período de lockup", "restrição de venda", "Tratamento de Dividendos", 
    "equivalente em dividendos", "proventos", "Stock Options", "opções de ações", "SOP",
    "Ações Restritas", "RSU", "restricted shares", "Eventos Corporativos", 
    "IPO", "grupamento", "desdobramento"
]

# --- FUNÇÃO SEGURA PARA API ---
def safe_api_call(url, payload, headers, timeout=90):
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        return None, f"Erro de API com código {e.response.status_code}: {e.response.reason}"
    except requests.exceptions.RequestException:
        return None, "Erro de conexão. Verifique sua internet."

# --- FUNÇÕES PRINCIPAIS ---
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
                        results.append({
                            'text': chunk_text, 'path': document_path, 'index': i,
                            'source': index_name, 'tag_found': tag
                        })
                        st.write(f"     -> Encontrado chunk com tag '{tag}' em {document_path}")
                        break
    return results

@st.cache_resource
def load_all_artifacts():
    """Carrega artefatos da pasta data do repositório"""
    artifacts = {}
    canonical_company_names = set()
    
    # VERIFICAÇÃO: Confirma se a pasta data existe
    if not os.path.exists(GOOGLE_DRIVE_PATH):
        st.error(f"❌ ERRO CRÍTICO: Pasta 'data' não encontrada em: {GOOGLE_DRIVE_PATH}")
        st.info("Certifique-se de que a pasta 'data' está no mesmo diretório do app.py")
        return None, None, None
    
    st.info("📁 Carregando artefatos da pasta 'data' do repositório...")
    
    with st.spinner("Carregando modelo de embedding..."):
        model = SentenceTransformer(MODEL_NAME)
    
    # BUSCA: Procura pelos arquivos FAISS na pasta data
    index_files = glob.glob(os.path.join(GOOGLE_DRIVE_PATH, '*_faiss_index.bin'))
    
    if not index_files:
        st.error(f"❌ NENHUM arquivo *_faiss_index.bin encontrado em: {GOOGLE_DRIVE_PATH}")
        st.info("Arquivos esperados: item_8_4_faiss_index.bin, plano_remuneracao_faiss_index.bin")
        return None, None, None
    
    st.success(f"✅ Encontrados {len(index_files)} arquivo(s) FAISS:")
    for file in index_files:
        st.write(f"  - {os.path.basename(file)}")
    
    progress_bar = st.progress(0)
    total_files = len(index_files)
    
    for idx, index_file in enumerate(index_files):
        category = os.path.basename(index_file).replace('_faiss_index.bin', '')
        chunks_file = os.path.join(GOOGLE_DRIVE_PATH, f"{category}_chunks_map.json")
        
        try:
            st.info(f"Carregando '{category}'...")
            
            # Carrega o índice FAISS
            index = faiss.read_index(index_file)
            
            # Carrega os chunks correspondentes
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            artifacts[category] = {'index': index, 'chunks': chunk_data}
            
            # Extrai nomes das empresas
            for mapping in chunk_data.get('map', []):
                company_name = mapping['document_path'].split('/')[0]
                canonical_company_names.add(company_name)
            
            progress_bar.progress((idx + 1) / total_files)
            
        except FileNotFoundError:
            st.error(f"❌ Arquivo '{chunks_file}' não encontrado!")
            continue
        except Exception as e:
            st.error(f"❌ Erro ao carregar '{category}': {e}")
            continue
    
    if not artifacts:
        st.error("❌ NENHUM artefato foi carregado com sucesso!")
        return None, None, None
    
    st.success(f"✅ {len(artifacts)} categorias carregadas: {list(artifacts.keys())}")
    st.success(f"✅ {len(canonical_company_names)} empresas identificadas")
    
    return artifacts, model, list(canonical_company_names)

def create_dynamic_analysis_plan(query, company_catalog, available_indices):
    """Cria plano de análise usando API do Gemini"""
    
    # ACESSO SEGURO À API KEY
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = st.session_state.get('gemini_api_key')
        if not api_key:
            st.error("❌ API Key do Gemini não configurada!")
            return {"status": "error", "message": "API Key não encontrada"}
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # IDENTIFICAÇÃO DE EMPRESAS (lógica robusta do código original)
    mentioned_companies = []
    query_clean = query.lower().strip()
    
    st.write(f"🔍 Buscando empresas na query: '{query_clean}'")
    
    for canonical_name in company_catalog:
        # Busca por substring
        if canonical_name.lower() in query_clean:
            mentioned_companies.append(canonical_name)
            st.write(f"   ✅ Encontrada: {canonical_name}")
            continue
        
        # Busca por partes do nome
        company_parts = canonical_name.split(' ')
        for part in company_parts:
            if len(part) > 2 and part.lower() in query_clean:
                if canonical_name not in mentioned_companies:
                    mentioned_companies.append(canonical_name)
                    st.write(f"   ✅ Encontrada por parte '{part}': {canonical_name}")
                break
    
    # CHAMADA PARA ANÁLISE DE TÓPICOS
    prompt = f"""
Você é um planejador de análise. Analise a pergunta e identifique os tópicos de interesse.

**Instruções:**
- Se for pergunta genérica (ex: "resumo dos planos"), inclua TODOS os tópicos
- Se for específica (ex: "vesting da empresa X"), inclua apenas tópicos relevantes
- Retorne APENAS uma lista JSON de strings

**Tópicos Disponíveis:** {json.dumps(AVAILABLE_TOPICS, indent=2)}

**Pergunta:** "{query}"

**Tópicos (JSON apenas):**
"""
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    response_data, error_message = safe_api_call(url, payload, headers)
    
    if error_message:
        st.warning(f"⚠️ {error_message}")
        plan = {"empresas": mentioned_companies, "topicos": AVAILABLE_TOPICS}
        return {"status": "success", "plan": plan}
    
    try:
        text_response = response_data['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        if json_match:
            topics = json.loads(json_match.group(0))
            plan = {"empresas": mentioned_companies, "topicos": topics}
            return {"status": "success", "plan": plan}
        else:
            plan = {"empresas": mentioned_companies, "topicos": AVAILABLE_TOPICS}
            return {"status": "success", "plan": plan}
    except Exception as e:
        st.error(f"❌ Erro ao processar resposta: {e}")
        plan = {"empresas": mentioned_companies, "topicos": AVAILABLE_TOPICS}
        return {"status": "success", "plan": plan}

def execute_dynamic_plan(plan, query_intent, artifacts, model):
    """Executa o plano de busca (lógica do código original)"""
    full_context = ""
    all_retrieved_docs = set()
    
    if query_intent == 'item_8_4_query':
        st.write("📋 Estratégia: BUSCA EXAUSTIVA ITEM 8.4")
        
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            
            # BUSCA EXAUSTIVA NO ITEM 8.4
            if 'item_8_4' in artifacts:
                artifact_data = artifacts['item_8_4']
                chunk_data = artifact_data['chunks']
                
                chunks_8_4 = []
                for i, mapping in enumerate(chunk_data.get('map', [])):
                    document_path = mapping['document_path']
                    if re.search(re.escape(empresa.split(' ')[0]), document_path, re.IGNORECASE):
                        chunk_text = chunk_data["chunks"][i]
                        all_retrieved_docs.add(str(document_path))
                        chunks_8_4.append({'text': chunk_text, 'path': document_path})
                
                st.write(f"   📄 {len(chunks_8_4)} chunks do Item 8.4 para {empresa}")
                
                full_context += f"=== SEÇÃO COMPLETA DO ITEM 8.4 - {empresa.upper()} ===\n\n"
                for chunk in chunks_8_4:
                    full_context += f"--- Chunk Item 8.4 (Doc: {chunk['path']}) ---\n{chunk['text']}\n\n"
                full_context += f"=== FIM DA SEÇÃO ITEM 8.4 - {empresa.upper()} ===\n\n"
    
    else:
        st.write("📋 Estratégia: BUSCA GERAL COM TAGS")
        
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            
            # BUSCA POR TAGS
            target_tags = []
            for topico in plan.get("topicos", []):
                target_tags.extend(expand_search_terms(topico))
            
            target_tags = list(set([tag.title() for tag in target_tags if len(tag) > 3]))
            st.write(f"   🏷️ {len(target_tags)} tags para {empresa}")
            
            tagged_chunks = search_by_tags(artifacts, empresa, target_tags)
            
            if tagged_chunks:
                full_context += f"=== CHUNKS COM TAGS ESPECÍFICAS - {empresa.upper()} ===\n\n"
                for chunk in tagged_chunks:
                    full_context += f"--- Tag '{chunk['tag_found']}' (Doc: {chunk['path']}) ---\n{chunk['text']}\n\n"
                    all_retrieved_docs.add(str(chunk['path']))
                full_context += f"=== FIM DOS CHUNKS COM TAGS - {empresa.upper()} ===\n\n"
    
    return full_context, [str(doc) for doc in all_retrieved_docs]

def get_final_unified_answer(query, context):
    """Gera resposta final usando API do Gemini"""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = st.session_state.get('gemini_api_key')
        if not api_key:
            return "❌ API Key não configurada para gerar resposta!"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # Detecta tipo de análise
    has_complete_8_4 = "=== SEÇÃO COMPLETA DO ITEM 8.4" in context
    has_tagged_chunks = "=== CHUNKS COM TAGS ESPECÍFICAS" in context
    
    if has_complete_8_4:
        structure_instruction = """
**ESTRUTURA PARA ITEM 8.4:**
Organize seguindo a estrutura oficial:
a) Termos e condições gerais
b) Data de aprovação e órgão responsável  
c) Número máximo de ações abrangidas
d) Número máximo de opções a serem outorgadas
e) Condições de aquisição de ações
f) Critérios para fixação do preço de aquisição ou exercício
g) Critérios para fixação do prazo de aquisição ou exercício
h) Forma de liquidação
i) Restrições à transferência das ações
j) Critérios e eventos de suspensão, alteração ou extinção
k) Efeitos da saída do administrador
"""
    else:
        structure_instruction = "Organize a resposta de forma lógica usando Markdown."
    
    prompt = f"""
Você é um analista financeiro especializado em Formulários de Referência da CVM.

**PERGUNTA:** "{query}"

**CONTEXTO DOS DOCUMENTOS:**
{context}

{structure_instruction}

**INSTRUÇÕES:**
1. Responda diretamente à pergunta
2. **PRIORIZE** informações da SEÇÃO COMPLETA DO ITEM 8.4 quando disponível
3. **PRIORIZE** informações dos CHUNKS COM TAGS ESPECÍFICAS quando disponível
4. Seja detalhado, preciso e profissional
5. Transcreva dados importantes (valores, datas, percentuais)
6. Se informação não disponível: "Informação não encontrada nas fontes"

**RELATÓRIO FINAL:**
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}
    }
    headers = {'Content-Type': 'application/json'}
    
    response_data, error_message = safe_api_call(url, payload, headers, timeout=180)
    
    if error_message:
        return f"❌ Erro ao gerar resposta: {error_message}"
    
    try:
        return response_data['candidates'][0]['content']['parts'][0]['text'].strip()
    except:
        return "❌ Erro ao processar resposta final"

# --- INTERFACE STREAMLIT ---
def main():
    st.title("🚀 Agente de Análise LTIP")
    st.markdown("**Análise de Formulários de Referência da CVM**")
    
    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # API Key
        gemini_api_key = st.text_input("🔑 API Key Gemini", type="password")
        if gemini_api_key:
            st.session_state['gemini_api_key'] = gemini_api_key
            st.success("✅ API Key configurada!")
        
        # Info da pasta data
        st.subheader("📁 Arquivos")
        st.info(f"Pasta: {GOOGLE_DRIVE_PATH}")
        
        if st.button("🔄 Recarregar"):
            st.cache_resource.clear()
            st.rerun()
    
    # CARREGAMENTO
    if 'loaded_artifacts' not in st.session_state:
        if not st.session_state.get('gemini_api_key') and 'GEMINI_API_KEY' not in st.secrets:
            st.warning("⚠️ Configure a API Key do Gemini na barra lateral")
            return
        
        artifacts, model, company_catalog = load_all_artifacts()
        
        if artifacts is None:
            st.error("❌ Falha no carregamento dos artefatos")
            return
        
        st.session_state['loaded_artifacts'] = artifacts
        st.session_state['embedding_model'] = model
        st.session_state['company_catalog'] = company_catalog

    # INTERFACE PRINCIPAL
    loaded_artifacts = st.session_state['loaded_artifacts']
    company_catalog = st.session_state['company_catalog']
    
    # Métricas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Categorias", len(loaded_artifacts))
    with col2:
        st.metric("🏢 Empresas", len(company_catalog))
    
    # Empresas disponíveis
    with st.expander("🏢 Ver Empresas Disponíveis"):
        for empresa in sorted(company_catalog):
            st.write(f"• {empresa}")
    
    st.divider()
    
    # EXEMPLOS
    st.subheader("💡 Exemplos")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Item 8.4 Vibra"):
            st.session_state['exemplo'] = "Descreva o item 8.4 da Vibra"
    with col2:
        if st.button("⚡ Vesting CCR"):
            st.session_state['exemplo'] = "Como funciona o vesting da CCR?"
    with col3:
        if st.button("💰 Liquidação Vale"):
            st.session_state['exemplo'] = "Forma de liquidação da Vale"
    
    # CONSULTA
    st.subheader("💬 Sua Pergunta")
    user_query = st.text_area(
        "Digite aqui:",
        value=st.session_state.get('exemplo', ''),
        height=100
    )
    
    if 'exemplo' in st.session_state:
        del st.session_state['exemplo']
    
    if st.button("🔍 Analisar", type="primary", disabled=not user_query.strip()):
        try:
            # PLANEJAMENTO
            st.header("📋 Plano de Análise")
            with st.expander("Ver detalhes", expanded=True):
                plan_response = create_dynamic_analysis_plan(
                    user_query, company_catalog, list(loaded_artifacts.keys())
                )
                
                if plan_response['status'] != 'success':
                    st.error("❌ Erro no planejamento")
                    return
                
                plan = plan_response['plan']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🏢 Empresas:**")
                    for empresa in plan.get('empresas', []):
                        st.write(f"• {empresa}")
                with col2:
                    st.write("**📝 Tópicos:**")
                    for i, topico in enumerate(plan.get('topicos', [])[:5]):
                        st.write(f"{i+1}. {topico}")
                    if len(plan.get('topicos', [])) > 5:
                        st.write(f"... +{len(plan.get('topicos', [])) - 5} tópicos")
                
                if not plan.get("empresas"):
                    st.error("❌ Nenhuma empresa identificada")
                    return
            
            # DETECÇÃO DE INTENÇÃO
            query_intent = 'item_8_4_query' if any(term in user_query.lower() for term in 
                                                  ['8.4', '8-4', 'item 8.4', 'formulário']) else 'general_query'
            st.info(f"**Estratégia:** {query_intent}")
            
            # EXECUÇÃO
            st.header("🔍 Recuperação de Contexto")
            with st.expander("Ver busca", expanded=True):
                retrieved_context, sources = execute_dynamic_plan(
                    plan, query_intent, loaded_artifacts, st.session_state['embedding_model']
                )
                
                if not retrieved_context.strip():
                    st.warning("⚠️ Nenhuma informação relevante encontrada")
                    return
                
                st.success(f"✅ Contexto de {len(set(sources))} documento(s)")
            
            # RESPOSTA
            st.header("📄 Resposta")
            with st.spinner("Gerando resposta..."):
                final_answer = get_final_unified_answer(user_query, retrieved_context)
            
            st.markdown(final_answer)
            
            # FONTES
            st.divider()
            with st.expander(f"📚 Fontes ({len(set(sources))})"):
                for source in sorted(set(sources)):
                    st.write(f"• {source}")
                    
        except Exception as e:
            st.error(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
