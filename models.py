# models.py
# Remova a importação do streamlit se não for mais usada em nenhum outro lugar no arquivo.
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging

logger = logging.getLogger(__name__)

def get_embedding_model():
    logger.info("Carregando modelo de embedding: 'paraphrase-multilingual-mpnet-base-v2'")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    return model

def get_cross_encoder_model():
    logger.info("Carregando cross-encoder: 'cross-encoder/ms-marco-MiniLM-L-6-v2'")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return model
