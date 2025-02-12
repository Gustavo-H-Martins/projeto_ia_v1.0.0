import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub

import faiss
import tempfile
import os
import time
import pandas as pd
import json

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

# Configurações do Streamlit
st.set_page_config(page_title="Converse com documentos 📚", page_icon="📚")
st.title("Converse com documentos 📚")

# Menu de seleção na barra lateral
model_class = st.sidebar.selectbox(
    "Escolha o modelo de IA:",
    options=["DeepSeek", "Hugging Face", "OpenAI", "Llama"],
    index=0  # Padrão: DeepSeek
)

# Mapeamento para o valor usado no código
model_mapping = {
    "Hugging Face": "hf_hub",
    "OpenAI": "openai",
    "Llama": "ollama",
    "DeepSeek": "deepseek"
}
model_class = model_mapping[model_class]

## Provedores de modelos
def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    llm = HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 512,
        }
    )
    return llm

def model_openai(model="gpt-4", temperature=0.1):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    )
    return llm

def model_ollama(model="phi3", temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm

def model_deepseek(model="deepseek-ai/deepseek-llm-7b-chat", temperature=0.1):
    llm = HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 512,
        }
    )
    return llm

# Indexação e recuperação
def config_retriever(uploads):
    # Carregar documentos
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        # Verificar o tipo de arquivo e carregar o conteúdo
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())
        elif file.name.endswith(".csv"):
            df = pd.read_csv(temp_filepath)
            text = df.to_string()
            docs.append({"page_content": text, "metadata": {"source": file.name}})
        elif file.name.endswith(".xlsx") or file.name.endswith(".xls"):
            df = pd.read_excel(temp_filepath)
            text = df.to_string()
            docs.append({"page_content": text, "metadata": {"source": file.name}})
        elif file.name.endswith(".txt"):
            with open(temp_filepath, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"page_content": text, "metadata": {"source": file.name}})
        elif file.name.endswith(".json"):
            with open(temp_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = json.dumps(data, indent=2)
            docs.append({"page_content": text, "metadata": {"source": file.name}})
    
    # Divisão em pedaços de texto / split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Armazenamento
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    # Configuração do retriever
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 4})
    return retriever

# Configuração da chain
def config_rag_chain(model_class, retriever):
    # Carregamento da LLM
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()
    elif model_class == "deepseek":
        llm = model_deepseek()

    # Para definição dos prompts
    if model_class.startswith("hf") or model_class == "deepseek":
        token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s, token_e = "", ""

    # Prompt de contextualização
    context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = "Question: {input}" + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt),
        ]
    )

    # Chain para contextualização
    history_aware_retriever = create_history_aware_retriever(llm=llm,
                                                             retriever=retriever,
                                                             prompt=context_q_prompt)
    
    # Prompt para perguntas e respostas (Q&A)
    qa_prompt_template = """Você é um assistente virtual prestativo e está respondendo perguntas gerais. 
    Use os seguintes pedaços de contexto recuperado para responder à pergunta. 
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa. 
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Configurar LLM e Chain para perguntas e respostas (Q&A)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain

# Criação de painel lateral na interface
uploads = st.sidebar.file_uploader(
    label="Enviar arquivos", 
    type=["pdf", "csv", "xlsx", "xls", "txt", "json"],
    accept_multiple_files=True
)
if not uploads:
    st.info("Por favor, envie algum arquivo para continuar")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou o seu assistente virtual! Como posso ajudar você?"),
    ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

start = time.time()
user_query = st.chat_input("Digite sua mensagem aqui...")

if user_query is not None and user_query != "" and uploads is not None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # Verificar se os documentos foram alterados ou se o retriever ainda não foi configurado
        if st.session_state.docs_list != uploads or st.session_state.retriever is None:
            st.session_state.docs_list = uploads
            st.session_state.retriever = config_retriever(uploads)
        
        # Verificar se o retriever foi configurado corretamente
        if st.session_state.retriever is None:
            st.error("Erro ao configurar o retriever. Por favor, tente novamente.")
            st.stop()
        
        # Configurar a chain RAG
        rag_chain = config_rag_chain(model_class, st.session_state.retriever)

        # Invocar a chain RAG
        result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})

        resp = result['answer']
        st.write(resp)

        # Mostrar a fonte
        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Página não especificada')

            # Fonte 1: documento.pdf - p. 2
            ref = f":link: Fonte {idx}: *{file} - p. {page}*"
            with st.popover(ref):
                st.caption(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=resp))

end = time.time()
print("Tempo: ", end - start)