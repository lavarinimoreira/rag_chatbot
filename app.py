"""
Aplicação Streamlit para Chat com Documentos (RAG)
--------------------------------------------------
Este script implementa um fluxo completo de *Retrieval‑Augmented Generation* (RAG).
O usuário faz upload de PDFs, o conteúdo é vetorizado com ChromaDB + OpenAI Embeddings
e uma LLM da OpenAI responde perguntas usando o contexto recuperado.

Bibliotecas principais envolvidas
---------------------------------
- **Streamlit**: constrói a interface web.
- **LangChain**: orquestra carregamento, chunking, embeddings e cadeias de geração.
- **Chroma**: armazena vetores localmente (persistência em disco).
- **decouple**: lê a variável de ambiente OPENAI_API_KEY de um arquivo `.env`.
"""

# ------------- Imports padrão e de terceiros -------------
import os
import tempfile

import streamlit as st  # Interface web
from decouple import config  # Lê variáveis do .env

# LangChain – construção da cadeia RAG
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ------------- Configurações globais -------------
# API Key lida do arquivo .env e injetada na variável de ambiente requerida pela OpenAI
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

# Pasta onde o banco vetorial em Chroma será salvo / lido
persist_directory = "db"


# ------------- Funções utilitárias -------------


def process_pdf(file):
    """
    Converte um arquivo PDF enviado via Streamlit em *chunks* de texto.

    Etapas:
    1. Salva o arquivo de upload em um arquivo temporário (o loader precisa de um path).
    2. Usa PyPDFLoader para extrair o texto.
    3. Fragmenta o texto em pedaços sobrepostos para manter a coerência
       (chunk_size=1000 tokens, chunk_overlap=400).
    4. Remove o arquivo temporário e devolve a lista de documentos fragmentados.
    """
    # Salva o upload em disco para ser lido pelo loader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    # Extrai o texto completo de cada página
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Limpeza do arquivo temporário
    os.remove(temp_file_path)

    # Fragmenta o texto para melhorar a granularidade das buscas
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_splitter.split_documents(documents=docs)
    return chunks


def load_existing_vector_store():
    """
    Verifica se já existe um banco vetorial gravado em `persist_directory`.
    Caso exista, abre o Chroma em modo de leitura (sem recriar embeddings).
    """
    if os.path.exists(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    return None


def add_to_vector_store(chunks, vector_store=None):
    """
    Adiciona novos documentos ao banco vetorial existente ou cria um novo.

    Parâmetros
    ----------
    chunks : list
        Lista de objetos `Document` prontos para indexação.
    vector_store : Chroma | None
        Instância existente, se houver, para reaproveitar o índice.
    """
    if vector_store:
        # Incremental: adiciona somente os novos chunks
        vector_store.add_documents(chunks)
    else:
        # Primeira vez: cria a base completa e persiste em disco
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store


def ask_question(model, query, vector_store):
    """
    Executa a cadeia RAG:
    1. Recupera os documentos mais relevantes via o *retriever* do Chroma.
    2. Repassa o contexto recuperado para a LLM selecionada (ChatOpenAI).
    3. Retorna apenas a resposta gerada (campo 'answer').

    O prompt inclui:
    - Mensagem *system* com instruções sobre formato das respostas.
    - Histórico de chat armazenado em `st.session_state.messages`.
    - A pergunta atual do usuário.
    """
    llm = ChatOpenAI(model=model)

    # Estratégia de busca: embed + KNN
    retriever = vector_store.as_retriever()

    # Prompt base que orienta a LLM a usar o contexto
    system_prompt = """
    Use o contexto para responder às perguntas.
    Se não encontrar uma resposta no contexto,
    explique que não há informações disponíveis.
    Responda em formato de markdown e, quando útil,
    use visualizações interativas.
    Contexto: {context}
    """

    # Construção do histórico para manter a conversa
    messages = [("system", system_prompt)]
    for message in st.session_state.messages:
        messages.append((message["role"], message["content"]))
    messages.append(("human", "{input}"))  # Placeholder será preenchido pelo Chain

    prompt = ChatPromptTemplate.from_messages(messages)

    # Cadeias LangChain: combinação de recuperação + geração
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    # Execução e retorno somente do texto final
    response = chain.invoke({"input": query})
    return response.get("answer")


# ------------- Inicialização do banco vetorial (se existir) -------------
vector_store = load_existing_vector_store()


# ------------- Configuração da interface Streamlit -------------
st.set_page_config(
    page_title="Chat PyGPT",
    page_icon="📄",  # Ícone exibido na aba do navegador
)
st.header("🤖 Chat com seus documentos (RAG)")

# ---------- Barra lateral: upload de PDFs e escolha de modelo ----------
with st.sidebar:
    st.header("Upload de arquivos 📄")
    uploaded_files = st.file_uploader(
        label="Faça o upload de arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # Processa arquivos assim que forem enviados
    if uploaded_files:
        with st.spinner("Processando documentos..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(uploaded_file)
                all_chunks.extend(chunks)

            # Atualiza ou cria o banco vetorial com os novos documentos
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )

    # Seleção dinâmica de modelos da OpenAI disponíveis
    model_options = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o-mini",
        "gpt-4o",
    ]
    selected_model = st.selectbox(
        label="Selecione o modelo LLM",
        options=model_options,
    )

# ------------- Sessão de chat propriamente dita -------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Caixa de entrada unificada do chat
question = st.chat_input("Como posso ajudar?")

# Executa somente se há um banco vetorial carregado e a pergunta foi feita
if vector_store and question:
    # Renderiza o histórico na interface antes de processar a nova pergunta
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    # Adiciona a pergunta atual ao histórico
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Buscando resposta..."):
        # Chama a cadeia RAG e exibe a resposta
        response = ask_question(
            model=selected_model,
            query=question,
            vector_store=vector_store,
        )
        st.chat_message("ai").write(response)
        st.session_state.messages.append({"role": "ai", "content": response})
