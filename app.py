import streamlit as st


st.set_page_config(page_title="Chat Gotalk", page_icon="🗨️")
st.header("Chat com seus documentos (RAG)")

# Navegação lateral para upload de arquivos
with st.sidebar:
    st.header("📂 Upload de arquivos")
    uploaded_files = st.file_uploader(
        label="Faça o upload de arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # Modelos disponíveis
    model_options = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o-mini",
        "gpt-4o",
    ]
    selected_model = st.sidebar.selectbox(
        label="Selecione o modelo LLM",
        options=model_options,
    )

# Componente visual do streamlit para receber input do usuário
question = st.chat_input("Como posso ajudar?")
